#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.geometry import project_points, root_mean_square, to_homogeneous
from utils.media import PHOTO_SUFFIXES, image_paths, read_image, resize_to_fit, save_image


def project(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return project_points(points, homography, min_scale=1e-10)


def corners(image):
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)


def estimate_pair(source, target, ratio=0.75, threshold=3.0):
    detector = cv2.SIFT_create(nfeatures=8000)
    ks, ds = detector.detectAndCompute(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY), None)
    kt, dt = detector.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    if ds is None or dt is None or len(ds) < 4 or len(dt) < 4:
        raise ValueError('Too few features; use textured images with more overlap.')
    candidates = cv2.BFMatcher(cv2.NORM_L2).knnMatch(ds, dt, k=2)
    good = [pair[0] for pair in candidates if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance]
    if len(good) < 4:
        raise ValueError(f'Only {len(good)} ratio-test matches; at least four are needed.')
    ps = np.float64([ks[m.queryIdx].pt for m in good])
    pt = np.float64([kt[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(ps, pt, cv2.RANSAC, threshold, maxIters=10000, confidence=0.999)
    if H is None or mask is None or int(mask.sum()) < 4:
        raise ValueError('RANSAC failed to find a supported homography.')
    H = H / H[2, 2]
    inliers = mask.ravel().astype(bool)
    errors = np.linalg.norm(project(ps, H) - pt, axis=1)
    report = {
        'homography_source_to_target': H.tolist(),
        'keypoints_source': len(ks), 'keypoints_target': len(kt),
        'ratio_matches': len(good), 'inliers': int(inliers.sum()),
        'inlier_fraction': float(inliers.mean()),
        'inlier_rmse_px': root_mean_square(errors[inliers]),
        'inlier_median_error_px': float(np.median(errors[inliers])),
        'point_checks': [dict(source=ps[i].tolist(), observed_target=pt[i].tolist(),
                              projected_target=project(ps[i:i+1], H)[0].tolist(),
                              error_px=float(errors[i])) for i in np.flatnonzero(inliers)[:20]],
    }
    drawing = cv2.drawMatches(source, ks, target, kt, good, None,
                              matchesMask=inliers.astype(int).tolist(),
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return H, report, drawing


def canvas_geometry(images, transforms, max_pixels):
    all_corners = []
    for image, H in zip(images, transforms):
        c = corners(image)
        denominator = to_homogeneous(c) @ H[2, :]
        if np.min(denominator) <= 0 <= np.max(denominator):
            raise ValueError('A projected image crosses infinity; check matches and capture geometry.')
        all_corners.append(project(c, H))
    points = np.concatenate(all_corners)
    lower = np.floor(points.min(axis=0)).astype(np.int64)
    upper = np.ceil(points.max(axis=0)).astype(np.int64)
    width, height = (upper - lower + 1).tolist()
    if width <= 0 or height <= 0 or width * height > max_pixels:
        raise ValueError(f'Unsafe canvas {width} x {height}; inspect homographies or resize inputs.')
    translation = np.array([[1., 0., -lower[0]], [0., 1., -lower[1]], [0., 0., 1.]])
    return (width, height), translation, lower.tolist(), upper.tolist()


def inverse_warp(image, H, size):
    """Map canvas pixels back to source pixels; remap supplies bilinear sampling."""
    width, height = size
    inverse = np.linalg.inv(H)
    warped = np.zeros((height, width, 3), np.uint8)
    valid = np.zeros((height, width), np.uint8)
    # Chunk the grid so coordinate maps do not consume several full canvases.
    for start in range(0, height, 256):
        end = min(start + 256, height)
        x, y = np.meshgrid(np.arange(width, dtype=float), np.arange(start, end, dtype=float))
        denominator = inverse[2, 0] * x + inverse[2, 1] * y + inverse[2, 2]
        safe = np.abs(denominator) > 1e-10
        denominator = np.where(safe, denominator, 1.)
        sx = (inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2]) / denominator
        sy = (inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2]) / denominator
        inside = safe & (sx >= -1e-5) & (sx <= image.shape[1] - 1 + 1e-5) & (sy >= -1e-5) & (sy <= image.shape[0] - 1 + 1e-5)
        warped[start:end] = cv2.remap(image, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR)
        valid[start:end] = inside.astype(np.uint8) * 255
    return warped, valid


def stitch(paths, output, args):
    images = []
    for path in paths:
        image, _ = resize_to_fit(read_image(path), 1600)
        images.append(image)
    output.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(42)
    # Each adjacent matrix maps image i+1 into image i.
    chain = [np.eye(3)]
    pairs = []
    for i in range(len(images) - 1):
        H, report, drawing = estimate_pair(images[i+1], images[i])
        chain.append(chain[-1] @ H)
        report.update(source=paths[i+1].name, target=paths[i].name)
        pairs.append(report)
        save_image(output / f'matches_{i+1:02d}_to_{i:02d}.jpg', drawing)
    anchor = len(images) // 2
    reference_inverse = np.linalg.inv(chain[anchor])
    transforms = [reference_inverse @ H for H in chain]
    size, translation, lower, upper = canvas_geometry(images, transforms, 25000000)
    total = np.zeros((size[1], size[0], 3), np.float32)
    weights = np.zeros((size[1], size[0]), np.float32)
    image_reports = []
    for i, (path, image, H) in enumerate(zip(paths, images, transforms)):
        canvas_H = translation @ H
        warped, mask = inverse_warp(image, canvas_H, size)
        save_image(output / f'warped_{i:02d}.png', warped)
        # Padding ensures distance falls toward every edge, even at canvas borders.
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        weight = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        total += warped.astype(np.float32) * weight[:, :, None]
        weights += weight
        image_reports.append(dict(file=str(path), processed_shape=list(image.shape),
                                  homography_to_reference=H.tolist(),
                                  homography_to_canvas=canvas_H.tolist(),
                                  projected_corners=project(corners(image), canvas_H).tolist()))
    panorama = np.clip(total / np.maximum(weights[:, :, None], 1e-10), 0, 255).astype(np.uint8)
    save_image(output / 'panorama.png', panorama)
    report = dict(reference_index=anchor, canvas_width=size[0], canvas_height=size[1],
                  reference_bounds_min=lower, reference_bounds_max=upper,
                  parameters=vars(args), pairs=pairs, images=image_reports)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Saved {size[0]} x {size[1]} panorama and diagnostics to {output}')


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(root / 'images' / 'main'), help='Directory of ordered photos')
    parser.add_argument('--output', default=str(root / 'output' / 'main'))
    parser.add_argument('--extra-credit', action='store_true', help='Require at least ten images instead of four')
    args = parser.parse_args()
    directory = Path(args.input)
    paths = image_paths(directory, suffixes=PHOTO_SUFFIXES | {'.tif', '.tiff', '.bmp'})
    required = 10 if args.extra_credit else 4
    if len(paths) < required:
        parser.error(f'Need at least {required} images in {directory}; found {len(paths)}. Add photos with ordered filenames (01.jpg, 02.jpg, ...).')
    try:
        stitch(paths, Path(args.output), args)
    except (ValueError, OSError, cv2.error, np.linalg.LinAlgError) as error:
        parser.exit(1, f'Stitching failed: {error}\n')


if __name__ == '__main__':
    main()
