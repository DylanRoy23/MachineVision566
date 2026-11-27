#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.geometry import project_points, root_mean_square, to_homogeneous
from utils.media import open_video, resize_to_fit, save_image, video_frames

MAX_OVERLAP = 0.8   # assignment limit for any two neighbouring photos
FRAME_STEP = 3      # test every third video frame when picking frames
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def project(points, homography):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return project_points(points, homography, min_scale=1e-10)


def corners(image):
    h, w = image.shape[:2]
    return np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float)


def overlap_fraction(source, target, H):
    """Share of the target image covered by the source image once mapped through H."""
    projected = project(corners(source), H).astype(np.float32)
    frame = corners(target).astype(np.float32)
    area, _ = cv2.intersectConvexConvex(projected, frame)
    return float(area / cv2.contourArea(frame))


def frame_homography(a, b, detector):
    """Homography mapping frame a into frame b (both small greyscale images), or None."""
    ka, da = detector.detectAndCompute(a, None)
    kb, db = detector.detectAndCompute(b, None)
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return None
    good = [m for m, n in (p for p in cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2) if len(p) == 2)
            if m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None
    pa = np.float64([ka[m.queryIdx].pt for m in good])
    pb = np.float64([kb[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
    return H if H is not None and mask.sum() >= 12 else None


def frames_from_video(video, output, target, rotate=0):
    """Keep a frame each time its overlap with the last kept frame falls to the target.

    Frames are compared at 640 px. If matching fails before the target is reached,
    the last frame that still matched is kept instead, so the sequence stays connected.
    Kept frames are saved as numbered JPEGs in <output>/frames and returned with their names.
    """
    directory = output / 'frames'
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob('*.jpg'):
        old.unlink()
    detector = cv2.SIFT_create(nfeatures=1500)
    kept = []
    last_small = candidate = None
    with open_video(video) as capture:
        for index, frame in enumerate(video_frames(capture)):
            if index % FRAME_STEP:
                continue
            if rotate:
                frame = cv2.rotate(frame, ROTATIONS[rotate])
            small = cv2.cvtColor(resize_to_fit(frame, 640)[0], cv2.COLOR_BGR2GRAY)
            if last_small is None:
                kept.append(frame)
                last_small = small
                continue
            H = frame_homography(small, last_small, detector)
            overlap = overlap_fraction(small, last_small, H) if H is not None else None
            if overlap is not None and overlap > target:
                candidate = (frame, small)
                continue
            if overlap is None and candidate is not None:
                frame, small = candidate     # matching lost: keep the last frame that matched
            kept.append(frame)
            last_small, candidate = small, None
    names = [f'{number:02d}.jpg' for number in range(1, len(kept) + 1)]
    for name, frame in zip(names, kept):
        save_image(directory / name, frame)
    print(f'Kept {len(kept)} frames from {video.name} at about {target:.0%} overlap; saved to {directory}')
    return kept, names


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


def stitch(frames, names, output, args):
    images = [resize_to_fit(frame, 1600)[0] for frame in frames]
    cv2.setRNGSeed(42)
    # Each adjacent matrix maps image i+1 into image i.
    chain = [np.eye(3)]
    pairs = []
    for i in range(len(images) - 1):
        H, report, drawing = estimate_pair(images[i+1], images[i])
        chain.append(chain[-1] @ H)
        overlap = overlap_fraction(images[i+1], images[i], H)
        report.update(source=names[i+1], target=names[i], overlap_fraction=overlap)
        flag = '' if overlap <= MAX_OVERLAP else f'  (above the {MAX_OVERLAP:.0%} limit)'
        print(f'{names[i+1]} -> {names[i]}: {report["inliers"]} inliers, '
              f'{report["inlier_rmse_px"]:.2f} px RMSE, {overlap:.0%} overlap{flag}')
        pairs.append(report)
        save_image(output / f'matches_{i+1:02d}_to_{i:02d}.jpg', drawing)
    anchor = len(images) // 2
    reference_inverse = np.linalg.inv(chain[anchor])
    transforms = [reference_inverse @ H for H in chain]
    size, translation, lower, upper = canvas_geometry(images, transforms, 25000000)
    total = np.zeros((size[1], size[0], 3), np.float32)
    weights = np.zeros((size[1], size[0]), np.float32)
    image_reports = []
    for i, (name, image, H) in enumerate(zip(names, images, transforms)):
        canvas_H = translation @ H
        warped, mask = inverse_warp(image, canvas_H, size)
        save_image(output / f'warped_{i:02d}.png', warped)
        # Padding ensures distance falls toward every edge, even at canvas borders.
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        weight = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
        total += warped.astype(np.float32) * weight[:, :, None]
        weights += weight
        image_reports.append(dict(file=name, processed_shape=list(image.shape),
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
    parser = argparse.ArgumentParser(description='Build a panorama from a video of a slow pan.')
    parser.add_argument('--input', default=str(root / 'videos' / 'house.mp4'), help='Video of a slow pan')
    parser.add_argument('--output', default=str(root / 'output' / 'house'))
    parser.add_argument('--frame-overlap', type=float, default=0.7,
                        help='Overlap between kept frames (default: %(default)s)')
    parser.add_argument('--rotate', type=int, choices=[0, 90, 180, 270], default=270,
                        help='Rotate frames clockwise by this many degrees; the bundled video is stored '
                             'sideways (default: %(default)s)')
    args = parser.parse_args()
    if not 0 < args.frame_overlap < MAX_OVERLAP:
        parser.error(f'--frame-overlap must be between 0 and {MAX_OVERLAP}.')
    output = Path(args.output)
    try:
        frames, names = frames_from_video(Path(args.input), output, args.frame_overlap, args.rotate)
        if len(frames) < 4:
            parser.error(f'Need at least 4 frames; kept {len(frames)}. '
                         'Pan farther, or raise --frame-overlap to keep more frames.')
        stitch(frames, names, output, args)
    except (ValueError, OSError, cv2.error, np.linalg.LinAlgError) as error:
        parser.exit(1, f'Stitching failed: {error}\n')


if __name__ == '__main__':
    main()
