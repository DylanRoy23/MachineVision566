"""Two-view (task 3) and whole-video (task 4) reconstruction of a stationary object."""
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .bundle_adjustment import bundle_adjust
from .features import build_keyframe_tracks, match_pair
from .geometry import distort_pixels, fit_plane
from .reconstruction import save_cloud, save_comparison
from .regions import bundled_outlines, outline_mask
from .sfm import IncrementalSfM, camera_centers, projection_matrices, triangulate_robust
from .tracking import object_mask, save_tracking, track_object

KEYFRAME_STEP = 3          # every third video frame is a keyframe
PAIR_SEARCH_FRAMES = 150   # the two-view partner of frame 0 is searched in the first 150 frames
PAIR_ANGLE_DEG = 5.        # triangulation angle at which the two-view score stops rewarding parallax
OUTLINE_MARGIN_PX = 8      # tolerance when testing projected points against object outlines


class Cloud:
    """3D points with their observations: point_id -> xyz, keyframe rows, undistorted pixels."""

    def __init__(self):
        self.xyz, self.rows, self.pixels, self.source, self.color = {}, {}, {}, {}, {}

    def add(self, point_id, xyz, rows, pixels, source, color):
        self.xyz[point_id] = np.asarray(xyz, float)
        self.rows[point_id] = np.asarray(rows)
        self.pixels[point_id] = np.asarray(pixels, float)
        self.source[point_id] = source
        self.color[point_id] = np.asarray(color, float)

    def subset(self, ids):
        out = Cloud()
        for i in ids:
            out.add(i, self.xyz[i], self.rows[i], self.pixels[i], self.source[i], self.color[i])
        return out

    def ids(self):
        return list(self.xyz)

    def array(self):
        return np.array([self.xyz[i] for i in self.xyz]).reshape(-1, 3)

    def __len__(self):
        return len(self.xyz)


# --------------------------------------------------------------------- task 3
def choose_two_view_pair(keyframes, K, first_mask, requested):
    """Pick frame 0's partner and reconstruct the object from those two images only.

    SIFT features of frame 0 are matched directly to each candidate keyframe
    (ratio test, then a RANSAC essential matrix). Depth uncertainty of a
    triangulated point grows roughly as 1/sin(angle) for the angle between the
    two viewing rays, so a wider baseline is better. A wider baseline also
    leaves fewer object features visible in both images, so each candidate is
    scored as

        score = (object points passing all checks) * min(median angle / 5 deg, 1)

    which rewards parallax up to 5 degrees and after that just counts points.
    """
    frames = keyframes.frames
    if frames[0] != 0:
        raise ValueError('Frame 0 must be a keyframe.')
    if requested is not None:
        rows = np.flatnonzero(frames == requested)
    else:
        rows = np.flatnonzero((frames > 0) & (frames <= PAIR_SEARCH_FRAMES))
    raw0, und0, _ = keyframes.keypoints[0]
    height, width = first_mask.shape
    x, y = np.rint(raw0).astype(int).T
    inside_all = first_mask[np.clip(y, 0, height - 1), np.clip(x, 0, width - 1)] > 0
    best = None
    for row in rows:
        match = match_pair((und0, keyframes.descriptors[0]),
                           (keyframes.keypoints[row][1], keyframes.descriptors[row]), K, threshold=1.)
        if match is None:
            continue
        ia, ib = match
        inside = inside_all[ia]
        if inside.sum() < 8:
            continue
        a, b = und0[ia], keyframes.keypoints[row][1][ib]
        E, mask = cv2.findEssentialMat(a, b, K, method=cv2.RANSAC, prob=.999, threshold=1.)
        if E is None or E.shape != (3, 3):
            continue
        _, R, t, mask = cv2.recoverPose(E, a, b, K, mask=mask)
        P0, P1 = K @ np.eye(3, 4), K @ np.column_stack((R, t))
        X = cv2.triangulatePoints(P0, P1, np.ascontiguousarray(a.T), np.ascontiguousarray(b.T))
        X = (X[:3] / X[3]).T
        second_center = -R.T @ t.ravel()
        ray_a = X / np.linalg.norm(X, axis=1, keepdims=True)
        ray_b = (X - second_center) / np.linalg.norm(X - second_center, axis=1, keepdims=True)
        angles = np.degrees(np.arccos(np.clip(np.sum(ray_a * ray_b, axis=1), -1, 1)))
        z0, z1 = X[:, 2], (X @ R.T + t.ravel())[:, 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            e0 = np.linalg.norm(X[:, :2] / z0[:, None] * (K[0, 0], K[1, 1]) + K[:2, 2] - a, axis=1)
            q1 = X @ (K @ R).T + (K @ t).ravel()
            e1 = np.linalg.norm(q1[:, :2] / q1[:, 2:] - b, axis=1)
        valid = (mask.ravel() > 0) & (z0 > 0) & (z1 > 0) & (e0 < 2) & (e1 < 2) & (angles > 1.)
        good = valid & inside
        if good.sum() < 8:
            continue
        angle = float(np.median(angles[good]))
        score = good.sum() * min(angle / PAIR_ANGLE_DEG, 1.)
        if best is None or score > best['score']:
            best = dict(score=score, row=int(row), frame=int(frames[row]), R=R, t=t.ravel(),
                        features=ia[good], X=X[good], pixels=np.stack((a[good], b[good]), axis=1),
                        angle=angle, shared=len(ia), inliers=int(valid.sum()),
                        object_shared=int(inside.sum()),
                        motion=float(np.median(np.linalg.norm(b[good] - a[good], axis=1))))
    if best is None:
        raise ValueError('No image pair with frame 0 shares eight well-triangulated object points.')
    return best


def two_view_cloud(pair, keyframes, first_id):
    """Two-view points keyed by SIFT track id (shared with the whole-video cloud) when the
    frame-0 feature joined a track, otherwise by a new id starting at first_id."""
    cloud = Cloud()
    rows = np.array([0, pair['row']])
    raw0, _, colors0 = keyframes.keypoints[0]
    for feature, xyz, pixels in zip(pair['features'], pair['X'], pair['pixels']):
        track = keyframes.track_of[0][feature]
        point_id = int(track) if track >= 0 else first_id + int(feature)
        cloud.add(point_id, xyz, rows, pixels, 'sift', colors0[feature])
    poses = {0: np.eye(3, 4), pair['row']: np.column_stack((pair['R'], pair['t']))}
    return cloud, poses


# --------------------------------------------------------------------- task 4
def whole_video_cloud(sfm, keyframes, sequence, K, distortion):
    """SfM points plus KLT object tracks triangulated against the refined keyframe poses."""
    cloud = Cloud()
    usable = sfm.usable()
    registered = np.flatnonzero(sfm.registered)
    for point_id in np.flatnonzero(sfm.triangulated):
        rows = registered[usable[registered, point_id]]
        cloud.add(int(point_id), sfm.X[point_id], rows, sfm.T[rows, point_id], 'sift',
                  keyframes.colors[point_id])

    # KLT tracks start inside the object outlines, so they add points on the
    # object's weakly textured surfaces. They are observed in every video frame;
    # here they use the keyframe rows, whose poses are known.
    frames = keyframes.frames[registered]
    P = projection_matrices(K, sfm.cams[registered])
    centers = camera_centers(sfm.cams[registered])
    offset = sfm.n_tracks
    klt_ids, klt_xyz, klt_rows, klt_pixels = [], [], [], []
    for track in range(sequence.observations.shape[1]):
        pixels = sequence.observations[frames, track].astype(np.float64)
        seen = np.flatnonzero(np.isfinite(pixels[:, 0]))
        if len(seen) < 3:
            continue
        result = triangulate_robust(P[seen], centers[seen], pixels[seen])
        if result is None or result[1].sum() < 3:
            continue
        xyz, inliers = result
        klt_ids.append(offset + int(track))
        klt_xyz.append(xyz)
        klt_rows.append(registered[seen[inliers]])
        klt_pixels.append(pixels[seen[inliers]])
    if klt_ids:
        # Refine the DLT estimates by minimizing reprojection error with every camera fixed.
        point_index = np.concatenate([np.full(len(r), i) for i, r in enumerate(klt_rows)])
        rows = np.concatenate(klt_rows)
        slot = np.searchsorted(registered, rows)
        _, _, refined, _ = bundle_adjust(
            sfm.cams[registered, :3], sfm.cams[registered, 3:], np.array(klt_xyz), slot, point_index,
            np.concatenate(klt_pixels), K, fixed=np.ones(len(registered), bool), iterations=10)
        for point_id, xyz, rows, pixels in zip(klt_ids, refined, klt_rows, klt_pixels):
            raw = distort_pixels(pixels[:1], K, distortion)[0]
            image = keyframes.images[int(keyframes.frames[rows[0]])]
            x = int(np.clip(round(raw[0]), 0, image.shape[1] - 1))
            y = int(np.clip(round(raw[1]), 0, image.shape[0] - 1))
            cloud.add(point_id, xyz, rows, pixels, 'klt', image[y, x])
    return cloud


def floor_plane(points, cameras):
    """Dominant plane (the floor) with its normal pointing toward the cameras, or None."""
    scale = np.median(np.linalg.norm(points - np.median(points, axis=0), axis=1))
    origin, normal, inliers = fit_plane(points, .01 * scale)
    if inliers.mean() < .3:
        return None
    if np.median((cameras - origin) @ normal) < 0:
        normal = -normal
    noise = float(np.std((points[inliers] - origin) @ normal))
    return dict(origin=origin, normal=normal, noise=noise, inliers=int(inliers.sum()))


def select_object(cloud, sfm, keyframes, outlines, first_mask, K, distortion, floor):
    """Keep points that project inside the object outline in the reference views,
    stand above the floor, and are not isolated."""
    ids = cloud.ids()
    points = cloud.array()
    height, width = first_mask.shape
    row_of = {int(f): r for r, f in enumerate(keyframes.frames)}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * OUTLINE_MARGIN_PX + 1,) * 2)
    references = {0: cv2.dilate(first_mask, kernel)}
    for frame, outline in outlines.items():
        row = row_of.get(int(frame))
        if row is not None and sfm.registered[row]:
            references[row] = outline_mask(first_mask.shape, outline, OUTLINE_MARGIN_PX)
    tested = np.zeros(len(points), int)
    accepted = np.zeros(len(points), int)
    for row, mask in references.items():
        rvec, tvec = sfm.cams[row, :3], sfm.cams[row, 3:]
        pixels = cv2.projectPoints(points, rvec, tvec, K, distortion)[0].reshape(-1, 2)
        depth = points @ cv2.Rodrigues(rvec)[0][2] + tvec[2]
        visible = (depth > 0) & np.isfinite(pixels).all(axis=1)
        visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        x, y = pixels[visible].astype(int).T
        tested[visible] += 1
        accepted[visible] += mask[y, x] > 0
    keep = (tested >= min(3, len(references))) & (accepted >= .7 * tested)
    if floor is not None:
        heights = (points - floor['origin']) @ floor['normal']
        keep &= heights > 4 * floor['noise']
        if len(references) == 1:
            # Only the frame-0 region is known, and points hidden behind the object
            # also project into it. Seed with points actually observed inside the
            # region in frame 0, then keep everything above the floor inside the
            # seeds' footprint, so the far side of the object is included.
            seeds = keep & np.array([0 in cloud.rows[i] for i in ids])
            if seeds.sum() >= 8:
                flat = points - np.outer(heights, floor['normal'])
                center = np.median(flat[seeds], axis=0)
                distance = np.linalg.norm(flat - center, axis=1)
                radius = 1.5 * np.percentile(distance[seeds], 95)
                top = 1.2 * np.percentile(heights[seeds], 99)
                keep = (heights > 4 * floor['noise']) & (distance <= radius) & (heights <= top)
    # Statistical outlier removal: drop points far from their 6 nearest neighbours.
    if keep.sum() > 10:
        kept = points[keep]
        distance = cKDTree(kept).query(kept, k=7)[0][:, 1:].mean(axis=1)
        mad = np.median(np.abs(distance - np.median(distance)))
        dense = distance < np.median(distance) + 4 * 1.4826 * mad
        keep[np.flatnonzero(keep)[~dense]] = False
    return cloud.subset([i for i, k in zip(ids, keep) if k])


def estimate_pixel_sigma(sfm):
    """Robust 1-sigma pixel noise per axis from bundle-adjustment residual lengths.

    For isotropic Gaussian noise with per-axis sigma, the residual length is
    Rayleigh distributed with median sigma * sqrt(2 ln 2).
    """
    _, _, errors = sfm.observation_errors()
    inliers = errors[errors < 2.]
    return float(np.median(inliers) / np.sqrt(2 * np.log(2)))


# ---------------------------------------------------------------------- main
def reconstruct_full_video(args, K, distortion, calibration_size, first_outline, log=print):
    start = time.time()
    outlines = bundled_outlines(args.video.resolve().name)
    sequence = track_object(args, K, distortion, calibration_size, first_outline, outlines)
    width, height = sequence.frame_size
    first_mask = object_mask(np.empty((height, width), np.uint8), args.roi, first_outline)
    log(f'  KLT: {len(sequence.observations)} frames, {sequence.observations.shape[1]} tracks')

    required = set(outlines) | ({args.pair_frame} if args.pair_frame is not None else set())
    if args.pair_frame is not None and args.pair_frame >= len(sequence.observations):
        raise ValueError('--pair-frame exceeds the video length.')
    keyframes = build_keyframe_tracks(args.video, K, distortion, KEYFRAME_STEP, required,
                                      keep_descriptors_until=max(PAIR_SEARCH_FRAMES, args.pair_frame or 0),
                                      log=log)

    # Task 3: two images.
    pair = choose_two_view_pair(keyframes, K, first_mask, args.pair_frame)
    unused_id = keyframes.pixels.shape[1] + sequence.observations.shape[1]   # after SIFT and KLT ids
    pair_cloud, pair_poses = two_view_cloud(pair, keyframes, unused_id)
    log(f'  Two-view: frames 0 and {pair["frame"]}, {len(pair_cloud)} object points, '
        f'median angle {pair["angle"]:.1f} deg')

    # Task 4: whole video.
    sfm = IncrementalSfM(keyframes.pixels, K, log=log).run(keyframes.frames)
    if not sfm.registered[0] or not sfm.registered[pair['row']]:
        raise ValueError('Frame 0 or the two-view partner could not be registered.')
    # Same gauge as the two-view result: camera 0 is the world frame and the
    # camera centre of the partner frame is one unit from it.
    sfm.normalize(0, pair['row'])
    sigma = args.pixel_sigma if args.pixel_sigma is not None else estimate_pixel_sigma(sfm)
    video_R = cv2.Rodrigues(sfm.cams[pair['row'], :3])[0]
    video_t = sfm.cams[pair['row'], 3:]
    pose_rotation_difference = float(np.degrees(np.linalg.norm(cv2.Rodrigues(video_R @ pair['R'].T)[0])))
    pose_direction_difference = float(np.degrees(np.arccos(np.clip(
        video_t @ pair['t'] / np.linalg.norm(video_t) / np.linalg.norm(pair['t']), -1, 1))))
    scene = whole_video_cloud(sfm, keyframes, sequence, K, distortion)
    centers = camera_centers(sfm.cams[sfm.registered])
    scene_points = sfm.X[sfm.triangulated]
    floor = floor_plane(scene_points, centers)
    cloud = select_object(scene, sfm, keyframes, outlines, first_mask, K, distortion, floor)
    if not len(cloud):
        raise ValueError('No reconstructed points passed the object outline checks.')

    def video_camera(row):
        return np.column_stack((cv2.Rodrigues(sfm.cams[row, :3])[0], sfm.cams[row, 3:]))

    # Outputs: the object clouds, the comparison, and the tracking demo.
    out = args.output
    ec_sigma = sigma if args.extra_credit else None
    pair_std = save_cloud(out / 'two_view', pair_cloud, lambda r: pair_poses[int(r)], K, ec_sigma)
    video_std = save_cloud(out / 'whole_video', cloud, video_camera, K, ec_sigma)
    comparison = save_comparison(pair_cloud.xyz, cloud.xyz, out)
    save_tracking(args, sequence)

    object_xyz = cloud.array()
    if floor is not None:
        object_height = float(np.percentile((object_xyz - floor['origin']) @ floor['normal'], 99))
    else:
        object_height = float(np.ptp(object_xyz @ np.linalg.svd(object_xyz - object_xyz.mean(0))[2][0]))
    log(f'Two-view (task 3): frames 0 and {pair["frame"]}, {len(pair_cloud)} object points, '
        f'median triangulation angle {pair["angle"]:.1f} deg, {pair["inliers"]} essential-matrix inliers')
    log(f'Whole video (task 4): {len(cloud)} object points, {sfm.registered.sum()}/{len(keyframes.frames)} '
        f'keyframes registered, {len(keyframes.loop_pairs)} loop closures')
    if comparison['points']:
        log(f'Comparison: {comparison["points"]} shared points, median difference {comparison["median"]:.3f} '
            f'({100 * comparison["median"] / object_height:.1f}% of object height); two-view camera differs from '
            f'the whole-video camera by {pose_rotation_difference:.2f} deg rotation and '
            f'{pose_direction_difference:.2f} deg translation direction')
    if args.extra_credit:
        log(f'EC: pixel sigma {sigma:.3f} px; median position std {pair_std:.4f} (two view), '
            f'{video_std:.4f} (whole video)')
    log(f'Saved to {out}. Units are arbitrary: camera 0 is the origin and the two-view partner '
        f'camera is 1 unit away. ({time.time() - start:.0f} s)')
