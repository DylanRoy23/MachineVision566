"""Triangulate newly visible object points over the full camera orbit."""
import json

import cv2
import numpy as np

from utils.geometry import camera_center, from_homogeneous

from .geometry import triangulate
from .reconstruction import IDENTITY, initialize_pair, save_cloud, save_comparison
from .regions import bundled_outlines, outline_mask
from .sequence_poses import choose_pair, recover_sequence
from .tracking import object_mask, save_scene_tracking, track_scene

MAX_REPROJECTION_PX = 3.
MIN_ANGLE_DEG = 1.4


def score_points(points, projections, pixels):
    transformed = np.einsum('vij,pj->pvi', projections[:, :, :3], points)
    transformed += projections[None, :, :, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.linalg.norm(transformed[:, :, :2] / transformed[:, :, 2:] - pixels, axis=2)
    return np.where(transformed[:, :, 2] > 0, errors, np.inf)


def triangulate_track(projections, centers, pixels):
    """Fit a point to its agreeing views, using sampled view pairs as hypotheses."""
    rows = pixels[:, :, None] * projections[:, None, 2] - projections[:, :2]
    sampled = np.linspace(0, len(rows) - 1, min(9, len(rows)), dtype=int)
    a, b = np.triu_indices(len(sampled), k=1)
    pairs = np.concatenate((rows[sampled[a]], rows[sampled[b]]), axis=1)
    coordinates = np.linalg.svd(pairs)[2][:, -1]
    valid = np.abs(coordinates[:, 3]) > 1e-10
    if not valid.any():
        return None
    candidates = from_homogeneous(coordinates[valid])
    rays_a = candidates - centers[sampled[a[valid]]]
    rays_b = candidates - centers[sampled[b[valid]]]
    lens = np.linalg.norm(rays_a, axis=1) * np.linalg.norm(rays_b, axis=1)
    angles = np.sum(rays_a * rays_b, axis=1) < lens * np.cos(np.deg2rad(MIN_ANGLE_DEG))
    if not angles.any():
        return None
    candidates = candidates[angles]
    errors = score_points(candidates, projections, pixels)
    supports = errors <= MAX_REPROJECTION_PX
    clipped_error = np.minimum(errors, MAX_REPROJECTION_PX).sum(axis=1)
    best = np.lexsort((clipped_error, -supports.sum(axis=1)))[0]
    support = supports[best]
    if support.sum() < 4:
        return None
    point = triangulate(projections[support], pixels[support])
    support = score_points(point[None], projections, pixels)[0] <= MAX_REPROJECTION_PX
    if support.sum() < 4:
        return None
    rays = point - centers[support]
    lengths = np.linalg.norm(rays, axis=1)
    if not np.any(rays @ rays[0] < lengths * lengths[0] * np.cos(np.deg2rad(MIN_ANGLE_DEG))):
        return None
    return point, support


def reconstruct_object(sequence, poses, K):
    tracks = sequence.observations
    frame_indices = np.array(sorted(poses))
    cameras = np.array([poses[index] for index in frame_indices])
    projections = K @ cameras
    centers = np.array([camera_center(pose[:, :3], pose[:, 3]) for pose in cameras])
    reference_by_id = {}
    for index, ids in sequence.object_views.items():
        for point_id in ids:
            reference_by_id.setdefault(int(point_id), []).append(index)
    cloud, views = {}, {}
    for point_id, references in reference_by_id.items():
        pixels = tracks[frame_indices, point_id]
        finite = np.isfinite(pixels).all(axis=1)
        if finite.sum() < 4:
            continue
        try:
            estimate = triangulate_track(projections[finite], centers[finite], pixels[finite])
        except ValueError:
            continue
        if estimate is None:
            continue
        point, good = estimate
        support = frame_indices[finite][good]
        if np.isin(support, references).any():
            cloud[point_id], views[point_id] = point, support.tolist()
    return cloud, views


def filter_foreground(cloud, views, poses, outlines, first_mask, K, distortion, plane, depth):
    """Keep points whose projections agree with the object in reference views."""
    if not cloud:
        return {}, {}
    ids = list(cloud)
    points = np.array([cloud[point_id] for point_id in ids])
    height, width = first_mask.shape
    references = {index: outline_mask(first_mask.shape, outline, margin=20)
                  for index, outline in outlines.items() if index in poses}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    references[0] = cv2.dilate(first_mask, kernel)
    tested = np.zeros(len(points), dtype=int)
    accepted = np.zeros(len(points), dtype=int)
    for index, mask in references.items():
        pose = poses[index]
        rvec = cv2.Rodrigues(pose[:, :3])[0]
        pixels = cv2.projectPoints(points, rvec, pose[:, 3], K, distortion)[0].reshape(-1, 2)
        positive = (points @ pose[2, :3] + pose[2, 3]) > 0
        visible = positive & np.isfinite(pixels).all(axis=1)
        visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        visible &= (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
        x, y = pixels[visible].astype(int).T
        tested[visible] += 1
        accepted[visible] += mask[y, x] != 0
    above_floor = (points - plane.origin) @ plane.normal > -.025 * depth
    keep = above_floor & (tested >= min(3, len(references))) & (accepted >= .65 * tested)
    return ({point_id: cloud[point_id] for point_id, good in zip(ids, keep) if good},
            {point_id: views[point_id] for point_id, good in zip(ids, keep) if good})


def two_view_result(tracks, K, initial_ids, initial_scene, seed, poses, requested):
    second, pose, ids, xyz, angles, shared = seed
    if requested is None or requested == second:
        selected = np.isin(ids, initial_scene)
        cloud = dict(zip(ids[selected].tolist(), xyz[selected]))
        pair_pixels = tracks[[0, second]][:, ids[selected]]
        info = dict(
            first_frame=0, second_frame=second,
            shared_tracks=int(np.isfinite(tracks[second, initial_scene]).all(axis=1).sum()),
            shared_scene_tracks=shared,
            reconstructed_points=len(cloud),
            median_motion_px=float(np.median(np.linalg.norm(pair_pixels[1] - pair_pixels[0], axis=1))),
            median_triangulation_angle_deg=float(np.median(angles[selected])),
            selection='Early calibrated pair with strong scene support and at least eight usable object tracks',
        )
        return {0: IDENTITY, second: pose}, cloud, info

    if requested not in poses:
        raise ValueError('The requested pair frame has no registered camera for scale comparison.')
    try:
        pair_poses, points, info = initialize_pair(tracks[:, initial_ids], K, requested)
    except ValueError:
        # A dense set of corners on a single surface can dominate the fit.
        # Retry with the more widely spaced scene tracks inside the first outline.
        initial_ids = initial_scene
        pair_poses, points, info = initialize_pair(tracks[:, initial_ids], K, requested)
    # Both reconstructions use frame zero as the world origin. Match their
    # arbitrary units using the length of the same camera baseline.
    scale = float(np.linalg.norm(poses[requested][:, 3]))
    if scale <= 1e-8:
        raise ValueError('The requested pair has no usable camera baseline.')
    pair_poses[requested][:, 3] *= scale
    cloud = {int(initial_ids[index]): xyz * scale for index, xyz in points.items()}
    info.update(selection='Requested calibrated pair from object tracks', pair_scale=scale)
    return pair_poses, cloud, info


def reconstruct_full_video(args, K, distortion, calibration_size, first_outline):
    outlines = bundled_outlines(args.video.resolve().name)
    sequence = track_scene(args, K, distortion, calibration_size, first_outline, outlines)
    tracks = sequence.observations
    if args.pair_frame is not None and args.pair_frame >= len(tracks):
        raise ValueError('--pair-frame exceeds the video length.')
    width, height = sequence.frame_size
    initial_ids = sequence.object_views[0]
    initial_scene = initial_ids[initial_ids < sequence.scene_count]
    scene = tracks[:, :sequence.scene_count]
    seed = choose_pair(scene, K, initial_scene)
    poses, plane, depth = recover_sequence(scene, K, height, initial_scene, seed)
    if len(poses) < 3:
        raise ValueError('Need at least three registered camera frames for full-video reconstruction.')

    pair_poses, pair_cloud, info = two_view_result(tracks, K, initial_ids, initial_scene, seed, poses, args.pair_frame)
    second = info['second_frame']
    pair_views = {point_id: [0, second] for point_id in pair_cloud}
    cloud, views = reconstruct_object(sequence, poses, K)
    first_mask = object_mask(np.empty((height, width), np.uint8), args.roi, first_outline)
    cloud, views = filter_foreground(cloud, views, poses, outlines, first_mask, K, distortion, plane, depth)
    if not cloud:
        raise ValueError('No object tracks passed multiview triangulation and outline checks.')

    save_cloud('two_view', pair_cloud, pair_views, tracks, pair_poses, K, args)
    save_cloud('whole_video', cloud, views, tracks, poses, K, args)
    save_comparison(pair_cloud, cloud, args.output)
    object_points = np.zeros(tracks.shape[1], dtype=bool)
    for point_ids in sequence.object_views.values():
        object_points[point_ids] = True
    save_scene_tracking(args, sequence, object_points)

    contributing = set().union(*views.values())
    by_quarter = [sum(int(frame) in contributing for frame in quarter)
                  for quarter in np.array_split(np.arange(len(tracks)), 4)]
    center = np.median(list(cloud.values()), axis=0)
    cameras = np.array([camera_center(poses[index][:, :3], poses[index][:, 3]) for index in contributing])
    around_object = (cameras - center) @ plane.axes
    azimuth = np.mod(np.degrees(np.arctan2(around_object[:, 1], around_object[:, 0])), 360)
    by_azimuth = np.histogram(azimuth, bins=np.arange(0, 361, 45))[0].tolist()
    info.update(
        camera_registration='Track the floor plane with RANSAC homographies and refine each pose using PnP',
        frames_read=len(tracks), registered_frames=len(poses),
        contributing_frames=len(contributing), contributing_frames_by_quarter=by_quarter,
        contributing_frames_by_45_degree_sector=by_azimuth,
        first_contributing_frame=min(contributing), last_contributing_frame=max(contributing),
        object_outline_frames=sorted(sequence.object_views),
        whole_video_points=len(cloud),
        points_from_later_outlines=len(cloud.keys() - set(initial_ids)),
        comparison_points=len(cloud.keys() & pair_cloud.keys()),
    )
    (args.output / 'results.json').write_text(json.dumps(info, indent=2) + '\n')
    print(f'Two-view frames 0 and {second}: {len(pair_cloud)} object points')
    print(f'Whole video: {len(cloud)} object points, {len(contributing)}/{len(tracks)} contributing frames '
          f'({len(poses)} registered)')
    print(f'Saved tracking, point clouds, and comparison to {args.output}. Coordinates have arbitrary scale.')
