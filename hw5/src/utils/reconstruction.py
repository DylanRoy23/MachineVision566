import csv
import json

import cv2
import numpy as np

from utils.geometry import camera_center, from_homogeneous, project_points

from .geometry import point_quality, position_std, triangulate

IDENTITY = np.column_stack((np.eye(3), np.zeros(3)))


def pair_geometry(tracks, K, index):
    ids = np.flatnonzero(np.isfinite(tracks[[0, index]]).all(axis=(0, 2)))
    if len(ids) < 8:
        return None
    a, b = tracks[0, ids], tracks[index, ids]
    E, mask = cv2.findEssentialMat(
        a, b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or E.shape != (3, 3):
        return None
    _, R, t, mask = cv2.recoverPose(E, a, b, K, mask=mask)
    second = np.column_stack((R, t))
    good = ids[mask.ravel() != 0]
    if len(good) < 8:
        return None
    homogeneous = cv2.triangulatePoints(
        K @ IDENTITY, K @ second, tracks[0, good].T, tracks[index, good].T
    ).T.astype(float)
    finite = np.abs(homogeneous[:, 3]) >= 1e-10
    good = good[finite]
    xyz = from_homogeneous(homogeneous[finite])
    center = camera_center(R, t.ravel())
    rays = xyz - center
    cosine = np.sum(xyz * rays, axis=1) / (
        np.linalg.norm(xyz, axis=1) * np.linalg.norm(rays, axis=1)
    )
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    positive = (xyz[:, 2] > 0) & ((xyz @ R.T + t.ravel())[:, 2] > 0)
    return second, good[positive], xyz[positive], angles[positive], len(ids)


def initialize_pair(tracks, K, requested):
    candidates = []
    indices = (
        [requested]
        if requested is not None
        else np.unique(
            np.linspace(1, len(tracks) - 1, min(30, len(tracks) - 1), dtype=int)
        )
    )
    for index in indices:
        if index >= len(tracks):
            raise ValueError("--pair-frame exceeds the video length.")
        ids = np.flatnonzero(np.isfinite(tracks[index]).all(axis=1))
        if len(ids) >= 8:
            motion = np.median(
                np.linalg.norm(tracks[index, ids] - tracks[0, ids], axis=1)
            )
            candidates.append((float(motion), int(index)))
    for motion, index in sorted(
        candidates, key=lambda item: item[0], reverse=True
    ):
        pair = pair_geometry(tracks, K, index)
        if pair is None:
            continue
        second, good, points, angles, shared = pair
        cloud = {}
        kept_angles = []
        for point_id, xyz, angle in zip(good, points, angles):
            valid, _ = point_quality(
                xyz, [IDENTITY, second], tracks[[0, index], point_id], K
            )
            if valid and angle > 1.0:
                cloud[int(point_id)] = xyz
                kept_angles.append(angle)
        if len(cloud) >= 8:
            info = dict(
                first_frame=0,
                second_frame=index,
                shared_tracks=shared,
                reconstructed_points=len(cloud),
                median_motion_px=motion,
                median_triangulation_angle_deg=float(np.median(kept_angles)),
                selection=(
                    "Largest tested median track motion passing cheirality, "
                    "3px reprojection, and >1 degree triangulation checks"
                ),
            )
            return {0: IDENTITY.copy(), index: second}, cloud, info
    raise ValueError(
        "No usable image pair. Use textured, nonplanar geometry with "
        "camera translation and shared visible points."
    )


def estimate_cameras(tracks, K, poses, cloud):
    ids = np.array(sorted(cloud))
    xyz = np.array([cloud[i] for i in ids])
    for index, observations in enumerate(tracks):
        if index in poses:
            continue
        valid = np.isfinite(observations[ids]).all(axis=1)
        if valid.sum() < 6:
            continue
        points, pixels = xyz[valid], observations[ids[valid]]
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            points,
            pixels,
            K,
            None,
            iterationsCount=200,
            reprojectionError=3.0,
            confidence=0.999,
        )
        if not ok or inliers is None or len(inliers) < 6:
            continue
        kept = inliers.ravel()
        ok, rvec, tvec = cv2.solvePnP(
            points[kept], pixels[kept], K, None, rvec, tvec, True
        )
        if ok:
            R = cv2.Rodrigues(rvec)[0]
            if np.all((R @ points[kept].T + tvec)[2] > 0):
                poses[index] = np.column_stack((R, tvec))


def reconstruct_video(tracks, K, poses, initial):
    cloud, views = {}, {}
    for point_id, seed in initial.items():
        indices = []
        for frame_index in sorted(poses):
            observed_pixel = tracks[frame_index, point_id]
            if not np.isfinite(observed_pixel).all():
                continue

            predicted_pixel = project_points(seed, K @ poses[frame_index])
            error_pixels = np.linalg.norm(predicted_pixel - observed_pixel)
            if error_pixels <= 3.0:
                indices.append(frame_index)

        if len(indices) < 3:
            continue
        cameras = [poses[i] for i in indices]
        pixels = tracks[indices, point_id]
        try:
            xyz = triangulate([K @ camera for camera in cameras], pixels)
        except ValueError:
            continue
        valid, _ = point_quality(xyz, cameras, pixels, K)
        if valid:
            cloud[point_id], views[point_id] = xyz, indices
    if not cloud:
        raise ValueError("No points reconstructed from at least three frames.")
    return cloud, views


def save_cloud(name, cloud, views, tracks, poses, K, args):
    with (args.output / f"{name}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        header = ["point_id", "x", "y", "z", "views", "reprojection_rmse_px"]
        if args.extra_credit:
            header += ["x_std", "y_std", "z_std"]
        writer.writerow(header)
        for point_id, xyz in sorted(cloud.items()):
            indices = views[point_id]
            cameras = [poses[i] for i in indices]
            _, error = point_quality(xyz, cameras, tracks[indices, point_id], K)
            row = [point_id, *xyz, len(indices), error]
            if args.extra_credit:
                row += position_std(
                    xyz, [K @ camera for camera in cameras], args.pixel_sigma
                )
            writer.writerow(row)
    with (args.output / f"{name}.ply").open("w") as file:
        file.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(cloud)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        for xyz in cloud.values():
            file.write(" ".join(str(float(v)) for v in xyz) + "\n")


def reconstruct(tracks, K, args):
    poses, pair_cloud, info = initialize_pair(tracks, K, args.pair_frame)
    pair_indices = [0, info["second_frame"]]
    pair_views = {i: pair_indices for i in pair_cloud}
    save_cloud("two_view", pair_cloud, pair_views, tracks, poses, K, args)
    estimate_cameras(tracks, K, poses, pair_cloud)
    if len(poses) < 3:
        raise ValueError(
            "Only two cameras recovered; video reconstruction needs "
            "more tracked views."
        )
    video_cloud, views = reconstruct_video(tracks, K, poses, pair_cloud)
    save_cloud("whole_video", video_cloud, views, tracks, poses, K, args)
    save_comparison(pair_cloud, video_cloud, args.output)
    (args.output / "results.json").write_text(json.dumps(info, indent=2) + "\n")
    print(
        f"Two-view frames: 0 and {info['second_frame']}; "
        f"{len(pair_cloud)} points"
    )
    print(
        f"Video: {len(video_cloud)} points using "
        f"{len(poses)}/{len(tracks)} camera frames"
    )
    print(
        f"Saved tracking, point clouds, and comparison to {args.output}. "
        "Coordinates have arbitrary scale."
    )


def save_comparison(pair_cloud, video_cloud, output):
    with (output / "comparison.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "point_id",
                "two_view_x",
                "two_view_y",
                "two_view_z",
                "video_x",
                "video_y",
                "video_z",
                "position_difference",
            ]
        )
        for point_id in sorted(pair_cloud.keys() & video_cloud.keys()):
            xyz = video_cloud[point_id]
            initial = pair_cloud[point_id]
            writer.writerow(
                [point_id, *initial, *xyz, np.linalg.norm(xyz - initial)]
            )
