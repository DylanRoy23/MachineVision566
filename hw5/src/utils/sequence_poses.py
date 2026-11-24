"""Register an orbiting camera against the textured floor of the supplied clips."""
from dataclasses import dataclass

import cv2
import numpy as np

from utils.geometry import camera_center, to_homogeneous

from .reconstruction import IDENTITY, pair_geometry


def reprojection_errors(points, projection, pixels):
    projected = points @ projection[:, :3].T + projection[:, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        errors = np.linalg.norm(projected[:, :2] / projected[:, 2:] - pixels, axis=1)
    return np.where(projected[:, 2] > 0, errors, np.inf)


def choose_pair(tracks, K, object_ids):
    indices = range(5, min(95, len(tracks)), 4)
    best = None
    best_score = 0
    for index in indices:
        pair = pair_geometry(tracks, K, index)
        if pair is None:
            continue
        pose, ids, xyz, angles, shared = pair
        valid = (angles > 1.4)
        valid &= reprojection_errors(xyz, K @ IDENTITY, tracks[0, ids]) < 2.
        valid &= reprojection_errors(xyz, K @ pose, tracks[index, ids]) < 2.
        if np.count_nonzero(valid & np.isin(ids, object_ids)) < 8 or np.count_nonzero(valid) < 20:
            continue
        angle = float(np.median(angles[valid]))
        score = valid.sum() * min(angle / 4., 1.)
        if score > best_score:
            best_score = score
            best = index, pose, ids[valid], xyz[valid], angles[valid], shared
    if best is None:
        raise ValueError('Could not initialize a scene pair with eight usable object tracks.')
    return best


@dataclass
class FloorPlane:
    origin: np.ndarray
    normal: np.ndarray
    axes: np.ndarray

    def intersect(self, pose, pixels, inverse_K):
        rotation = pose[:, :3]
        center = camera_center(rotation, pose[:, 3])
        rays = to_homogeneous(pixels) @ inverse_K.T @ rotation
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = ((self.origin - center) @ self.normal) / (rays @ self.normal)
        return center + distance[:, None] * rays


def find_floor(xyz, is_object):
    rng = np.random.default_rng(42)
    tolerance = np.median(xyz[:, 2]) * .015
    best = np.zeros(len(xyz), dtype=bool)
    for indices in rng.choice(len(xyz), (450, 3)):
        a, b, c = xyz[indices]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        inliers = (np.abs((xyz - a) @ normal) < tolerance) & ~is_object
        if inliers.sum() > best.sum():
            best = inliers
        if best.sum() > len(xyz) * .9:
            break
    if best.sum() < max(12, len(xyz) * .4):
        raise ValueError('Could not identify a dominant floor plane for camera registration.')
    inliers = xyz[best]
    origin = inliers.mean(axis=0)
    _, _, axes = np.linalg.svd(inliers - origin, full_matrices=False)
    normal = axes[-1]
    if normal @ origin > 0:
        normal = -normal
    return FloorPlane(origin, normal, np.linalg.svd(normal[None])[2][1:].T), best


def refine_pose(pose, points, pixels, K):
    rvec = cv2.Rodrigues(pose[:, :3])[0]
    tvec = pose[:, 3:4].copy()
    ok, rvec, tvec = cv2.solvePnP(points, pixels, K, None, rvec, tvec, True)
    if ok:
        return np.column_stack((cv2.Rodrigues(rvec)[0], tvec))
    return None


def register_from_floor(anchor, current_pixels, prediction, known_points, plane, K, inverse_K, depth):
    anchor_pose, anchor_pixels = anchor
    world = plane.intersect(anchor_pose, anchor_pixels, inverse_K)
    good = np.isfinite(world).all(axis=1) & (np.linalg.norm(world, axis=1) < 100. * depth)
    if good.sum() < 18:
        return None
    world, pixels = world[good], current_pixels[good]
    H, inliers = cv2.findHomography((world - plane.origin) @ plane.axes, pixels, cv2.RANSAC, 1.6)
    if H is None or inliers.sum() < 18:
        return None
    inside = inliers.ravel().astype(bool)
    selected = np.flatnonzero(good)[inside]
    points, pixels = world[inside], pixels[inside]
    known = known_points[selected]
    fixed = np.isfinite(known[:, 0])
    if fixed.sum() >= 12:
        fixed[fixed] = reprojection_errors(known[fixed], K @ anchor_pose, anchor_pixels[selected[fixed]]) < 1.5
        points[fixed] = known[fixed]
    pose = refine_pose(prediction, points, pixels, K)
    if pose is None:
        return None
    valid = reprojection_errors(points, K @ pose, pixels) < 2.5
    if valid.sum() < 12 or valid.mean() < .5:
        return None
    pose = refine_pose(pose, points[valid], pixels[valid], K)
    if pose is None:
        return None
    return pose, selected[valid], points[valid]


def recover_sequence(tracks, K, height, initial_ids, seed):
    second_index, second_pose, seed_ids, xyz, _, _ = seed
    plane, floor_inliers = find_floor(xyz, np.isin(seed_ids, initial_ids))
    depth = float(np.median(xyz[:, 2]))
    scene = np.full((tracks.shape[1], 3), np.nan)
    floor_xyz = xyz[floor_inliers]
    scene[seed_ids[floor_inliers]] = (
        floor_xyz - ((floor_xyz - plane.origin) @ plane.normal)[:, None] * plane.normal
    )
    inverse_K = np.linalg.inv(K)
    poses = {0: IDENTITY}
    recent = [0]
    # The bottom of these clips gives the floor most of the tracking support.
    floor_pixels = tracks[:, :, 1] > height * .47
    floor_pixels[:, initial_ids] = False
    for index in range(1, len(tracks)):
        last = recent[-1]
        prediction = poses[last]
        for previous in reversed(recent[-5:]):
            ids = np.flatnonzero(floor_pixels[index] & floor_pixels[previous] &
                                 (tracks[previous, :, 1] > height * .49))
            if len(ids) < 18:
                continue
            estimate = register_from_floor(
                (poses[previous], tracks[previous, ids]), tracks[index, ids],
                prediction, scene[ids], plane, K, inverse_K, depth,
            )
            if estimate is None:
                continue
            pose, selected, points = estimate
            distance = np.linalg.norm(
                camera_center(pose[:, :3], pose[:, 3]) - camera_center(prediction[:, :3], prediction[:, 3])
            )
            cosine = (np.trace(prediction[:, :3] @ pose[:, :3].T) - 1.) / 2.
            rotation = np.arccos(np.clip(cosine, -1., 1.))
            gap = index - last
            # A bad planar match can otherwise send the camera across the scene.
            if distance > depth * (.065 * gap + .035) or rotation > .15 * gap + .08:
                continue
            poses[index] = second_pose if index == second_index else pose
            accepted = ids[selected]
            new = ~np.isfinite(scene[accepted, 0])
            scene[accepted[new]] = points[new]
            recent.append(index)
            break
    poses[second_index] = second_pose
    return poses, plane, depth
