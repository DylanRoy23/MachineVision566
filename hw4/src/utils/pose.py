"""Marker pose, relative camera motion, and optional noise estimation."""
import cv2
import numpy as np

from utils.geometry import camera_center, root_mean_square


def marker_points(side):
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)


def pose(points, pixels, K, distortion):
    ok, rvec, tvec = cv2.solvePnP(points, pixels, K, distortion,
                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0:
        raise ValueError('No valid marker pose.')
    R = cv2.Rodrigues(rvec)[0]
    return rvec, tvec, R, camera_center(R, tvec)


def displacement(reference, current):
    delta = current[3] - reference[3]
    distance = float(np.linalg.norm(delta))
    cosine = (np.trace(current[2] @ reference[2].T) - 1) / 2
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return delta, distance, angle


def fixed_orientation_displacement(reference, current):
    """Camera displacement assuming the camera translated without rotating.

    The camera centre in marker coordinates is C = -R^T t. A small, distant
    marker gives a noisy R, and the error is multiplied by the marker distance
    |t| (1 degree at 1.9 m moves C by 3.3 cm). If the rig only translates, R is
    the same in both frames, so C_i - C_0 = -R^T (t_i - t_0) and the distance
    moved is |t_i - t_0|. That depends only on the marker position, which is
    well conditioned.
    """
    return float(np.linalg.norm(current[1] - reference[1]))


def reprojection_error(points, pixels, estimate, K, distortion):
    projected = cv2.projectPoints(points, estimate[0], estimate[1], K, distortion)[0]
    return root_mean_square(projected.reshape(4, 2) - pixels, vector_axis=1)


def pose_uncertainty(points, initial_pixels, pixels, K, distortion, sigma, rng):
    samples = []
    nominal_initial = pose(points, initial_pixels, K, distortion)
    nominal_current = pose(points, pixels, K, distortion)
    nominal_rotation = nominal_initial[2] @ nominal_current[2].T
    for _ in range(100):
        try:
            initial = pose(points, initial_pixels + rng.normal(0, sigma, (4, 2)), K, distortion)
            current = pose(points, pixels + rng.normal(0, sigma, (4, 2)), K, distortion)
        except ValueError:
            continue
        delta = current[3] - initial[3]
        relative_rotation = initial[2] @ current[2].T
        rotation_error = cv2.Rodrigues(relative_rotation @ nominal_rotation.T)[0].ravel()
        samples.append([*delta, *np.degrees(rotation_error), np.linalg.norm(delta),
                        fixed_orientation_displacement(initial, current)])
    return np.std(samples, axis=0, ddof=1).tolist() if len(samples) > 1 else [''] * 8
