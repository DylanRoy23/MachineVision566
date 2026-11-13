"""Marker pose, relative camera motion, and optional noise estimation."""
import cv2
import numpy as np


def marker_points(side):
    s = side / 2
    return np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], np.float64)


def pose(points, pixels, K, distortion):
    ok, rvec, tvec = cv2.solvePnP(points, pixels, K, distortion,
                                 flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not np.all(np.isfinite(tvec)) or tvec[2, 0] <= 0:
        raise ValueError('No valid marker pose.')
    R = cv2.Rodrigues(rvec)[0]
    return rvec, tvec, R, (-R.T @ tvec).ravel()


def displacement(reference, current):
    delta = current[3] - reference[3]
    distance = float(np.linalg.norm(delta))
    cosine = (np.trace(current[2] @ reference[2].T) - 1) / 2
    angle = float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))
    return delta, distance, angle


def reprojection_error(points, pixels, estimate, K, distortion):
    projected = cv2.projectPoints(points, estimate[0], estimate[1], K, distortion)[0]
    return float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - pixels) ** 2, axis=1))))


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
        samples.append([*delta, *np.degrees(rotation_error), np.linalg.norm(delta)])
    return np.std(samples, axis=0, ddof=1).tolist() if len(samples) > 1 else [''] * 7
