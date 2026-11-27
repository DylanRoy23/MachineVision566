import cv2
import numpy as np

from utils.geometry import to_homogeneous


def position_std(xyz, projections, sigma):
    """EC: 1-sigma X/Y/Z uncertainty of a triangulated point with the cameras held fixed.

    Each view contributes a 2x3 Jacobian J of the projected pixel with respect
    to X. With independent pixel noise sigma, the first-order covariance is
    Sigma_X = sigma^2 (J^T J)^-1, and the square roots of its diagonal are the
    returned standard deviations.
    """
    rows = []
    point = to_homogeneous(xyz)
    for P in projections:
        q = P @ point
        rows.extend([(P[i, :3] * q[2] - q[i] * P[2, :3]) / q[2] ** 2 for i in range(2)])
    J = np.asarray(rows)
    information = J.T @ J
    if np.linalg.cond(information) > 1e12:
        return [float('nan')] * 3
    return np.sqrt(np.maximum(np.diag(sigma ** 2 * np.linalg.inv(information)), 0)).tolist()


def distort_pixels(pixels, K, distortion):
    """Map undistorted pixels back to raw image pixels (inverse of cv2.undistortPoints)."""
    pixels = np.asarray(pixels, float).reshape(-1, 2)
    normalized = (pixels - K[:2, 2]) / (K[0, 0], K[1, 1])
    rays = np.column_stack((normalized, np.ones(len(normalized))))
    raw, _ = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), K, distortion)
    return raw.reshape(-1, 2)


def fit_plane(points, tolerance, iterations=600, seed=0):
    """RANSAC plane: returns (origin, unit normal, inlier mask) refined by SVD on the inliers."""
    rng = np.random.default_rng(seed)
    best = np.zeros(len(points), bool)
    for sample in rng.choice(len(points), (iterations, 3)):
        a, b, c = points[sample]
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        inliers = np.abs((points - a) @ (normal / length)) < tolerance
        if inliers.sum() > best.sum():
            best = inliers
    origin = points[best].mean(axis=0)
    normal = np.linalg.svd(points[best] - origin, full_matrices=False)[2][-1]
    best = np.abs((points - origin) @ normal) < tolerance
    return origin, normal, best
