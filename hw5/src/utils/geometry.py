import numpy as np

from utils.geometry import from_homogeneous, project_points, root_mean_square, to_homogeneous


def triangulate(projections, pixels):
    rows = []
    for P, (u, v) in zip(projections, pixels):
        rows.extend([u * P[2] - P[0], v * P[2] - P[1]])
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    point = vt[-1]
    if abs(point[3]) < 1e-10:
        raise ValueError('Point is at infinity.')
    return from_homogeneous(point)


def point_quality(xyz, poses, pixels, K):
    if not np.isfinite(xyz).all():
        return False, float('inf')
    point = to_homogeneous(xyz)
    if any((pose @ point)[2] <= 0 for pose in poses):
        return False, float('inf')
    errors = [np.linalg.norm(project_points(xyz, K @ pose) - pixel) for pose, pixel in zip(poses, pixels)]
    return max(errors) <= 3., root_mean_square(errors)


def position_std(xyz, projections, sigma):
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
