import numpy as np


def project(P, xyz):
    q = P @ np.append(xyz, 1.)
    return q[:2] / q[2]


def triangulate(projections, pixels):
    rows = []
    for P, (u, v) in zip(projections, pixels):
        rows.extend([u * P[2] - P[0], v * P[2] - P[1]])
    _, _, vt = np.linalg.svd(rows, full_matrices=False)
    point = vt[-1]
    if abs(point[3]) < 1e-10:
        raise ValueError('Point is at infinity.')
    return point[:3] / point[3]


def point_quality(xyz, poses, pixels, K):
    if not np.isfinite(xyz).all():
        return False, float('inf')
    if any((pose @ np.append(xyz, 1.))[2] <= 0 for pose in poses):
        return False, float('inf')
    errors = [np.linalg.norm(project(K @ pose, xyz) - pixel) for pose, pixel in zip(poses, pixels)]
    return max(errors) <= 3., float(np.sqrt(np.mean(np.square(errors))))


def position_std(xyz, projections, sigma):
    rows = []
    for P in projections:
        q = P @ np.append(xyz, 1.)
        rows.extend([(P[i, :3] * q[2] - q[i] * P[2, :3]) / q[2] ** 2 for i in range(2)])
    J = np.asarray(rows)
    information = J.T @ J
    if np.linalg.cond(information) > 1e12:
        return [float('nan')] * 3
    return np.sqrt(np.maximum(np.diag(sigma ** 2 * np.linalg.inv(information)), 0)).tolist()
