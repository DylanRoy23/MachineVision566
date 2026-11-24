"""Projective geometry and error measurements."""
import numpy as np


def to_homogeneous(points):
    points = np.asarray(points)
    ones = np.ones((*points.shape[:-1], 1))
    return np.concatenate((points, ones), axis=-1)


def from_homogeneous(points):
    return points[..., :-1] / points[..., -1:]


def project_points(points, matrix, *, min_scale=0.):
    """Project a point or a row array of points using homogeneous coordinates."""
    coordinates = to_homogeneous(points)
    projected = matrix @ coordinates if coordinates.ndim == 1 else coordinates @ matrix.T
    if min_scale and np.any(np.abs(projected[..., -1]) < min_scale):
        raise ValueError('Projection sends points to infinity.')
    return from_homogeneous(projected)


def camera_center(rotation, translation):
    return (-rotation.T @ translation).ravel()


def root_mean_square(values, *, vector_axis=None):
    """Compute RMS for scalars or for the lengths of residual vectors."""
    squares = np.square(values)
    if vector_axis is not None:
        squares = np.sum(squares, axis=vector_axis)
    return float(np.sqrt(np.mean(squares)))
