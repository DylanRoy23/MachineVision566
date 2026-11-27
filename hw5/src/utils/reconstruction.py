"""Write the object point clouds and the two-view comparison."""
import csv

import numpy as np

from .geometry import position_std


def _reprojection_rmse(xyz, cameras, pixels, K):
    errors = []
    for camera, pixel in zip(cameras, pixels):
        q = K @ (camera[:, :3] @ xyz + camera[:, 3])
        errors.append(np.linalg.norm(q[:2] / q[2] - pixel))
    return float(np.sqrt(np.mean(np.square(errors))))


def save_cloud(stem, cloud, camera_of_row, K, sigma=None):
    """Save <stem>.csv (coordinates, views, RMSE, optional EC std) and a coloured <stem>.ply.

    Returns the median length of the (x, y, z) standard deviation when sigma is given.
    """
    ids = cloud.ids()
    stds = []
    with stem.with_suffix('.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        header = ['point_id', 'source', 'x', 'y', 'z', 'views', 'reprojection_rmse_px']
        if sigma is not None:
            header += ['x_std', 'y_std', 'z_std']
        writer.writerow(header)
        for point_id in ids:
            cameras = [camera_of_row(r) for r in cloud.rows[point_id]]
            xyz = cloud.xyz[point_id]
            row = [point_id, cloud.source[point_id], *xyz, len(cameras),
                   _reprojection_rmse(xyz, cameras, cloud.pixels[point_id], K)]
            if sigma is not None:
                std = position_std(xyz, [K @ c for c in cameras], sigma)
                stds.append(std)
                row += std
            writer.writerow(row)
    with stem.with_suffix('.ply').open('w') as file:
        file.write(f'ply\nformat ascii 1.0\nelement vertex {len(ids)}\n'
                   'property float x\nproperty float y\nproperty float z\n'
                   'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for point_id in ids:
            x, y, z = cloud.xyz[point_id]
            b, g, r = np.clip(np.rint(cloud.color[point_id]), 0, 255).astype(int)
            file.write(f'{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n')
    if sigma is None or not stds:
        return None
    return float(np.nanmedian(np.linalg.norm(np.array(stds), axis=1)))


def save_comparison(pair_cloud, video_cloud, output):
    """Per-point difference between the two-view and whole-video positions (same gauge)."""
    shared = sorted(pair_cloud.keys() & video_cloud.keys())
    differences = []
    with (output / 'comparison.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['point_id', 'two_view_x', 'two_view_y', 'two_view_z',
                         'video_x', 'video_y', 'video_z', 'position_difference'])
        for point_id in shared:
            initial, xyz = pair_cloud[point_id], video_cloud[point_id]
            difference = float(np.linalg.norm(xyz - initial))
            differences.append(difference)
            writer.writerow([point_id, *initial, *xyz, difference])
    return dict(points=len(shared), median=float(np.median(differences)) if differences else None)
