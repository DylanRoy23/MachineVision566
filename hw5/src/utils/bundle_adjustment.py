import numpy as np
import scipy.linalg
import scipy.sparse


def skew(v):
    zero = np.zeros(len(v))
    return np.stack((
        np.stack((zero, -v[:, 2], v[:, 1]), axis=-1),
        np.stack((v[:, 2], zero, -v[:, 0]), axis=-1),
        np.stack((-v[:, 1], v[:, 0], zero), axis=-1),
    ), axis=1)


def rodrigues(rvecs):
    """Rotation vectors (N,3) to rotation matrices (N,3,3)."""
    theta = np.linalg.norm(rvecs, axis=1)
    small = theta < 1e-12
    k = rvecs / np.where(small, 1., theta)[:, None]
    K = skew(k)
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    R = np.eye(3) + s * K + (1 - c) * (K @ K)
    R[small] = np.eye(3) + skew(rvecs[small])
    return R


def rotation_vectors(R):
    """Rotation matrices (N,3,3) to rotation vectors (N,3)."""
    cosine = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    theta = np.arccos(cosine)
    axis = np.stack((R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), axis=1)
    sin = np.sin(theta)
    out = np.zeros((len(R), 3))
    regular = sin > 1e-6
    out[regular] = axis[regular] * (theta[regular] / (2 * sin[regular]))[:, None]
    tiny = theta <= 1e-6
    out[tiny] = axis[tiny] / 2
    # theta close to pi: take the axis from the symmetric part
    flipped = ~regular & ~tiny
    for index in np.flatnonzero(flipped):
        B = (R[index] + np.eye(3)) / 2
        column = int(np.argmax(np.diag(B)))
        v = B[:, column] / np.sqrt(max(B[column, column], 1e-12))
        out[index] = v * theta[index]
    return out


def project(R, t, X, K):
    """Project points X (N,3) with per-row rotations R (N,3,3) and translations t (N,3)."""
    p = np.einsum('nij,nj->ni', R, X) + t
    uv = p[:, :2] / p[:, 2:3]
    return uv * (K[0, 0], K[1, 1]) + (K[0, 2], K[1, 2]), p


def _block_sum(index, blocks, count):
    """Sum per-observation blocks (N,a,b) into per-owner blocks (count,a,b)."""
    shape = blocks.shape[1:]
    size = int(np.prod(shape))
    flat = (index[:, None] * size + np.arange(size)).ravel()
    out = np.bincount(flat, weights=blocks.reshape(-1), minlength=count * size)
    return out.reshape(count, *shape)


def _huber_weights(errors, huber):
    return np.where(errors <= huber, 1., huber / np.maximum(errors, 1e-12))


def _huber_cost(errors, huber):
    return float(np.sum(np.where(errors <= huber, errors ** 2, 2 * huber * errors - huber ** 2)))


def bundle_adjust(rvecs, tvecs, points, cam_index, point_index, pixels, K, fixed=None,
                  huber=1.0, iterations=30, tolerance=1e-5):
    """Refine camera poses and points.

    rvecs, tvecs: (C,3) world-to-camera poses. points: (P,3).
    cam_index, point_index, pixels: one row per observation.
    fixed: boolean mask (C,) of cameras held constant (at least one fixes the gauge).
    Returns refined (rvecs, tvecs, points, residual_norms).
    """
    n_cams, n_points = len(rvecs), len(points)
    fixed = np.zeros(n_cams, bool) if fixed is None else np.asarray(fixed, bool)
    free = np.flatnonzero(~fixed)
    slot = -np.ones(n_cams, int)
    slot[free] = np.arange(len(free))
    n_free = len(free)
    R = rodrigues(np.asarray(rvecs, float))
    t = np.asarray(tvecs, float).copy()
    X = np.asarray(points, float).copy()
    fx, fy = K[0, 0], K[1, 1]

    def evaluate(R, t, X):
        uv, p = project(R[cam_index], t[cam_index], X[point_index], K)
        residual = uv - pixels
        errors = np.linalg.norm(residual, axis=1)
        errors[p[:, 2] <= 0] = 1e6
        return residual, errors, p

    block_rows = np.repeat(np.arange(3 * n_points), 3)
    block_cols = (np.arange(n_points)[:, None, None] * 3 + np.zeros((1, 3, 1), int) + np.arange(3)).ravel()
    residual, errors, p = evaluate(R, t, X)
    cost = _huber_cost(errors, huber)
    damping = 1e-4
    for _ in range(iterations):
        w = _huber_weights(errors, huber)
        x, y, z = p.T
        J_proj = np.zeros((len(p), 2, 3))
        J_proj[:, 0, 0] = fx / z
        J_proj[:, 0, 2] = -fx * x / z ** 2
        J_proj[:, 1, 1] = fy / z
        J_proj[:, 1, 2] = -fy * y / z ** 2
        rotated = p - t[cam_index]
        J_cam = np.concatenate((-J_proj @ skew(rotated), J_proj), axis=2)      # (N,2,6)
        J_point = J_proj @ R[cam_index]                                          # (N,2,3)

        wJc = J_cam * w[:, None, None]
        wJp = J_point * w[:, None, None]
        V = _block_sum(point_index, np.einsum('nki,nkj->nij', wJp, J_point), n_points)
        g_point = _block_sum(point_index, np.einsum('nki,nk->ni', wJp, residual)[:, :, None], n_points)[:, :, 0]

        on_free = slot[cam_index] >= 0
        free_slot = slot[cam_index[on_free]]
        U = _block_sum(free_slot, np.einsum('nki,nkj->nij', wJc[on_free], J_cam[on_free]), n_free)
        g_cam = _block_sum(free_slot, np.einsum('nki,nk->ni', wJc[on_free], residual[on_free])[:, :, None],
                           n_free)[:, :, 0]
        W_blocks = np.einsum('nki,nkj->nij', wJc[on_free], J_point[on_free])  # (M,6,3)
        rows = (free_slot[:, None, None] * 6 + np.arange(6)[None, :, None]) * np.ones((1, 1, 3), int)
        cols = (point_index[on_free][:, None, None] * 3 + np.arange(3)[None, None, :]) * np.ones((1, 6, 1), int)
        W = scipy.sparse.csr_matrix((W_blocks.ravel(), (rows.ravel(), cols.ravel())),
                                    shape=(6 * n_free, 3 * n_points))

        improved = False
        while damping < 1e10:
            Vd = V + damping * (np.einsum('nii->ni', V)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3))
            V_inv = np.linalg.inv(Vd)
            V_inv_sparse = scipy.sparse.csr_matrix((V_inv.ravel(), (block_rows, block_cols)),
                                                   shape=(3 * n_points, 3 * n_points))
            Ud = U + damping * (np.einsum('nii->ni', U)[:, :, None] * np.eye(6) + 1e-9 * np.eye(6))
            WV = W @ V_inv_sparse
            S = scipy.linalg.block_diag(*Ud) if n_free else np.zeros((0, 0))
            S = S - (WV @ W.T).toarray()
            rhs = -g_cam.ravel() + WV @ g_point.ravel()
            try:
                dc = scipy.linalg.solve(S, rhs, assume_a='sym') if n_free else np.zeros(0)
            except (np.linalg.LinAlgError, ValueError):
                damping *= 10
                continue
            dX = np.einsum('nij,nj->ni', V_inv, -g_point - (W.T @ dc).reshape(-1, 3))
            dc = dc.reshape(-1, 6)
            R_new, t_new = R.copy(), t.copy()
            if n_free:
                R_new[free] = rodrigues(dc[:, :3]) @ R[free]
                t_new[free] = t[free] + dc[:, 3:]
            X_new = X + dX
            residual_new, errors_new, p_new = evaluate(R_new, t_new, X_new)
            cost_new = _huber_cost(errors_new, huber)
            if cost_new < cost:
                relative = (cost - cost_new) / max(cost, 1e-12)
                R, t, X = R_new, t_new, X_new
                residual, errors, p, cost = residual_new, errors_new, p_new, cost_new
                damping = max(damping / 5, 1e-8)
                improved = True
                break
            damping *= 5
        if not improved or relative < tolerance:
            break
    return rotation_vectors(R), t, X, errors
