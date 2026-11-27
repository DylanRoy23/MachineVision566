import time

import cv2
import numpy as np

from .bundle_adjustment import bundle_adjust, project, rodrigues

MAX_ERROR_PX = 2.
MIN_ANGLE_DEG = 1.5


def camera_centers(cams):
    R = rodrigues(cams[:, :3])
    return -np.einsum('nji,nj->ni', R, cams[:, 3:])


def projection_matrices(K, cams):
    return K @ np.concatenate((rodrigues(cams[:, :3]), cams[:, 3:, None]), axis=2)


def dlt(P, uv):
    """Linear triangulation: each view adds rows u p3 - p1 and v p3 - p2 to A, and A X = 0."""
    A = (uv[:, :, None] * P[:, 2:3, :] - P[:, :2, :]).reshape(-1, 4)
    X = np.linalg.svd(A)[2][-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-12 else None


def reprojection(P, X, uv):
    q = P[:, :, :3] @ X + P[:, :, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.linalg.norm(q[:, :2] / q[:, 2:3] - uv, axis=1), q[:, 2]


def max_ray_angle(X, centers):
    rays = X - centers
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(np.min(rays @ rays.T), -1, 1)))


def triangulate_robust(P, centers, uv, threshold=MAX_ERROR_PX, min_angle=MIN_ANGLE_DEG):
    """Triangulate one track. Returns (X, inlier mask) or None."""
    X = dlt(P, uv)
    if X is None:
        return None
    errors, depth = reprojection(P, X, uv)
    inliers = (errors < threshold) & (depth > 0)
    if not inliers.all() and len(uv) > 2:
        sample = np.unique(np.linspace(0, len(uv) - 1, min(len(uv), 8)).astype(int))
        best = inliers
        for a_pos, a in enumerate(sample):
            for b in sample[a_pos + 1:]:
                Y = dlt(P[[a, b]], uv[[a, b]])
                if Y is None:
                    continue
                e, d = reprojection(P, Y, uv)
                candidate = (e < threshold) & (d > 0)
                if candidate.sum() > best.sum():
                    best = candidate
        if best.sum() >= 2:
            X = dlt(P[best], uv[best])
            if X is None:
                return None
            errors, depth = reprojection(P, X, uv)
            inliers = (errors < threshold) & (depth > 0)
    if inliers.sum() < 2 or max_ray_angle(X, centers[inliers]) < min_angle:
        return None
    return X, inliers


class IncrementalSfM:
    def __init__(self, pixels, K, log=print):
        self.T = np.asarray(pixels, float)
        self.K = K
        self.n_frames, self.n_tracks = self.T.shape[:2]
        self.visible = np.isfinite(self.T[..., 0])
        self.rejected = np.zeros_like(self.visible)
        self.cams = np.full((self.n_frames, 6), np.nan)   # rvec | tvec, world to camera
        self.registered = np.zeros(self.n_frames, bool)
        self.X = np.full((self.n_tracks, 3), np.nan)
        self.triangulated = np.zeros(self.n_tracks, bool)
        self.anchor = None
        self.log = log

    def usable(self):
        return self.visible & ~self.rejected

    # -------------------------------------------------------------- start
    def initialize(self, min_angle=2.):
        T, K = self.T, self.K
        best = None
        for a in range(0, self.n_frames, max(1, self.n_frames // 30)):
            for gap in (3, 5, 7, 9, 12):
                b = a + gap
                if b >= self.n_frames:
                    continue
                ids = np.flatnonzero(self.visible[a] & self.visible[b])
                if len(ids) < 100:
                    continue
                pa, pb = T[a, ids], T[b, ids]
                E, mask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, .999, 1.)
                if E is None or E.shape != (3, 3):
                    continue
                _, R, t, mask = cv2.recoverPose(E, pa, pb, K, mask=mask)
                good = mask.ravel() > 0
                if good.sum() < 50:
                    continue
                X = cv2.triangulatePoints(K @ np.eye(3, 4), K @ np.column_stack((R, t)),
                                          np.ascontiguousarray(pa[good].T), np.ascontiguousarray(pb[good].T))
                X = (X[:3] / X[3]).T
                second_center = -R.T @ t.ravel()
                ray_a = X / np.linalg.norm(X, axis=1, keepdims=True)
                ray_b = X - second_center
                ray_b /= np.linalg.norm(ray_b, axis=1, keepdims=True)
                angle = np.degrees(np.arccos(np.clip(np.sum(ray_a * ray_b, axis=1), -1, 1)))
                ok = (X[:, 2] > 0) & ((X @ R.T + t.ravel())[:, 2] > 0) & (angle > min_angle)
                if ok.sum() < 50:
                    continue
                score = ok.sum() * min(np.median(angle[ok]) / 6., 1.)
                if best is None or score > best[0]:
                    best = (score, a, b, R, t, ids[good][ok], X[ok], np.median(angle[ok]))
        if best is None:
            raise ValueError('No keyframe pair has enough parallax to start the reconstruction.')
        _, a, b, R, t, ids, X, angle = best
        self.cams[a] = 0.
        self.cams[b] = np.hstack((cv2.Rodrigues(R)[0].ravel(), t.ravel()))
        self.registered[[a, b]] = True
        self.anchor = a
        self.X[ids] = X
        self.triangulated[ids] = True
        self.bundle(np.array([a, b]), iterations=20)
        return a, b, len(ids), angle

    # ---------------------------------------------------------- register
    def register(self, frame, threshold=3., min_inliers=30):
        ids = np.flatnonzero(self.usable()[frame] & self.triangulated)
        if len(ids) < min_inliers:
            return False
        X, uv = self.X[ids], self.T[frame, ids]
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            X, uv, self.K, None, iterationsCount=1000, reprojectionError=threshold,
            confidence=.9999, flags=cv2.SOLVEPNP_EPNP)
        if not ok or inliers is None or len(inliers) < min_inliers:
            return False
        inliers = inliers.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(X[inliers], uv[inliers], self.K, None, rvec, tvec)
        cam = np.hstack((rvec.ravel(), tvec.ravel()))
        R = np.repeat(rodrigues(cam[None, :3]), len(X), axis=0)
        projected, p = project(R, np.repeat(cam[None, 3:], len(X), axis=0), X, self.K)
        good = (np.linalg.norm(projected - uv, axis=1) < threshold) & (p[:, 2] > 0)
        if good.sum() < min_inliers or good.mean() < .25:
            return False
        self.rejected[frame, ids[~good]] = True
        self.cams[frame] = cam
        self.registered[frame] = True
        return True

    # ------------------------------------------------------- triangulate
    def triangulate(self, tracks):
        frames = np.flatnonzero(self.registered)
        P = projection_matrices(self.K, self.cams[frames])
        centers = camera_centers(self.cams[frames])
        usable = self.usable()
        added = 0
        for track in tracks:
            seen = np.flatnonzero(usable[frames, track])
            if len(seen) < 2:
                continue
            result = triangulate_robust(P[seen], centers[seen], self.T[frames[seen], track])
            if result is None:
                continue
            X, inliers = result
            self.X[track] = X
            self.triangulated[track] = True
            self.rejected[frames[seen[~inliers]], track] = True
            added += 1
        return added

    # ------------------------------------------------------------- adjust
    def bundle(self, free_frames, tracks=None, iterations=10):
        usable = self.usable()
        if tracks is None:
            tracks = np.flatnonzero(self.triangulated & usable[free_frames].any(axis=0))
        frames = np.flatnonzero(self.registered & usable[:, tracks].any(axis=1))
        sub = usable[np.ix_(frames, tracks)]
        fi, ti = np.nonzero(sub)
        fixed = ~np.isin(frames, free_frames)
        if not fixed.any():
            fixed[np.searchsorted(frames, self.anchor)] = True
        rv, tv, X, errors = bundle_adjust(
            self.cams[frames, :3], self.cams[frames, 3:], self.X[tracks], fi, ti,
            self.T[frames[fi], tracks[ti]], self.K, fixed=fixed, iterations=iterations)
        self.cams[frames] = np.hstack((rv, tv))
        self.X[tracks] = X
        return len(frames), len(tracks), len(errors), float(np.median(errors))

    def local_bundle(self, frame, neighbours=10):
        frames = np.flatnonzero(self.registered)
        shared = (self.visible[frames] & self.visible[frame] & self.triangulated).sum(axis=1)
        free = np.sort(frames[np.argsort(-shared)][:neighbours + 1])
        tracks = np.flatnonzero(self.triangulated & self.usable()[frame])
        return self.bundle(free, tracks, iterations=5)

    def global_bundle(self, iterations=10):
        return self.bundle(np.flatnonzero(self.registered), iterations=iterations)

    def observation_errors(self):
        frames = np.flatnonzero(self.registered)
        tracks = np.flatnonzero(self.triangulated)
        fi, ti = np.nonzero(self.visible[np.ix_(frames, tracks)])
        f, t = frames[fi], tracks[ti]
        R = rodrigues(self.cams[f, :3])
        projected, p = project(R, self.cams[f, 3:], self.X[t], self.K)
        errors = np.linalg.norm(projected - self.T[f, t], axis=1)
        errors[p[:, 2] <= 0] = np.inf
        return f, t, errors

    def filter(self, threshold=MAX_ERROR_PX):
        """Reject observations above threshold, re-admit ones that fit again, drop weak points."""
        f, t, errors = self.observation_errors()
        outlier = errors > threshold
        self.rejected[f, t] = outlier
        counts = np.bincount(t[~outlier], minlength=self.n_tracks)
        drop = self.triangulated & (counts < 2)
        self.triangulated[drop] = False
        self.X[drop] = np.nan
        return int(outlier.sum()), int(drop.sum())

    def retriangulate(self):
        candidates = np.flatnonzero(~self.triangulated & (self.usable()[self.registered].sum(axis=0) >= 2))
        return self.triangulate(candidates)

    # -------------------------------------------------------------- run
    def run(self, frame_labels=None):
        labels = np.arange(self.n_frames) if frame_labels is None else np.asarray(frame_labels)
        start = time.time()
        a, b, count, angle = self.initialize()
        self.log(f'  SfM start: keyframes at frames {labels[a]} and {labels[b]}, {count} points, '
                 f'median triangulation angle {angle:.1f} deg')
        last_global = 2
        failures = np.zeros(self.n_frames, int)
        while True:
            candidates = ~self.registered & (failures < 3)
            counts = (self.usable() & self.triangulated).sum(axis=1)
            counts[~candidates] = -1
            frame = int(np.argmax(counts))
            if counts[frame] < 30:
                break
            if not self.register(frame):
                failures[frame] += 1
                continue
            self.triangulate(np.flatnonzero(self.visible[frame] & ~self.triangulated))
            self.local_bundle(frame)
            registered = int(self.registered.sum())
            if registered >= 1.25 * last_global or registered - last_global >= 40:
                stats = self.global_bundle()
                self.filter()
                self.retriangulate()
                self.log(f'    {registered}/{self.n_frames} keyframes, {self.triangulated.sum()} points, '
                         f'median error {stats[3]:.2f} px  [{time.time() - start:.0f} s]')
                last_global = registered
        for _ in range(2):
            self.global_bundle(iterations=30)
            self.filter()
            self.retriangulate()
        stats = self.global_bundle(iterations=30)
        self.filter()
        self.log(f'  SfM done: {self.registered.sum()}/{self.n_frames} keyframes, {self.triangulated.sum()} points, '
                 f'median error {stats[3]:.3f} px [{time.time() - start:.0f} s]')
        return self

    def normalize(self, origin, scale_frame):
        """Express everything in the origin camera's frame, scaled so |C(scale_frame)| = 1."""
        R0 = rodrigues(self.cams[origin:origin + 1, :3])[0]
        t0 = self.cams[origin, 3:].copy()
        frames = np.flatnonzero(self.registered)
        R = rodrigues(self.cams[frames, :3]) @ R0.T
        t = self.cams[frames, 3:] - np.einsum('nij,j->ni', R, t0)
        X = self.X[self.triangulated] @ R0.T + t0
        cams = np.full_like(self.cams, np.nan)
        cams[frames, :3] = np.array([cv2.Rodrigues(r)[0].ravel() for r in R])
        cams[frames, 3:] = t
        baseline = np.linalg.norm(camera_centers(cams[[scale_frame]])[0])
        if baseline < 1e-9:
            raise ValueError('The scale keyframe coincides with the origin keyframe.')
        cams[frames, 3:] /= baseline
        self.cams = cams
        self.X[self.triangulated] = X / baseline
        return baseline
