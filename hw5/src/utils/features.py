import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from utils.media import open_video, video_frames

SEQUENTIAL_WINDOW = 5     # match each keyframe with the next 5 keyframes
LOOP_STRIDE = 3           # test loop pairs on every 3rd keyframe
LOOP_FEATURES = 1000      # strongest features used to screen loop pairs
MIN_PAIR_INLIERS = 20
MIN_LOOP_INLIERS = 40


@dataclass
class KeyframeTracks:
    frames: np.ndarray        # video frame index of each keyframe row
    pixels: np.ndarray        # (keyframes, tracks, 2) undistorted pixels, NaN if unobserved
    colors: np.ndarray        # (tracks, 3) BGR colour of each track
    images: dict              # keyframe index -> BGR image (for colouring other points)
    loop_pairs: list          # (frame_a, frame_b, inliers) for pairs > 150 frames apart
    keypoints: list           # per keyframe: (raw pixels, undistorted pixels, BGR colours)
    track_of: list            # per keyframe: track id of each keypoint, -1 if it joined no track
    descriptors: dict         # keyframe row -> SIFT descriptors (only rows up to keep_descriptors_until)


def _flann():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=4), dict(checks=48))


def match_pair(a, b, K, ratio=.8, threshold=1.5, minimum=MIN_PAIR_INLIERS):
    """Ratio-test SIFT matches verified by an essential matrix fitted with RANSAC.

    a, b are (undistorted_pixels, descriptors). Returns index arrays (ia, ib) or None.
    """
    if len(a[1]) < minimum or len(b[1]) < minimum:
        return None
    knn = _flann().knnMatch(a[1], b[1], k=2)
    pairs = [(m.queryIdx, m.trainIdx) for m, n in (p for p in knn if len(p) == 2)
             if m.distance < ratio * n.distance]
    if len(pairs) < minimum:
        return None
    ia, ib = np.array(pairs).T
    _, unique = np.unique(ib, return_index=True)     # one match per feature in b
    ia, ib = ia[unique], ib[unique]
    cv2.setRNGSeed(0)   # OpenCV's RNG is per thread; seeding keeps RANSAC repeatable
    E, mask = cv2.findEssentialMat(a[0][ia], b[0][ib], K, method=cv2.RANSAC,
                                   prob=.999, threshold=threshold)
    if E is None or mask is None:
        return None
    inliers = mask.ravel() > 0
    if inliers.sum() < minimum:
        return None
    return ia[inliers], ib[inliers]


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        parent = self.parent
        parent.setdefault(item, item)
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != root:
            parent[item], item = root, parent[item]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def keyframe_indices(frame_count, step, required=()):
    return np.array(sorted(set(range(0, frame_count, step)) | {f for f in required if f < frame_count}))


def read_keyframes(video, step, required=()):
    with open_video(video) as capture:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        wanted = set(keyframe_indices(max(frame_count, 1), step, required).tolist())
        images = {}
        for index, frame in enumerate(video_frames(capture)):
            if index in wanted or (frame_count <= 0 and index % step == 0):
                images[index] = frame
    return images


def build_keyframe_tracks(video, K, distortion, step=3, required=(), features=3000,
                          keep_descriptors_until=0, log=print):
    images = read_keyframes(video, step, required)
    frames = np.array(sorted(images))
    sift = cv2.SIFT_create(nfeatures=features)
    data = []
    for frame in frames:
        gray = cv2.cvtColor(images[frame], cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        raw = np.float64([k.pt for k in keypoints]).reshape(-1, 1, 2)
        if descriptors is None or not len(keypoints):
            data.append((np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 128), np.float32),
                         np.empty(0), np.empty((0, 3))))
            continue
        undistorted = cv2.undistortPoints(raw, K, distortion, P=K).reshape(-1, 2)
        x, y = np.rint(raw.reshape(-1, 2)).astype(int).T
        h, w = gray.shape
        colors = images[frame][np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)]
        response = np.array([k.response for k in keypoints])
        data.append((raw.reshape(-1, 2), undistorted, descriptors, response, colors))

    # OpenCV releases the GIL while matching, so pairs are matched in parallel threads.
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        def match_all(pairs, source, **options):
            results = pool.map(lambda pair: match_pair(source[pair[0]], source[pair[1]], K, **options), pairs)
            return {pair: result for pair, result in zip(pairs, results) if result is not None}

        full = [d[1:3] for d in data]
        pairs = [(i, j) for i in range(len(frames))
                 for j in range(i + 1, min(i + 1 + SEQUENTIAL_WINDOW, len(frames)))]
        matches = match_all(pairs, full)
        sequential = len(matches)

        # Loop closure: screen distant pairs with the strongest features, then match fully.
        strongest = []
        for raw, und, des, response, _ in data:
            keep = np.argsort(-response)[:LOOP_FEATURES]
            strongest.append((und[keep], des[keep]))
        sampled = range(0, len(frames), LOOP_STRIDE)
        distant = [(i, j) for i in sampled for j in sampled if j - i > SEQUENTIAL_WINDOW]
        screened = match_all(distant, strongest, ratio=.75, minimum=MIN_LOOP_INLIERS // 2)
        verified = match_all(sorted(screened), full, ratio=.75, minimum=MIN_LOOP_INLIERS)
    matches.update(verified)
    loops = [(int(frames[i]), int(frames[j]), len(result[0])) for (i, j), result in sorted(verified.items())]

    union = _UnionFind()
    for (i, j), (ia, ib) in matches.items():
        for a, b in zip(ia.tolist(), ib.tolist()):
            union.union((i, a), (j, b))
    groups = {}
    for key in list(union.parent):
        groups.setdefault(union.find(key), []).append(key)
    # A track that contains two features from the same keyframe joined two
    # different points through a bad match, so it is discarded.
    tracks = [members for members in groups.values()
              if len(members) >= 3 and len({k for k, _ in members}) == len(members)]
    pixels = np.full((len(frames), len(tracks), 2), np.nan)
    colors = np.zeros((len(tracks), 3))
    for track, members in enumerate(tracks):
        samples = []
        for keyframe, feature in members:
            pixels[keyframe, track] = data[keyframe][1][feature]
            samples.append(data[keyframe][4][feature])
        colors[track] = np.median(samples, axis=0)
    track_of = [np.full(len(d[0]), -1) for d in data]
    for track, members in enumerate(tracks):
        for keyframe, feature in members:
            track_of[keyframe][feature] = track
    keypoints = [(d[0], d[1], d[4]) for d in data]
    descriptors = {row: data[row][2] for row in range(len(frames)) if frames[row] <= keep_descriptors_until}
    long_loops = [loop for loop in loops if loop[1] - loop[0] > 150]
    log(f'  SIFT: {len(frames)} keyframes, {sequential} sequential pairs, '
        f'{len(loops)} loop pairs ({len(long_loops)} more than 150 frames apart), {len(tracks)} tracks')
    return KeyframeTracks(frames, pixels, colors, images, long_loops,
                          keypoints, track_of, descriptors)
