"""Select object features and track them with forward/backward optical flow."""
import csv

import cv2
import numpy as np


def follow(previous, gray, points):
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01))
    current, forward, _ = cv2.calcOpticalFlowPyrLK(previous, gray, points, None, **options)
    if current is None:
        return points, np.zeros(len(points), dtype=bool)
    back, backward, _ = cv2.calcOpticalFlowPyrLK(gray, previous, current, None, **options)
    if back is None:
        return current, np.zeros(len(points), dtype=bool)
    xy = current.reshape(-1, 2)
    valid = (forward.ravel() == 1) & (backward.ravel() == 1)
    valid &= np.linalg.norm((points - back).reshape(-1, 2), axis=1) < 1.
    valid &= np.isfinite(xy).all(axis=1)
    valid &= (xy[:, 0] >= 0) & (xy[:, 0] < gray.shape[1])
    valid &= (xy[:, 1] >= 0) & (xy[:, 1] < gray.shape[0])
    return current, valid


def track_video(args):
    fx, fy, cx, cy = args.intrinsics
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
    distortion = np.asarray(args.distortion, float)
    capture = cv2.VideoCapture(args.video)
    video = None
    tracks = []
    try:
        ok, first = capture.read()
        if not ok:
            raise ValueError('Cannot read the input video.')
        height, width = first.shape[:2]
        x, y, w, h = args.roi
        if x + w > width or y + h > height:
            raise ValueError('ROI extends outside the video frame.')
        gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        points = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=.01, minDistance=8, mask=mask)
        if points is None or len(points) < 8:
            raise ValueError('Select an object region with at least eight textured corners.')
        ids = np.arange(len(points))
        count = len(points)
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.
        video = cv2.VideoWriter(str(args.output / 'tracking.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not video.isOpened():
            raise ValueError('Cannot create tracking.mp4.')
        with (args.output / 'tracks.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'point_id', 'x_px', 'y_px'])
            frame = first
            previous = gray
            while True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.shape != previous.shape:
                    raise ValueError('Video resolution changed.')
                if tracks and len(points):
                    current, valid = follow(previous, gray, points)
                    points, ids = current[valid], ids[valid]
                observations = np.full((count, 2), np.nan)
                if len(points):
                    observations[ids] = cv2.undistortPoints(points, K, distortion, P=K).reshape(-1, 2)
                tracks.append(observations)
                for point_id, point in zip(ids, points.reshape(-1, 2)):
                    writer.writerow([len(tracks)-1, point_id, *point])
                    location = tuple(np.round(point).astype(int))
                    cv2.circle(frame, location, 3, (0, 255, 0), -1)
                    cv2.putText(frame, str(point_id), location, cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 255), 1)
                cv2.putText(frame, f'Frame {len(tracks)-1}: {len(ids)} / {count} points', (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1)
                video.write(frame)
                if len(tracks) == 1:
                    if not cv2.imwrite(str(args.output / 'selected_points.jpg'), frame):
                        raise ValueError('Cannot save selected_points.jpg.')
                previous = gray
                ok, frame = capture.read()
                if not ok:
                    break
    finally:
        capture.release()
        if video is not None:
            video.release()
    if len(tracks) < 3:
        raise ValueError('Need at least three video frames for the comparison.')
    return np.asarray(tracks), K
