"""Select object features and track them with forward/backward optical flow."""
import csv
from dataclasses import dataclass
from itertools import chain

import cv2
import numpy as np

from utils.calibration import check_calibration_size
from utils.media import frame_size, open_video, open_video_writer, save_image, video_frames

from .regions import outline_mask


@dataclass
class SceneTracks:
    observations: np.ndarray
    raw_frames: list
    object_views: dict
    frame_size: tuple
    scene_count: int


def draw_tracks(frame, writer, index, ids, pixels, total=None):
    for point_id, point in zip(ids, pixels):
        writer.writerow([index, point_id, *point])
        location = tuple(np.round(point).astype(int))
        cv2.circle(frame, location, 3, (0, 255, 0), -1)
        cv2.putText(frame, str(point_id), location, cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 255), 1)
    count = f'{len(ids)} / {total} points' if total is not None else f'{len(ids)} object tracks'
    cv2.putText(frame, f'Frame {index}: {count}', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 1)


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


def object_mask(gray, roi, region):
    height, width = gray.shape
    mask = np.zeros_like(gray)
    if roi is not None:
        x, y, w, h = roi
        if x + w > width or y + h > height:
            raise ValueError('ROI extends outside the video frame.')
        mask[y:y+h, x:x+w] = 255
    else:
        if (width, height) != (406, 720):
            raise ValueError('The bundled object outline expects a 406-by-720 video. Supply --roi.')
        cv2.fillPoly(mask, [np.asarray(region, dtype=np.int32)], 255)
    return mask


def add_corners(gray, mask, occupied, limit, distance):
    for point in np.rint(occupied[:, 0]).astype(int):
        cv2.circle(mask, tuple(point), distance, 0, -1)
    return cv2.goodFeaturesToTrack(
        gray, maxCorners=limit, qualityLevel=.005, minDistance=distance, mask=mask,
    )


def inside_mask(points, mask):
    x, y = np.rint(points[:, 0]).astype(int).T
    return mask[np.clip(y, 0, mask.shape[0] - 1), np.clip(x, 0, mask.shape[1] - 1)] != 0


def track_scene(args, K, distortion, calibration_size, first_outline, outlines):
    """Replenish tracks across the image and record which ones cross object outlines."""
    records = []
    foreground_records = []
    undistorted = []
    object_views = {}
    foreground_views = {}
    points = np.empty((0, 1, 2), np.float32)
    ids = np.empty(0, dtype=int)
    foreground = points.copy()
    foreground_ids = ids.copy()
    count = 0
    foreground_count = 0
    previous = None
    with open_video(args.video) as capture:
        for index, frame in enumerate(video_frames(capture)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if index == 0:
                size = frame_size(frame)
                check_calibration_size(size, calibration_size)
            elif gray.shape != previous.shape:
                raise ValueError('Video resolution changed.')
            tracked = np.concatenate((points, foreground))
            if len(tracked):
                split = len(points)
                tracked, valid = follow(previous, gray, tracked)
                points, ids = tracked[:split][valid[:split]], ids[valid[:split]]
                foreground = tracked[split:][valid[split:]]
                foreground_ids = foreground_ids[valid[split:]]
            if index % 4 == 0 and len(points) < 520:
                available = np.full_like(gray, 255)
                new = add_corners(gray, available, points, 600 - len(points), 7)
                if new is not None:
                    points = np.concatenate((points, new))
                    ids = np.concatenate((ids, np.arange(count, count + len(new))))
                    count += len(new)
            mask = None
            if index == 0:
                mask = object_mask(gray, args.roi, first_outline)
            elif index in outlines:
                mask = outline_mask(gray.shape, outlines[index])
            if mask is not None:
                object_views[index] = ids[inside_mask(points, mask)]
                occupied = np.concatenate((points, foreground))
                new = add_corners(gray, mask.copy(), occupied, 160, 4)
                if new is not None:
                    foreground = np.concatenate((foreground, new))
                    foreground_ids = np.concatenate((foreground_ids, np.arange(
                        foreground_count, foreground_count + len(new),
                    )))
                    foreground_count += len(new)
                foreground_views[index] = foreground_ids[inside_mask(foreground, mask)]
            records.append((ids, points.reshape(-1, 2)))
            foreground_records.append((foreground_ids, foreground.reshape(-1, 2)))
            combined = np.concatenate((points, foreground))
            normalized = combined.reshape(-1, 2)
            if len(combined):
                normalized = cv2.undistortPoints(combined, K, distortion, P=K).reshape(-1, 2)
            undistorted.append(normalized)
            previous = gray
    if len(records) < 3:
        raise ValueError('Need at least three video frames for reconstruction.')

    observations = np.full((len(records), count + foreground_count, 2), np.nan, np.float32)
    for index, ((ids, pixels), (extra_ids, extra_pixels)) in enumerate(zip(records, foreground_records)):
        combined_ids = np.concatenate((ids, extra_ids + count))
        observations[index, combined_ids] = undistorted[index]
        records[index] = combined_ids, np.concatenate((pixels, extra_pixels))
    for index in object_views:
        object_views[index] = np.concatenate((object_views[index], foreground_views[index] + count))
    return SceneTracks(observations, records, object_views, size, count)


def save_scene_tracking(args, tracks, object_points):
    with open_video(args.video) as capture, \
         open_video_writer(args.output / 'tracking.mp4', capture, tracks.frame_size) as video, \
         (args.output / 'tracks.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame', 'point_id', 'x_px', 'y_px'])
        for index, (frame, (ids, pixels)) in enumerate(zip(video_frames(capture), tracks.raw_frames)):
            selected = np.isin(ids, tracks.object_views[0]) if index == 0 else object_points[ids]
            draw_tracks(frame, writer, index, ids[selected], pixels[selected])
            video.write(frame)
            if index == 0:
                save_image(args.output / 'selected_points.jpg', frame)


def track_video(args, K, distortion, calibration_size, region):
    tracks = []
    with open_video(args.video) as capture:
        frames = video_frames(capture)
        first = next(frames, None)
        if first is None:
            raise ValueError(f'Cannot read the input video: {args.video}')
        size = frame_size(first)
        check_calibration_size(size, calibration_size)
        gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
        mask = object_mask(gray, args.roi, region)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=.01, minDistance=8, mask=mask)
        if points is None or len(points) < 8:
            raise ValueError('Select an object region with at least eight textured corners.')
        ids = np.arange(len(points))
        count = len(points)
        video_path = args.output / 'tracking.mp4'
        tracks_path = args.output / 'tracks.csv'
        with open_video_writer(video_path, capture, size) as video, tracks_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['frame', 'point_id', 'x_px', 'y_px'])
            previous = gray
            for index, frame in enumerate(chain((first,), frames)):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if gray.shape != previous.shape:
                    raise ValueError('Video resolution changed.')
                if index and len(points):
                    current, valid = follow(previous, gray, points)
                    points, ids = current[valid], ids[valid]
                observations = np.full((count, 2), np.nan)
                if len(points):
                    observations[ids] = cv2.undistortPoints(points, K, distortion, P=K).reshape(-1, 2)
                tracks.append(observations)
                draw_tracks(frame, writer, index, ids, points.reshape(-1, 2), count)
                video.write(frame)
                if index == 0:
                    save_image(args.output / 'selected_points.jpg', frame)
                previous = gray
    if len(tracks) < 3:
        raise ValueError('Need at least three video frames for the comparison.')
    return np.asarray(tracks)
