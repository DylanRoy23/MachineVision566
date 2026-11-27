from dataclasses import dataclass

import cv2
import numpy as np

from utils.calibration import check_calibration_size
from utils.media import frame_size, open_video, open_video_writer, save_image, video_frames

from .regions import outline_mask


@dataclass
class ObjectTracks:
    observations: np.ndarray   # (frames, tracks, 2) undistorted pixels, NaN once a track is lost
    raw_frames: list           # per frame: (track ids, raw pixels), for the tracking video
    frame_size: tuple


def draw_tracks(frame, index, ids, pixels):
    for point_id, point in zip(ids, pixels):
        location = tuple(np.round(point).astype(int))
        cv2.circle(frame, location, 3, (0, 255, 0), -1)
        cv2.putText(frame, str(point_id), location, cv2.FONT_HERSHEY_SIMPLEX, .3, (0, 255, 255), 1)
    cv2.putText(frame, f'Frame {index}: {len(ids)} object tracks', (10, 25),
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


def track_object(args, K, distortion, calibration_size, first_outline, outlines):
    """Select corners inside the object outline (frame 0 and each outline frame) and follow them with KLT."""
    records, undistorted = [], []
    points = np.empty((0, 1, 2), np.float32)
    ids = np.empty(0, dtype=int)
    count = 0
    previous = None
    with open_video(args.video) as capture:
        for index, frame in enumerate(video_frames(capture)):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if index == 0:
                size = frame_size(frame)
                check_calibration_size(size, calibration_size)
            elif gray.shape != previous.shape:
                raise ValueError('Video resolution changed.')
            if len(points):
                points, valid = follow(previous, gray, points)
                points, ids = points[valid], ids[valid]
            mask = None
            if index == 0:
                mask = object_mask(gray, args.roi, first_outline)
            elif index in outlines:
                mask = outline_mask(gray.shape, outlines[index])
            if mask is not None:
                new = add_corners(gray, mask.copy(), points, 160, 4)
                if new is not None:
                    points = np.concatenate((points, new))
                    ids = np.concatenate((ids, np.arange(count, count + len(new))))
                    count += len(new)
            records.append((ids, points.reshape(-1, 2)))
            undistorted.append(cv2.undistortPoints(points, K, distortion, P=K).reshape(-1, 2)
                               if len(points) else np.empty((0, 2)))
            previous = gray
    if len(records) < 3:
        raise ValueError('Need at least three video frames for reconstruction.')
    observations = np.full((len(records), count, 2), np.nan, np.float32)
    for index, (track_ids, _) in enumerate(records):
        observations[index, track_ids] = undistorted[index]
    return ObjectTracks(observations, records, size)


def save_tracking(args, tracks):
    """tracking.mp4 shows every object track; selected_points.jpg shows the points chosen in frame 0."""
    with open_video(args.video) as capture, \
         open_video_writer(args.output / 'tracking.mp4', capture, tracks.frame_size) as video:
        for index, (frame, (ids, pixels)) in enumerate(zip(video_frames(capture), tracks.raw_frames)):
            draw_tracks(frame, index, ids, pixels)
            video.write(frame)
            if index == 0:
                save_image(args.output / 'selected_points.jpg', frame)
