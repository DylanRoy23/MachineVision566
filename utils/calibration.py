"""Shared checkerboard calibration for HW4 and HW5."""
from contextlib import closing

import cv2
import numpy as np

from .media import frame_size, image_paths, open_video, read_image, resize_to_fit, video_fps

BOARD = (9, 6)
REFINEMENT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, .01)


def _photo_frames(directory):
    for path in image_paths(directory):
        yield path.name, read_image(path, cv2.IMREAD_GRAYSCALE)


def _sampled_video_frames(path):
    with open_video(path) as capture:
        frame_step = max(1, round(video_fps(capture)))
        index = 0
        while capture.grab():
            if index % frame_step == 0:
                ok, frame = capture.retrieve()
                if not ok:
                    raise ValueError(f'Cannot read calibration video frame {index}.')
                yield f'frame {index}', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            index += 1


def calibrate(source, *, flags=0):
    """Estimate intrinsics and distortion at the input resolution."""
    photos = source.is_dir()
    if not photos and not source.is_file():
        raise ValueError(f'Cannot open calibration source: {source}')
    frames = _photo_frames(source) if photos else _sampled_video_frames(source)
    board_points = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    board_points[:, :2] = np.mgrid[:BOARD[0], :BOARD[1]].T.reshape(-1, 2)
    image_size = None
    observations = []
    with closing(frames):
        for label, gray in frames:
            size = frame_size(gray)
            if image_size is not None and size != image_size:
                raise ValueError(f'Calibration resolution changed at {label}.')
            image_size = size
            detection_image, scale = resize_to_fit(gray, 1000) if photos else (gray, 1.)
            found, corners = cv2.findChessboardCornersSB(
                detection_image, BOARD, cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                if photos:
                    corners = (corners + .5) / scale - .5
                    corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), REFINEMENT)
                observations.append(corners)
    units = 'photos' if photos else 'video frames'
    if len(observations) < 6:
        raise ValueError(f'Need at least six {units} showing the full '
                         f'{BOARD[0]}-by-{BOARD[1]} inner-corner checkerboard.')
    rms, K, distortion, _, _ = cv2.calibrateCamera(
        [board_points] * len(observations), observations, image_size, None, None,
        flags=flags,
    )
    print(f'Calibration: {len(observations)} {units}, {rms:.2f} px RMS reprojection error')
    return K, distortion, image_size


def camera_parameters(source, intrinsics, distortion_override, *, flags=0):
    if intrinsics is None:
        K, distortion, size = calibrate(source, flags=flags)
    else:
        fx, fy, cx, cy = intrinsics
        K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        distortion = np.zeros(5)
        size = None
    if distortion_override is not None:
        distortion = np.asarray(distortion_override, dtype=float)
    return K, distortion, size


def check_calibration_size(size, calibrated_size):
    if calibrated_size is not None and size != calibrated_size:
        raise ValueError(f'Source resolution {size} does not match calibration {calibrated_size}.')
