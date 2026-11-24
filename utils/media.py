"""Image and video I/O for the homework scripts."""
from contextlib import contextmanager

import cv2
import numpy as np

PHOTO_SUFFIXES = frozenset({'.jpg', '.jpeg', '.png'})


def image_paths(directory, *, suffixes=PHOTO_SUFFIXES):
    return sorted(path for path in directory.glob('*')
                  if path.suffix.lower() in suffixes and path.is_file())


def read_image(path, flags=cv2.IMREAD_COLOR):
    image = cv2.imread(str(path), flags)
    if image is None:
        raise ValueError(f'Cannot read image: {path}')
    return image


def save_image(path, image):
    if not cv2.imwrite(str(path), image):
        raise OSError(f'Could not write {path}')


def frame_size(image):
    """Return the OpenCV size tuple (width, height)."""
    return image.shape[1::-1]


def resize_to_fit(image, max_side):
    """Downsize the longest side to max_side and return the applied scale."""
    scale = min(1., max_side / max(frame_size(image)))
    if scale < 1.:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return image, scale


@contextmanager
def open_video(source):
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open camera or video: {source}')
        yield capture
    finally:
        capture.release()


def video_fps(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    return fps if np.isfinite(fps) and fps > 0 else 30.


def video_frames(capture):
    while True:
        ok, frame = capture.read()
        if not ok:
            return
        yield frame


@contextmanager
def open_video_writer(path, capture, size):
    video = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), video_fps(capture), size)
    try:
        if not video.isOpened():
            raise ValueError(f'Cannot create video: {path}')
        yield video
    finally:
        video.release()
