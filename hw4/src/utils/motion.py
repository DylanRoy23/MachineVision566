"""Track one fixed marker and save camera-motion measurements."""
import csv
from itertools import chain
from pathlib import Path

import cv2
import numpy as np

from utils.calibration import camera_parameters, check_calibration_size
from utils.media import (
    frame_size, image_paths, open_video, open_video_writer, read_image, save_image, video_frames,
)

from .cli import PHOTO_STEP_CM
from .markers import DICTIONARY
from .pose import displacement, pose_uncertainty, marker_points, pose, reprojection_error

COLUMNS = ['frame', 'camera_x_m', 'camera_y_m', 'camera_z_m',
           'dx_m', 'dy_m', 'dz_m', 'displacement_m', 'rotation_deg',
           'reprojection_rmse_px', 'dx_std_m', 'dy_std_m', 'dz_std_m',
           'rotation_x_std_deg', 'rotation_y_std_deg', 'rotation_z_std_deg', 'distance_std_m']


class MarkerTracker:
    def __init__(self, args):
        self.args = args
        self.K, self.distortion, self.image_size = camera_parameters(
            args.calibration, args.intrinsics, args.distortion, flags=cv2.CALIB_FIX_K3,
        )
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(DICTIONARY, parameters)
        self.points = marker_points(args.marker_size)
        self.reference = None
        self.reference_pixels = None
        self.last_distance = None
        self.rng = np.random.default_rng(42)

    def process(self, frame, index):
        """Draw detections and return a CSV row, or None when no pose is valid."""
        check_calibration_size(frame_size(frame), self.image_size)
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if ids is None:
            return None
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        matches = np.flatnonzero(ids.ravel() == self.args.marker_id)
        if not len(matches):
            return None
        pixels = corners[matches[0]].reshape(4, 2).astype(np.float64)
        try:
            current = pose(self.points, pixels, self.K, self.distortion)
        except ValueError:
            return None
        cv2.drawFrameAxes(frame, self.K, self.distortion, *current[:2], self.args.marker_size / 2)
        first = self.reference is None
        if first:
            self.reference = current
            self.reference_pixels = pixels.copy()
            save_image(self.args.output / 'initial_frame.jpg', frame)
        delta, distance, angle = displacement(self.reference, current)
        error = reprojection_error(self.points, pixels, current, self.K, self.distortion)
        std = [0.] * 7 if first and self.args.extra_credit else [''] * 7
        if self.args.extra_credit and not first:
            std = pose_uncertainty(self.points, self.reference_pixels, pixels,
                                   self.K, self.distortion, self.args.corner_sigma, self.rng)
        self.last_distance = distance
        return [index, *current[3], *delta, distance, angle, error, *std]


def annotate(frame, row):
    message = 'No valid target-marker pose'
    if row is not None:
        distance, angle, error = row[7:10]
        message = f'Motion: {distance * 100:.1f} cm | rotation: {angle:.1f} deg | error: {error:.2f} px'
        if distance >= .30:
            message += ' | >= 30 cm'
    scale = max(1., frame.shape[1] / 800)
    cv2.putText(frame, message, (round(10 * scale), round(30 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, .55 * scale, (0, 255, 255), round(scale))


def process_video(capture, tracker, output):
    with (output / 'motion.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(COLUMNS)
        frames = video_frames(capture)
        first = next(frames, None)
        if first is None:
            return
        with open_video_writer(output / 'annotated.mp4', capture, frame_size(first)) as video:
            for index, frame in enumerate(chain((first,), frames)):
                row = tracker.process(frame, index)
                annotate(frame, row)
                video.write(frame)
                if row is not None:
                    writer.writerow(row)
                    save_image(output / 'final_frame.jpg', frame)


def process_images(source, tracker, output, step_cm):
    paths = image_paths(source)
    if not paths:
        raise ValueError(f'No JPEG or PNG images in {source}')
    annotated = output / 'annotated_frames'
    annotated.mkdir(exist_ok=True)
    reference_index = None
    expected_cm = None
    final_frame = None
    with (output / 'motion.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([*COLUMNS, 'image', 'expected_displacement_cm', 'displacement_error_cm'])
        for index, path in enumerate(paths):
            frame = read_image(path)
            row = tracker.process(frame, index)
            annotate(frame, row)
            save_image(annotated / path.name, frame)
            if row is not None:
                if reference_index is None:
                    reference_index = index
                error_cm = None
                if step_cm is not None:
                    expected_cm = (index - reference_index) * step_cm
                    error_cm = row[7] * 100 - expected_cm
                writer.writerow([*row, path.name, expected_cm, error_cm])
                final_frame = frame
    if final_frame is not None:
        save_image(output / 'final_frame.jpg', final_frame)
    return expected_cm


def motion(args, output):
    tracker = MarkerTracker(args)
    source_path = Path(args.source)
    expected_cm = None
    if source_path.is_dir() or args.bundled_photos:
        step_cm = PHOTO_STEP_CM if args.bundled_photos else None
        expected_cm = process_images(source_path, tracker, output, step_cm)
        saved = 'CSV and annotated photos'
    else:
        source = int(args.source) if args.source.isdecimal() else args.source
        with open_video(source) as capture:
            process_video(capture, tracker, output)
        saved = 'CSV, annotated video, and frames'
    if tracker.last_distance is None:
        raise ValueError('No valid target-marker poses were found.')
    measured_cm = tracker.last_distance * 100
    print(f'Marker black-square side used: {args.marker_size * 100:.2f} cm')
    print(f'Final valid displacement: {measured_cm:.2f} cm')
    if args.expected_cm is not None:
        expected_cm = args.expected_cm
    if expected_cm is not None:
        print(f'Ruler displacement: {expected_cm:.2f} cm; signed error: {measured_cm - expected_cm:.2f} cm')
    print(f'Saved {saved} to {output}')
