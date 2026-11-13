"""Track one fixed marker and save camera-motion measurements."""
import csv

import cv2
import numpy as np

from .markers import DICTIONARY, save_image
from .pose import displacement, pose_uncertainty, marker_points, pose, reprojection_error

COLUMNS = ['frame', 'camera_x_m', 'camera_y_m', 'camera_z_m',
           'dx_m', 'dy_m', 'dz_m', 'displacement_m', 'rotation_deg',
           'reprojection_rmse_px', 'dx_std_m', 'dy_std_m', 'dz_std_m',
           'rotation_x_std_deg', 'rotation_y_std_deg', 'rotation_z_std_deg', 'distance_std_m']


class MarkerTracker:
    def __init__(self, args):
        self.args = args
        fx, fy, cx, cy = args.intrinsics
        self.K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1.]])
        self.distortion = np.asarray(args.distortion, dtype=float)
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
    cv2.putText(frame, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 1)


def open_video_writer(capture, output, size):
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.
    video = cv2.VideoWriter(str(output / 'annotated.mp4'),
                            cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not video.isOpened():
        video.release()
        raise ValueError('Cannot create annotated video.')
    return video


def process_video(capture, tracker, output):
    video = None
    try:
        with (output / 'motion.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(COLUMNS)
            index = 0
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if video is None:
                    video = open_video_writer(capture, output, frame.shape[1::-1])
                row = tracker.process(frame, index)
                annotate(frame, row)
                video.write(frame)
                if row is not None:
                    writer.writerow(row)
                    save_image(output / 'final_frame.jpg', frame)
                index += 1
    finally:
        if video is not None:
            video.release()


def motion(args, output):
    tracker = MarkerTracker(args)
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open camera/video: {args.source}')
        process_video(capture, tracker, output)
    finally:
        capture.release()
    if tracker.last_distance is None:
        raise ValueError('No valid target-marker poses were found.')
    measured_cm = tracker.last_distance * 100
    print(f'Final valid displacement: {measured_cm:.2f} cm')
    if args.expected_cm is not None:
        print(f'Ruler displacement: {args.expected_cm:.2f} cm; signed error: {measured_cm - args.expected_cm:.2f} cm')
    print(f'Saved CSV, annotated video, and frames to {output}')
