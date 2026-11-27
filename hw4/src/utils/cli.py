"""Options for marker creation and camera motion."""
import argparse
from pathlib import Path

from utils.cli import add_calibration_arguments, resolve_calibration_arguments

ROOT = Path(__file__).resolve().parents[2]
CAPTURES = ROOT / 'hw4_captures'
MARKER_PHOTOS = CAPTURES / 'aruco_caps'
CALIBRATION_PHOTOS = CAPTURES / 'cal_caps'
PHOTO_STEP_CM = 3 * 2.54


def parse_args():
    parser = argparse.ArgumentParser(description=(
        'Generate an ArUco marker or measure camera motion from marker photos. Motion defaults to '
        'the bundled marker and 9-by-6 checkerboard photos.'))
    parser.add_argument('mode', choices=['marker', 'motion'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--marker-id', type=int, default=0)
    parser.add_argument('--marker-size', type=float, default=.10,
                        help='Measured black-square side in meters (default: %(default)s)')
    parser.add_argument('--source', type=Path, default=MARKER_PHOTOS,
                        help='Directory of marker photos in capture order (default: bundled photos)')
    parser.add_argument('--step-cm', type=float, default=PHOTO_STEP_CM,
                        help='Ruler distance the camera moved between photos (default: 3 in = %(default).2f cm)')
    add_calibration_arguments(parser)
    parser.add_argument('--extra-credit', action='store_true', help='Estimate position and orientation uncertainty')
    parser.add_argument('--corner-sigma', type=float, default=.5, help='EC: assumed corner noise in pixels')
    args = parser.parse_args()
    if not 0 <= args.marker_id < 50 or min(args.marker_size, args.corner_sigma, args.step_cm) <= 0:
        parser.error('Marker ID must be 0–49; size, step and noise must be positive.')
    if args.mode == 'motion':
        if not args.source.is_dir():
            parser.error(f'Photo directory not found: {args.source}')
        bundled = args.source.resolve() == MARKER_PHOTOS
        resolve_calibration_arguments(parser, args, CALIBRATION_PHOTOS if bundled else None)
    return args
