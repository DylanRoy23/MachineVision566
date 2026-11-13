"""Options for marker creation and camera motion."""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=(
        'Print the marker and measure its black-square side. Keep it fixed; '
        'move the camera along a ruler by at least 30 cm. '
        'Supply camera intrinsics in pixels for the video resolution.'))
    parser.add_argument('mode', choices=['marker', 'motion'])
    parser.add_argument('--output', type=Path, default=ROOT / 'output')
    parser.add_argument('--marker-id', type=int, default=0)
    parser.add_argument('--marker-size', type=float, default=.10, help='Black-square side in meters')
    parser.add_argument('--source', default='0', help='Camera index or video path')
    parser.add_argument('--intrinsics', type=float, nargs=4, metavar=('FX', 'FY', 'CX', 'CY'),
                        help='Camera focal lengths and principal point in pixels')
    parser.add_argument('--distortion', type=float, nargs=5, default=[0.] * 5,
                        metavar=('K1', 'K2', 'P1', 'P2', 'K3'), help='Lens coefficients; default assumes no distortion')
    parser.add_argument('--expected-cm', type=float, help='Ruler-measured final displacement')
    parser.add_argument('--extra-credit', action='store_true', help='Estimate position and orientation uncertainty')
    parser.add_argument('--corner-sigma', type=float, default=.5, help='EC: assumed corner noise in pixels')
    args = parser.parse_args()
    if not 0 <= args.marker_id < 50 or min(args.marker_size, args.corner_sigma) <= 0:
        parser.error('Marker ID must be 0–49; size and noise must be positive.')
    if args.mode == 'motion' and (args.intrinsics is None or min(args.intrinsics[:2]) <= 0):
        parser.error('Motion requires --intrinsics FX FY CX CY with positive focal lengths.')
    return args
