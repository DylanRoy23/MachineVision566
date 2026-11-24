"""Calibration options shared by the camera-motion and reconstruction CLIs."""
from pathlib import Path


def add_calibration_arguments(parser):
    parser.add_argument('--calibration', type=Path,
                        help='Checkerboard photos or video at the source resolution (9-by-6 inner corners)')
    parser.add_argument('--intrinsics', type=float, nargs=4, metavar=('FX', 'FY', 'CX', 'CY'),
                        help='Camera focal lengths and principal point in pixels')
    parser.add_argument('--distortion', type=float, nargs=5, metavar=('K1', 'K2', 'P1', 'P2', 'K3'),
                        help='Override lens coefficients (zero by default with manual intrinsics)')


def resolve_calibration_arguments(parser, args, default_source):
    if args.intrinsics is not None:
        if min(args.intrinsics[:2]) <= 0:
            parser.error('Focal lengths must be positive.')
    elif args.calibration is None:
        if default_source is None:
            parser.error('A custom source requires --calibration or --intrinsics FX FY CX CY.')
        args.calibration = default_source
