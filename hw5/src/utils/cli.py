"""Only the inputs required for tracking and reconstruction."""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=(
        'A sfm implementation. Move the camera around a stationary object while keeping it in view'))
    parser.add_argument('--video', required=True, help='Object video path')
    parser.add_argument('--intrinsics', required=True, type=float, nargs=4,
                        metavar=('FX', 'FY', 'CX', 'CY'), help='Focal lengths and principal point in pixels')
    parser.add_argument('--roi', required=True, type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'))
    parser.add_argument('--distortion', type=float, nargs=5, default=[0.] * 5,
                        help='k1 k2 p1 p2 k3; default assumes no lens distortion')
    parser.add_argument('--pair-frame', type=int, help='Second frame index (zero-based); default chooses from track geometry')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2] / 'output')
    parser.add_argument('--extra-credit', action='store_true', help='Save conditional 3D position uncertainty')
    parser.add_argument('--pixel-sigma', type=float, default=.5, help='EC: assumed 1-sigma tracking noise in pixels')
    args = parser.parse_args()
    if min(args.intrinsics[:2]) <= 0 or args.pixel_sigma <= 0:
        parser.error('Focal lengths and pixel noise must be positive.')
    if min(args.roi[:2]) < 0 or min(args.roi[2:]) <= 0:
        parser.error('ROI needs nonnegative X/Y and positive WIDTH/HEIGHT.')
    if args.pair_frame is not None and args.pair_frame < 1:
        parser.error('--pair-frame must be at least 1.')
    return args
