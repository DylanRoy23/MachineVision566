"""Only the inputs required for tracking and reconstruction."""
import argparse
from pathlib import Path

from utils.cli import add_calibration_arguments, resolve_calibration_arguments

ROOT = Path(__file__).resolve().parents[2]
MEDIA = ROOT / 'ObjectsCameraCal'
CALIBRATION_VIDEO = MEDIA / 'CameraCalIPhone.mp4'
# Outlines inside each object's silhouette in the first 406-by-720 frame.
BUNDLED_REGIONS = {
    MEDIA / 'Object1.mp4': (
        (178, 281), (201, 266), (230, 270), (246, 285), (243, 379),
        (239, 453), (225, 469), (194, 469), (181, 458), (173, 370), (170, 300),
    ),
    MEDIA / 'Object2.mp4': (
        (100, 230), (156, 190), (249, 151), (287, 174), (286, 191),
        (307, 208), (320, 245), (303, 278), (269, 287), (228, 264),
        (200, 282), (181, 353), (204, 361), (247, 393), (242, 419),
        (164, 460), (131, 467), (103, 446), (113, 396), (147, 281),
        (119, 275), (103, 255),
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description=(
        'Reconstruct a stationary object from a moving camera. Defaults to both '
        'bundled object videos and the bundled iPhone calibration video.'))
    parser.add_argument('--video', type=Path, help='Run on one object video')
    add_calibration_arguments(parser)
    parser.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'WIDTH', 'HEIGHT'),
                        help='First-frame object region (required for custom videos)')
    parser.add_argument('--pair-frame', type=int, help='Second frame index (zero-based); default chooses from track geometry')
    parser.add_argument('--output', type=Path, default=ROOT / 'output',
                        help='Output directory (one subdirectory per clip in the default run)')
    parser.add_argument('--extra-credit', action='store_true', help='Save conditional 3D position uncertainty')
    parser.add_argument('--pixel-sigma', type=float, default=.5, help='EC: assumed 1-sigma tracking noise in pixels')
    args = parser.parse_args()
    custom_video = args.video is not None and args.video.resolve() not in BUNDLED_REGIONS
    resolve_calibration_arguments(parser, args, None if custom_video else CALIBRATION_VIDEO)
    if args.pixel_sigma <= 0:
        parser.error('Pixel noise must be positive.')
    if custom_video and args.roi is None:
        parser.error('A custom video requires --roi X Y WIDTH HEIGHT.')
    if args.roi is not None and (min(args.roi[:2]) < 0 or min(args.roi[2:]) <= 0):
        parser.error('ROI needs nonnegative X/Y and positive WIDTH/HEIGHT.')
    if args.pair_frame is not None and args.pair_frame < 1:
        parser.error('--pair-frame must be at least 1.')
    return args
