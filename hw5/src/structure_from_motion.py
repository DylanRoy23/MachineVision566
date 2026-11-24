"""HW5: run with --help for inputs. Reconstruction has arbitrary scale."""
import sys
from copy import copy
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hw5.src.utils.cli import BUNDLED_REGIONS, parse_args
from hw5.src.utils.full_video import reconstruct_full_video
from hw5.src.utils.reconstruction import reconstruct
from hw5.src.utils.tracking import track_video
from utils.calibration import camera_parameters


def main():
    args = parse_args()
    try:
        K, distortion, calibration_size = camera_parameters(
            args.calibration, args.intrinsics, args.distortion,
        )
        videos = [args.video] if args.video is not None else BUNDLED_REGIONS
        for video in videos:
            run = copy(args)
            run.video = video
            if args.video is None:
                run.output = args.output / video.stem
            run.output.mkdir(parents=True, exist_ok=True)
            cv2.setRNGSeed(42)
            print(f'Processing {video.name}')
            region = BUNDLED_REGIONS.get(video.resolve())
            if region is None:
                tracks = track_video(run, K, distortion, calibration_size, region)
                reconstruct(tracks, K, run)
            else:
                reconstruct_full_video(run, K, distortion, calibration_size, region)
    except (ValueError, OSError, cv2.error) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
