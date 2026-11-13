"""HW5: run with --help for inputs. Reconstruction has arbitrary scale."""
import cv2

from utils.cli import parse_args
from utils.reconstruction import reconstruct
from utils.tracking import track_video


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(42)
    try:
        tracks, K = track_video(args)
        reconstruct(tracks, K, args)
    except (ValueError, OSError, cv2.error) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
