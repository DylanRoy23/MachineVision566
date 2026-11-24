"""HW4 entry point. Run with --help for capture instructions and options."""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hw4.src.utils.cli import parse_args
from hw4.src.utils.markers import create_marker
from hw4.src.utils.motion import motion


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    actions = {'marker': create_marker, 'motion': motion}
    try:
        actions[args.mode](args, args.output)
    except (ValueError, OSError, KeyError, cv2.error) as error:
        raise SystemExit(f'Error: {error}') from error


if __name__ == '__main__':
    main()
