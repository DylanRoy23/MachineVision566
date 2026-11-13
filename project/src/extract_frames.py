import argparse
import shutil
import subprocess
from pathlib import Path

from utils.datasets import IMAGE_TYPES


def extract_frames(video, output):
    if not video.is_file():
        raise ValueError(f"Video not found: {video}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ValueError("FFmpeg is required; install it and add it to PATH.")
    output.mkdir(parents=True, exist_ok=True)
    if any(path.suffix.lower() in IMAGE_TYPES for path in output.iterdir()):
        raise ValueError(
            f"{output} already contains images. Use an empty output folder."
        )
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-n",
            "-i",
            str(video.resolve()),
            "-map",
            "0:v:0",
            "-vf",
            "fps=2",
            "-q:v",
            "2",
            "-start_number",
            "1",
            str(output.resolve() / "%04d.jpg"),
        ],
        check=True,
    )
    count = len(list(output.glob("*.jpg")))
    if count == 0:
        raise ValueError("The video produced no frames.")
    print(f"Saved {count} images, one every 0.5 seconds, to {output}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to the input video")
    parser.add_argument("--output", type=Path, default=root / "data/images")
    args = parser.parse_args()
    try:
        extract_frames(args.video, args.output)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
