import argparse
import json
from pathlib import Path

from utils.datasets import prepare
from utils.evaluation import evaluate
from utils.reconstruction import reconstruct


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "reconstruct", "evaluate"])
    parser.add_argument("--config", type=Path, default=root / "config.json")
    args = parser.parse_args()
    actions = {
        "prepare": prepare,
        "reconstruct": reconstruct,
        "evaluate": evaluate,
    }
    try:
        config = json.loads(args.config.read_text())
        actions[args.stage](config, args.config.resolve().parent)
    except (ValueError, OSError, KeyError, NotImplementedError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
