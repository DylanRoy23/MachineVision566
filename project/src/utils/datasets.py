import csv
import json
from pathlib import Path

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ENDPOINT_COLUMNS = ["name", "ax", "ay", "az", "bx", "by", "bz"]


def prepare(config, root):
    image_dir = root / config["images"]
    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_TYPES
    )
    if len(images) < 32:
        raise ValueError(
            "Add at least 32 ordered photos to data/images; aim for 120–160."
        )
    with (root / config["measurements"]).open(newline="") as file:
        measurements = list(csv.DictReader(file))
    for name, stride in config["datasets"].items():
        if not isinstance(stride, int) or stride < 1:
            raise ValueError("Dataset strides must be positive integers.")
        folder = root / config["output"] / name
        folder.mkdir(parents=True, exist_ok=True)
        selected = images[::stride]
        manifest = {
            "dataset": name,
            "stride": stride,
            "image_count": len(selected),
            "images": [str(path.resolve()) for path in selected],
        }
        path = folder / "manifest.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous != manifest:
                raise ValueError(
                    f"{path} describes different inputs; choose a new "
                    "output folder in config.json for the new capture."
                )
        path.write_text(json.dumps(manifest, indent=2) + "\n")
        endpoints = folder / "endpoints.csv"
        if not endpoints.exists():
            with endpoints.open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(ENDPOINT_COLUMNS)
                for measurement in measurements:
                    writer.writerow([measurement["name"], *([""] * 6)])
        print(f"{name}: {len(selected)} images; {path}")


def image_names(manifest, image_dir):
    names = []
    for image in manifest["images"]:
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"Manifest image is missing: {path}")
        if not path.is_relative_to(image_dir):
            raise ValueError(f"Manifest image is outside {image_dir}: {path}")
        names.append(path.relative_to(image_dir).as_posix())
    return names
