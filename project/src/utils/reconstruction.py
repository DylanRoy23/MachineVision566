import json
import shutil
from pathlib import Path

from .colmap import colmap_version, stage_paths, validate_options
from .datasets import image_names
from .environment import describe_hardware
from .pipeline import run_dataset


def reconstruct(config, root):
    settings = config["reconstruction"]
    options = settings["options"]
    image_dir = (root / config["images"]).resolve()
    validate_options(options, stage_paths(Path(), image_dir))
    colmap = shutil.which(settings["colmap"])
    if colmap is None:
        raise ValueError(
            f"COLMAP executable not found: {settings['colmap']}. "
            "Install a CUDA "
            "build of COLMAP or set reconstruction.colmap in config.json."
        )
    version = colmap_version(colmap)

    pending = []
    for name in config["datasets"]:
        folder = root / config["output"] / name
        manifest_path = folder / "manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing {manifest_path}; run prepare first.")
        result_path = folder / "reconstruction.json"
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if (previous["colmap_version"], previous["options"]) != (
                version,
                options,
            ):
                raise ValueError(
                    f"{result_path} used a different COLMAP version or "
                    "options; "
                    "choose a new output folder in config.json and rerun "
                    "prepare and reconstruct for every dataset."
                )
            print(
                f"{name}: already recorded as {previous['status']}; "
                f"{result_path}"
            )
            continue
        manifest = json.loads(manifest_path.read_text())
        pending.append((name, folder, image_names(manifest, image_dir)))

    hardware = describe_hardware() if pending else None
    for name, folder, names in pending:
        record = run_dataset(
            colmap, version, options, hardware, image_dir, name, folder, names
        )
        result_path = folder / "reconstruction.json"
        result_path.write_text(json.dumps(record, indent=2) + "\n")
        if record["status"] == "succeeded":
            print(
                f"{name}: {record['registered_images']}/"
                f"{record['image_count']} "
                f"images registered in component "
                f"{record['selected_component']} "
                f"of {record['component_count']}; "
                f"{record['dense_points']} dense "
                f"points; {result_path}"
            )
        else:
            print(
                f"{name}: failed at {record['failed_stage']}: "
                f"{record['failure']} Recorded in {result_path}"
            )
