import os
import shutil
import time
from datetime import datetime

from .colmap import StageFailure, run_stage, stage_paths
from .model_io import count_database_images, count_ply_vertices, read_components


def run_dataset(
    colmap, version, options, hardware, image_dir, name, folder, names
):
    workspace = folder / "colmap"
    if workspace.exists():
        print(f"{name}: discarding incomplete workspace {workspace}")
        shutil.rmtree(workspace)
    (folder / "cloud.ply").unlink(missing_ok=True)
    (workspace / "logs").mkdir(parents=True)
    (workspace / "sparse").mkdir()
    paths = stage_paths(workspace, image_dir)
    paths["feature_extractor"]["image_list_path"].write_text(
        "\n".join(names) + "\n"
    )

    record = {
        "dataset": name,
        "status": "failed",
        "failed_stage": None,
        "failure": None,
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "colmap_version": version,
        "options": options,
        "hardware": hardware,
        "image_count": len(names),
        "images_in_database": None,
        "component_count": None,
        "selected_component": None,
        "registered_images": None,
        "registered_fraction": None,
        "dense_points": None,
        "cloud": None,
        "wall_time_s": None,
        "stages": [],
        "components": [],
    }

    def run(stage):
        run_stage(colmap, stage, paths, options, folder, record)

    sparse = paths["mapper"]["output_path"]
    fused = paths["stereo_fusion"]["output_path"]
    start = time.perf_counter()
    try:
        run("feature_extractor")
        record["images_in_database"] = count_database_images(
            paths["feature_extractor"]["database_path"]
        )
        if record["images_in_database"] < len(names):
            print(f"{name}: COLMAP could not read some images; see the log.")
        run("exhaustive_matcher")
        run("mapper")
        components = read_components(sparse, folder)
        record["component_count"] = len(components)
        record["components"] = components
        if not components:
            raise StageFailure("mapper", "COLMAP produced no sparse model.")
        selected = max(
            components, key=lambda component: component["registered_images"]
        )
        record.update(
            selected_component=selected["index"],
            registered_images=selected["registered_images"],
            registered_fraction=selected["registered_images"] / len(names),
        )
        paths["image_undistorter"]["input_path"] = sparse / str(
            selected["index"]
        )
        run("image_undistorter")
        run("patch_match_stereo")
        run("stereo_fusion")
        points = count_ply_vertices(fused) if fused.exists() else 0
        if points == 0:
            raise StageFailure(
                "stereo_fusion", "COLMAP produced an empty dense cloud."
            )
        os.replace(fused, folder / "cloud.ply")
        record.update(
            status="succeeded", dense_points=points, cloud="cloud.ply"
        )
    except StageFailure as failure:
        record.update(failed_stage=failure.stage, failure=str(failure))
    record["wall_time_s"] = time.perf_counter() - start
    return record
