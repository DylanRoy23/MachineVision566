import subprocess
import time


class StageFailure(Exception):
    def __init__(self, stage, message):
        super().__init__(message)
        self.stage = stage


def stage_paths(workspace, image_dir, component=None):
    """Path options set by this module for each COLMAP command, in run order."""
    database = workspace / "database.db"
    dense = workspace / "dense"
    return {
        "feature_extractor": {
            "database_path": database,
            "image_path": image_dir,
            "image_list_path": workspace / "image_list.txt",
        },
        "exhaustive_matcher": {"database_path": database},
        "mapper": {
            "database_path": database,
            "image_path": image_dir,
            "output_path": workspace / "sparse",
        },
        "image_undistorter": {
            "image_path": image_dir,
            "input_path": component,
            "output_path": dense,
        },
        "patch_match_stereo": {"workspace_path": dense},
        "stereo_fusion": {
            "workspace_path": dense,
            "output_path": dense / "fused.ply",
        },
    }


def validate_options(options, paths):
    unknown = set(options) - set(paths)
    if unknown:
        raise ValueError(
            f"Unknown COLMAP stages in config.json: {sorted(unknown)}"
        )
    for stage, reserved in paths.items():
        clash = set(reserved) & set(options.get(stage, {}))
        if clash:
            raise ValueError(
                f"{stage} paths are set by the pipeline: {sorted(clash)}"
            )


def run_stage(colmap, stage, paths, options, folder, record):
    args = [colmap, stage]
    for key, value in [*paths[stage].items(), *options.get(stage, {}).items()]:
        if isinstance(value, bool):
            value = int(value)
        args += [f"--{key}", str(value)]
    log = folder / "colmap" / "logs" / f"{stage}.log"
    print(f"{record['dataset']}: {stage}...", end=" ", flush=True)
    start = time.perf_counter()
    with log.open("w") as file:
        returncode = subprocess.run(
            args, stdout=file, stderr=subprocess.STDOUT
        ).returncode
    seconds = time.perf_counter() - start
    print(f"{seconds:.1f} s")
    record["stages"].append(
        {
            "stage": stage,
            "seconds": seconds,
            "returncode": returncode,
            "log": log.relative_to(folder).as_posix(),
            "command": args,
        }
    )
    if returncode != 0:
        raise StageFailure(stage, f"exited with code {returncode}; see {log}.")


def colmap_version(colmap):
    result = subprocess.run([colmap, "help"], capture_output=True, text=True)
    lines = (result.stdout + result.stderr).splitlines()
    version = next(
        (line.strip() for line in lines if line.startswith("COLMAP")), None
    )
    if result.returncode != 0 or version is None:
        raise OSError(
            f"Could not read the COLMAP version from '{colmap} help'."
        )
    if "without CUDA" in version:
        raise ValueError(
            f"{version}: patch_match_stereo requires COLMAP built with CUDA."
        )
    return version
