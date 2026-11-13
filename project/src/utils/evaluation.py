import csv
import json
import math
from statistics import mean


def read_measurements(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("Measurement names must be unique.")
    for row in rows:
        if row["role"] not in {"scale", "evaluate"}:
            raise ValueError("Each measurement role must be scale or evaluate.")
        distance = float(row["distance_m"])
        if not math.isfinite(distance) or distance <= 0:
            raise ValueError("Measured distances must be positive and finite.")
        row["distance_m"] = distance
    reference = [row for row in rows if row["role"] == "scale"]
    checks = [row for row in rows if row["role"] == "evaluate"]
    if len(reference) != 1 or len(checks) != 5:
        raise ValueError("Use exactly one scale distance and five checks.")
    return reference[0], checks


def read_distances(path):
    distances = {}
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            values = [row[key] for key in ["ax", "ay", "az", "bx", "by", "bz"]]
            if not any(values):
                continue
            if not all(values):
                raise ValueError(f"Incomplete endpoint coordinates in {path}.")
            coordinates = [float(value) for value in values]
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError("Endpoint coordinates must be finite.")
            name = row["name"]
            if name in distances:
                raise ValueError(f"Duplicate endpoint name: {name}.")
            distances[name] = math.dist(coordinates[:3], coordinates[3:])
    return distances


def evaluate(config, root):
    reference, checks = read_measurements(root / config["measurements"])
    for dataset in config["datasets"]:
        folder = root / config["output"] / dataset
        distances = read_distances(folder / "endpoints.csv")
        length = distances.get(reference["name"], 0.0)
        if length <= 0:
            raise ValueError(f"{dataset}: select the scale endpoints first.")
        scale = reference["distance_m"] / length
        results = []
        for check in checks:
            reconstructed = distances.get(check["name"])
            result = {
                "name": check["name"],
                "available": reconstructed is not None,
            }
            if reconstructed is not None:
                predicted = scale * reconstructed
                error = abs(predicted - check["distance_m"])
                result.update(
                    measured_m=check["distance_m"],
                    reconstructed_m=predicted,
                    absolute_error_m=error,
                    percentage_error=100 * error / check["distance_m"],
                )
            results.append(result)
        errors = [
            row["absolute_error_m"] for row in results if row["available"]
        ]
        summary = {
            "dataset": dataset,
            "scale_m_per_model_unit": scale,
            "available_checks": len(errors),
            "total_checks": len(checks),
            "mean_absolute_error_m": mean(errors) if errors else None,
            "measurements": results,
        }
        (folder / "evaluation.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        print(
            f"{dataset}: {len(errors)}/{len(checks)} check distances available"
        )
