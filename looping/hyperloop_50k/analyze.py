"""Verify downloaded records and apply the predeclared paired comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save, validate_config
from looping.hyperloop_50k.common import protocol, run_config, run_name
from runtime_utils import file_sha256


def compare(profiles, reference="baseline"):
    settings = protocol()
    required = [(arm, seed) for arm in settings["arms"] for seed in settings["seeds"]]
    if any(key not in profiles for key in required):
        return {"status": "pending"}
    if any(profiles[key] is None for key in required):
        return {"status": "not_promising", "reason": "failed or nonfinite run"}
    differences = {str(h): [profiles["gated_four", seed][str(h)] - profiles[reference, seed][str(h)]
                            for seed in settings["seeds"]] for h in (1024, 4096)}
    means = {h: float(np.mean(values)) for h, values in differences.items()}
    routes = {}
    for route, horizon in (("accuracy_route", "1024"), ("stability_route", "4096")):
        criteria = settings["decision"][route]
        routes[route] = (means["1024"] >= criteria["minimum_mean_1024_gain"] - 1e-12
                         and means["4096"] >= criteria["minimum_mean_4096_gain"] - 1e-12
                         and sum(value > 0 for value in differences[horizon]) >= criteria[f"improved_{horizon}_pairs"])
    return {"status": "promising" if any(routes.values()) else "not_promising", "reference": reference,
            "paired_differences": differences, "mean_differences": means, "routes": routes}


def analyze(root, data_dir):
    root, settings = Path(root), protocol()
    from looping.hyperloop_50k.common import validate_data
    validate_data(data_dir, names=("development.npz",))
    with np.load(Path(data_dir) / "development.npz", allow_pickle=False) as data:
        digits, targets, indices = data["digits"], data["targets"], data["indices"]
    records, profiles, paired = {}, {}, {}
    for arm in settings["arms"]:
        for seed in settings["seeds"]:
            name = run_name(arm, seed)
            directory = root / "runs" / name
            if not (directory / "result.json").exists():
                records[name] = {"status": "pending"}
                continue
            training = json.loads((directory / "result.json").read_text())
            validate_config(training["config"], run_config(arm, seed))
            if training["status"] != "complete":
                records[name] = {"status": training["status"]}
                profiles[arm, seed] = None
                continue
            if training["updates"] != settings["training"]["steps"]:
                raise ValueError(f"Wrong training budget: {name}")
            pair = {key: training[key] for key in ("base_initial_state_sha256", "sample_digest", "work_counts", "data_sha256", "source_sha256")}
            validate_config(paired.setdefault(seed, pair), pair)
            record = {"status": "complete", "parameters": training["parameters"], "timings_seconds": training["timings_seconds"],
                      "selections": {}}
            late = [row["scores"]["1024"]["accuracy"] for row in training["history"]
                    if row["updates"] >= settings["training"]["late_floor_start"]]
            record["late_probe"] = {"mean": float(np.mean(late)), "minimum": min(late)}
            for selection in ("final", "best_validation"):
                path = directory / "evaluations" / selection / "result.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                identity = result["identity"]
                validate_config(identity["config"], training["config"])
                validate_config(identity["source_sha256"], training["source_sha256"])
                for key, expected in settings["evaluation"].items():
                    if identity[key] != expected:
                        raise ValueError(f"Unexpected {key}: {name}/{selection}")
                expected_step = training["updates"] if selection == "final" else training["best_validation"]["updates"]
                if identity["checkpoint_selection"] != selection or identity["updates"] != expected_step:
                    raise ValueError("Evaluation used the wrong checkpoint selection")
                if result["predictions_sha256"] != file_sha256(path.parent / "predictions.npz"):
                    raise ValueError("Evaluation predictions checksum mismatch")
                export_path = directory / f"{selection}.pt"
                if identity["weights_sha256"] != file_sha256(export_path):
                    raise ValueError("Evaluated weights do not match the downloaded export")
                with np.load(path.parent / "predictions.npz", allow_pickle=False) as predictions:
                    np.testing.assert_array_equal(predictions["indices"], indices)
                    for horizon, score in result["scores"].items():
                        solved = predictions[f"solved_{horizon}"]
                        recomputed = ((predictions[f"predictions_{horizon}"] == targets) | (digits != 0)).all(-1)
                        recomputed &= predictions[f"finite_{horizon}"]
                        np.testing.assert_array_equal(solved, recomputed)
                        if len(solved) != 25000 or int(solved.sum()) != score["solved"] or score["total"] != 25000:
                            raise ValueError("Reported score differs from per-puzzle results")
                        if abs(score["accuracy"] - float(solved.mean())) > 1e-12:
                            raise ValueError("Reported accuracy differs from per-puzzle results")
                record["selections"][selection] = result["scores"]
                if selection == "final":
                    profiles[arm, seed] = ({h: score["accuracy"] for h, score in result["scores"].items()}
                                           if not any(score["nonfinite"] for score in result["scores"].values()) else None)
                    profile, criteria = profiles[arm, seed], settings["decision"]["reliability"]
                    record["reliable_final"] = bool(profile is not None
                        and profile["1024"] >= criteria["minimum_final_1024"]
                        and profile["4096"] >= criteria["minimum_final_4096"]
                        and profile["1024"] - profile["4096"] <= criteria["maximum_1024_to_4096_drop"])
            records[name] = record
    return {"study_id": settings["study_id"], "runs": records,
            "versus_baseline": compare(profiles)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file; existing reports are not overwritten")
    report = analyze(args.root, args.data_dir)
    atomic_json_save(report, args.output)
    print(json.dumps({name: report[name] for name in ("versus_baseline",)}, indent=2))


if __name__ == "__main__":
    main()
