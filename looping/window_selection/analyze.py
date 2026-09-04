"""Apply the recorded decision rule without dropping failed or pending runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save, validate_config
from looping.window_selection.common import protocol, run_config, run_name
from runtime_utils import file_sha256


def decision(profiles):
    settings = protocol()
    missing = [(selector, seed) for selector in settings["selectors"] for seed in settings["seeds"]
               if (selector, seed) not in profiles]
    if missing:
        return {"status": "pending", "missing": missing}
    if any(value is None for value in profiles.values()):
        return {"status": "not_promising", "reason": "At least one run failed"}
    differences = {str(horizon): [100 * (profiles["confidence", seed][str(horizon)]
                                        - profiles["random", seed][str(horizon)])
                                 for seed in settings["seeds"]] for horizon in (1024, 4096)}
    criteria = settings["decision"]
    improved = sum(value > 0 for value in differences["1024"])
    mean_gain = float(np.mean(differences["1024"]))
    deep_gain = float(np.mean(differences["4096"]))
    promising = (mean_gain >= criteria["minimum_mean_1024_gain_percentage_points"] - 1e-9
                 and improved >= criteria["minimum_improved_seed_pairs"]
                 and deep_gain >= criteria["minimum_mean_4096_gain_percentage_points"] - 1e-9)
    latest_gain = float(np.mean([100 * (profiles["confidence", seed]["1024"]
                                       - profiles["latest", seed]["1024"]) for seed in settings["seeds"]]))
    return {"status": "promising" if promising else "not_promising",
            "paired_gain_percentage_points": differences, "mean_1024_gain": mean_gain,
            "improved_seed_pairs": improved, "mean_4096_gain": deep_gain,
            "mean_1024_gain_over_latest": latest_gain,
            "adaptive_selection_advantage": latest_gain > 0,
            "recommendation_changed": False}


def summarize(root):
    root = Path(root)
    settings = protocol()
    profiles, rows, matching = {}, {}, {}
    for selector in settings["selectors"]:
        for seed in settings["seeds"]:
            name = run_name(selector, seed)
            directory = root / "runs" / name
            path = directory / "result.json"
            if not path.exists():
                rows[name] = {"status": "pending_training"}
                continue
            result = json.loads(path.read_text())
            validate_config(result["config"], run_config(selector, seed))
            if result["status"] != "complete":
                profiles[selector, seed] = None
                rows[name] = {"status": result["status"]}
                continue
            if result["updates"] != settings["training"]["steps"]:
                raise ValueError("Incorrect training budget")
            pair = {key: result[key] for key in ("sample_digest", "initial_state_sha256", "data_sha256", "work_counts", "source_sha256")}
            if seed in matching:
                validate_config(pair, matching[seed])
            matching[seed] = pair
            history = [row["scores"]["1024"]["accuracy"] for row in result["history"]
                       if row["updates"] >= settings["training"]["late_floor_start"]]
            rows[name] = {"status": "pending_evaluation", "selected_counts": result["selected_counts"],
                          "late_training_mean": float(np.mean(history)),
                          "late_training_minimum": min(history), "timing_seconds": result["timing_seconds"]}
            if (directory / "evaluation_failure.json").exists():
                rows[name]["status"] = "evaluation_numerical_failure"
                profiles[selector, seed] = None
                continue
            for selection in ("final", "best_validation"):
                eval_path = directory / "evaluations" / selection / "result.json"
                if not eval_path.exists():
                    continue
                evaluation = json.loads(eval_path.read_text())
                if evaluation["identity"]["weights_sha256"] != file_sha256(directory / f"{selection}.pt"):
                    raise ValueError("Evaluation uses a different checkpoint")
                if evaluation["per_puzzle_sha256"] != file_sha256(eval_path.parent / "per_puzzle.npz"):
                    raise ValueError("Prediction array checksum mismatch")
                validate_evaluation(evaluation)
                rows[name][selection] = evaluation["scores"]
                rows[name][f"{selection}_retention"] = evaluation.get("solution_tracking")
                if selection == "final":
                    profiles[selector, seed] = {str(h): evaluation["scores"][str(h)]["solved"] / 25000
                                               for h in settings["evaluation"]["iterations"]}
            if "final" in rows[name] and "best_validation" in rows[name]:
                rows[name]["status"] = "complete"
    return {"study_id": settings["study_id"], "runs": rows, "decision": decision(profiles)}


def validate_evaluation(evaluation):
    settings = protocol()["evaluation"]
    benchmark = json.loads((Path(__file__).parents[2] / settings["benchmark"]).read_text())
    expected = {"benchmark_rows_sha256": benchmark["rows_sha256"],
                "iterations": settings["iterations"], "precision": "fp32",
                "matmul_precision": "highest", "batch_size": settings["batch_size"],
                "compiled": False, "track_solutions_every_iteration": True}
    validate_config({key: evaluation["identity"][key] for key in expected}, expected)
    for horizon in settings["iterations"]:
        row = evaluation["scores"][str(horizon)]
        if row["total"] != 25000 or not isinstance(row["solved"], int) or not 0 <= row["solved"] <= 25000:
            raise ValueError("Invalid full-set solved count")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = summarize(args.root)
    if args.output:
        atomic_json_save(report, args.output)
    print(json.dumps(report, indent=2))
