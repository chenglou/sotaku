"""Summarize every seed without treating puzzles as independent training runs."""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save, validate_config
from looping.weight_tying.common import protocol, protocol_sha256, run_name
from runtime_utils import file_sha256


def training_summary(result):
    metric = "16" if result["config"]["regime"] == "early" else "1024"
    floor_start = protocol()["training"]["late_floor_start"]
    late = [row["scores"][metric]["accuracy"] for row in result["history"] if row["updates"] >= floor_start]
    return {"status": result["status"], "updates": result["updates"], "parameters": result["parameters"],
            "primary_iteration": int(metric), "best_validation": result["best_validation"],
            "late_validation_mean": float(np.mean(late)) if late else None,
            "late_validation_minimum": min(late) if late else None,
            "timings_seconds": result["timings_seconds"], "estimated_model_flops": result["estimated_model_flops"],
            "processed_puzzles": result["processed_puzzles"], "sample_digest": result["sample_digest"],
            "final_validation": result["history"][-1]["scores"] if result["history"] else {}}


def read_evaluation(root, name, selection, entry):
    path = root / "evaluations" / name / selection / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    identity = result["identity"]
    if identity["protocol_sha256"] != protocol_sha256() or identity["run_name"] != name:
        raise ValueError(f"Evaluation identity mismatch: {path}")
    if identity["weights_sha256"] != entry["exports"][selection]["weights_sha256"]:
        raise ValueError(f"Evaluation is not for the selected model: {path}")
    if identity["selection"] != selection or identity["cohort_lock_sha256"] != file_sha256(root / "cohort_lock.json"):
        raise ValueError(f"Evaluation selection or cohort lock mismatch: {path}")
    for dataset, expected in result["prediction_sha256"].items():
        predictions_path = path.parent / f"{dataset}_predictions.npz"
        data_path = root / "data" / f"{dataset}.npz"
        if file_sha256(predictions_path) != expected:
            raise ValueError(f"Prediction checksum mismatch: {path}/{dataset}")
        if file_sha256(data_path) != result["dataset_sha256"][dataset]:
            raise ValueError(f"Evaluation data checksum mismatch: {path}/{dataset}")
        with np.load(data_path, allow_pickle=False) as data, np.load(predictions_path, allow_pickle=False) as predictions:
            verify_scores(data, predictions, result["scores"][dataset], identity["iterations"])
    return result


def verify_scores(data, predictions, scores, iterations):
    """Recompute exact-board accuracy from saved predictions, not saved scores."""
    previous_solved = None
    np.testing.assert_array_equal(data["labels"], predictions["labels"])
    for horizon in iterations:
        predicted = predictions[f"predictions_{horizon}"]
        finite = predictions[f"finite_{horizon}"]
        if predicted.shape != data["targets"].shape or finite.shape != (len(predicted),) or finite.dtype != bool:
            raise ValueError("Invalid prediction or finite-mask shape/type")
        if np.any(predicted > 8) or np.any(predicted < 0):
            raise ValueError("Predicted digit is outside the class range")
        solved = ((predicted == data["targets"]) | (data["digits"] != 0)).all(-1) & finite
        np.testing.assert_array_equal(solved, predictions[f"solved_{horizon}"])
        expected = {"solved": int(solved.sum()), "total": len(solved), "accuracy": float(solved.mean()),
                    "nonfinite": int((~finite).sum()),
                    "difficulty": {str(label): {"solved": int(solved[data["labels"] == label].sum()),
                                                 "total": int((data["labels"] == label).sum())}
                                   for label in np.unique(data["labels"])}}
        if previous_solved is not None:
            expected["lost_since_previous"] = int((previous_solved & ~solved).sum())
            expected["gained_since_previous"] = int((~previous_solved & solved).sum())
        validate_config(scores[str(horizon)], expected)
        previous_solved = solved


def reliability_summary(results, architecture, regime):
    settings = protocol()
    members = [results.get(run_name(architecture, regime, seed)) for seed in settings["seeds"]]
    summary = {"planned": len(members), "completed": 0, "numerical_failures": 0, "pending_training": 0,
               "datasets": {name: {"evaluated": 0, "healthy": 0, "pending_evaluation": 0}
                            for name in ("validation", "development", "holdout")}}
    for result in members:
        if result is None or result["status"] not in ("complete", "numerical_failure"):
            summary["pending_training"] += 1
            continue
        if result["status"] == "numerical_failure":
            summary["numerical_failures"] += 1
            continue
        summary["completed"] += 1
        for dataset, counts in summary["datasets"].items():
            scores = (result.get("final_validation") if dataset == "validation"
                      else result.get("evaluations", {}).get("final", {}).get(dataset))
            if scores is None or "4096" not in scores:
                counts["pending_evaluation"] += 1
                continue
            counts["evaluated"] += 1
            accuracy_1024 = scores["1024"]["accuracy"]
            drop = accuracy_1024 - scores["4096"]["accuracy"]
            if (accuracy_1024 >= settings["evaluation"]["healthy_1024_accuracy"]
                    and drop <= settings["evaluation"]["maximum_1024_to_4096_drop"] + 1e-12):
                counts["healthy"] += 1
    return summary


def paired_puzzle_counts(left, right):
    if left.dtype != bool or right.dtype != bool or left.shape != right.shape or left.ndim != 1:
        raise ValueError("Paired solved masks must be matching boolean vectors")
    return {"both_solved": int((left & right).sum()), "tied_only": int((left & ~right).sum()),
            "untied_only": int((~left & right).sum()), "neither_solved": int((~left & ~right).sum()),
            "total": len(left)}


def summarize(root, output, *, partial=False):
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    settings = protocol()
    lock_path = root / "cohort_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    if not partial and lock is None:
        raise ValueError("Final analysis requires a sealed cohort")
    results, missing, histories = {}, [], {}
    for architecture in settings["architectures"]:
        for regime in settings["regimes"]:
            for seed in settings["seeds"]:
                name = run_name(architecture, regime, seed)
                path = root / "runs" / name / "result.json"
                if not path.exists():
                    missing.append(name)
                    continue
                result = json.loads(path.read_text())
                if result["config"]["protocol_sha256"] != protocol_sha256() or result["config"]["smoke"]:
                    raise ValueError(f"Wrong protocol or fixture in analysis: {name}")
                if lock and file_sha256(path) != lock["identity"]["runs"][name]["result_sha256"]:
                    raise ValueError(f"Training result changed after cohort lock: {name}")
                histories[name] = result["history"]
                summary = training_summary(result)
                summary["evaluations"] = {}
                if lock and result["status"] == "complete":
                    entry = lock["identity"]["runs"][name]
                    for selection in ("final", "best_validation"):
                        evaluation = read_evaluation(root, name, selection, entry)
                        if evaluation is None:
                            missing.append(f"{name}/{selection} evaluation")
                        else:
                            summary["evaluations"][selection] = evaluation["scores"]
                results[name] = summary
    if missing and not partial:
        raise ValueError("Incomplete final analysis: " + ", ".join(missing))
    comparisons = {}
    for regime in settings["regimes"]:
        primary_iteration = "16" if regime == "early" else "1024"
        for comparison in ("untied_compute", "untied_parameters"):
            seed_differences = []
            for seed in settings["seeds"]:
                left = results.get(run_name("tied", regime, seed), {})
                right = results.get(run_name(comparison, regime, seed), {})
                left_scores = left.get("evaluations", {}).get("final", {}).get("holdout")
                right_scores = right.get("evaluations", {}).get("final", {}).get("holdout")
                if left_scores is None or right_scores is None:
                    continue
                differences = {str(horizon): 100 * (left_scores[str(horizon)]["accuracy"]
                                                    - right_scores[str(horizon)]["accuracy"])
                               for horizon in settings["evaluation"]["iterations"]}
                left_path = root / "evaluations" / run_name("tied", regime, seed) / "final/holdout_predictions.npz"
                right_path = root / "evaluations" / run_name(comparison, regime, seed) / "final/holdout_predictions.npz"
                with np.load(left_path, allow_pickle=False) as left_predictions, np.load(right_path, allow_pickle=False) as right_predictions:
                    puzzle_counts = {str(horizon): paired_puzzle_counts(left_predictions[f"solved_{horizon}"],
                                                                       right_predictions[f"solved_{horizon}"])
                                     for horizon in settings["evaluation"]["iterations"]}
                seed_differences.append({"seed": seed, "tied_minus_untied_percentage_points": differences,
                                         "paired_puzzle_counts": puzzle_counts})
            primary_differences = [row["tied_minus_untied_percentage_points"][primary_iteration] for row in seed_differences]
            comparisons[f"{regime}/{comparison}"] = {
                "primary_iteration": int(primary_iteration), "paired_seeds": seed_differences,
                "mean_primary_difference_percentage_points": float(np.mean(primary_differences)) if primary_differences else None,
                "seed_count": len(primary_differences),
                "missing_or_failed_pairs": len(settings["seeds"]) - len(primary_differences),
                "interpretation": "Positive differences favor tied weights. Averages include only pairs with both final evaluations; consult reliability counts for failures. Three seeds do not establish a precise reliability probability.",
            }
    reliability = {f"{architecture}/{regime}": reliability_summary(results, architecture, regime)
                   for architecture in settings["architectures"] for regime in settings["regimes"]}
    report = {"protocol_sha256": protocol_sha256(), "partial": partial, "missing": missing,
              "runs": results, "comparisons": comparisons, "reliability": reliability}
    atomic_json_save(report, output / "report.json")
    atomic_json_save(histories, output / "learning_curves.json")
    (output / "report.md").write_text(render_markdown(report))
    return report


def render_markdown(report):
    horizons = protocol()["evaluation"]["iterations"]
    lines = ["# Weight-Tying Results", "", "Partial report; no final conclusions." if report["partial"] else "All preregistered runs are included.", "",
             "## Training", "", "The validation floor and mean cover updates 12K-20K. The measured inference count is 16 for models trained only on iterations 1-16, and 1024 for models also trained on later iterations.", "",
             "| Run | Status | Parameters | Training Hours | Total Recorded Hours | Validation Mean | Validation Floor |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, summary in report["runs"].items():
        floor = summary["late_validation_minimum"]
        floor_text = f"{100 * floor:.2f}%" if floor is not None else "not reached"
        mean = summary["late_validation_mean"]
        mean_text = f"{100 * mean:.2f}%" if mean is not None else "not reached"
        timings = summary["timings_seconds"]
        total = sum(value for category, value in timings.items() if category != "optimizer")
        lines.append(f"| {name} | {summary['status']} | {summary['parameters']:,} | "
                     f"{timings['training'] / 3600:.2f} | {total / 3600:.2f} | {mean_text} | {floor_text} |")
    for selection, heading in (("final", "Final Checkpoints (Primary)"),
                               ("best_validation", "Validation-Selected Checkpoints (Secondary)")):
        lines.extend(["", f"## {heading}"])
        for dataset, label in (("development", "Reused 25K Sudoku-Extreme Benchmark"),
                               ("holdout", "New 10K QQWing Test Set")):
            lines.extend(["", f"### {label}", "",
                          "| Run | Selected Update | " + " | ".join(f"@{horizon}" for horizon in horizons) + " |",
                          "|---|---:|" + "---:|" * len(horizons)])
            for name, summary in report["runs"].items():
                scores = summary["evaluations"].get(selection, {}).get(dataset, {})
                missing = "not evaluated (failed)" if summary["status"] == "numerical_failure" else "pending"
                values = [f"{100 * scores[str(horizon)]['accuracy']:.2f}%" if str(horizon) in scores else missing
                          for horizon in horizons]
                updates = summary["updates"] if selection == "final" else summary["best_validation"].get("updates", 0)
                lines.append(f"| {name} | {updates} | {' | '.join(values)} |")
    lines.extend(["", "## Reliability", "",
                  "Healthy means at least 90% at 1024 iterations and no more than a 5-point drop by 4096. Numerical failures remain in the denominator. Pending runs are not failures.", "",
                  "| Architecture / Regime | Completed | Numerical Failures | Pending Training | Healthy Validation | Healthy Development | Healthy Holdout |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name, counts in report["reliability"].items():
        health = [f"{counts['datasets'][dataset]['healthy']}/{counts['planned']} ({counts['datasets'][dataset]['evaluated']} evaluated)"
                  for dataset in ("validation", "development", "holdout")]
        lines.append(f"| {name} | {counts['completed']} | {counts['numerical_failures']} | {counts['pending_training']} | {' | '.join(health)} |")
    lines.extend(["", "## Paired Comparisons", "",
                  "These are differences between three paired training seeds, not confidence intervals obtained by treating puzzles as independent training runs.", ""])
    for name, comparison in report["comparisons"].items():
        value = comparison["mean_primary_difference_percentage_points"]
        rendered = "pending" if value is None else f"{value:+.2f} percentage points"
        lines.append(f"- {name}, iteration {comparison['primary_iteration']}: {rendered}; {comparison['seed_count']} completed pairs, {comparison['missing_or_failed_pairs']} missing or failed pairs.")
    lines.extend(["", "Early-trained untied stacks beyond iteration 16 are explicit stack-repetition diagnostics. Late-trained stacks already repeat during training. Neither result represents a fully untied network with thousands of independently trained stages.", "",
                  "Training time includes forward/backward passes, data transfer, optimizer work, and synchronization. Optimizer time is a subset, not an additional category. Total recorded hours add preparation, compilation, evaluation, and checkpoint time, but exclude queueing and container setup. New compiler signatures can add compilation time to the first training batches. The JSON records these categories and nominal FLOP estimates separately.", "",
                  "The new QQWing set and the reused sudoku-extreme benchmark are different distributions and must not be pooled. The paired comparisons above use the new set. Three training seeds do not establish a precise probability of a successful run."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partial", action="store_true")
    arguments = parser.parse_args()
    report = summarize(arguments.root, arguments.output, partial=arguments.partial)
    print(json.dumps({"runs": len(report["runs"]), "missing": report["missing"]}))


if __name__ == "__main__":
    main()
