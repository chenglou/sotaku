"""Verify downloaded evaluations and compare them with training-time validation."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save, validate_config
from looping.weight_tying.analyze import read_evaluation
from looping.weight_tying.common import DIRECTORY, protocol, protocol_sha256, run_config, run_name
from runtime_utils import file_sha256


def validation_positions(development, validation):
    indices = development["indices"]
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate development indices")
    positions = {int(index): position for position, index in enumerate(indices)}
    try:
        selected = np.array([positions[int(index)] for index in validation["indices"]])
    except KeyError as error:
        raise ValueError("Validation puzzle is missing from development data") from error
    for key in ("digits", "targets", "labels"):
        np.testing.assert_array_equal(development[key][selected], validation[key])
    return selected


def validation_replay(predictions, positions, original_scores):
    comparisons = {}
    for horizon, score in original_scores.items():
        solved = int(predictions[f"solved_{horizon}"][positions].sum())
        comparisons[horizon] = {"training_probe_solved": score["solved"],
                                "full_evaluation_subset_solved": solved,
                                "difference": solved - score["solved"]}
    return comparisons


def audit(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {output}")
    settings = protocol()
    lock_path = root / "cohort_lock.json"
    lock = json.loads(lock_path.read_text())
    lock_hash = file_sha256(lock_path)
    before = json.loads((DIRECTORY / "pre_evaluation_audit.json").read_text())
    manifest = json.loads((root / "data/manifest.json").read_text())
    if file_sha256(root / "data/manifest.json") != before["data_manifest_sha256"]:
        raise ValueError("Data manifest differs from the pre-evaluation audit")
    for name in ("development", "validation", "holdout"):
        if file_sha256(root / "data" / f"{name}.npz") != manifest["files"][f"{name}.npz"]:
            raise ValueError(f"Dataset hash mismatch: {name}")
    cache = json.loads((root / "received_evaluations.json").read_text())
    if cache["errors"]:
        raise ValueError("Unresolved result-collection errors")
    with np.load(root / "data/development.npz", allow_pickle=False) as development, np.load(root / "data/validation.npz", allow_pickle=False) as validation:
        positions = validation_positions(development, validation)
    verified, repeated = {}, {}
    for architecture in settings["architectures"]:
        for regime in settings["regimes"]:
            for seed in settings["seeds"]:
                name = run_name(architecture, regime, seed)
                training_path = root / "runs" / name / "result.json"
                training = json.loads(training_path.read_text())
                entry = lock["identity"]["runs"][name]
                if file_sha256(training_path) != entry["result_sha256"]:
                    raise ValueError(f"Training result changed after locking: {name}")
                validate_config(training["config"], run_config(architecture, regime, seed))
                validate_config(training["source_sha256"], before["source_sha256"])
                for selection in ("final", "best_validation"):
                    key = f"{name}/{selection}"
                    evaluation = read_evaluation(root, name, selection, entry)
                    if evaluation is None or evaluation != cache["results"].get(key):
                        raise ValueError(f"Volume result is missing or differs from returned result: {key}")
                    identity = {"protocol_sha256": protocol_sha256(), "run_name": name,
                                "selection": selection, "cohort_lock_sha256": lock_hash,
                                "weights_sha256": entry["exports"][selection]["weights_sha256"],
                                "model": settings["architectures"][architecture],
                                "iterations": settings["evaluation"]["iterations"],
                                "precision": "fp32", "compiled": False, "tf32_matmul": False,
                                "repeat_stack": architecture != "tied"}
                    validate_config(evaluation["identity"], identity)
                    if set(evaluation["scores"]) != {"development", "holdout"}:
                        raise ValueError(f"Unexpected datasets: {key}")
                    directory = root / "evaluations" / name / selection
                    environment = json.loads((directory / "environment.json").read_text())
                    validate_config(environment["source_sha256"], before["source_sha256"])
                    for setting, expected in (("matmul_precision", "highest"), ("tf32_matmul", False),
                                              ("gpu", "NVIDIA H200"), ("cuda", "12.8")):
                        if environment[setting] != expected:
                            raise ValueError(f"Unexpected {setting}: {key}")
                    if environment["packages"]["torch"] != "2.10.0+cu128":
                        raise ValueError(f"Unexpected PyTorch version: {key}")
                    selected_step = entry["exports"][selection]["updates"]
                    original = next(row["scores"] for row in training["history"] if row["updates"] == selected_step)
                    with np.load(directory / "development_predictions.npz", allow_pickle=False) as predictions:
                        replay = validation_replay(predictions, positions, original)
                    verified[key] = {"result_sha256": file_sha256(directory / "result.json"),
                                     "environment_sha256": file_sha256(directory / "environment.json"),
                                     "prediction_sha256": evaluation["prediction_sha256"],
                                     "weights_sha256": identity["weights_sha256"],
                                     "selected_update": selected_step, "validation_replay": replay}
                if entry["exports"]["final"]["updates"] == entry["exports"]["best_validation"]["updates"]:
                    repeated[name] = {}
                    for dataset in ("development", "holdout"):
                        directory = root / "evaluations" / name
                        with np.load(directory / "final" / f"{dataset}_predictions.npz", allow_pickle=False) as final, np.load(directory / "best_validation" / f"{dataset}_predictions.npz", allow_pickle=False) as best:
                            if set(final.files) != set(best.files):
                                raise ValueError(f"Repeated export has different prediction arrays: {name}")
                            repeated[name][dataset] = {key: int(np.count_nonzero(final[key] != best[key])) for key in final.files}
    if set(verified) != set(cache["results"]):
        raise ValueError("Result cache contains an unexpected evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {"created_at": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_sha256(),
              "cohort_lock_sha256": lock_hash, "evaluations_verified": len(verified),
              "checks": ["Frozen training results, configs, and source hashes",
                         "Dataset and prediction checksums; scores recomputed from all saved predictions",
                         "FP32 eager H200 environments with TF32 matmul disabled",
                         "Identical results from durable Volume files and returned function calls",
                         "Validation puzzle identity and selected-checkpoint subset replay"],
              "evaluations": verified, "same_step_export_differences": repeated,
              "maximum_validation_solved_count_difference": max(abs(row["difference"]) for entry in verified.values() for row in entry["validation_replay"].values())}
    atomic_json_save(result, output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = audit(arguments.root, arguments.output)
    print(f"Verified {result['evaluations_verified']} full evaluations; maximum validation count difference: {result['maximum_validation_solved_count_difference']}")
