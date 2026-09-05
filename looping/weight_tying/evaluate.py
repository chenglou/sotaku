"""FP32 evaluation and a cohort-wide lock before opening the new test set."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, validate_config
from looping.weight_tying.common import SOURCE_PATHS, atomic_npz, protocol, protocol_sha256, run_config, run_name
from looping.weight_tying.data import load_manifest
from looping.weight_tying.model import StudyTransformer
from runtime_utils import file_sha256, runtime_manifest


@torch.inference_mode()
def evaluate_arrays(model, digits, targets, iterations, *, batch_size=256, repeat=False):
    if list(iterations) != sorted(set(iterations)) or not iterations or iterations[0] <= 0:
        raise ValueError("Iteration counts must be positive, unique, and increasing")
    if model.period > 1 and iterations[-1] > model.period and not repeat:
        raise ValueError("Long evaluation of an untied stack requires explicit repetition")
    was_training = model.training
    previous_precision = torch.get_float32_matmul_precision()
    model.eval()
    torch.set_float32_matmul_precision("highest")
    device = next(model.parameters()).device
    predictions = {horizon: [] for horizon in iterations}
    finite = {horizon: [] for horizon in iterations}
    try:
        for start in range(0, len(digits), batch_size):
            inputs = F.one_hot(torch.as_tensor(digits[start:start + batch_size], device=device).long(), 10).float()
            hidden, probabilities = model.initial_state(inputs)
            for iteration in range(iterations[-1]):
                hidden, probabilities, logits = model.step(hidden, probabilities, iteration, repeat=repeat)
                horizon = iteration + 1
                if horizon in predictions:
                    predictions[horizon].append(logits.argmax(-1).to(torch.uint8).cpu().numpy())
                    finite[horizon].append((torch.isfinite(logits).all(dim=(-2, -1))
                                            & torch.isfinite(hidden).all(dim=(-2, -1))).cpu().numpy())
    finally:
        model.train(was_training)
        torch.set_float32_matmul_precision(previous_precision)
    arrays, scores = {}, {}
    previous_solved = None
    for horizon in iterations:
        predicted = np.concatenate(predictions[horizon])
        finite_rows = np.concatenate(finite[horizon])
        solved = ((predicted == targets) | (digits != 0)).all(-1) & finite_rows
        arrays[f"predictions_{horizon}"] = predicted
        arrays[f"finite_{horizon}"] = finite_rows
        arrays[f"solved_{horizon}"] = solved
        score = {"solved": int(solved.sum()), "total": len(digits),
                 "accuracy": float(solved.mean()), "nonfinite": int((~finite_rows).sum())}
        if previous_solved is not None:
            score["lost_since_previous"] = int((previous_solved & ~solved).sum())
            score["gained_since_previous"] = int((~previous_solved & solved).sum())
        scores[str(horizon)] = score
        previous_solved = solved
    return scores, arrays


def load_export(path, *, device):
    path = Path(path)
    manifest = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if manifest["weights_sha256"] != file_sha256(path):
        raise ValueError("Export checksum mismatch")
    if manifest["config"]["protocol_sha256"] != protocol_sha256():
        raise ValueError("Export belongs to a different protocol")
    model = StudyTransformer(**manifest["config"]["model"]).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model, manifest


def seal_cohort(root):
    root = Path(root)
    settings = protocol()
    entries = {}
    paired_digests = {}
    data_digest = file_sha256(root / "data" / "manifest.json") if (root / "data" / "manifest.json").exists() else None
    for architecture in settings["architectures"]:
        for regime in settings["regimes"]:
            for seed in settings["seeds"]:
                name = run_name(architecture, regime, seed)
                directory = root / "runs" / name
                result_path = directory / "result.json"
                if not result_path.exists():
                    raise ValueError(f"Cohort incomplete: {name}")
                result = json.loads(result_path.read_text())
                validate_config(result["config"], run_config(architecture, regime, seed))
                if result["config"]["protocol_sha256"] != protocol_sha256() or result["config"]["smoke"]:
                    raise ValueError(f"Wrong protocol or smoke result: {name}")
                if result["status"] not in ("complete", "numerical_failure"):
                    raise ValueError(f"Unfinished cohort member: {name}")
                if result["data_manifest_sha256"] != data_digest:
                    raise ValueError(f"Different data in cohort member: {name}")
                if result["status"] == "complete" and result["updates"] != settings["training"]["steps"]:
                    raise ValueError(f"Wrong training budget: {name}")
                exports = {}
                if result["status"] == "complete":
                    pair = (regime, seed)
                    paired_digests.setdefault(pair, result["sample_digest"])
                    if paired_digests[pair] != result["sample_digest"]:
                        raise ValueError(f"Puzzle/horizon samples are not paired: {name}")
                    for selection in ("final", "best_validation"):
                        path = directory / f"{selection}.pt"
                        metadata = json.loads(path.with_suffix(".pt.json").read_text())
                        validate_config(metadata["config"], result["config"])
                        expected_updates = result["updates"] if selection == "final" else result["best_validation"]["updates"]
                        if metadata["updates"] != expected_updates or metadata["data_manifest_sha256"] != data_digest:
                            raise ValueError(f"Wrong selected step or data identity: {name}/{selection}")
                        digest = file_sha256(path)
                        if metadata["weights_sha256"] != digest:
                            raise ValueError(f"Unverified checkpoint: {name}/{selection}")
                        exports[selection] = {"weights_sha256": digest,
                                              "manifest_sha256": file_sha256(path.with_suffix(".pt.json")),
                                              "updates": metadata["updates"]}
                entries[name] = {"result_sha256": file_sha256(result_path),
                                 "status": result["status"], "exports": exports}
    data_manifest = load_manifest(root / "data")
    identity = {"protocol_sha256": protocol_sha256(), "runs": entries,
                "holdout_sha256": data_manifest["files"]["holdout.npz"]}
    path = root / "cohort_lock.json"
    if path.exists():
        saved = json.loads(path.read_text())
        validate_config(saved["identity"], identity)
        return saved
    locked = {"created_at": datetime.now(timezone.utc).isoformat(), "identity": identity}
    atomic_json_save(locked, path)
    return locked


def evaluate_run(root, architecture, regime, seed, selection="final"):
    root = Path(root)
    if not (root / "cohort_lock.json").exists():
        raise ValueError("Run the separate cohort-sealing job before any held-out evaluation")
    lock = seal_cohort(root)
    if selection not in ("final", "best_validation"):
        raise ValueError("Unsupported selection")
    name = run_name(architecture, regime, seed)
    entry = lock["identity"]["runs"][name]
    if entry["status"] != "complete":
        return {"run_name": name, "status": entry["status"]}
    directory = root / "evaluations" / name / selection
    directory.mkdir(parents=True, exist_ok=True)
    metadata = load_manifest(root / "data", verify=("holdout.npz", "development.npz"))
    model, export = load_export(root / "runs" / name / f"{selection}.pt", device="cuda")
    torch.set_float32_matmul_precision("highest")
    identity = {"protocol_sha256": protocol_sha256(), "cohort_lock_sha256": file_sha256(root / "cohort_lock.json"),
                "weights_sha256": export["weights_sha256"], "selection": selection,
                "run_name": name, "iterations": protocol()["evaluation"]["iterations"],
                "precision": "fp32", "compiled": False, "tf32_matmul": False,
                "repeat_stack": model.period > 1, "model": export["config"]["model"]}
    complete = directory / "result.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        validate_config(result["identity"], identity)
        return result
    atomic_json_save(runtime_manifest(SOURCE_PATHS), directory / "environment.json")
    scores = {}
    started = time.perf_counter()
    for dataset_name in ("development", "holdout"):
        arrays = np.load(root / "data" / f"{dataset_name}.npz", allow_pickle=False)
        dataset_scores, predictions = evaluate_arrays(model, arrays["digits"], arrays["targets"], identity["iterations"],
                                                     batch_size=protocol()["evaluation"]["batch_size"], repeat=True)
        for horizon in identity["iterations"]:
            solved = predictions[f"solved_{horizon}"]
            dataset_scores[str(horizon)]["difficulty"] = {
                str(label): {"solved": int(solved[arrays["labels"] == label].sum()),
                             "total": int((arrays["labels"] == label).sum())}
                for label in np.unique(arrays["labels"])
            }
        predictions["labels"] = arrays["labels"]
        atomic_npz(directory / f"{dataset_name}_predictions.npz", **predictions)
        scores[dataset_name] = dataset_scores
        print(f"{name}/{selection}/{dataset_name}: {json.dumps(dataset_scores)}", flush=True)
    result = {"identity": identity, "scores": scores, "elapsed_seconds": time.perf_counter() - started,
              "dataset_sha256": {key: metadata["files"][f"{key}.npz"] for key in scores},
              "prediction_sha256": {key: file_sha256(directory / f"{key}_predictions.npz") for key in scores}}
    atomic_json_save(result, complete)
    return result
