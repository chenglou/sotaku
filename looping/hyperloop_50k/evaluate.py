"""Fixed-count FP32 evaluation of the study's own recurrence and tensor exports."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save, validate_config
from inference import validate_iterations
from looping.hyperloop_50k.common import SOURCE_PATHS, build_model, protocol, run_config, validate_data
from looping.weight_tying.common import atomic_npz
from runtime_utils import file_sha256, runtime_manifest


def export_model(model, path, config, updates, data_identity, source_identity):
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    return export_state(state, path, config, updates, data_identity, source_identity)


def export_state(state, path, config, updates, data_identity, source_identity):
    atomic_torch_save(state, path)
    manifest = {"artifact_type": "sotaku-hyperloop-50k-inference", "schema_version": 1,
                "weights_sha256": file_sha256(path), "config": config, "updates": updates,
                "data_sha256": data_identity, "source_sha256": source_identity}
    atomic_json_save(manifest, str(path) + ".json")
    return manifest


def load_export(path, *, device="cpu"):
    path = Path(path)
    manifest = json.loads(Path(str(path) + ".json").read_text())
    if manifest.get("artifact_type") != "sotaku-hyperloop-50k-inference" or manifest.get("schema_version") != 1:
        raise ValueError("Not a Hyperloop study inference export")
    config = manifest["config"]
    validate_config(config, run_config(config["arm"], config["seed"], smoke=config["smoke"]))
    if file_sha256(path) != manifest["weights_sha256"]:
        raise ValueError("Inference checksum mismatch")
    model = build_model(config).float().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True), strict=True)
    return model.eval(), manifest


@torch.inference_mode()
def evaluate_arrays(model, digits, targets, iterations, *, batch_size=256, track_solutions=False, progress=None):
    iterations = validate_iterations(iterations)
    if len(digits) == 0 or digits.shape != targets.shape or digits.shape[1:] != (81,) or batch_size <= 0:
        raise ValueError("Expected nonempty, matching 81-cell arrays and a positive batch size")
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError("Evaluation requires FP32 model weights")
    was_training, previous_precision = model.training, torch.get_float32_matmul_precision()
    model.eval()
    torch.set_float32_matmul_precision("highest")
    device = next(model.parameters()).device
    predictions, finite_rows = {h: [] for h in iterations}, {h: [] for h in iterations}
    first_solved, regression_counts, ever_solved = [], [], []
    diagnostics = {}
    try:
        with torch.autocast(device.type, enabled=False):
            for start in range(0, len(digits), batch_size):
                given = torch.as_tensor(digits[start:start + batch_size], device=device).long()
                answers = torch.as_tensor(targets[start:start + batch_size], device=device).long()
                inputs = F.one_hot(given, 10).float()
                hidden, probabilities = model.initial_state(inputs)
                previous_solved = torch.zeros(len(inputs), dtype=torch.bool, device=device)
                first = torch.zeros(len(inputs), dtype=torch.int32, device=device)
                regressions = torch.zeros_like(first)
                for step in range(1, iterations[-1] + 1):
                    hidden, probabilities, logits = model.step(hidden, probabilities)
                    if track_solutions or step in predictions:
                        predicted = logits.argmax(-1)
                        finite = torch.isfinite(logits).flatten(1).all(-1) & torch.isfinite(hidden).flatten(1).all(-1)
                        solved = ((predicted == answers) | (given != 0)).all(-1) & finite
                        if track_solutions:
                            first = torch.where((first == 0) & solved, step, first)
                            regressions += (previous_solved & ~solved).int()
                            previous_solved = solved
                    if step in predictions:
                        predictions[step].append(predicted.to(torch.uint8).cpu().numpy())
                        finite_rows[step].append(finite.cpu().numpy())
                        if start == 0:
                            diagnostics[str(step)] = model.state_diagnostics(hidden[:64])
                if track_solutions:
                    first_solved.append(first.cpu().numpy())
                    regression_counts.append(regressions.cpu().numpy())
                    ever_solved.append((first > 0).cpu().numpy())
                if progress:
                    progress(f"EVAL {min(start + batch_size, len(digits))}/{len(digits)} through {iterations[-1]}")
    finally:
        model.train(was_training)
        torch.set_float32_matmul_precision(previous_precision)
    arrays, scores, previous = {}, {}, None
    for horizon in iterations:
        predicted, finite = np.concatenate(predictions[horizon]), np.concatenate(finite_rows[horizon])
        solved = ((predicted == targets) | (digits != 0)).all(-1) & finite
        arrays.update({f"predictions_{horizon}": predicted, f"finite_{horizon}": finite, f"solved_{horizon}": solved})
        score = {"accuracy": float(solved.mean()), "solved": int(solved.sum()), "total": len(digits),
                 "nonfinite": int((~finite).sum())}
        if previous is not None:
            score.update({"lost_since_previous": int((previous & ~solved).sum()),
                          "gained_since_previous": int((~previous & solved).sum())})
        scores[str(horizon)], previous = score, solved
    if track_solutions:
        arrays.update({"first_solved_iteration": np.concatenate(first_solved),
                       "regression_count": np.concatenate(regression_counts), "ever_solved": np.concatenate(ever_solved)})
        arrays["stayed_solved_after_first"] = arrays["ever_solved"] & (arrays["regression_count"] == 0)
    return scores, arrays, diagnostics


def evaluate_run(data_dir, run_dir, selection, *, device="cuda", checkpoint_callback=None):
    if selection not in ("final", "best_validation"):
        raise ValueError("Unknown checkpoint selection")
    data_dir, run_dir = Path(data_dir), Path(run_dir)
    training = json.loads((run_dir / "result.json").read_text())
    if training["status"] != "complete":
        return {"status": "not_evaluated", "training_status": training["status"]}
    if training["config"]["smoke"]:
        raise ValueError("Do not report smoke fixtures as full evaluation")
    model, manifest = load_export(run_dir / f"{selection}.pt", device=device)
    validate_config(manifest["config"], training["config"])
    validate_config(manifest["source_sha256"], training["source_sha256"])
    validate_config(manifest["source_sha256"], runtime_manifest(SOURCE_PATHS)["source_sha256"])
    expected_step = training["updates"] if selection == "final" else training["best_validation"]["updates"]
    if manifest["updates"] != expected_step:
        raise ValueError("Export has the wrong checkpoint step")
    validate_config(manifest["data_sha256"], training["data_sha256"])
    data_identity = validate_data(data_dir, names=("development.npz",))
    settings = protocol()["evaluation"]
    torch.set_float32_matmul_precision("highest")
    identity = {**settings, "weights_sha256": manifest["weights_sha256"], "updates": expected_step,
                "data_sha256": data_identity, "checkpoint_selection": selection,
                "config": manifest["config"], "source_sha256": manifest["source_sha256"]}
    output = run_dir / "evaluations" / selection
    output.mkdir(parents=True, exist_ok=True)
    completed = output / "result.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        validate_config(result["identity"], identity)
        if result["predictions_sha256"] != file_sha256(output / "predictions.npz"):
            raise ValueError("Saved evaluation predictions have changed")
        return result
    benchmark = json.loads((Path(__file__).parents[2] / settings["benchmark"]).read_text())
    with np.load(data_dir / "development.npz", allow_pickle=False) as arrays:
        if len(arrays["digits"]) != 25000 or not np.array_equal(arrays["indices"], benchmark["indices"]):
            raise ValueError("Full evaluation must use the exact frozen 25K benchmark indices")
        started = time.perf_counter()
        with (output / "eval.log").open("a") as log:
            def progress(message):
                print(message, flush=True)
                log.write(message + "\n")
                log.flush()
            scores, predictions, diagnostics = evaluate_arrays(
                model, arrays["digits"], arrays["targets"], settings["iterations"],
                batch_size=settings["batch_size"], track_solutions=True, progress=progress)
        predictions["indices"], predictions["labels"] = arrays["indices"], arrays["labels"]
        for horizon in settings["iterations"]:
            solved = predictions[f"solved_{horizon}"]
            scores[str(horizon)]["difficulty"] = {
                str(label): {"solved": int(solved[arrays["labels"] == label].sum()),
                             "total": int((arrays["labels"] == label).sum())}
                for label in np.unique(arrays["labels"])}
    atomic_npz(output / "predictions.npz", **predictions)
    result = {"identity": identity, "scores": scores, "state_diagnostics_first_batch": diagnostics,
              "predictions_sha256": file_sha256(output / "predictions.npz"),
              "elapsed_seconds": time.perf_counter() - started}
    atomic_json_save(runtime_manifest(SOURCE_PATHS), output / "environment.json")
    atomic_json_save(result, completed)
    print(f"FULL_RESULT {selection} {json.dumps(scores, sort_keys=True)}", flush=True)
    if checkpoint_callback:
        checkpoint_callback()
    return result
