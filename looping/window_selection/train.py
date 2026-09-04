"""Matched scan/replay training and ordinary FP32 evaluation."""

import copy
import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save, validate_config
from inference import RecurrentRunner
from looping.weight_tying.common import update_sample_digest
from looping.weight_tying.train import PairedSampler, learning_rate, restore_rng, rng_state
from looping.window_selection.common import SOURCE_PATHS, protocol, run_config, state_sha256, validate_data
from looping.window_selection.selection import (
    WindowTransformer, confidence_score, scan_candidates, select_index, window_diagnostics,
)
from model_io import write_model_manifest
from runtime_utils import runtime_manifest


@torch.inference_mode()
def monitor(model, digits, targets, horizons, batch_size=256):
    was_training = model.training
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    runner = RecurrentRunner(model)
    device = next(model.parameters()).device
    counts = {str(horizon): 0 for horizon in horizons}
    try:
        for start in range(0, len(digits), batch_size):
            inputs = F.one_hot(torch.as_tensor(digits[start:start + batch_size], device=device).long(), 10).float()
            answers = torch.as_tensor(targets[start:start + batch_size], device=device).long()
            try:
                outputs, _ = runner.run_batch(inputs, horizons)
            except ValueError as error:
                if "non-finite logits or recurrent states" not in str(error):
                    raise
                raise FloatingPointError(str(error)) from error
            for horizon, logits in outputs.items():
                counts[str(horizon)] += int(((logits.argmax(-1) == answers) | ~inputs[:, :, 0].bool()).all(-1).sum())
    finally:
        model.train(was_training)
        torch.set_float32_matmul_precision(previous_precision)
    return {key: {"solved": value, "total": len(digits), "accuracy": value / len(digits)}
            for key, value in counts.items()}


def export(model, path, config, updates, data_identity):
    atomic_torch_save({key: value.detach().cpu() for key, value in model.state_dict().items()}, path)
    write_model_manifest(path, model, training={"config": config, "updates": updates},
                         provenance={"data_sha256": data_identity})


def train_run(data_dir, output_dir, selector, seed, *, smoke=False, stop_after=None,
              device="cuda", checkpoint_callback=None):
    config = run_config(selector, seed, smoke=smoke)
    specification = config["protocol"]
    settings = copy.deepcopy(specification["training"])
    if smoke:
        settings.update(steps=4, batch_size=2, warmup_steps=1, probe_every=2,
                        phases=[[0, 4, 0]])
    starts = (16, 32) if smoke else tuple(specification["candidate_starts"])
    probability = 1.0 if smoke else specification["late_probability"]
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_identity = validate_data(data_dir, smoke=smoke)
    source_identity = runtime_manifest(SOURCE_PATHS)["source_sha256"]
    result_path = output_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        validate_config(result["config"], config)
        validate_config(result["source_sha256"], source_identity)
        validate_config(result["data_sha256"], data_identity)
        return result

    with (output_dir / "train.log").open("a", buffering=1) as log_file:
        def log(message):
            line = f"{datetime.now(timezone.utc).isoformat()} | {message}"
            print(line, flush=True)
            log_file.write(line + "\n")

        with np.load(data_dir / "train.npz", allow_pickle=False) as data:
            digits, targets, ratings = data["digits"], data["targets"], data["ratings"]
        with np.load(data_dir / "validation.npz", allow_pickle=False) as data:
            validation_digits, validation_targets = data["digits"], data["targets"]
        if not smoke and (len(digits) != settings["train_rows"] or len(validation_digits) != 1000):
            raise ValueError("Incorrect training or monitoring dataset size")
        torch.manual_seed(seed)
        model = WindowTransformer().to(device)
        initial_weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        initial_digest = state_sha256(initial_weights)
        torch.manual_seed(seed + 10_000_019)
        sampler = PairedSampler(ratings, seed, {"burnin_probability": probability,
                                               "burnin_iterations": starts}, settings)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                     betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
        updates, history, best = 0, [], {"accuracy": -1.0, "updates": 0}
        digest, selected_counts = "0" * 64, {str(start): 0 for start in starts}
        selection_history = []
        timing = {key: 0.0 for key in ("compilation", "training", "scanning", "monitoring", "checkpoint")}
        scan_count, replay_count, ordinary_count = 0, 0, 0
        checkpoint_path = output_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            validate_config(saved["config"], config)
            validate_config(saved["source_sha256"], source_identity)
            validate_config(saved["data_sha256"], data_identity)
            if saved["initial_state_sha256"] != initial_digest:
                raise ValueError("Initialization differs from the saved run")
            model.load_state_dict(saved["model_state_dict"])
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            sampler.load_state_dict(saved["sampler"])
            restore_rng(saved["rng"])
            updates, history, best = saved["updates"], saved["history"], saved["best"]
            digest, selected_counts = saved["sample_digest"], saved["selected_counts"]
            selection_history, timing = saved["selection_history"], saved["timing"]
            scan_count, replay_count, ordinary_count = saved["work_counts"]
            log(f"RESUME updates={updates}")
        if (output_dir / "initial.pt").exists():
            if state_sha256(torch.load(output_dir / "initial.pt", weights_only=True)) != initial_digest:
                raise ValueError("Saved initialization has different tensor contents")
        else:
            atomic_torch_save(initial_weights, output_dir / "initial.pt")
        del initial_weights
        environment = runtime_manifest(SOURCE_PATHS)
        atomic_json_save(environment, output_dir / f"environment_{time.time_ns()}.json")
        if not (output_dir / "source.tar.gz").exists():
            with tarfile.open(output_dir / "source.tar.gz", "w:gz") as archive:
                for relative in SOURCE_PATHS:
                    archive.add(Path(__file__).parents[2] / relative, arcname=relative)
        log("CONFIG " + json.dumps(config, sort_keys=True))
        log("DATA " + json.dumps(data_identity, sort_keys=True))
        log(f"PARAMETERS {sum(parameter.numel() for parameter in model.parameters())}; "
            f"selection_unit=whole_batch; starts={starts}; supervised=16; "
            f"scan_through={starts[-1] + 16}; late_probability={probability}")
        kind = torch.device(device).type
        torch.set_float32_matmul_precision("high")
        forward = torch.compile(model) if kind == "cuda" else model
        advance = torch.compile(model.scan_window) if kind == "cuda" else model.scan_window

        def autocast():
            return torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda")

        def synchronize():
            if kind == "cuda":
                torch.cuda.synchronize()

        def batch(indices):
            inputs = F.one_hot(torch.as_tensor(digits[indices], device=device).long(), 10).float()
            answers = torch.as_tensor(targets[indices], device=device).long()
            return inputs, answers

        warmup_started, saved_rng = time.perf_counter(), rng_state()
        inputs, answers = batch(np.arange(settings["batch_size"]) % len(digits))
        model.train()
        with autocast():
            loss, _, _ = forward(inputs, answers)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast():
            initial = model.initial_state(inputs)
            initial, _ = advance(*initial)
        with autocast():
            loss, _, _ = forward(inputs, answers, initial_state=initial)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        restore_rng(saved_rng)
        timing["compilation"] += time.perf_counter() - warmup_started
        del inputs, answers, initial, loss
        log(f"COMPILED starting_update={updates}; compilation_seconds={timing['compilation']:.3f}")

        def checkpoint():
            started = time.perf_counter()
            atomic_torch_save({"config": config, "source_sha256": source_identity,
                "data_sha256": data_identity, "updates": updates,
                "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "sampler": sampler.state_dict(), "rng": rng_state(), "history": history, "best": best,
                "sample_digest": digest, "selected_counts": selected_counts,
                "selection_history": selection_history, "timing": timing,
                "initial_state_sha256": initial_digest,
                "work_counts": [scan_count, replay_count, ordinary_count]}, checkpoint_path)
            if checkpoint_callback:
                checkpoint_callback()
            timing["checkpoint"] += time.perf_counter() - started

        limit = settings["steps"] if stop_after is None else stop_after
        if not updates <= limit <= settings["steps"]:
            raise ValueError("Invalid requested stopping update")
        status = "paused"
        while updates < limit:
            synchronize()
            started = time.perf_counter()
            indices, random_start = sampler.sample(updates, settings["batch_size"])
            inputs, answers = batch(indices)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(updates, settings)
            record, end_rng = None, None
            if random_start:
                scan_started = time.perf_counter()
                with autocast():
                    scan = scan_candidates(model, inputs, advance, starts)
                synchronize()
                timing["scanning"] += time.perf_counter() - scan_started
                try:
                    selected = select_index(scan.scores, selector, starts.index(random_start))
                except FloatingPointError:
                    status = "numerical_failure"
                    log(f"NONFINITE scan before update {updates + 1}")
                    break
                # Answer labels are used only after selection, for recorded diagnostics.
                diagnostics = [window_diagnostics(logits, inputs[:, :, 0].bool(), answers)
                               for logits in scan.logits]
                initial, end_rng = scan.states[selected], scan.end_rng
                record = {"updates": updates + 1, "random_start": random_start,
                          "selected_start": starts[selected], "confidence": scan.scores,
                          "diagnostics": diagnostics}
                restore_rng(scan.rng_states[selected])
                del scan
                with autocast():
                    loss, _, replay_logits = forward(inputs, answers, initial_state=initial)
                with torch.no_grad():
                    record["replay_confidence"] = float(confidence_score(replay_logits, inputs[:, :, 0].bool()))
                del initial, replay_logits
            else:
                with autocast():
                    loss, _, _ = forward(inputs, answers)
            if not torch.isfinite(loss).item():
                status = "numerical_failure"
                log(f"NONFINITE loss before update {updates + 1}")
                break
            loss.backward()
            if end_rng is not None:
                restore_rng(end_rng)
            finite_gradients = torch.stack([parameter.grad.isfinite().all() for parameter in model.parameters()
                                           if parameter.grad is not None]).all()
            if not finite_gradients.item():
                status = "numerical_failure"
                log(f"NONFINITE gradients before update {updates + 1}")
                break
            optimizer.step()
            synchronize()
            timing["training"] += time.perf_counter() - started
            digest = update_sample_digest(digest, indices, random_start)
            if record is not None:
                selected_counts[str(record["selected_start"])] += 1
                selection_history.append(record)
                scan_count += starts[-1] + 16
                replay_count += 16
                if len(selection_history) <= 3 or updates % 100 == 0 or smoke:
                    log("SELECTION " + json.dumps(record, sort_keys=True))
            else:
                ordinary_count += 16
            updates += 1
            if updates % 100 == 0 or smoke:
                log(f"UPDATE {updates} loss={loss.item():.6f} training_seconds={timing['training']:.3f} "
                    f"sample_digest={digest} selected_counts={json.dumps(selected_counts)}")
            if updates % settings["probe_every"] == 0 or updates == limit:
                monitor_started = time.perf_counter()
                horizons = [16] if smoke else [16, 128, 1024]
                try:
                    scores = monitor(model, validation_digits, validation_targets, horizons,
                                     batch_size=2 if smoke else 256)
                except FloatingPointError:
                    status = "numerical_failure"
                    log(f"NONFINITE validation after update {updates}")
                    break
                timing["monitoring"] += time.perf_counter() - monitor_started
                history.append({"updates": updates, "scores": scores})
                key = "16" if smoke else "1024"
                if scores[key]["accuracy"] > best["accuracy"]:
                    best = {"accuracy": scores[key]["accuracy"], "updates": updates}
                    export(model, output_dir / "best_validation.pt", config, updates, data_identity)
                log(f"VALIDATION {updates} {json.dumps(scores, sort_keys=True)}")
                checkpoint()
            elif updates % 250 == 0:
                checkpoint()
        if status != "numerical_failure" and updates == settings["steps"]:
            status = "complete"
            export(model, output_dir / "final.pt", config, updates, data_identity)
        result = {"config": config, "status": status, "updates": updates,
                  "source_sha256": source_identity, "data_sha256": data_identity,
                  "history": history, "best_validation": best, "sample_digest": digest,
                  "selected_counts": selected_counts, "selection_history": selection_history,
                  "timing_seconds": timing, "work_counts": {"scan_iterations": scan_count,
                  "replay_iterations": replay_count, "ordinary_iterations": ordinary_count},
                  "initial_state_sha256": initial_digest}
        if status != "paused":
            atomic_json_save(result, result_path)
            if checkpoint_callback:
                checkpoint_callback()
        log(f"RESULT {status} updates={updates} sample_digest={digest}")
        return result


def evaluate_run(directory, *, checkpoint_callback=None):
    from iters.eval_more_iters import evaluate
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        return {"status": result["status"]}
    settings = protocol()["evaluation"]
    for selection in ("final", "best_validation"):
        path = directory / f"{selection}.pt"
        metadata = json.loads(Path(str(path) + ".json").read_text())
        validate_config(metadata["training"]["config"], result["config"])
        expected_updates = result["updates"] if selection == "final" else result["best_validation"]["updates"]
        if metadata["training"]["updates"] != expected_updates:
            raise ValueError("Checkpoint has the wrong selected update")
        try:
            evaluate(path, iter_counts=settings["iterations"], benchmark_path=settings["benchmark"],
                     precision="fp32", compiled=False, batch_size=settings["batch_size"],
                     track_solutions=True, output_dir=directory / "evaluations" / selection)
        except ValueError as error:
            if "non-finite logits or recurrent states" not in str(error):
                raise
            failure = {"status": "numerical_failure", "checkpoint": selection, "error": str(error)}
            atomic_json_save(failure, directory / "evaluation_failure.json")
            if checkpoint_callback:
                checkpoint_callback()
            return failure
        if checkpoint_callback:
            checkpoint_callback()
    return {"status": "complete"}
