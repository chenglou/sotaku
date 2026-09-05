"""Matched training with unchanged CE, short gradient windows, and exact resumption."""

import json
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save, validate_config
from looping.hyperloop.common import SOURCE_PATHS, build_model, run_config, state_sha256, validate_data
from looping.hyperloop.evaluate import evaluate_arrays, export_model, export_state, load_export
from looping.weight_tying.common import update_sample_digest
from looping.weight_tying.train import PairedSampler, learning_rate, restore_rng, rng_state
from runtime_utils import file_sha256, runtime_manifest


def train_run(data_dir, output_dir, arm, seed, *, smoke=False, stop_after=None,
              device="cuda", checkpoint_callback=None):
    config = run_config(arm, seed, smoke=smoke)
    spec, settings = config["protocol"], config["protocol"]["training"]
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    data_identity = validate_data(data_dir, smoke=smoke)
    environment = runtime_manifest(SOURCE_PATHS)
    source_identity = environment["source_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "result.json").exists():
        result = json.loads((output_dir / "result.json").read_text())
        for name, expected in (("config", config), ("data_sha256", data_identity), ("source_sha256", source_identity)):
            validate_config(result[name], expected)
        if result["status"] == "complete":
            if result["updates"] != settings["steps"]:
                raise ValueError("Completed run has the wrong number of updates")
            for selection in ("final", "best_validation"):
                _, manifest = load_export(output_dir / f"{selection}.pt")
                for name, expected in (("config", config), ("data_sha256", data_identity), ("source_sha256", source_identity)):
                    validate_config(manifest[name], expected)
                expected_step = result["updates"] if selection == "final" else result["best_validation"]["updates"]
                if manifest["updates"] != expected_step:
                    raise ValueError("Completed export has the wrong training step")
        return result
    with np.load(data_dir / "train.npz", allow_pickle=False) as data:
        digits, targets, ratings = data["digits"], data["targets"], data["ratings"]
    with np.load(data_dir / "validation.npz", allow_pickle=False) as data:
        validation_digits, validation_targets = data["digits"], data["targets"]
    if not smoke and (len(digits) != settings["train_rows"] or len(validation_digits) != 1000):
        raise ValueError("Wrong training or monitoring dataset size")
    torch.manual_seed(seed)
    model = build_model(config).to(device)
    initial = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    initial_digest = state_sha256(initial)
    base_digest = state_sha256({name: value for name, value in initial.items() if not name.startswith("gates.")})
    torch.manual_seed(seed + 10_000_019)
    sampler = PairedSampler(ratings, seed, {"burnin_probability": spec["burnin_probability"],
                                          "burnin_iterations": spec["burnin_iterations"]}, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
    updates, history, best = 0, [], {"accuracy": -1.0, "updates": 0}
    best_state = None
    digest, horizon_counts = "0" * 64, {}
    timings = {name: 0.0 for name in ("compilation", "training", "monitoring", "checkpoint")}
    work_counts = {"gradient_free_iterations": 0, "supervised_iterations": 0}
    checkpoint_path = output_dir / "checkpoint.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        for name, expected in (("config", config), ("source_sha256", source_identity), ("data_sha256", data_identity)):
            validate_config(saved[name], expected)
        if saved["initial_state_sha256"] != initial_digest:
            raise ValueError("Initialization differs from resumed run")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        sampler.load_state_dict(saved["sampler"])
        restore_rng(saved["rng"])
        updates, history, best = saved["updates"], saved["history"], saved["best"]
        best_state = saved["best_model_state_dict"]
        if best_state is not None:
            export_state(best_state, output_dir / "best_validation.pt", config, best["updates"], data_identity, source_identity)
        digest, horizon_counts = saved["sample_digest"], saved["horizon_counts"]
        timings, work_counts = saved["timings_seconds"], saved["work_counts"]
    initial_path = output_dir / "initial.pt"
    if initial_path.exists():
        if state_sha256(torch.load(initial_path, weights_only=True)) != initial_digest:
            raise ValueError("Saved initial tensors do not match the requested run")
    else:
        atomic_torch_save(initial, initial_path)
    del initial
    archive_path = output_dir / "source.tar.gz"
    if not archive_path.exists():
        with tarfile.open(archive_path, "w:gz") as archive:
            for relative in SOURCE_PATHS:
                archive.add(Path(__file__).parents[2] / relative, arcname=relative)
    atomic_json_save(environment, output_dir / f"environment_{time.time_ns()}.json")
    kind = torch.device(device).type

    def synchronize():
        if kind == "cuda":
            torch.cuda.synchronize()

    def autocast():
        return torch.autocast(kind, dtype=torch.bfloat16, enabled=kind == "cuda")

    def batch(indices):
        inputs = F.one_hot(torch.as_tensor(digits[indices], device=device).long(), 10).float()
        answers = torch.as_tensor(targets[indices], device=device).long()
        return inputs, answers

    torch.set_float32_matmul_precision("high")
    forward = torch.compile(model) if kind == "cuda" else model
    advance = torch.compile(model.advance) if kind == "cuda" else model.advance
    with (output_dir / "train.log").open("a") as logger:
        def log(message):
            line = f"{datetime.now(timezone.utc).isoformat()} | {message}"
            print(line, flush=True)
            logger.write(line + "\n")
            logger.flush()

        log("CONFIG " + json.dumps(config, sort_keys=True))
        log("DATA " + json.dumps(data_identity, sort_keys=True))
        log(f"PARAMETERS {sum(p.numel() for p in model.parameters())}; base_sha256={base_digest}; "
            f"streams={model.streams}; starting_update={updates}; no_scan; ordinary_fp32_eval")
        started, saved_rng = time.perf_counter(), rng_state()
        inputs, answers = batch(np.arange(settings["batch_size"]) % len(digits))
        model.train()
        with autocast():
            loss, _, _ = forward(inputs, answers)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), autocast():
            state = advance(*model.initial_state(inputs))
        with autocast():
            loss, _, _ = forward(inputs, answers, initial_state=state)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
        synchronize()
        restore_rng(saved_rng)
        timings["compilation"] += time.perf_counter() - started
        del inputs, answers, loss, state
        log(f"WARMUP_DONE compilation_seconds={timings['compilation']:.3f}; starting_update={updates}")

        def checkpoint():
            started = time.perf_counter()
            atomic_torch_save({"config": config, "source_sha256": source_identity, "data_sha256": data_identity,
                "updates": updates, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "sampler": sampler.state_dict(), "rng": rng_state(), "history": history, "best": best,
                "best_model_state_dict": best_state,
                "sample_digest": digest, "horizon_counts": horizon_counts, "timings_seconds": timings,
                "work_counts": work_counts, "initial_state_sha256": initial_digest}, checkpoint_path)
            if checkpoint_callback:
                checkpoint_callback()
            timings["checkpoint"] += time.perf_counter() - started

        limit = settings["steps"] if stop_after is None else stop_after
        if not updates <= limit <= settings["steps"]:
            raise ValueError("Invalid stopping step")
        status = "paused"
        while updates < limit:
            synchronize()
            started = time.perf_counter()
            indices, horizon = sampler.sample(updates, settings["batch_size"])
            inputs, answers = batch(indices)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate(updates, settings)
            state = None
            if horizon:
                with torch.no_grad(), autocast():
                    state = model.initial_state(inputs)
                    for _ in range(horizon // spec["window_length"]):
                        state = advance(*state)
                    state = tuple(value.detach() for value in state)
            with autocast():
                loss, _, _ = forward(inputs, answers, initial_state=state)
            if not torch.isfinite(loss).item():
                status = "numerical_failure"
                log(f"NONFINITE_LOSS before update {updates + 1}")
                break
            loss.backward()
            if not torch.stack([parameter.grad.isfinite().all() for parameter in model.parameters()
                                if parameter.grad is not None]).all().item():
                status = "numerical_failure"
                log(f"NONFINITE_GRADIENT before update {updates + 1}")
                break
            optimizer.step()
            synchronize()
            timings["training"] += time.perf_counter() - started
            updates += 1
            digest = update_sample_digest(digest, indices, horizon)
            horizon_counts[str(horizon)] = horizon_counts.get(str(horizon), 0) + 1
            work_counts["gradient_free_iterations"] += horizon
            work_counts["supervised_iterations"] += spec["window_length"]
            if updates % 100 == 0 or smoke:
                log(f"UPDATE {updates} loss={loss.item():.6f} training_seconds={timings['training']:.3f}; "
                    f"sample_digest={digest}; horizons={json.dumps(horizon_counts, sort_keys=True)}")
            if updates % settings["probe_every"] == 0 or updates == limit:
                started = time.perf_counter()
                horizons = settings["probe_iterations"]
                if updates == settings["steps"] and not smoke:
                    horizons = spec["evaluation"]["iterations"]
                scores, _, diagnostics = evaluate_arrays(model, validation_digits, validation_targets, horizons,
                                                         batch_size=2 if smoke else 256)
                timings["monitoring"] += time.perf_counter() - started
                history.append({"updates": updates, "scores": scores, "state_diagnostics_first_batch": diagnostics,
                                "training_seconds": timings["training"]})
                metric = "16" if smoke else "1024"
                if scores[metric]["accuracy"] > best["accuracy"]:
                    best = {"accuracy": scores[metric]["accuracy"], "updates": updates, "metric": metric}
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                    export_state(best_state, output_dir / "best_validation.pt", config, updates, data_identity, source_identity)
                log(f"PROBE {updates} {json.dumps(scores, sort_keys=True)}")
                log(f"STATE {updates} {json.dumps(diagnostics, sort_keys=True)}")
                checkpoint()
        if status != "numerical_failure" and updates == settings["steps"]:
            status = "complete"
            export_model(model, output_dir / "final.pt", config, updates, data_identity, source_identity)
        result = {"config": config, "status": status, "updates": updates, "history": history, "best_validation": best,
                  "parameters": sum(p.numel() for p in model.parameters()), "initial_state_sha256": initial_digest,
                  "base_initial_state_sha256": base_digest, "sample_digest": digest, "horizon_counts": horizon_counts,
                  "work_counts": work_counts, "timings_seconds": timings, "processed_puzzles": updates * settings["batch_size"],
                  "data_sha256": data_identity, "source_sha256": source_identity,
                  "source_archive_sha256": file_sha256(archive_path), "finished_at": datetime.now(timezone.utc).isoformat()}
        if status != "paused":
            atomic_json_save(result, output_dir / "result.json")
            if checkpoint_callback:
                checkpoint_callback()
        log(f"RESULT status={status}; updates={updates}; sample_digest={digest}")
        return result
