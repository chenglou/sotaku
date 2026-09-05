"""Paired 20K training runs, independent samplers, and complete resumption."""

import copy
import json
import math
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, atomic_torch_save, validate_config
from looping.weight_tying.common import SOURCE_PATHS, protocol, run_config, update_sample_digest
from looping.weight_tying.data import load_manifest
from looping.weight_tying.evaluate import evaluate_arrays
from looping.weight_tying.model import StudyTransformer, forward_flops_per_iteration, parameter_count
from runtime_utils import file_sha256, runtime_manifest


def learning_rate(step, settings):
    warmup = settings["warmup_steps"]
    if step < warmup:
        return settings["learning_rate"] * (step + 1) / warmup
    progress = (step - warmup) / (settings["steps"] - warmup)
    ratio = settings["minimum_lr_ratio"]
    return settings["learning_rate"] * (ratio + (1 - ratio) * (1 + math.cos(math.pi * progress)) / 2)


class PairedSampler:
    def __init__(self, ratings, seed, regime, settings):
        self.puzzle_rng = np.random.default_rng(seed)
        self.horizon_rng = np.random.default_rng(seed + 1_000_003)
        self.regime = regime
        self.settings = settings
        self.pools = {minimum: np.flatnonzero(ratings >= minimum)
                      for _, _, minimum in settings["phases"]}

    def sample(self, step, batch_size):
        minimum = next(minimum for begin, end, minimum in self.settings["phases"] if begin <= step < end)
        pool = self.pools[minimum]
        if len(pool) == 0:
            raise ValueError(f"Training pool is empty at minimum rating {minimum}")
        indices = pool[self.puzzle_rng.integers(0, len(pool), size=batch_size)]
        horizon = 0
        if self.horizon_rng.random() < self.regime["burnin_probability"]:
            horizon = int(self.horizon_rng.choice(self.regime["burnin_iterations"]))
        return indices, horizon

    def state_dict(self):
        return {"puzzles": copy.deepcopy(self.puzzle_rng.bit_generator.state),
                "horizons": copy.deepcopy(self.horizon_rng.bit_generator.state)}

    def load_state_dict(self, state):
        self.puzzle_rng.bit_generator.state = state["puzzles"]
        self.horizon_rng.bit_generator.state = state["horizons"]


def rng_state():
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def export_model(model, path, config, updates, data_sha256):
    atomic_torch_save({name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}, path)
    atomic_json_save({"artifact_type": "weight-tying-study-inference", "schema_version": 1,
                      "config": config, "updates": updates, "weights_sha256": file_sha256(path),
                      "data_manifest_sha256": data_sha256}, str(path) + ".json")


def train_run(data_dir, output_dir, architecture, regime, seed, *, smoke=False,
              stop_after=None, device="cuda", checkpoint_callback=None):
    prepared_at = time.perf_counter()
    config = run_config(architecture, regime, seed, smoke=smoke)
    settings = config["training"]
    output_dir, data_dir = Path(output_dir), Path(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = (output_dir / "train.log").open("a")

    def log(message):
        text = f"{datetime.now(timezone.utc).isoformat()} | {message}"
        print(text, flush=True)
        logger.write(text + "\n")
        logger.flush()

    data_manifest = load_manifest(data_dir, verify=("train.npz", "validation.npz"))
    data_identity = file_sha256(data_dir / "manifest.json")
    if (output_dir / "result.json").exists():
        result = json.loads((output_dir / "result.json").read_text())
        validate_config(result["config"], config)
        if result["data_manifest_sha256"] != data_identity:
            raise ValueError("Completed run uses different data")
        logger.close()
        return result
    arrays = np.load(data_dir / "train.npz", allow_pickle=False)
    digits, targets, ratings = arrays["digits"], arrays["targets"], arrays["ratings"]
    arrays.close()
    validation = np.load(data_dir / "validation.npz", allow_pickle=False)
    if not smoke and len(digits) != settings["train_rows"]:
        raise ValueError("Unexpected number of training examples")
    torch.manual_seed(seed)
    model = StudyTransformer(**config["model"]).to(device)
    torch.manual_seed(seed + 10_000_019)
    sampler_settings = copy.deepcopy(settings)
    if smoke:
        sampler_settings["phases"] = [[0, 20000, 0]]
    sampler = PairedSampler(ratings, seed, config["regime_settings"], sampler_settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                 betas=tuple(settings["adam_betas"]), weight_decay=settings["weight_decay"])
    training_runtime = runtime_manifest(SOURCE_PATHS)
    source_identity = training_runtime["source_sha256"]
    atomic_json_save(training_runtime, output_dir / f"environment_{time.time_ns()}.json")
    source_archive = output_dir / "source.tar.gz"
    if not source_archive.exists():
        with tarfile.open(source_archive, "w:gz") as archive:
            for relative in SOURCE_PATHS:
                archive.add(Path(__file__).parents[2] / relative, arcname=relative)
    checkpoint_path = output_dir / "checkpoint.pt"
    updates, history, best = 0, [], {"accuracy": -1.0, "updates": 0}
    sample_digest = "0" * 64
    horizon_counts = {}
    timings = {name: 0.0 for name in ("preparation", "compilation", "training", "optimizer", "evaluation", "checkpoint")}
    cumulative_flops = 0
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        validate_config(saved["config"], config)
        validate_config(saved["source_sha256"], source_identity)
        if saved["data_manifest_sha256"] != data_identity:
            raise ValueError("Resume data identity differs")
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        sampler.load_state_dict(saved["sampler"])
        restore_rng(saved["rng"])
        updates, history, best = saved["updates"], saved["history"], saved["best"]
        sample_digest, horizon_counts = saved["sample_digest"], saved["horizon_counts"]
        timings, cumulative_flops = saved["timings"], saved["estimated_model_flops"]
        log(f"Resuming after {updates} optimizer updates")
    parameter_total = parameter_count(model)
    log("CONFIG " + json.dumps(config, sort_keys=True))
    log(f"PARAMETERS {parameter_total}; data manifest {data_identity}; device {device}")
    batch_size = 4 if smoke else settings["batch_size"]
    limit = 4 if smoke else settings["steps"]
    if stop_after is not None:
        if not updates <= stop_after <= limit:
            raise ValueError("Requested stop is outside the remaining schedule")
        limit = stop_after
    device_type = torch.device(device).type

    def synchronize():
        if device_type == "cuda":
            torch.cuda.synchronize()

    def batch(indices):
        inputs = F.one_hot(torch.as_tensor(digits[indices], device=device).long(), 10).float()
        answers = torch.as_tensor(targets[indices], device=device).long()
        return inputs, answers

    torch.set_float32_matmul_precision("high")
    compiled = torch.compile(model) if device_type == "cuda" else model
    advance = torch.compile(model.advance) if device_type == "cuda" else model.advance
    timings["preparation"] += time.perf_counter() - prepared_at
    # Compile without changing parameters, sample streams, or dropout RNG state.
    warmup_at, saved_rng = time.perf_counter(), rng_state()
    inputs, answers = batch(np.arange(batch_size) % len(digits))
    model.train()
    with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
        loss, _ = compiled(inputs, answers)
    loss.backward()
    optimizer.zero_grad(set_to_none=True)
    if config["regime_settings"]["burnin_probability"]:
        with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            initial = model.initial_state(inputs)
            initial = advance(*initial)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            loss, _ = compiled(inputs, answers, initial_state=initial)
        loss.backward()
        optimizer.zero_grad(set_to_none=True)
    synchronize()
    restore_rng(saved_rng)
    timings["compilation"] += time.perf_counter() - warmup_at
    log(f"Compilation/warm-up completed; starting at update {updates}")
    del inputs, answers, loss

    def checkpoint():
        started = time.perf_counter()
        synchronize()
        state = {"config": config, "source_sha256": source_identity,
                 "data_manifest_sha256": data_identity, "updates": updates,
                 "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                 "sampler": sampler.state_dict(), "rng": rng_state(), "history": history, "best": best,
                 "sample_digest": sample_digest, "horizon_counts": horizon_counts,
                 "timings": timings, "estimated_model_flops": cumulative_flops}
        atomic_torch_save(state, checkpoint_path)
        if checkpoint_callback:
            checkpoint_callback()
        timings["checkpoint"] += time.perf_counter() - started

    status = "paused"
    while updates < limit:
        synchronize()
        started = time.perf_counter()
        indices, horizon = sampler.sample(updates, batch_size)
        inputs, answers = batch(indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(updates, settings)
        initial = None
        if horizon:
            with torch.no_grad(), torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
                initial = model.initial_state(inputs)
                for _ in range(horizon // 16):
                    initial = advance(*initial)
        with torch.autocast(device_type, dtype=torch.bfloat16, enabled=device_type == "cuda"):
            loss, _ = compiled(inputs, answers, initial_state=initial)
        if not torch.isfinite(loss).item():
            status = "numerical_failure"
            log(f"Nonfinite training loss before update {updates + 1}")
            break
        loss.backward()
        synchronize()
        optimizer_started = time.perf_counter()
        optimizer.step()
        synchronize()
        timings["optimizer"] += time.perf_counter() - optimizer_started
        timings["training"] += time.perf_counter() - started
        sample_digest = update_sample_digest(sample_digest, indices, horizon)
        horizon_counts[str(horizon)] = horizon_counts.get(str(horizon), 0) + 1
        cumulative_flops += (horizon + 3 * 16) * forward_flops_per_iteration(
            model.width, model.feedforward_width, batch_size)
        cumulative_flops += 2 * batch_size * 81 * 10 * model.width * (1 if horizon else 3)
        updates += 1
        if updates % 100 == 0 or smoke:
            log(f"UPDATE {updates} loss={loss.item():.6f} training_seconds={timings['training']:.3f} "
                f"samples_sha256={sample_digest}")
        if updates % settings["probe_every"] == 0 or updates == limit:
            started = time.perf_counter()
            iterations = [16] if smoke else settings["probe_iterations"]
            if not smoke and updates == settings["steps"]:
                iterations = sorted(set(iterations + [4096]))
            scores, _ = evaluate_arrays(model, validation["digits"], validation["targets"], iterations,
                                       batch_size=4 if smoke else 256, repeat=True)
            timings["evaluation"] += time.perf_counter() - started
            history.append({"updates": updates, "scores": scores, "training_seconds": timings["training"]})
            metric = "16" if regime == "early" or smoke else "1024"
            if scores[metric]["accuracy"] > best["accuracy"]:
                best = {"accuracy": scores[metric]["accuracy"], "updates": updates, "metric": metric}
                export_model(model, output_dir / "best_validation.pt", config, updates, data_identity)
            log(f"VALIDATION {updates} {json.dumps(scores, sort_keys=True)}")
            checkpoint()
    completed_steps = 4 if smoke else settings["steps"]
    if status != "numerical_failure" and updates == completed_steps:
        status = "complete"
        export_model(model, output_dir / "final.pt", config, updates, data_identity)
    result = {"config": config, "status": status, "updates": updates, "parameters": parameter_total,
              "data_manifest_sha256": data_identity, "source_sha256": source_identity,
              "source_archive_sha256": file_sha256(source_archive),
              "history": history, "best_validation": best, "timings_seconds": timings,
              "estimated_model_flops": cumulative_flops, "sample_digest": sample_digest,
              "horizon_counts": horizon_counts, "processed_puzzles": updates * batch_size,
              "finished_at": datetime.now(timezone.utc).isoformat()}
    if status != "paused":
        atomic_json_save(result, output_dir / "result.json")
        if checkpoint_callback:
            checkpoint_callback()
    log(f"RESULT status={status} updates={updates} sample_digest={sample_digest} timings={json.dumps(timings)}")
    validation.close()
    logger.close()
    return result
