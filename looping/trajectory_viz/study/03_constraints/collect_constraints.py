"""Collect recurrent states for the Sudoku-constraint geometry study."""

import hashlib
import os
import re
import time

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


SNAPSHOTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
SPLIT_NAMES = ("discovery", "validation", "final")


def assign_balanced_splits(bucket_names, examples_per_split_bucket):
    """Assign consecutive sampled examples within each bucket to three splits."""
    if examples_per_split_bucket <= 0:
        raise ValueError("examples_per_split_bucket must be positive")
    expected_per_bucket = examples_per_split_bucket * len(SPLIT_NAMES)
    seen = {}
    assignments = []
    for bucket_name in bucket_names:
        offset = seen.get(bucket_name, 0)
        split_index = offset // examples_per_split_bucket
        if split_index >= len(SPLIT_NAMES):
            raise ValueError(
                f"bucket {bucket_name!r} has more than {expected_per_bucket} examples"
            )
        assignments.append(SPLIT_NAMES[split_index])
        seen[bucket_name] = offset + 1
    if not seen or any(count != expected_per_bucket for count in seen.values()):
        raise ValueError(
            "each rating bucket must contain exactly "
            f"{expected_per_bucket} examples"
        )
    return assignments


def collect_snapshots(model, inputs, snapshots=SNAPSHOTS):
    """Return hidden states and output logits after each requested iteration."""
    if not snapshots or tuple(sorted(set(snapshots))) != tuple(snapshots):
        raise ValueError("snapshots must be strictly increasing")
    if snapshots[0] < 0:
        raise ValueError("snapshots must be non-negative")

    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    snapshot_set = set(snapshots)
    states = []
    logits = []

    with torch.no_grad():
        hidden_state = model.initial_encoder(inputs)
        predictions = torch.zeros(
            inputs.size(0), 81, 9, device=inputs.device
        )
        for iteration in range(snapshots[-1] + 1):
            current_logits = model.output_head(hidden_state)
            if iteration in snapshot_set:
                states.append(hidden_state.detach().float().cpu())
                logits.append(current_logits.detach().float().cpu())
            if iteration == snapshots[-1]:
                break
            hidden_state = model.recurrent_step(
                hidden_state,
                predictions,
                rope_cos,
                rope_sin,
            )
            predictions = F.softmax(model.output_head(hidden_state), dim=-1)

    return torch.stack(states, dim=1), torch.stack(logits, dim=1)


def collect(
    output_path,
    *,
    examples_per_split_bucket=4,
    seed=20260811,
    device="cuda",
    model_configs=DEFAULT_MODELS,
):
    """Collect all four checkpoints and atomically save one analysis payload."""
    if not re.fullmatch(r"[A-Za-z0-9_./-]+\.pt", output_path):
        raise ValueError(f"unsafe output path: {output_path!r}")
    if examples_per_split_bucket < 4:
        raise ValueError("the study protocol requires at least 20 puzzles per split")

    resolved_device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    examples_per_bucket = examples_per_split_bucket * len(SPLIT_NAMES)
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    split_names = assign_balanced_splits(
        bucket_names,
        examples_per_split_bucket,
    )
    inputs_device = inputs.to(resolved_device)

    output_directory = os.path.dirname(output_path)
    os.makedirs(output_directory, exist_ok=True)
    log_path = os.path.splitext(output_path)[0] + ".log"
    started_at = time.time()
    payload = {
        "format_version": 1,
        "config": {
            "snapshots": list(SNAPSHOTS),
            "examples_per_split_bucket": examples_per_split_bucket,
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(puzzles),
            "seed": seed,
            "device": str(resolved_device),
            "models": list(model_configs),
        },
        "sample": {
            "inputs": inputs,
            "targets": targets,
            "empty_mask": empty_mask,
            "bucket_names": bucket_names,
            "split_names": split_names,
            "puzzle_hashes": [
                hashlib.sha256(puzzle.encode("ascii")).hexdigest()[:16]
                for puzzle in puzzles
            ],
            "solution_hashes": [
                hashlib.sha256(solution.encode("ascii")).hexdigest()[:16]
                for solution in solutions
            ],
        },
        "models": {},
    }

    with open(log_path, "w") as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        split_counts = {
            split_name: split_names.count(split_name)
            for split_name in SPLIT_NAMES
        }
        log(
            f"Constraint study collection: {len(puzzles)} puzzles, "
            f"splits={split_counts}, snapshots={list(SNAPSHOTS)}"
        )
        for model_config in model_configs:
            model_name = model_config["name"]
            model_started_at = time.time()
            log(f"MODEL {model_name}: {model_config['path']}")
            model = _load_model(model_config, resolved_device)
            states, logits = collect_snapshots(model, inputs_device)
            payload["models"][model_name] = {
                "model_config": model_config,
                "states": states,
                "logits": logits,
                "elapsed_seconds": time.time() - model_started_at,
            }
            log(
                f"  states={tuple(states.shape)}, logits={tuple(logits.shape)}, "
                f"elapsed={time.time() - model_started_at:.1f}s"
            )
            del model, states, logits
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

        payload["elapsed_seconds"] = time.time() - started_at
        temporary_path = output_path + ".tmp"
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)
        log(f"Saved {output_path}")
        log(f"Total time: {payload['elapsed_seconds']:.1f}s")
    return payload


if __name__ == "__main__":
    collect("/tmp/constraints_states.pt", device="cpu")
