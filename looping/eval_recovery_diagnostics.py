"""Measure whether recurrent Sudoku states recover over the next 16 iterations."""

import json
import os
import re
import time

import torch
import torch.nn.functional as F

from looping.eval_loop_diagnostics import (
    DEFAULT_MODELS,
    LATE_STATE_MODELS,
    _load_balanced_sample,
    _load_model,
)
import stabilize.exp_testbed_20k as model_module


DEFAULT_HORIZONS = (16, 32, 64, 128, 256, 512, 1024)
RECOVERY_WINDOW = 16
QUALITY_BUCKETS = (
    (0, 0, "solved"),
    (1, 2, "near_miss"),
    (3, 10, "semi_bad"),
    (11, 81, "bad"),
)

LATE_SWITCH_MODELS = (
    {
        "name": "late_switch_consistency_best",
        "path": (
            "/outputs/looping/"
            "model_loop_stay_late_switch_consistency_from39k_best_probe.pt"
        ),
        "model_kwargs": {},
    },
)

REFERENCE_RECOVERY_MODELS = (
    *DEFAULT_MODELS,
    LATE_STATE_MODELS[0],
    *LATE_SWITCH_MODELS,
)

EXTREME_TIMING_MODELS = (
    {
        "name": "late_aux_every_batch_from_start",
        "path": (
            "/outputs/looping/"
            "model_loop_late_random_aux_p100_trial0_best_probe.pt"
        ),
        "model_kwargs": {},
    },
    {
        "name": "late_aux_every_batch_after_10k",
        "path": (
            "/outputs/looping/"
            "model_loop_late_random_aux_p100_after_10k_trial0_best_probe.pt"
        ),
        "model_kwargs": {},
    },
)

DELAYED_SWITCH_TRAJECTORY_MODELS = tuple(
    {
        "name": f"late_aux_after_10k_step_{step}",
        "path": (
            "/outputs/looping/"
            "loop_late_random_aux_p100_after_10k_trial0_"
            f"checkpoint_step{step}.pt"
        ),
        "model_kwargs": {},
        "trusted_full_checkpoint": True,
    }
    for step in (9000, 10000, 11000, 12000, 13000)
)

MODEL_PRESETS = {
    "reference_models": REFERENCE_RECOVERY_MODELS,
    "extreme_timing_models": EXTREME_TIMING_MODELS,
    "delayed_switch_trajectory": DELAYED_SWITCH_TRAJECTORY_MODELS,
}


def classify_wrong_counts(wrong_counts):
    labels = torch.empty_like(wrong_counts, dtype=torch.long)
    for bucket_index, (minimum, maximum, _) in enumerate(QUALITY_BUCKETS):
        in_bucket = (wrong_counts >= minimum) & (wrong_counts <= maximum)
        labels[in_bucket] = bucket_index
    return labels


def _masked_per_puzzle_mean(values, empty_mask):
    denominator = empty_mask.sum(dim=1).clamp_min(1)
    return (values * empty_mask).sum(dim=1) / denominator


def _minimum_empty_value(values, empty_mask):
    return values.masked_fill(~empty_mask, torch.inf).min(dim=1).values


def _snapshot(model, hidden_state, logits, targets, empty_mask):
    logits = logits.float()
    predictions = logits.argmax(dim=-1)
    wrong_counts = ((predictions != targets) & empty_mask).sum(dim=1)

    log_probabilities = F.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    target_log_probabilities = log_probabilities.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    other_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=9).bool(),
        -torch.inf,
    )
    margins = target_logits - other_logits.max(dim=-1).values
    entropy = -(probabilities * log_probabilities).sum(dim=-1)

    normalized_hidden = F.normalize(hidden_state.float(), dim=-1)
    directional_logits = F.linear(
        normalized_hidden,
        model.output_head.weight.float(),
        bias=None,
    )
    directional_predictions = directional_logits.argmax(dim=-1)
    directional_wrong_counts = (
        (directional_predictions != targets) & empty_mask
    ).sum(dim=1)
    directional_target_logits = directional_logits.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    directional_other_logits = directional_logits.masked_fill(
        F.one_hot(targets, num_classes=9).bool(),
        -torch.inf,
    )
    directional_margins = (
        directional_target_logits
        - directional_other_logits.max(dim=-1).values
    )

    return {
        "wrong_counts": wrong_counts.cpu(),
        "predictions": predictions.to(torch.uint8).cpu(),
        "cross_entropy": _masked_per_puzzle_mean(
            -target_log_probabilities,
            empty_mask,
        ).cpu(),
        "correct_probability": _masked_per_puzzle_mean(
            probabilities.gather(
                -1,
                targets.unsqueeze(-1),
            ).squeeze(-1),
            empty_mask,
        ).cpu(),
        "mean_target_margin": _masked_per_puzzle_mean(
            margins,
            empty_mask,
        ).cpu(),
        "minimum_target_margin": _minimum_empty_value(
            margins,
            empty_mask,
        ).cpu(),
        "entropy": _masked_per_puzzle_mean(
            entropy,
            empty_mask,
        ).cpu(),
        "state_rms": hidden_state.float().flatten(1).square().mean(
            dim=1
        ).sqrt().cpu(),
        "directional_wrong_counts": directional_wrong_counts.cpu(),
        "directional_minimum_target_margin": _minimum_empty_value(
            directional_margins,
            empty_mask,
        ).cpu(),
    }


def _distribution(values):
    values = values.float()
    if values.numel() == 0:
        return None
    return {
        "mean": values.mean().item(),
        "p10": torch.quantile(values, 0.1).item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, 0.9).item(),
    }


def summarize_transition(
    start_snapshot,
    current_snapshot,
    empty_mask,
    selection,
    direction_cosine,
):
    count = int(selection.sum().item())
    if count == 0:
        return {"count": 0}

    start_wrong = start_snapshot["wrong_counts"][selection]
    current_wrong = current_snapshot["wrong_counts"][selection]
    start_solved = start_wrong == 0
    start_unsolved = ~start_solved
    current_solved = current_wrong == 0
    selected_empty = empty_mask[selection]
    prediction_changes = (
        start_snapshot["predictions"][selection]
        != current_snapshot["predictions"][selection]
    ) & selected_empty

    solved_retention = None
    if start_solved.any():
        solved_retention = (
            current_solved[start_solved].float().mean().item()
        )
    unsolved_recovery = None
    if start_unsolved.any():
        unsolved_recovery = (
            current_solved[start_unsolved].float().mean().item()
        )

    empty_cells = selected_empty.sum().clamp_min(1)
    directional_wrong = current_snapshot[
        "directional_wrong_counts"
    ][selection]
    return {
        "count": count,
        "puzzle_accuracy": current_solved.float().mean().item(),
        "blank_cell_accuracy": (
            1 - current_wrong.sum().float() / empty_cells
        ).item(),
        "mean_wrong_cells": current_wrong.float().mean().item(),
        "improved_fraction": (
            current_wrong < start_wrong
        ).float().mean().item(),
        "unchanged_fraction": (
            current_wrong == start_wrong
        ).float().mean().item(),
        "worsened_fraction": (
            current_wrong > start_wrong
        ).float().mean().item(),
        "solved_retention": solved_retention,
        "unsolved_recovery": unsolved_recovery,
        "prediction_change_fraction": (
            prediction_changes.sum().float() / empty_cells
        ).item(),
        "cross_entropy": current_snapshot["cross_entropy"][
            selection
        ].mean().item(),
        "correct_probability": current_snapshot["correct_probability"][
            selection
        ].mean().item(),
        "mean_target_margin": current_snapshot["mean_target_margin"][
            selection
        ].mean().item(),
        "minimum_target_margin": _distribution(
            current_snapshot["minimum_target_margin"][selection]
        ),
        "entropy": current_snapshot["entropy"][selection].mean().item(),
        "state_rms": _distribution(
            current_snapshot["state_rms"][selection]
        ),
        "state_direction_cosine_from_start": _distribution(
            direction_cosine[selection]
        ),
        "state_direction_cosine_to_reference": _distribution(
            current_snapshot[
                "state_direction_cosine_to_reference"
            ][selection]
        ),
        "directional_puzzle_accuracy": (
            directional_wrong == 0
        ).float().mean().item(),
        "directional_minimum_target_margin": _distribution(
            current_snapshot[
                "directional_minimum_target_margin"
            ][selection]
        ),
    }


def build_recovery_curves(
    snapshots,
    direction_cosines,
    empty_mask,
    horizons,
):
    curves = {}
    all_examples = torch.ones(empty_mask.size(0), dtype=torch.bool)
    for horizon in horizons:
        start_snapshot = snapshots[horizon]
        quality_labels = classify_wrong_counts(
            start_snapshot["wrong_counts"]
        )
        curve = []
        for offset in range(RECOVERY_WINDOW + 1):
            current_snapshot = snapshots[horizon + offset]
            cosine = direction_cosines[horizon][offset]
            by_quality = {}
            for bucket_index, (_, _, bucket_name) in enumerate(
                QUALITY_BUCKETS
            ):
                by_quality[bucket_name] = summarize_transition(
                    start_snapshot,
                    current_snapshot,
                    empty_mask,
                    quality_labels == bucket_index,
                    cosine,
                )
            curve.append({
                "offset": offset,
                "overall": summarize_transition(
                    start_snapshot,
                    current_snapshot,
                    empty_mask,
                    all_examples,
                    cosine,
                ),
                "by_start_quality": by_quality,
            })
        curves[str(horizon)] = {
            "start_quality_counts": {
                bucket_name: int(
                    (quality_labels == bucket_index).sum().item()
                )
                for bucket_index, (_, _, bucket_name) in enumerate(
                    QUALITY_BUCKETS
                )
            },
            "curve": curve,
        }
    return curves


def capture_trajectory(
    model,
    inputs,
    targets,
    empty_mask,
    horizons,
    batch_size,
    device,
):
    observation_steps = {
        horizon + offset
        for horizon in horizons
        for offset in range(RECOVERY_WINDOW + 1)
    }
    snapshot_chunks = {
        step: {}
        for step in observation_steps
    }
    cosine_chunks = {
        horizon: {
            offset: []
            for offset in range(RECOVERY_WINDOW + 1)
        }
        for horizon in horizons
    }
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    maximum_iteration = max(observation_steps)
    reference_horizon = horizons[0]

    autocast = torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )
    with torch.inference_mode(), autocast:
        for batch_start in range(0, inputs.size(0), batch_size):
            batch_end = min(batch_start + batch_size, inputs.size(0))
            batch_inputs = inputs[batch_start:batch_end].to(device)
            batch_targets = targets[batch_start:batch_end].to(device)
            batch_empty = empty_mask[batch_start:batch_end].to(device)
            hidden_state = model.initial_encoder(batch_inputs)
            predictions = torch.zeros(
                batch_inputs.size(0),
                81,
                9,
                device=device,
            )
            start_states = {}
            reference_state = None

            for iteration in range(1, maximum_iteration + 1):
                hidden_state = model.recurrent_step(
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                )
                logits = model.output_head(hidden_state)
                predictions = F.softmax(logits, dim=-1)

                if iteration == reference_horizon:
                    reference_state = hidden_state.clone()
                if iteration in horizons:
                    start_states[iteration] = hidden_state.clone()

                if iteration in observation_steps:
                    snapshot = _snapshot(
                        model,
                        hidden_state,
                        logits,
                        batch_targets,
                        batch_empty,
                    )
                    snapshot[
                        "state_direction_cosine_to_reference"
                    ] = F.cosine_similarity(
                        reference_state.float().flatten(1),
                        hidden_state.float().flatten(1),
                        dim=1,
                        eps=1e-12,
                    ).clamp(-1, 1).cpu()
                    for key, values in snapshot.items():
                        snapshot_chunks[iteration].setdefault(
                            key,
                            [],
                        ).append(values)

                for horizon, start_state in tuple(start_states.items()):
                    offset = iteration - horizon
                    if 0 <= offset <= RECOVERY_WINDOW:
                        cosine = F.cosine_similarity(
                            start_state.float().flatten(1),
                            hidden_state.float().flatten(1),
                            dim=1,
                            eps=1e-12,
                        ).clamp(-1, 1)
                        cosine_chunks[horizon][offset].append(
                            cosine.cpu()
                        )
                    if offset == RECOVERY_WINDOW:
                        del start_states[horizon]

    snapshots = {
        step: {
            key: torch.cat(chunks)
            for key, chunks in fields.items()
        }
        for step, fields in snapshot_chunks.items()
    }
    direction_cosines = {
        horizon: {
            offset: torch.cat(chunks)
            for offset, chunks in offsets.items()
        }
        for horizon, offsets in cosine_chunks.items()
    }
    return snapshots, direction_cosines


def evaluate(
    model_configs=REFERENCE_RECOVERY_MODELS,
    horizons=DEFAULT_HORIZONS,
    examples_per_bucket=200,
    batch_size=100,
    seed=42,
    device="cuda",
    output_dir=None,
    output_prefix="recovery_diagnostics",
):
    horizons = tuple(horizons)
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be strictly increasing")
    if horizons[0] <= 0:
        raise ValueError("horizons must be positive")
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = open(
            os.path.join(output_dir, f"{output_prefix}.log"),
            "w",
        )
    else:
        log_file = None

    def log(message=""):
        print(message, flush=True)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    result = {
        "config": {
            "models": list(model_configs),
            "horizons": list(horizons),
            "direction_reference_horizon": horizons[0],
            "recovery_window": RECOVERY_WINDOW,
            "quality_buckets": [
                {
                    "name": name,
                    "minimum_wrong_cells": minimum,
                    "maximum_wrong_cells": maximum,
                }
                for minimum, maximum, name in QUALITY_BUCKETS
            ],
            "examples_per_bucket": examples_per_bucket,
            "sample_size": inputs.size(0),
            "batch_size": batch_size,
            "seed": seed,
            "device": str(device),
            "inference_policy": "native undamped recurrence",
        },
        "sample": {
            "puzzles": puzzles,
            "solutions": solutions,
            "rating_buckets": bucket_names,
        },
        "models": {},
    }
    started_at = time.time()
    log(
        f"Recovery diagnostics: {inputs.size(0)} puzzles, "
        f"models={len(model_configs)}, horizons={list(horizons)}"
    )

    for model_config in model_configs:
        model_name = model_config["name"]
        model_started_at = time.time()
        log(f"\nMODEL {model_name}: {model_config['path']}")
        model = _load_model(model_config, device)
        snapshots, direction_cosines = capture_trajectory(
            model,
            inputs,
            targets,
            empty_mask,
            horizons,
            batch_size,
            device,
        )
        recovery = build_recovery_curves(
            snapshots,
            direction_cosines,
            empty_mask,
            horizons,
        )
        result["models"][model_name] = {
            "model_config": model_config,
            "recovery": recovery,
            "elapsed_seconds": time.time() - model_started_at,
        }
        for horizon in horizons:
            start = recovery[str(horizon)]["curve"][0]["overall"]
            end = recovery[str(horizon)]["curve"][-1]["overall"]
            semi_bad = recovery[str(horizon)]["curve"][-1][
                "by_start_quality"
            ]["semi_bad"]
            log(
                f"  h={horizon:4d}: solved "
                f"{100 * start['puzzle_accuracy']:.1f}% -> "
                f"{100 * end['puzzle_accuracy']:.1f}%, "
                f"retained={end['solved_retention']}, "
                f"semi-bad improved={semi_bad.get('improved_fraction')}"
            )
        log(
            f"  done in "
            f"{result['models'][model_name]['elapsed_seconds']:.1f}s"
        )
        del model, snapshots, direction_cosines
        if device.type == "cuda":
            torch.cuda.empty_cache()

    result["elapsed_seconds"] = time.time() - started_at
    if output_dir:
        result_path = os.path.join(
            output_dir,
            f"{output_prefix}.json",
        )
        temporary_path = result_path + ".tmp"
        with open(temporary_path, "w") as result_file:
            json.dump(result, result_file, indent=2)
            result_file.write("\n")
        os.replace(temporary_path, result_path)
        log(f"\nStructured results: {result_path}")
    log(f"Total time: {result['elapsed_seconds']:.1f}s")
    if log_file:
        log_file.close()
    return result


if __name__ == "__main__":
    evaluate()
