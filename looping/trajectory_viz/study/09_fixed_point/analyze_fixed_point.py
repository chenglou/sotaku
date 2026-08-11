"""Held-out diagnostics for fixed-point and settling behavior."""

import json
import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


SEED = 20260811
ITERATIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 640, 768, 896, 1024)
CONTRACTION_ITERATIONS = (16, 64, 128, 256, 512, 768, 1024)
SPLIT_ORDER = ("discovery", "validation", "final_holdout")


def stratified_split(bucket_names, per_bucket_per_split=4):
    """Return deterministic, balanced indices for three disjoint splits."""
    grouped = {}
    for index, bucket in enumerate(bucket_names):
        grouped.setdefault(bucket, []).append(index)
    required = per_bucket_per_split * len(SPLIT_ORDER)
    if any(len(indices) < required for indices in grouped.values()):
        raise ValueError(f"each bucket needs at least {required} examples")
    splits = {name: [] for name in SPLIT_ORDER}
    for indices in grouped.values():
        for split_number, name in enumerate(SPLIT_ORDER):
            start = split_number * per_bucket_per_split
            splits[name].extend(indices[start : start + per_bucket_per_split])
    return splits


def sequence_metrics(updates):
    """Summarize direction persistence and acceleration for [N,T,D] updates."""
    if updates.ndim != 3 or updates.size(1) < 2:
        raise ValueError("updates must have shape [puzzles, steps, features]")
    current = updates[:, :-1].float()
    following = updates[:, 1:].float()
    cosine = F.cosine_similarity(current, following, dim=-1, eps=1e-12)
    acceleration = (following - current).norm(dim=-1) / current.norm(
        dim=-1
    ).clamp_min(1e-12)
    return {
        "cosine_mean": cosine.mean().item(),
        "cosine_median": cosine.median().item(),
        "relative_acceleration_mean": acceleration.mean().item(),
        "relative_acceleration_median": acceleration.median().item(),
    }


def shuffled_time_control(updates, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(updates.size(1), generator=generator)
    return sequence_metrics(updates[:, permutation])


def normalized_distance(left, right):
    left = F.normalize(left.float(), dim=-1, eps=1e-12)
    right = F.normalize(right.float(), dim=-1, eps=1e-12)
    return (left - right).norm(dim=-1)


def endpoint_distances(states, endpoint_index=-1):
    """Measure actual and puzzle-shuffled distances to a terminal state."""
    if states.ndim != 3:
        raise ValueError("states must have shape [puzzles, steps, features]")
    endpoint = states[:, endpoint_index]
    shuffled_endpoint = endpoint.roll(1, dims=0)
    actual = normalized_distance(states, endpoint[:, None])
    shuffled = normalized_distance(states, shuffled_endpoint[:, None])
    return actual, shuffled


def _prediction_metrics(model, state, targets, empty_mask):
    predictions = model.output_head(state).argmax(dim=-1)
    correct = (predictions == targets) | ~empty_mask
    return {
        "cell_accuracy": correct.float().mean(dim=1).cpu(),
        "solved": correct.all(dim=1).float().cpu(),
    }


def _full_step(model, state, rope_cos, rope_sin):
    predictions = F.softmax(model.output_head(state), dim=-1)
    return model.recurrent_step(state, predictions, rope_cos, rope_sin)


def random_direction_gains(
    model,
    states,
    rope_cos,
    rope_sin,
    *,
    directions=3,
    relative_epsilon=1e-3,
    seed=SEED,
):
    """Estimate local recurrence gain in random hidden-state directions."""
    base_next = _full_step(model, states, rope_cos, rope_sin)
    gains = []
    generator = torch.Generator(device=states.device).manual_seed(seed)
    flattened = states.flatten(1)
    scale = flattened.norm(dim=1, keepdim=True) * relative_epsilon
    for _ in range(directions):
        direction = torch.randn(
            flattened.shape,
            generator=generator,
            device=states.device,
            dtype=states.dtype,
        )
        direction = F.normalize(direction, dim=1, eps=1e-12)
        perturbation = (direction * scale).view_as(states)
        perturbed_next = _full_step(
            model,
            states + perturbation,
            rope_cos,
            rope_sin,
        )
        output_delta = (perturbed_next - base_next).flatten(1).norm(dim=1)
        input_delta = perturbation.flatten(1).norm(dim=1).clamp_min(1e-12)
        gains.append((output_delta / input_delta).float().cpu())
    return torch.stack(gains, dim=1)


def specified_direction_gain(
    model,
    states,
    directions,
    rope_cos,
    rope_sin,
    *,
    relative_epsilon=1e-3,
):
    """Estimate local recurrence gain along one specified direction per puzzle."""
    flat_states = states.flatten(1)
    flat_directions = F.normalize(directions.flatten(1), dim=1, eps=1e-12)
    scale = flat_states.norm(dim=1, keepdim=True) * relative_epsilon
    perturbation = (flat_directions * scale).view_as(states)
    base_next = _full_step(model, states, rope_cos, rope_sin)
    perturbed_next = _full_step(
        model,
        states + perturbation,
        rope_cos,
        rope_sin,
    )
    return (
        (perturbed_next - base_next).flatten(1).norm(dim=1)
        / perturbation.flatten(1).norm(dim=1).clamp_min(1e-12)
    ).float().cpu()


def collect_model(model, inputs, targets, empty_mask):
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    selected = set(ITERATIONS)
    states = {}
    updates = {}
    predictions_at = {}
    dense_norms = []
    dense_cosines = []
    dense_accelerations = []
    previous_update = None
    with torch.no_grad():
        state = model.initial_encoder(inputs)
        predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
        for iteration in range(max(ITERATIONS) + 1):
            if iteration in selected:
                states[iteration] = state.float().cpu()
                predictions_at[iteration] = _prediction_metrics(
                    model, state, targets, empty_mask
                )
            next_state = model.recurrent_step(state, predictions, rope_cos, rope_sin)
            update = next_state - state
            dense_norms.append(update.flatten(1).norm(dim=1).float().cpu())
            if previous_update is not None:
                flat_previous = previous_update.flatten(1)
                flat_update = update.flatten(1)
                dense_cosines.append(
                    F.cosine_similarity(flat_previous, flat_update, dim=1).float().cpu()
                )
                dense_accelerations.append(
                    ((flat_update - flat_previous).norm(dim=1)
                     / flat_previous.norm(dim=1).clamp_min(1e-12)).float().cpu()
                )
            if iteration in selected:
                updates[iteration] = update.float().cpu()
            previous_update = update
            state = next_state
            predictions = F.softmax(model.output_head(state), dim=-1)

    contraction = {}
    update_direction_gain = {}
    with torch.no_grad():
        for iteration in CONTRACTION_ITERATIONS:
            iteration_states = states[iteration].to(inputs.device)
            contraction[iteration] = random_direction_gains(
                model,
                iteration_states,
                rope_cos,
                rope_sin,
                seed=SEED + iteration,
            )
            update_direction_gain[iteration] = specified_direction_gain(
                model,
                iteration_states,
                updates[iteration].to(inputs.device),
                rope_cos,
                rope_sin,
            )
    return {
        "states": torch.stack([states[i].flatten(1) for i in ITERATIONS], dim=1),
        "updates": torch.stack([updates[i].flatten(1) for i in ITERATIONS], dim=1),
        "cell_accuracy": torch.stack(
            [predictions_at[i]["cell_accuracy"] for i in ITERATIONS], dim=1
        ),
        "solved": torch.stack(
            [predictions_at[i]["solved"] for i in ITERATIONS], dim=1
        ),
        "dense_update_norm": torch.stack(dense_norms, dim=1),
        "dense_update_cosine": torch.stack(dense_cosines, dim=1),
        "dense_relative_acceleration": torch.stack(dense_accelerations, dim=1),
        "contraction_gain": torch.stack(
            [contraction[i] for i in CONTRACTION_ITERATIONS], dim=1
        ),
        "update_direction_gain": torch.stack(
            [update_direction_gain[i] for i in CONTRACTION_ITERATIONS], dim=1
        ),
    }


def _quantiles(values):
    values = values.float().flatten()
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "p10": torch.quantile(values, 0.1).item(),
        "p90": torch.quantile(values, 0.9).item(),
    }


def summarize_split(collected, indices):
    indices = torch.tensor(indices, dtype=torch.long)
    states = collected["states"][indices]
    updates = collected["updates"][indices]
    actual_endpoint, shuffled_endpoint = endpoint_distances(states)
    chronological = sequence_metrics(updates)
    shuffled = shuffled_time_control(updates, SEED)
    lag128 = []
    lag128_shuffled = []
    iteration_to_index = {iteration: index for index, iteration in enumerate(ITERATIONS)}
    for iteration in (256, 384, 512, 640, 768, 896, 1024):
        current = states[:, iteration_to_index[iteration]]
        previous = states[:, iteration_to_index[iteration - 128]]
        lag128.append(normalized_distance(current, previous))
        lag128_shuffled.append(normalized_distance(current, previous.roll(1, dims=0)))
    dense_norm = collected["dense_update_norm"][indices]
    tail_256 = dense_norm[:, 256:]
    x = torch.arange(tail_256.size(1), dtype=torch.float32)
    x = x - x.mean()
    slopes = ((tail_256 - tail_256.mean(dim=1, keepdim=True)) * x).sum(dim=1)
    slopes = slopes / x.square().sum()
    return {
        "sample_size": len(indices),
        "selected_update_sequence": chronological,
        "shuffled_time_control": shuffled,
        "time_order_cosine_effect": (
            chronological["cosine_mean"] - shuffled["cosine_mean"]
        ),
        "tail_update_norm_slope_per_iteration": _quantiles(slopes),
        "update_norm_by_iteration": {
            str(iteration): _quantiles(dense_norm[:, iteration])
            for iteration in ITERATIONS
        },
        "update_cosine_by_iteration": {
            str(iteration): _quantiles(
                collected["dense_update_cosine"][indices, iteration - 1]
            )
            for iteration in ITERATIONS if iteration >= 1
        },
        "relative_acceleration_by_iteration": {
            str(iteration): _quantiles(
                collected["dense_relative_acceleration"][indices, iteration - 1]
            )
            for iteration in ITERATIONS if iteration >= 1
        },
        "normalized_distance_to_final": {
            str(iteration): _quantiles(actual_endpoint[:, position])
            for position, iteration in enumerate(ITERATIONS)
        },
        "shuffled_endpoint_distance": {
            str(iteration): _quantiles(shuffled_endpoint[:, position])
            for position, iteration in enumerate(ITERATIONS)
        },
        "endpoint_identity_effect_at_768": (
            shuffled_endpoint[:, iteration_to_index[768]].mean()
            - actual_endpoint[:, iteration_to_index[768]].mean()
        ).item(),
        "lag128_normalized_distance": {
            str(iteration): _quantiles(values)
            for iteration, values in zip(
                (256, 384, 512, 640, 768, 896, 1024), lag128
            )
        },
        "lag128_shuffled_puzzle_control": {
            str(iteration): _quantiles(values)
            for iteration, values in zip(
                (256, 384, 512, 640, 768, 896, 1024), lag128_shuffled
            )
        },
        "random_direction_contraction_gain": {
            str(iteration): _quantiles(
                collected["contraction_gain"][indices, position]
            )
            for position, iteration in enumerate(CONTRACTION_ITERATIONS)
        },
        "fraction_random_directions_contracting": {
            str(iteration): (
                collected["contraction_gain"][indices, position] < 1.0
            ).float().mean().item()
            for position, iteration in enumerate(CONTRACTION_ITERATIONS)
        },
        "update_direction_gain": {
            str(iteration): _quantiles(
                collected["update_direction_gain"][indices, position]
            )
            for position, iteration in enumerate(CONTRACTION_ITERATIONS)
        },
        "fraction_update_directions_contracting": {
            str(iteration): (
                collected["update_direction_gain"][indices, position] < 1.0
            ).float().mean().item()
            for position, iteration in enumerate(CONTRACTION_ITERATIONS)
        },
        "cell_accuracy": {
            str(iteration): _quantiles(collected["cell_accuracy"][indices, position])
            for position, iteration in enumerate(ITERATIONS)
        },
        "solved_fraction": {
            str(iteration): collected["solved"][indices, position].mean().item()
            for position, iteration in enumerate(ITERATIONS)
        },
    }


def _series(summary, field, statistic="median"):
    return [summary[field][str(iteration)][statistic] for iteration in ITERATIONS]


def render_plots(result, output_dir):
    colors = {
        "stable_plain": "#16736d",
        "collapsed_plain": "#c44e52",
        "late_state_ce": "#3568a8",
        "combined_margin": "#8a6d1d",
    }
    holdout = {
        name: data["splits"]["final_holdout"]
        for name, data in result["models"].items()
    }
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for name, summary in holdout.items():
        color = colors[name]
        axes[0, 0].plot(
            ITERATIONS,
            _series(summary, "update_norm_by_iteration"),
            marker="o", label=name, color=color,
        )
        axes[0, 1].plot(
            ITERATIONS[1:],
            [summary["update_cosine_by_iteration"][str(i)]["median"] for i in ITERATIONS[1:]],
            marker="o", label=name, color=color,
        )
        axes[1, 0].plot(
            ITERATIONS,
            _series(summary, "normalized_distance_to_final"),
            marker="o", label=name, color=color,
        )
        axes[1, 1].plot(
            CONTRACTION_ITERATIONS,
            [summary["update_direction_gain"][str(i)]["median"] for i in CONTRACTION_ITERATIONS],
            marker="o", label=name, color=color,
        )
    axes[0, 0].set_title("Board update norm")
    axes[0, 1].set_title("Consecutive update direction cosine")
    axes[1, 0].set_title("Normalized-state distance to iteration 1024")
    axes[1, 1].set_title("Local gain along the model's update direction")
    axes[1, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    for axis in axes.flat:
        axis.set_xscale("symlog", linthresh=1)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "fixed_point_summary.png"), dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for name, summary in holdout.items():
        color = colors[name]
        axes[0].plot(
            ITERATIONS,
            _series(summary, "relative_acceleration_by_iteration") if "0" in summary["relative_acceleration_by_iteration"] else [np.nan] + [summary["relative_acceleration_by_iteration"][str(i)]["median"] for i in ITERATIONS[1:]],
            marker="o", label=name, color=color,
        )
        axes[1].plot(
            (256, 384, 512, 640, 768, 896, 1024),
            [summary["lag128_normalized_distance"][str(i)]["median"] for i in (256, 384, 512, 640, 768, 896, 1024)],
            marker="o", label=name, color=color,
        )
        axes[2].plot(
            ITERATIONS,
            [summary["solved_fraction"][str(i)] for i in ITERATIONS],
            marker="o", label=name, color=color,
        )
    axes[0].set_title("Relative acceleration")
    axes[1].set_title("Normalized-state movement over 128 iterations")
    axes[2].set_title("Solved fraction")
    for axis in axes:
        axis.set_xscale("symlog", linthresh=1)
        axis.grid(alpha=0.25)
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "settling_and_accuracy.png"), dpi=170)
    plt.close(figure)

    names = list(holdout)
    chronological = [holdout[name]["selected_update_sequence"]["cosine_mean"] for name in names]
    shuffled = [holdout[name]["shuffled_time_control"]["cosine_mean"] for name in names]
    actual_endpoint = [holdout[name]["normalized_distance_to_final"]["768"]["median"] for name in names]
    shuffled_endpoint = [holdout[name]["shuffled_endpoint_distance"]["768"]["median"] for name in names]
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(x - .18, chronological, .36, label="ordered")
    axes[0].bar(x + .18, shuffled, .36, label="shuffled time")
    axes[1].bar(x - .18, actual_endpoint, .36, label="own endpoint")
    axes[1].bar(x + .18, shuffled_endpoint, .36, label="other puzzle endpoint")
    axes[0].set_title("Temporal-order control")
    axes[1].set_title("Endpoint-identity control at iteration 768")
    for axis in axes:
        axis.set_xticks(x, names, rotation=20, ha="right")
        axis.grid(axis="y", alpha=.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "controls.png"), dpi=170)
    plt.close(figure)


def run(
    output_dir,
    *,
    examples_per_bucket=12,
    seed=SEED,
    device="cuda",
    model_configs=DEFAULT_MODELS,
):
    if examples_per_bucket < 12:
        raise ValueError("protocol requires 12 examples per bucket for 20 per split")
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, _, buckets = _load_balanced_sample(
        examples_per_bucket, seed
    )
    splits = stratified_split(buckets)
    inputs = inputs.to(resolved_device)
    targets = targets.to(resolved_device)
    empty_mask = empty_mask.to(resolved_device)
    result = {
        "protocol": {
            "seed": seed,
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(inputs),
            "iterations": list(ITERATIONS),
            "contraction_iterations": list(CONTRACTION_ITERATIONS),
            "split_sizes": {name: len(indices) for name, indices in splits.items()},
            "split_indices": splits,
            "device": str(resolved_device),
            "random_contraction_directions": 3,
            "relative_perturbation": 1e-3,
        },
        "sample": {"puzzles": puzzles, "buckets": buckets},
        "models": {},
    }
    started = time.time()
    for model_config in model_configs:
        name = model_config["name"]
        print(f"Collecting {name}", flush=True)
        model = _load_model(model_config, resolved_device)
        collected = collect_model(model, inputs, targets, empty_mask)
        result["models"][name] = {
            "checkpoint": model_config["path"],
            "splits": {
                split_name: summarize_split(collected, indices)
                for split_name, indices in splits.items()
            },
        }
        del model, collected
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    result["elapsed_seconds"] = time.time() - started
    render_plots(result, output_dir)
    output_path = os.path.join(output_dir, "fixed_point_metrics.json")
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    print(f"Wrote {output_path}", flush=True)
    return result


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu")
