"""Test for a puzzle-held-out one-dimensional progress coordinate."""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


DEFAULT_SEED = 20260811
DEFAULT_EXAMPLES_PER_BUCKET = 12
DEFAULT_ITERATIONS = (
    0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256,
    384, 512, 768, 1024,
)
TARGET_NAMES = (
    "correct_fraction",
    "first_solve_progress",
    "stable_solved_duration",
)
REPRESENTATION_NAMES = (
    "blank_mean_state",
    "blank_mean_delta",
    "all_mean_delta",
    "blank_mean_unit_state",
    "blank_mean_and_spread_delta",
)
RIDGE_STRENGTHS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
DEFAULT_CONTROL_REPETITIONS = 32
DEFAULT_RANDOM_DIRECTIONS = 64
BOOTSTRAP_REPETITIONS = 1000


@dataclass
class ProgressAxis:
    representation: str
    ridge_strength: float
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    weight: torch.Tensor
    target_mean: torch.Tensor
    target_scale: torch.Tensor
    output_intercept: torch.Tensor
    output_slope: torch.Tensor
    train_coordinate_mean: float
    train_coordinate_scale: float
    effective_ridge: float


def stratified_three_way_split(bucket_names, seed=DEFAULT_SEED):
    """Split complete puzzles into balanced discovery, validation, and final sets."""

    bucket_names = tuple(bucket_names)
    if not bucket_names:
        raise ValueError("bucket_names must not be empty")
    generator = torch.Generator().manual_seed(seed)
    splits = {"discovery": [], "validation": [], "final": []}
    for bucket in dict.fromkeys(bucket_names):
        indices = torch.tensor(
            [index for index, value in enumerate(bucket_names) if value == bucket],
            dtype=torch.long,
        )
        if len(indices) < 12:
            raise ValueError(
                f"rating bucket {bucket!r} needs at least 12 puzzles for the study"
            )
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        split_size = len(indices) // 3
        splits["discovery"].extend(shuffled[:split_size].tolist())
        splits["validation"].extend(
            shuffled[split_size : 2 * split_size].tolist()
        )
        splits["final"].extend(shuffled[2 * split_size : 3 * split_size].tolist())
    for name, indices in splits.items():
        if len(indices) < 20:
            raise ValueError(f"{name} split has only {len(indices)} puzzles")
        splits[name] = sorted(indices)
    return splits


def build_progress_targets(correct_fraction, solved_history, iterations):
    """Create behavioral progress targets without using hidden states."""

    correct_fraction = torch.as_tensor(correct_fraction, dtype=torch.float32)
    solved_history = torch.as_tensor(solved_history, dtype=torch.bool)
    iterations = torch.as_tensor(iterations, dtype=torch.long)
    if correct_fraction.ndim != 2:
        raise ValueError("correct_fraction must have shape [puzzles, snapshots]")
    if solved_history.ndim != 2 or solved_history.size(0) != correct_fraction.size(0):
        raise ValueError("solved_history must have shape [puzzles, iterations]")
    if correct_fraction.size(1) != len(iterations):
        raise ValueError("correct_fraction and iterations do not align")
    if iterations.min() < 0 or iterations.max() >= solved_history.size(1):
        raise ValueError("snapshot iteration falls outside solved_history")

    puzzle_count = correct_fraction.size(0)
    first_solve = torch.full((puzzle_count,), -1, dtype=torch.long)
    solved_streak = torch.zeros_like(solved_history, dtype=torch.long)
    running_streak = torch.zeros(puzzle_count, dtype=torch.long)
    for iteration in range(solved_history.size(1)):
        solved = solved_history[:, iteration]
        newly_solved = solved & first_solve.lt(0)
        first_solve[newly_solved] = iteration
        running_streak = torch.where(
            solved,
            running_streak + 1,
            torch.zeros_like(running_streak),
        )
        solved_streak[:, iteration] = running_streak

    first_progress = torch.zeros_like(correct_fraction)
    for puzzle_index, solve_iteration in enumerate(first_solve.tolist()):
        if solve_iteration >= 0:
            denominator = max(solve_iteration, 1)
            first_progress[puzzle_index] = (
                iterations.float() / denominator
            ).clamp(max=1.0)

    maximum_iteration = solved_history.size(1) - 1
    streak_at_snapshots = solved_streak[:, iterations]
    stable_duration = torch.log1p(streak_at_snapshots.float()) / math.log1p(
        maximum_iteration + 1
    )
    return {
        "correct_fraction": correct_fraction,
        "remaining_incorrect_fraction": 1.0 - correct_fraction,
        "first_solve_progress": first_progress,
        "stable_solved_duration": stable_duration,
        "first_solve_iteration": first_solve,
        "solved_streak_at_snapshots": streak_at_snapshots,
    }


def _masked_mean(values, mask):
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _snapshot_features(hidden, initial_hidden, empty_mask):
    delta = hidden.float() - initial_hidden.float()
    blank_mean_delta = _masked_mean(delta, empty_mask)
    centered_delta = delta - blank_mean_delta[:, None, :]
    blank_spread_delta = _masked_mean(
        centered_delta.square(), empty_mask
    ).clamp_min(0).sqrt()
    return {
        "blank_mean_state": _masked_mean(hidden.float(), empty_mask),
        "blank_mean_delta": blank_mean_delta,
        "all_mean_delta": delta.mean(dim=1),
        "blank_mean_unit_state": _masked_mean(
            F.normalize(hidden.float(), dim=-1, eps=1e-12), empty_mask
        ),
        "blank_mean_and_spread_delta": torch.cat(
            (blank_mean_delta, blank_spread_delta), dim=-1
        ),
    }


def collect_progress_trajectory(model, inputs, targets, empty_mask, iterations):
    """Collect sparse board summaries and exact solve events through the horizon."""

    iterations = tuple(iterations)
    if iterations != tuple(sorted(set(iterations))) or iterations[0] != 0:
        raise ValueError("iterations must be unique, increasing, and start at zero")
    maximum_iteration = iterations[-1]
    requested = set(iterations)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    feature_snapshots = {name: [] for name in REPRESENTATION_NAMES}
    correct_snapshots = []
    solved_history = []

    with torch.inference_mode():
        hidden = model.initial_encoder(inputs)
        initial_hidden = hidden.detach().float()
        feedback = torch.zeros(
            inputs.size(0), 81, 9, device=inputs.device, dtype=hidden.dtype
        )
        for iteration in range(maximum_iteration + 1):
            logits = model.output_head(hidden)
            correct = logits.argmax(dim=-1).eq(targets)
            solved = (correct | ~empty_mask).all(dim=1)
            solved_history.append(solved.detach().cpu())
            if iteration in requested:
                features = _snapshot_features(hidden, initial_hidden, empty_mask)
                for name, values in features.items():
                    feature_snapshots[name].append(values.detach().float().cpu())
                blank_correct = (correct & empty_mask).sum(dim=1).float()
                blank_total = empty_mask.sum(dim=1).clamp_min(1)
                correct_snapshots.append(
                    (blank_correct / blank_total).detach().cpu()
                )
            if iteration == maximum_iteration:
                break
            hidden = model.recurrent_step(
                hidden, feedback, rope_cos, rope_sin
            )
            feedback = F.softmax(model.output_head(hidden), dim=-1)

    features = {
        name: torch.stack(values, dim=1)
        for name, values in feature_snapshots.items()
    }
    correct_fraction = torch.stack(correct_snapshots, dim=1)
    solved_history = torch.stack(solved_history, dim=1)
    progress_targets = build_progress_targets(
        correct_fraction, solved_history, iterations
    )
    return {
        "features": features,
        "targets": progress_targets,
        "solved_history": solved_history,
    }


def _flatten_split(model_data, representation, puzzle_indices):
    indices = torch.as_tensor(puzzle_indices, dtype=torch.long)
    features = model_data["features"][representation][indices]
    targets = torch.stack(
        [model_data["targets"][name][indices] for name in TARGET_NAMES], dim=-1
    )
    return features.flatten(0, 1), targets.flatten(0, 1)


def fit_progress_axis(feature_rows, target_rows, representation, ridge_strength):
    """Fit one hidden-state coordinate to all progress targets jointly."""

    x = torch.as_tensor(feature_rows, dtype=torch.float64)
    y = torch.as_tensor(target_rows, dtype=torch.float64)
    if x.ndim != 2 or y.ndim != 2 or x.size(0) != y.size(0):
        raise ValueError("feature_rows and target_rows must be aligned matrices")
    if y.size(1) != len(TARGET_NAMES):
        raise ValueError("target_rows has the wrong number of columns")
    if ridge_strength <= 0:
        raise ValueError("ridge_strength must be positive")

    feature_mean = x.mean(dim=0)
    feature_scale = x.std(dim=0, unbiased=False)
    feature_scale = torch.where(
        feature_scale > 1e-8, feature_scale, torch.ones_like(feature_scale)
    )
    standardized_x = (x - feature_mean) / feature_scale
    target_mean = y.mean(dim=0)
    target_scale = y.std(dim=0, unbiased=False)
    target_scale = torch.where(
        target_scale > 1e-8, target_scale, torch.ones_like(target_scale)
    )
    standardized_y = (y - target_mean) / target_scale

    mean_row_energy = standardized_x.square().sum(dim=1).mean().clamp_min(1e-12)
    effective_ridge = float(ridge_strength * mean_row_energy)
    gram = standardized_x.T @ standardized_x
    right_hand_side = standardized_x.T @ standardized_y
    coefficients = torch.linalg.solve(
        gram + effective_ridge * torch.eye(gram.size(0), dtype=gram.dtype),
        right_hand_side,
    )
    fitted_targets = standardized_x @ coefficients
    _, _, right_vectors = torch.linalg.svd(fitted_targets, full_matrices=False)
    weight = coefficients @ right_vectors[0]
    coordinate = standardized_x @ weight
    mean_progress = standardized_y.mean(dim=1)
    if torch.dot(
        coordinate - coordinate.mean(), mean_progress - mean_progress.mean()
    ) < 0:
        weight = -weight
        coordinate = -coordinate

    coordinate_scale = float(coordinate.std(unbiased=False).clamp_min(1e-12))
    coordinate_mean = float(coordinate.mean())
    normalized_coordinate = (coordinate - coordinate_mean) / coordinate_scale
    design = torch.stack(
        (torch.ones_like(normalized_coordinate), normalized_coordinate), dim=1
    )
    output_coefficients = torch.linalg.lstsq(
        design, standardized_y
    ).solution
    return ProgressAxis(
        representation=representation,
        ridge_strength=float(ridge_strength),
        feature_mean=feature_mean.float(),
        feature_scale=feature_scale.float(),
        weight=weight.float(),
        target_mean=target_mean.float(),
        target_scale=target_scale.float(),
        output_intercept=output_coefficients[0].float(),
        output_slope=output_coefficients[1].float(),
        train_coordinate_mean=coordinate_mean,
        train_coordinate_scale=coordinate_scale,
        effective_ridge=effective_ridge,
    )


def apply_progress_axis(axis, features):
    features = torch.as_tensor(features, dtype=torch.float32)
    standardized = (features - axis.feature_mean) / axis.feature_scale
    coordinate = standardized @ axis.weight
    normalized = (
        coordinate - axis.train_coordinate_mean
    ) / axis.train_coordinate_scale
    standardized_targets = (
        axis.output_intercept + normalized.unsqueeze(-1) * axis.output_slope
    )
    predicted_targets = (
        standardized_targets * axis.target_scale + axis.target_mean
    )
    return normalized, predicted_targets


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(first, second):
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    finite = np.isfinite(first) & np.isfinite(second)
    first = first[finite]
    second = second[finite]
    if len(first) < 3 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return float("nan")
    ranked_first = _rankdata(first)
    ranked_second = _rankdata(second)
    return float(np.corrcoef(ranked_first, ranked_second)[0, 1])


def _bootstrap_mean_interval(values, seed, repetitions=BOOTSTRAP_REPETITIONS):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [None, None]
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repetitions, len(values)), replace=True)
    return np.quantile(samples.mean(axis=1), (0.025, 0.975)).tolist()


def score_coordinate(
    coordinate,
    predicted_targets,
    model_data,
    puzzle_indices,
    iterations,
    seed,
):
    indices = torch.as_tensor(puzzle_indices, dtype=torch.long)
    coordinate = torch.as_tensor(coordinate).reshape(len(indices), -1).cpu().numpy()
    predicted_targets = torch.as_tensor(predicted_targets).reshape(
        len(indices), -1, len(TARGET_NAMES)
    ).cpu().numpy()
    targets = {
        name: model_data["targets"][name][indices].cpu().numpy()
        for name in TARGET_NAMES
    }
    target_results = {}
    selection_values = []
    for target_index, name in enumerate(TARGET_NAMES):
        values = targets[name]
        per_puzzle = [
            spearman_correlation(coordinate[puzzle], values[puzzle])
            for puzzle in range(len(indices))
        ]
        finite_per_puzzle = [value for value in per_puzzle if math.isfinite(value)]
        per_snapshot = [
            spearman_correlation(coordinate[:, snapshot], values[:, snapshot])
            for snapshot in range(coordinate.shape[1])
        ]
        finite_per_snapshot = [
            value for value in per_snapshot if math.isfinite(value)
        ]
        residual = values - predicted_targets[..., target_index]
        total = values - values.mean()
        r2 = 1.0 - float(np.square(residual).sum()) / max(
            float(np.square(total).sum()), 1e-30
        )
        mean_within = (
            float(np.mean(finite_per_puzzle)) if finite_per_puzzle else None
        )
        if mean_within is not None:
            selection_values.append(mean_within)
        target_results[name] = {
            "pooled_spearman": spearman_correlation(coordinate, values),
            "mean_within_puzzle_spearman": mean_within,
            "median_within_puzzle_spearman": (
                float(np.median(finite_per_puzzle)) if finite_per_puzzle else None
            ),
            "mean_within_puzzle_spearman_ci95": _bootstrap_mean_interval(
                finite_per_puzzle, seed + target_index
            ),
            "valid_puzzles": len(finite_per_puzzle),
            "mean_within_snapshot_puzzle_spearman": (
                float(np.mean(finite_per_snapshot))
                if finite_per_snapshot else None
            ),
            "median_within_snapshot_puzzle_spearman": (
                float(np.median(finite_per_snapshot))
                if finite_per_snapshot else None
            ),
            "valid_snapshots": len(finite_per_snapshot),
            "r2": r2,
        }

    iterations_array = np.asarray(iterations, dtype=np.float64)
    iteration_correlations = [
        spearman_correlation(coordinate[puzzle], iterations_array)
        for puzzle in range(len(indices))
    ]
    monotonicity = []
    for puzzle_coordinate in coordinate:
        tolerance = 1e-8 * max(float(np.std(puzzle_coordinate)), 1.0)
        monotonicity.append(float(np.mean(np.diff(puzzle_coordinate) >= -tolerance)))
    remaining = model_data["targets"]["remaining_incorrect_fraction"][indices]
    return {
        "selection_score": (
            float(np.mean(selection_values)) if selection_values else None
        ),
        "targets": target_results,
        "remaining_incorrect_pooled_spearman": spearman_correlation(
            coordinate, remaining.cpu().numpy()
        ),
        "iteration_mean_within_puzzle_spearman": float(
            np.nanmean(iteration_correlations)
        ),
        "nondecreasing_pair_fraction": float(np.mean(monotonicity)),
        "nondecreasing_pair_fraction_ci95": _bootstrap_mean_interval(
            monotonicity, seed + 100
        ),
    }


def fit_axis_for_split(
    model_data_by_name,
    model_names,
    representation,
    ridge_strength,
    puzzle_indices,
    target_overrides=None,
):
    feature_rows = []
    target_rows = []
    for model_name in model_names:
        model_data = model_data_by_name[model_name]
        x, y = _flatten_split(model_data, representation, puzzle_indices)
        if target_overrides and model_name in target_overrides:
            override = target_overrides[model_name]
            y = torch.stack(
                [override[name][puzzle_indices] for name in TARGET_NAMES], dim=-1
            ).flatten(0, 1)
        feature_rows.append(x)
        target_rows.append(y)
    return fit_progress_axis(
        torch.cat(feature_rows),
        torch.cat(target_rows),
        representation,
        ridge_strength,
    )


def evaluate_axis_on_models(
    axis,
    model_data_by_name,
    model_names,
    puzzle_indices,
    iterations,
    seed,
):
    by_model = {}
    for model_index, model_name in enumerate(model_names):
        features = model_data_by_name[model_name]["features"][axis.representation][
            puzzle_indices
        ]
        coordinate, predicted = apply_progress_axis(axis, features)
        by_model[model_name] = score_coordinate(
            coordinate,
            predicted,
            model_data_by_name[model_name],
            puzzle_indices,
            iterations,
            seed + model_index * 100,
        )
    scores = [
        value["selection_score"]
        for value in by_model.values()
        if value["selection_score"] is not None
    ]
    return {
        "mean_selection_score": float(np.mean(scores)),
        "models": by_model,
    }


def select_axis(
    model_data_by_name,
    model_names,
    discovery_indices,
    validation_indices,
    iterations,
    seed,
):
    candidates = []
    best = None
    for representation in REPRESENTATION_NAMES:
        for ridge_strength in RIDGE_STRENGTHS:
            axis = fit_axis_for_split(
                model_data_by_name,
                model_names,
                representation,
                ridge_strength,
                discovery_indices,
            )
            validation = evaluate_axis_on_models(
                axis,
                model_data_by_name,
                model_names,
                validation_indices,
                iterations,
                seed,
            )
            record = {
                "representation": representation,
                "ridge_strength": ridge_strength,
                "validation_score": validation["mean_selection_score"],
            }
            candidates.append(record)
            if best is None or record["validation_score"] > best[0]:
                best = (record["validation_score"], axis, validation)
    return best[1], best[2], candidates


def _permuted_targets(model_data, puzzle_indices, bucket_names, kind, generator):
    overrides = {
        name: model_data["targets"][name].clone() for name in TARGET_NAMES
    }
    indices = list(puzzle_indices)
    if kind == "iteration":
        for puzzle_index in indices:
            order = torch.randperm(
                overrides[TARGET_NAMES[0]].size(1), generator=generator
            )
            for name in TARGET_NAMES:
                overrides[name][puzzle_index] = overrides[name][puzzle_index, order]
    elif kind == "puzzle":
        for bucket in dict.fromkeys(bucket_names):
            bucket_indices = [
                index for index in indices if bucket_names[index] == bucket
            ]
            if len(bucket_indices) < 2:
                raise ValueError("puzzle shuffle needs two puzzles per bucket")
            destinations = torch.tensor(bucket_indices)[
                torch.randperm(len(bucket_indices), generator=generator)
            ].tolist()
            sources = destinations[1:] + destinations[:1]
            for destination, source in zip(destinations, sources):
                for name in TARGET_NAMES:
                    overrides[name][destination] = model_data["targets"][name][source]
    else:
        raise ValueError(f"unknown permutation kind {kind!r}")
    return overrides


def _null_summary(values, observed):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return {
        "repetitions": int(len(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
        "empirical_p_greater_or_equal": float(
            (1 + np.sum(values >= observed)) / (len(values) + 1)
        ),
        "values": values.tolist(),
    }


def run_controls(
    axis,
    model_data_by_name,
    model_names,
    discovery_indices,
    final_indices,
    bucket_names,
    iterations,
    seed,
    repetitions,
    random_directions,
):
    observed = evaluate_axis_on_models(
        axis,
        model_data_by_name,
        model_names,
        final_indices,
        iterations,
        seed,
    )["mean_selection_score"]
    time_only_model_scores = {}
    time_coordinate = torch.log1p(
        torch.as_tensor(iterations, dtype=torch.float32)
    ).repeat(len(final_indices), 1)
    for model_index, model_name in enumerate(model_names):
        model_data = model_data_by_name[model_name]
        target_shape = (len(final_indices), len(iterations), len(TARGET_NAMES))
        time_only_model_scores[model_name] = score_coordinate(
            time_coordinate,
            torch.zeros(target_shape),
            model_data,
            final_indices,
            iterations,
            seed + 500 + model_index,
        )["selection_score"]
    time_only_score = float(np.mean(list(time_only_model_scores.values())))
    generator = torch.Generator().manual_seed(seed)
    nulls = {"shuffled_iteration_fit": [], "shuffled_puzzle_fit": []}
    for repetition in range(repetitions):
        for kind, output_name in (
            ("iteration", "shuffled_iteration_fit"),
            ("puzzle", "shuffled_puzzle_fit"),
        ):
            overrides = {
                model_name: _permuted_targets(
                    model_data_by_name[model_name],
                    discovery_indices,
                    bucket_names,
                    kind,
                    generator,
                )
                for model_name in model_names
            }
            controlled_axis = fit_axis_for_split(
                model_data_by_name,
                model_names,
                axis.representation,
                axis.ridge_strength,
                discovery_indices,
                target_overrides=overrides,
            )
            score = evaluate_axis_on_models(
                controlled_axis,
                model_data_by_name,
                model_names,
                final_indices,
                iterations,
                seed + repetition,
            )["mean_selection_score"]
            nulls[output_name].append(score)

    time_shuffle_scores = []
    for repetition in range(repetitions):
        model_scores = []
        for model_index, model_name in enumerate(model_names):
            model_data = model_data_by_name[model_name]
            features = model_data["features"][axis.representation][final_indices]
            coordinate, predicted = apply_progress_axis(axis, features)
            shuffled = coordinate.clone()
            for puzzle in range(len(final_indices)):
                order = torch.randperm(shuffled.size(1), generator=generator)
                shuffled[puzzle] = shuffled[puzzle, order]
            score = score_coordinate(
                shuffled,
                predicted,
                model_data,
                final_indices,
                iterations,
                seed + repetition + model_index * 100,
            )["selection_score"]
            model_scores.append(score)
        time_shuffle_scores.append(float(np.mean(model_scores)))
    nulls["shuffled_final_iteration_order"] = time_shuffle_scores

    random_scores = []
    feature_dimension = axis.weight.numel()
    random_matrix = torch.randn(
        feature_dimension, random_directions, generator=generator
    )
    random_matrix, _ = torch.linalg.qr(random_matrix, mode="reduced")
    for direction_index in range(random_matrix.size(1)):
        random_axis = ProgressAxis(
            **{
                **axis.__dict__,
                "weight": random_matrix[:, direction_index],
                "train_coordinate_mean": 0.0,
                "train_coordinate_scale": 1.0,
            }
        )
        discovery_coordinates = []
        discovery_progress = []
        for model_name in model_names:
            data = model_data_by_name[model_name]
            values, _ = apply_progress_axis(
                random_axis,
                data["features"][axis.representation][discovery_indices],
            )
            discovery_coordinates.append(values.reshape(-1))
            discovery_progress.append(torch.stack([
                data["targets"][name][discovery_indices]
                for name in TARGET_NAMES
            ], dim=-1).mean(dim=-1).reshape(-1))
        if spearman_correlation(
            torch.cat(discovery_coordinates), torch.cat(discovery_progress)
        ) < 0:
            random_axis.weight = -random_axis.weight
        score = evaluate_axis_on_models(
            random_axis,
            model_data_by_name,
            model_names,
            final_indices,
            iterations,
            seed + direction_index,
        )["mean_selection_score"]
        random_scores.append(score)
    nulls["random_orthonormal_direction"] = random_scores
    return {
        "observed": observed,
        "iteration_only_baseline": {
            "score": time_only_score,
            "models": time_only_model_scores,
            "observed_minus_baseline": observed - time_only_score,
        },
        **{
            name: _null_summary(values, observed)
            for name, values in nulls.items()
        },
    }


def _axis_metadata(axis):
    return {
        "representation": axis.representation,
        "ridge_strength": axis.ridge_strength,
        "feature_dimension": int(axis.weight.numel()),
        "effective_ridge": axis.effective_ridge,
        "target_output_slopes": {
            name: float(axis.output_slope[index])
            for index, name in enumerate(TARGET_NAMES)
        },
    }


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(_json_ready(value), handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def _write_coordinate_csv(
    path,
    axes,
    pooled_axis,
    model_data_by_name,
    final_indices,
    iterations,
    bucket_names,
):
    columns = [
        "axis", "model", "puzzle_index", "rating_bucket", "iteration",
        "coordinate", *TARGET_NAMES, "remaining_incorrect_fraction",
    ]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for axis_name, axis in [*axes.items(), ("pooled", pooled_axis)]:
            for model_name, model_data in model_data_by_name.items():
                features = model_data["features"][axis.representation][final_indices]
                coordinate, _ = apply_progress_axis(axis, features)
                for local_puzzle, puzzle_index in enumerate(final_indices):
                    for time_index, iteration in enumerate(iterations):
                        writer.writerow({
                            "axis": axis_name,
                            "model": model_name,
                            "puzzle_index": puzzle_index,
                            "rating_bucket": bucket_names[puzzle_index],
                            "iteration": iteration,
                            "coordinate": float(coordinate[local_puzzle, time_index]),
                            **{
                                name: float(model_data["targets"][name][puzzle_index, time_index])
                                for name in TARGET_NAMES
                            },
                            "remaining_incorrect_fraction": float(
                                model_data["targets"]["remaining_incorrect_fraction"][
                                    puzzle_index, time_index
                                ]
                            ),
                        })


def _plot_trajectories(path, axes, model_data_by_name, final_indices, iterations):
    figure, plot_axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for plot_axis, (model_name, model_data) in zip(
        plot_axes.flat, model_data_by_name.items()
    ):
        axis = axes[model_name]
        coordinate, _ = apply_progress_axis(
            axis,
            model_data["features"][axis.representation][final_indices],
        )
        coordinate = coordinate.numpy()
        target = model_data["targets"]["correct_fraction"][final_indices].numpy()
        median = np.median(coordinate, axis=0)
        lower, upper = np.quantile(coordinate, (0.25, 0.75), axis=0)
        target_median = np.median(target, axis=0)
        plot_axis.fill_between(iterations, lower, upper, color="#4C78A8", alpha=0.2)
        plot_axis.plot(iterations, median, color="#4C78A8", label="hidden coordinate")
        target_axis = plot_axis.twinx()
        target_axis.plot(
            iterations, target_median, color="#E45756", linestyle="--",
            label="correct blank fraction",
        )
        plot_axis.set_xscale("symlog", linthresh=1)
        plot_axis.set_title(model_name.replace("_", " "))
        plot_axis.set_xlabel("recurrent iteration")
        plot_axis.set_ylabel("coordinate (discovery SD)", color="#4C78A8")
        target_axis.set_ylabel("correct blank fraction", color="#E45756")
        target_axis.set_ylim(0, 1.03)
        plot_axis.grid(alpha=0.2)
    figure.suptitle("Final-holdout progress coordinates; bands show puzzle IQR")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_target_tracking(path, axes, model_data_by_name, final_indices, metrics):
    figure, plot_axes = plt.subplots(4, 3, figsize=(12, 14), constrained_layout=True)
    for model_row, (model_name, model_data) in enumerate(model_data_by_name.items()):
        axis = axes[model_name]
        coordinate, _ = apply_progress_axis(
            axis,
            model_data["features"][axis.representation][final_indices],
        )
        coordinate = coordinate.numpy().reshape(-1)
        for target_column, target_name in enumerate(TARGET_NAMES):
            plot_axis = plot_axes[model_row, target_column]
            target = model_data["targets"][target_name][final_indices].numpy().reshape(-1)
            plot_axis.hexbin(
                coordinate, target, gridsize=30, mincnt=1, cmap="viridis"
            )
            rho = metrics[model_name]["targets"][target_name][
                "mean_within_puzzle_spearman"
            ]
            rho_text = "NA" if rho is None else f"{rho:.2f}"
            target_title = {
                "correct_fraction": "Blank-cell correctness",
                "first_solve_progress": "First-solve progress",
                "stable_solved_duration": "Stable solved duration",
            }[target_name]
            plot_axis.set_title(
                f"{target_title}\nmean puzzle rho={rho_text}"
            )
            plot_axis.set_xlabel("hidden coordinate")
            plot_axis.set_ylabel(target_name.replace("_", " "))
            if target_column == 0:
                plot_axis.annotate(
                    model_name.replace("_", " "),
                    xy=(-0.32, 0.5),
                    xycoords="axes fraction",
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
    figure.suptitle("Behavioral tracking on 20 unseen final puzzles")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_transfer(path, transfer, model_names, pooled_final):
    matrix = np.asarray([
        [transfer[source][target]["selection_score"] for target in model_names]
        for source in model_names
    ])
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    for row in range(len(model_names)):
        for column in range(len(model_names)):
            axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center")
    labels = [name.replace("_", " ") for name in model_names]
    axis.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("checkpoint receiving the frozen axis")
    axis.set_ylabel("checkpoint used to fit the axis")
    axis.set_title(
        "Direct checkpoint transfer\n"
        f"pooled shared-axis final score: {pooled_final['mean_selection_score']:.2f}"
    )
    figure.colorbar(image, ax=axis, label="mean held-out target Spearman")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_controls(path, controls_by_axis):
    names = list(controls_by_axis)
    control_names = (
        "shuffled_iteration_fit",
        "shuffled_puzzle_fit",
        "shuffled_final_iteration_order",
        "random_orthonormal_direction",
    )
    colors = ("#F58518", "#54A24B", "#B279A2", "#9D755D")
    figure, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 5), constrained_layout=True)
    if len(names) == 1:
        axes = [axes]
    for axis, name in zip(axes, names):
        controls = controls_by_axis[name]
        values = [
            [controls["iteration_only_baseline"]["score"]],
            *[controls[control]["values"] for control in control_names],
        ]
        box = axis.boxplot(values, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], ("#4C78A8", *colors)):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        axis.axhline(controls["observed"], color="#222222", linewidth=2, label="observed")
        axis.set_xticks(
            range(1, len(control_names) + 2),
            ("iteration only", "time fit", "puzzle fit", "time eval", "random axis"),
            rotation=35,
            ha="right",
        )
        axis.set_ylim(-1.02, 1.02)
        axis.set_title(name.replace("_", " "))
        axis.set_ylabel("final mean target Spearman")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend()
    figure.suptitle("Observed final coordinate compared with predefined controls")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(
    output_dir,
    *,
    examples_per_bucket=DEFAULT_EXAMPLES_PER_BUCKET,
    seed=DEFAULT_SEED,
    device="cuda",
    model_configs=DEFAULT_MODELS,
    iterations=DEFAULT_ITERATIONS,
    control_repetitions=DEFAULT_CONTROL_REPETITIONS,
    random_directions=DEFAULT_RANDOM_DIRECTIONS,
):
    if examples_per_bucket < 12:
        raise ValueError("the protocol requires at least 12 examples per rating bucket")
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    splits = stratified_three_way_split(bucket_names, seed)
    model_names = [config["name"] for config in model_configs]
    progress_path = os.path.join(output_dir, "progress.log")
    progress_handle = open(progress_path, "w")
    started_at = time.time()

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        progress_handle.write(line + "\n")
        progress_handle.flush()

    model_data_by_name = {}
    inputs = inputs.to(resolved_device)
    targets_device = targets.to(resolved_device)
    empty_mask_device = empty_mask.to(resolved_device)
    log(
        f"Starting {len(model_names)} checkpoints on {len(inputs)} puzzles; "
        f"splits={{{', '.join(f'{name}:{len(value)}' for name, value in splits.items())}}}"
    )
    for config in model_configs:
        log(f"MODEL {config['name']}: loading {config['path']}")
        model = _load_model(config, resolved_device)
        model_data_by_name[config["name"]] = collect_progress_trajectory(
            model, inputs, targets_device, empty_mask_device, iterations
        )
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
        first_solve = model_data_by_name[config["name"]]["targets"][
            "first_solve_iteration"
        ]
        log(
            f"MODEL {config['name']}: ever solved "
            f"{int(first_solve.ge(0).sum())}/{len(first_solve)}"
        )

    log("Selecting one axis independently for each checkpoint on validation")
    selected_axes = {}
    selection = {}
    final_metrics = {}
    for model_index, model_name in enumerate(model_names):
        axis, validation, candidates = select_axis(
            model_data_by_name,
            [model_name],
            splits["discovery"],
            splits["validation"],
            iterations,
            seed + model_index * 1000,
        )
        selected_axes[model_name] = axis
        final = evaluate_axis_on_models(
            axis,
            model_data_by_name,
            [model_name],
            splits["final"],
            iterations,
            seed + model_index * 1000,
        )
        final_metrics[model_name] = final["models"][model_name]
        selection[model_name] = {
            "axis": _axis_metadata(axis),
            "validation": validation,
            "candidates": candidates,
            "final": final,
        }
        log(
            f"MODEL {model_name}: selected {axis.representation}, "
            f"ridge={axis.ridge_strength:g}, final={final['mean_selection_score']:.3f}"
        )

    log("Selecting one coordinate pooled across all checkpoints")
    pooled_axis, pooled_validation, pooled_candidates = select_axis(
        model_data_by_name,
        model_names,
        splits["discovery"],
        splits["validation"],
        iterations,
        seed + 10_000,
    )
    pooled_final = evaluate_axis_on_models(
        pooled_axis,
        model_data_by_name,
        model_names,
        splits["final"],
        iterations,
        seed + 10_000,
    )
    log(
        f"POOLED: selected {pooled_axis.representation}, "
        f"ridge={pooled_axis.ridge_strength:g}, final={pooled_final['mean_selection_score']:.3f}"
    )

    log("Measuring direct checkpoint-to-checkpoint transfer")
    transfer = {}
    for source_index, source_name in enumerate(model_names):
        source_axis = selected_axes[source_name]
        transfer[source_name] = evaluate_axis_on_models(
            source_axis,
            model_data_by_name,
            model_names,
            splits["final"],
            iterations,
            seed + 20_000 + source_index * 1000,
        )["models"]

    log("Running shuffled-time, shuffled-puzzle, and random-direction controls")
    controls = {}
    for model_index, model_name in enumerate(model_names):
        controls[model_name] = run_controls(
            selected_axes[model_name],
            model_data_by_name,
            [model_name],
            splits["discovery"],
            splits["final"],
            bucket_names,
            iterations,
            seed + 30_000 + model_index * 1000,
            control_repetitions,
            random_directions,
        )
        log(f"CONTROLS {model_name}: complete")
    controls["pooled"] = run_controls(
        pooled_axis,
        model_data_by_name,
        model_names,
        splits["discovery"],
        splits["final"],
        bucket_names,
        iterations,
        seed + 40_000,
        control_repetitions,
        random_directions,
    )

    first_solve_summary = {}
    for model_name, model_data in model_data_by_name.items():
        first = model_data["targets"]["first_solve_iteration"]
        first_solve_summary[model_name] = {
            split_name: {
                "ever_solved": int(first[indices].ge(0).sum()),
                "puzzles": len(indices),
                "median_first_solve_if_observed": (
                    float(first[indices][first[indices].ge(0)].float().median())
                    if first[indices].ge(0).any()
                    else None
                ),
            }
            for split_name, indices in splits.items()
        }

    metrics = {
        "config": {
            "seed": seed,
            "examples_per_bucket": examples_per_bucket,
            "puzzle_count": len(inputs),
            "iterations": list(iterations),
            "targets": TARGET_NAMES,
            "representations": REPRESENTATION_NAMES,
            "ridge_strengths": RIDGE_STRENGTHS,
            "control_repetitions": control_repetitions,
            "random_directions": random_directions,
            "models": list(model_configs),
            "device": str(resolved_device),
        },
        "definitions": {
            "correct_fraction": "fraction of originally blank cells whose top prediction equals the answer",
            "remaining_incorrect_fraction": "one minus correct_fraction; it contains the same information with opposite sign",
            "first_solve_progress": "zero for puzzles never solved through 1024; otherwise min(iteration / first_solve_iteration, 1)",
            "stable_solved_duration": "log-scaled number of consecutive exactly solved integer iterations ending at the snapshot",
            "shared_coordinate": "rank-one ridge regression from a board-level hidden-state summary to the three standardized progress targets",
            "selection_score": "mean of the three mean within-puzzle Spearman correlations",
            "fixed_snapshot_diagnostic": "mean correlation across final puzzles at the same snapshot; added after the predefined puzzle-shuffle control exposed an iteration-clock confound and not used for selection",
            "fitting": "projection and normalization fit on discovery puzzles only; representation and ridge selected on validation; final puzzles used once",
        },
        "splits": splits,
        "sample": {
            "rating_buckets": bucket_names,
            "puzzles": puzzles,
            "solutions": solutions,
        },
        "first_solve_summary": first_solve_summary,
        "per_checkpoint": selection,
        "pooled": {
            "axis": _axis_metadata(pooled_axis),
            "validation": pooled_validation,
            "candidates": pooled_candidates,
            "final": pooled_final,
        },
        "direct_checkpoint_transfer": transfer,
        "controls": controls,
        "runtime_seconds": time.time() - started_at,
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    _atomic_json(metrics_path, metrics)
    _write_coordinate_csv(
        os.path.join(output_dir, "final_coordinates.csv"),
        selected_axes,
        pooled_axis,
        model_data_by_name,
        splits["final"],
        iterations,
        bucket_names,
    )
    axis_arrays = {}
    for name, axis in [*selected_axes.items(), ("pooled", pooled_axis)]:
        axis_arrays[f"{name}__weight"] = axis.weight.numpy()
        axis_arrays[f"{name}__feature_mean"] = axis.feature_mean.numpy()
        axis_arrays[f"{name}__feature_scale"] = axis.feature_scale.numpy()
    np.savez_compressed(os.path.join(output_dir, "selected_axes.npz"), **axis_arrays)
    _plot_trajectories(
        os.path.join(output_dir, "progress_trajectories.png"),
        selected_axes,
        model_data_by_name,
        splits["final"],
        iterations,
    )
    _plot_target_tracking(
        os.path.join(output_dir, "heldout_target_tracking.png"),
        selected_axes,
        model_data_by_name,
        splits["final"],
        final_metrics,
    )
    _plot_transfer(
        os.path.join(output_dir, "checkpoint_transfer.png"),
        transfer,
        model_names,
        pooled_final,
    )
    _plot_controls(
        os.path.join(output_dir, "controls.png"), controls
    )
    log(f"Finished in {time.time() - started_at:.1f}s")
    progress_handle.close()
    return metrics


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu")
