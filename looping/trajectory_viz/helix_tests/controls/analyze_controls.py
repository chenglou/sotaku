"""Falsification controls for digit/number-helix-like Sotaku geometry.

The primary statistics are basis-invariant and split by whole puzzle.  A
supervised two-dimensional digit readout is included only as a visualization
control: any linearly decodable nine-class representation can be mapped to a
circle in an arbitrary digit order.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
import os
import time
from collections import defaultdict
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


DIGIT_COUNT = 9
ITERATIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
PHASES = {
    "early_1_16": (1, 2, 4, 8, 16),
    "middle_32_128": (32, 64, 128),
    "late_256_1024": (256, 512, 1024),
    "all_1_1024": ITERATIONS[1:],
}
NATURAL_ORDER = tuple(range(DIGIT_COUNT))
MODEL_HEALTH = {
    "stable_plain": {"accuracy_128": 0.9531, "accuracy_1024": 0.9885},
    "collapsed_plain": {"accuracy_128": 0.9339, "accuracy_1024": 0.0564},
    "late_state_ce": {"accuracy_128": 0.9647, "accuracy_1024": 0.9880},
    "combined_margin": {"accuracy_128": 0.9626, "accuracy_1024": 0.9900},
}


def _write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _pearson(first, second):
    first = torch.as_tensor(first, dtype=torch.float64).flatten()
    second = torch.as_tensor(second, dtype=torch.float64).flatten()
    finite = torch.isfinite(first) & torch.isfinite(second)
    first = first[finite]
    second = second[finite]
    if len(first) < 2:
        return float("nan")
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if denominator <= 1e-20:
        return 0.0
    return float((first @ second / denominator).item())


def _r_squared(target, prediction):
    target = target.float()
    prediction = prediction.float()
    residual = (target - prediction).square().sum()
    total = (target - target.mean(0, keepdim=True)).square().sum()
    if total <= 1e-20:
        return 0.0
    return float((1 - residual / total).item())


def _midrank_percentile(values, observed):
    values = torch.as_tensor(values)
    observed = torch.as_tensor(observed, dtype=values.dtype)
    equal = torch.isclose(values, observed, rtol=1e-7, atol=1e-10)
    lower = values < observed
    lower &= ~equal
    return float((lower.double().sum() + 0.5 * equal.double().sum()) / len(values))


def _cycle_orders():
    """Return all 20,160 nine-digit cycles modulo rotation and reflection."""

    orders = []
    for tail in itertools.permutations(range(1, DIGIT_COUNT)):
        if tail[0] < tail[-1]:
            orders.append((0, *tail))
    return torch.tensor(orders, dtype=torch.long)


@dataclass
class CycleControls:
    orders: torch.Tensor
    natural_index: int
    harmonic_kernels: torch.Tensor
    ideal_rdm_normalized: torch.Tensor
    triangle_rows: torch.Tensor
    triangle_columns: torch.Tensor


def build_cycle_controls():
    orders = _cycle_orders()
    natural_matches = (orders == torch.arange(DIGIT_COUNT)).all(1).nonzero()
    natural_index = int(natural_matches[0, 0])
    positions = torch.empty_like(orders)
    position_values = torch.arange(DIGIT_COUNT).expand_as(orders)
    positions.scatter_(1, orders, position_values)
    angles = 2 * math.pi * positions.double() / DIGIT_COUNT
    code = torch.stack((angles.cos(), angles.sin()), dim=-1)
    orthonormal_code = code * math.sqrt(2.0 / DIGIT_COUNT)
    harmonic_kernels = torch.einsum(
        "oid,ojd->oij", orthonormal_code, orthonormal_code
    )
    rows, columns = torch.triu_indices(DIGIT_COUNT, DIGIT_COUNT, offset=1)
    ideal_rdm = 1 - torch.cos(angles[:, rows] - angles[:, columns])
    ideal_rdm = ideal_rdm - ideal_rdm.mean(1, keepdim=True)
    ideal_rdm_normalized = ideal_rdm / ideal_rdm.norm(dim=1, keepdim=True).clamp_min(1e-20)
    return CycleControls(
        orders=orders,
        natural_index=natural_index,
        harmonic_kernels=harmonic_kernels,
        ideal_rdm_normalized=ideal_rdm_normalized,
        triangle_rows=rows,
        triangle_columns=columns,
    )


CYCLE_CONTROLS = build_cycle_controls()


def periodic_code(order=NATURAL_ORDER, dtype=torch.float32):
    angles = torch.empty(DIGIT_COUNT, dtype=dtype)
    for position, digit in enumerate(order):
        angles[digit] = 2 * math.pi * position / DIGIT_COUNT
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def stratified_puzzle_folds(bucket_names, fold_count=5, seed=20260807):
    """Split every rating bucket across each puzzle-held-out fold."""

    by_bucket = defaultdict(list)
    for puzzle_index, bucket in enumerate(bucket_names):
        by_bucket[bucket].append(puzzle_index)
    generator = np.random.default_rng(seed)
    fold_tests = [[] for _ in range(fold_count)]
    for indices in by_bucket.values():
        indices = np.asarray(indices, dtype=np.int64)
        generator.shuffle(indices)
        for offset, puzzle_index in enumerate(indices.tolist()):
            fold_tests[offset % fold_count].append(puzzle_index)
    all_indices = set(range(len(bucket_names)))
    folds = []
    for fold_index, test_indices in enumerate(fold_tests):
        test_indices = sorted(test_indices)
        train_indices = sorted(all_indices.difference(test_indices))
        folds.append({
            "fold": fold_index,
            "train_puzzles": train_indices,
            "test_puzzles": test_indices,
        })
    return folds


def collect_cell_trajectories(model, inputs, targets, empty_mask):
    selected = set(ITERATIONS)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    states = []
    updates = []
    logits = []
    hidden = model.initial_encoder(inputs)
    feedback_predictions = torch.zeros(
        len(inputs), 81, DIGIT_COUNT, device=inputs.device
    )
    with torch.no_grad():
        for iteration in range(ITERATIONS[-1] + 1):
            current_logits = model.output_head(hidden)
            if iteration in selected:
                states.append(hidden.detach().float().cpu())
                logits.append(current_logits.detach().float().cpu())
            next_hidden = model.recurrent_step(
                hidden,
                feedback_predictions,
                rope_cos,
                rope_sin,
            )
            if iteration in selected:
                updates.append((next_hidden - hidden).detach().float().cpu())
            if iteration == ITERATIONS[-1]:
                break
            hidden = next_hidden
            feedback_predictions = F.softmax(
                model.output_head(hidden), dim=-1
            )

    states = torch.stack(states, dim=1)
    updates = torch.stack(updates, dim=1)
    logits = torch.stack(logits, dim=1)
    probabilities = logits.softmax(-1)
    top_two = probabilities.topk(2, dim=-1).values
    targets_cpu = targets.detach().cpu().long()
    true_logits = logits.gather(
        -1,
        targets_cpu[:, None, :, None].expand(-1, len(ITERATIONS), -1, 1),
    ).squeeze(-1)
    wrong_logits = logits.masked_fill(
        F.one_hot(targets_cpu, DIGIT_COUNT).bool()[:, None],
        -torch.inf,
    ).max(-1).values
    return {
        "states": states,
        "updates": updates,
        "logits": logits,
        "predicted": logits.argmax(-1),
        "confidence": probabilities.max(-1).values,
        "probability_margin": top_two[..., 0] - top_two[..., 1],
        "true_logit_margin": true_logits - wrong_logits,
        "entropy": -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1),
        "targets": targets_cpu,
        "empty_mask": empty_mask.detach().cpu().bool(),
    }


def output_digit_basis(model):
    centered = model.output_head.weight.detach().float().cpu()
    centered = centered - centered.mean(0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    rank = int((singular_values > singular_values.max() * 1e-6).sum())
    return right[:rank].T.contiguous()


def remove_basis(values, basis):
    return values - (values @ basis) @ basis.T


def build_model_arrays(collected, decoder_basis):
    states = collected["states"]
    updates = collected["updates"]
    unit_states = F.normalize(states, dim=-1, eps=1e-12)
    unit_updates = F.normalize(updates, dim=-1, eps=1e-12)
    output_null = remove_basis(unit_states, decoder_basis)
    output_null = F.normalize(output_null, dim=-1, eps=1e-12)
    puzzle_count, time_count, cell_count, _ = states.shape
    puzzle_ids = torch.arange(puzzle_count)[:, None, None].expand(
        -1, time_count, cell_count
    )
    time_indices = torch.arange(time_count)[None, :, None].expand(
        puzzle_count, -1, cell_count
    )
    iterations = torch.tensor(ITERATIONS)[None, :, None].expand(
        puzzle_count, -1, cell_count
    )
    positions = torch.arange(cell_count)[None, None, :].expand(
        puzzle_count, time_count, -1
    )
    empty = collected["empty_mask"][:, None, :].expand(
        -1, time_count, -1
    )
    targets = collected["targets"][:, None, :].expand(
        -1, time_count, -1
    )
    state_norm = states.norm(dim=-1)
    arrays = {
        "representations": {
            "raw_state": states.reshape(-1, states.size(-1)),
            "unit_state": unit_states.reshape(-1, states.size(-1)),
            "unit_update": unit_updates.reshape(-1, states.size(-1)),
            "output_null_state": output_null.reshape(-1, states.size(-1)),
        },
        "puzzle": puzzle_ids.reshape(-1),
        "time_index": time_indices.reshape(-1),
        "iteration": iterations.reshape(-1),
        "position": positions.reshape(-1),
        "row": (positions // 9).reshape(-1),
        "column": (positions % 9).reshape(-1),
        "is_empty": empty.reshape(-1),
        "true_digit": targets.reshape(-1),
        "predicted_digit": collected["predicted"].reshape(-1),
        "confidence": collected["confidence"].reshape(-1),
        "probability_margin": collected["probability_margin"].reshape(-1),
        "true_logit_margin": collected["true_logit_margin"].reshape(-1),
        "entropy": collected["entropy"].reshape(-1),
        "state_norm": state_norm.reshape(-1),
    }
    return arrays


def sample_mask(arrays, phase, scope="blank"):
    selected_iterations = torch.tensor(PHASES[phase])
    phase_mask = (arrays["iteration"][:, None] == selected_iterations).any(1)
    if scope == "blank":
        scope_mask = arrays["is_empty"]
    elif scope == "given":
        scope_mask = ~arrays["is_empty"]
    elif scope == "incorrect_blank":
        scope_mask = arrays["is_empty"] & (
            arrays["predicted_digit"] != arrays["true_digit"]
        )
    elif scope == "all":
        scope_mask = torch.ones_like(arrays["is_empty"])
    else:
        raise ValueError(f"unknown scope: {scope}")
    return phase_mask & scope_mask


def split_mask(arrays, puzzle_indices, extra_mask=None):
    puzzle_indices = torch.tensor(puzzle_indices, dtype=torch.long)
    mask = (arrays["puzzle"][:, None] == puzzle_indices).any(1)
    if extra_mask is not None:
        mask &= extra_mask
    return mask


def _continuous_design(values, train_statistics=None):
    values = values.float()
    if train_statistics is None:
        mean = values.mean(0, keepdim=True)
        scale = values.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
        train_statistics = (mean, scale)
    mean, scale = train_statistics
    return (values - mean) / scale, train_statistics


def nuisance_design(arrays, mask, statistics=None, include_scores=True):
    position = arrays["position"][mask]
    time_index = arrays["time_index"][mask]
    categorical = torch.cat((
        F.one_hot(position, 81).float(),
        F.one_hot(time_index, len(ITERATIONS)).float(),
        arrays["is_empty"][mask, None].float(),
    ), dim=1)
    continuous_values = torch.stack((
        torch.log2(arrays["iteration"][mask].float() + 1),
        torch.log1p(arrays["state_norm"][mask].float()),
        arrays["confidence"][mask].float(),
        arrays["probability_margin"][mask].float(),
        arrays["true_logit_margin"][mask].float(),
        arrays["entropy"][mask].float(),
    ), dim=1)
    if not include_scores:
        continuous_values = continuous_values[:, :2]
    continuous_statistics = None if statistics is None else statistics["continuous"]
    continuous, continuous_statistics = _continuous_design(
        continuous_values,
        continuous_statistics,
    )
    design = torch.cat((
        torch.ones(len(position), 1),
        categorical,
        continuous,
    ), dim=1)
    return design, {"continuous": continuous_statistics}


def _position_time_center(
    train_values,
    test_values,
    arrays,
    train_mask,
    test_mask,
):
    """Subtract train-only means for every cell-position × snapshot group."""

    train_groups = (
        arrays["position"][train_mask] * len(ITERATIONS)
        + arrays["time_index"][train_mask]
    )
    test_groups = (
        arrays["position"][test_mask] * len(ITERATIONS)
        + arrays["time_index"][test_mask]
    )
    group_count = 81 * len(ITERATIONS)
    means = torch.empty(group_count, train_values.size(1))
    fallback = train_values.mean(0)
    for group_index in range(group_count):
        selected = train_values[train_groups == group_index]
        means[group_index] = selected.mean(0) if len(selected) else fallback
    return train_values - means[train_groups], test_values - means[test_groups]


def continuous_score_design(
    arrays,
    mask,
    statistics=None,
    include_true_margin=False,
):
    columns = [
        torch.log1p(arrays["state_norm"][mask].float()),
        arrays["confidence"][mask].float(),
        arrays["probability_margin"][mask].float(),
        arrays["entropy"][mask].float(),
    ]
    if include_true_margin:
        columns.append(arrays["true_logit_margin"][mask].float())
    values = torch.stack(columns, dim=1)
    continuous_statistics = None if statistics is None else statistics["continuous"]
    standardized, continuous_statistics = _continuous_design(
        values,
        continuous_statistics,
    )
    return torch.cat((torch.ones(len(values), 1), standardized), dim=1), {
        "continuous": continuous_statistics,
    }


def residualize_train_test(
    train_values,
    test_values,
    train_design,
    test_design,
    ridge=1e-3,
):
    train_values = train_values.float()
    test_values = test_values.float()
    train_design = train_design.float()
    test_design = test_design.float()
    covariance = train_design.T @ train_design / len(train_design)
    cross_covariance = train_design.T @ train_values / len(train_design)
    penalty = torch.eye(covariance.size(0)) * ridge
    penalty[0, 0] = ridge * 1e-3
    coefficients = torch.linalg.solve(covariance + penalty, cross_covariance)
    return (
        train_values - train_design @ coefficients,
        test_values - test_design @ coefficients,
    )


@dataclass
class RidgeModel:
    mean: torch.Tensor
    scale: torch.Tensor
    weights: torch.Tensor
    target_mean: torch.Tensor

    def predict(self, values):
        standardized = (values.float() - self.mean) / self.scale
        return standardized @ self.weights + self.target_mean


def fit_ridge(values, targets, ridge=1e-2):
    values = values.float()
    targets = targets.float()
    mean = values.mean(0, keepdim=True)
    scale = values.std(0, unbiased=False, keepdim=True).clamp_min(1e-5)
    standardized = (values - mean) / scale
    target_mean = targets.mean(0, keepdim=True)
    centered_targets = targets - target_mean
    covariance = standardized.T @ standardized / len(standardized)
    cross_covariance = standardized.T @ centered_targets / len(standardized)
    weights = torch.linalg.solve(
        covariance + ridge * torch.eye(covariance.size(0)),
        cross_covariance,
    )
    return RidgeModel(mean, scale, weights, target_mean)


def _balanced_accuracy(targets, predictions, class_count):
    recalls = []
    for class_index in range(class_count):
        selected = targets == class_index
        if selected.any():
            recalls.append((predictions[selected] == class_index).float().mean())
    return float(torch.stack(recalls).mean()) if recalls else float("nan")


def probe_metrics(train_values, test_values, arrays, train_mask, test_mask):
    rows = []
    scalar_targets = {
        "iteration_linear": arrays["iteration"].float(),
        "iteration_log2": torch.log2(arrays["iteration"].float() + 1),
        "confidence": arrays["confidence"].float(),
        "probability_margin": arrays["probability_margin"].float(),
        "true_logit_margin": arrays["true_logit_margin"].float(),
        "entropy": arrays["entropy"].float(),
        "state_norm_log": torch.log1p(arrays["state_norm"].float()),
    }
    for name, target in scalar_targets.items():
        train_target = target[train_mask, None]
        test_target = target[test_mask, None]
        fitted = fit_ridge(train_values, train_target)
        prediction = fitted.predict(test_values)
        rows.append({
            "target": name,
            "metric": "r2",
            "value": _r_squared(test_target, prediction),
        })

    categorical_targets = {
        "true_digit": (arrays["true_digit"], DIGIT_COUNT),
        "predicted_digit": (arrays["predicted_digit"], DIGIT_COUNT),
        "row": (arrays["row"], 9),
        "column": (arrays["column"], 9),
        "cell_position": (arrays["position"], 81),
    }
    for name, (target, class_count) in categorical_targets.items():
        train_target = target[train_mask]
        test_target = target[test_mask]
        fitted = fit_ridge(
            train_values,
            F.one_hot(train_target, class_count).float(),
        )
        scores = fitted.predict(test_values)
        prediction = scores.argmax(1)
        rows.append({
            "target": name,
            "metric": "balanced_accuracy",
            "value": _balanced_accuracy(test_target, prediction, class_count),
        })
        rows.append({
            "target": name,
            "metric": "r2",
            "value": _r_squared(
                F.one_hot(test_target, class_count).float(),
                scores,
            ),
        })
    return rows


def position_only_digit_probe(arrays, train_mask, test_mask, label_key):
    train_position = arrays["position"][train_mask]
    test_position = arrays["position"][test_mask]
    train_time = arrays["time_index"][train_mask]
    test_time = arrays["time_index"][test_mask]
    train_group = train_position * len(ITERATIONS) + train_time
    test_group = test_position * len(ITERATIONS) + test_time
    train_labels = arrays[label_key][train_mask]
    test_labels = arrays[label_key][test_mask]
    group_count = 81 * len(ITERATIONS)
    global_distribution = F.one_hot(
        train_labels, DIGIT_COUNT
    ).float().mean(0)
    group_scores = torch.empty(group_count, DIGIT_COUNT)
    for group_index in range(group_count):
        selected = train_labels[train_group == group_index]
        if len(selected):
            counts = torch.bincount(selected, minlength=DIGIT_COUNT).float()
            # One global-distribution pseudo-count prevents empty digit scores.
            group_scores[group_index] = (
                counts + global_distribution
            ) / (len(selected) + 1)
        else:
            group_scores[group_index] = global_distribution
    scores = group_scores[test_group]
    return {
        "balanced_accuracy": _balanced_accuracy(
            test_labels,
            scores.argmax(1),
            DIGIT_COUNT,
        ),
        "r2": _r_squared(F.one_hot(test_labels, DIGIT_COUNT).float(), scores),
    }


def _puzzle_equal_centroids(values, labels, puzzle_ids):
    puzzle_centroids = []
    for puzzle_index in torch.unique(puzzle_ids).tolist():
        puzzle_mask = puzzle_ids == puzzle_index
        digit_centroids = []
        for digit in range(DIGIT_COUNT):
            selected = values[puzzle_mask & (labels == digit)]
            if len(selected):
                digit_centroids.append(selected.mean(0))
            else:
                digit_centroids.append(torch.full(
                    (values.size(1),), float("nan")
                ))
        puzzle_centroids.append(torch.stack(digit_centroids))
    stacked = torch.stack(puzzle_centroids)
    valid = torch.isfinite(stacked[..., 0])
    counts = valid.sum(0)
    summed = torch.nan_to_num(stacked).sum(0)
    centroids = summed / counts[:, None].clamp_min(1)
    if (counts == 0).any():
        return None, counts
    return centroids, counts


def _centroid_spectrum(centroids):
    centered = centroids.double() - centroids.double().mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square()
    fractions = eigenvalues / eigenvalues.sum().clamp_min(1e-20)
    effective_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-20)
    distances = torch.cdist(centroids.double(), centroids.double())
    rows, columns = torch.triu_indices(DIGIT_COUNT, DIGIT_COUNT, offset=1)
    pairwise = distances[rows, columns]
    return {
        "top2_fraction": float(fractions[:2].sum()),
        "top3_fraction": float(fractions[:3].sum()),
        "effective_rank": float(effective_rank),
        "pairwise_distance_cv": float(
            pairwise.std(unbiased=False) / pairwise.mean().clamp_min(1e-20)
        ),
        "eigenvalue_fractions": fractions.tolist(),
    }


def intrinsic_cycle_metrics(
    train_values,
    train_labels,
    train_puzzles,
    test_values,
    test_labels,
    test_puzzles,
):
    train_centroids, train_counts = _puzzle_equal_centroids(
        train_values, train_labels, train_puzzles
    )
    test_centroids, test_counts = _puzzle_equal_centroids(
        test_values, test_labels, test_puzzles
    )
    if train_centroids is None or test_centroids is None:
        return None
    train_centroids = train_centroids.double()
    test_centroids = test_centroids.double()
    train_centroids -= train_centroids.mean(0, keepdim=True)
    test_centroids -= test_centroids.mean(0, keepdim=True)
    cross_gram = train_centroids @ test_centroids.T
    total_cross_energy = torch.diagonal(cross_gram).sum()
    harmonic_energies = torch.einsum(
        "oij,ij->o",
        CYCLE_CONTROLS.harmonic_kernels,
        cross_gram,
    )
    natural_energy = harmonic_energies[CYCLE_CONTROLS.natural_index]
    test_gram = test_centroids @ test_centroids.T
    natural_test_energy = (
        CYCLE_CONTROLS.harmonic_kernels[CYCLE_CONTROLS.natural_index]
        * test_gram
    ).sum()
    total_test_energy = torch.diagonal(test_gram).sum()

    rows = CYCLE_CONTROLS.triangle_rows
    columns = CYCLE_CONTROLS.triangle_columns
    train_differences = train_centroids[rows] - train_centroids[columns]
    test_differences = test_centroids[rows] - test_centroids[columns]
    cross_distances = (train_differences * test_differences).sum(1)
    normalized_cross_distances = cross_distances - cross_distances.mean()
    normalized_cross_distances /= normalized_cross_distances.norm().clamp_min(1e-20)
    order_rdm_correlations = (
        CYCLE_CONTROLS.ideal_rdm_normalized @ normalized_cross_distances
    )
    natural_rdm = order_rdm_correlations[CYCLE_CONTROLS.natural_index]

    train_distances = torch.cdist(train_centroids, train_centroids)
    test_distances = torch.cdist(test_centroids, test_centroids)
    orders = CYCLE_CONTROLS.orders
    following = orders.roll(-1, dims=1)
    train_cycle_lengths = train_distances[orders, following].sum(1)
    test_cycle_lengths = test_distances[orders, following].sum(1)
    natural_length = test_cycle_lengths[CYCLE_CONTROLS.natural_index]
    selected_order_index = int(train_cycle_lengths.argmin())
    selected_test_length = test_cycle_lengths[selected_order_index]
    train_pairwise = train_distances[rows, columns]
    test_pairwise = test_distances[rows, columns]
    denominator = total_cross_energy
    crossvalidated_harmonic_fraction = (
        natural_energy / denominator
        if denominator.abs() > 1e-20 else torch.tensor(float("nan"))
    )
    return {
        "train_min_digit_count": int(train_counts.min()),
        "test_min_digit_count": int(test_counts.min()),
        "total_crossvalidated_digit_energy": float(total_cross_energy),
        "natural_first_harmonic_fraction": float(
            natural_test_energy / total_test_energy.clamp_min(1e-20)
        ),
        "crossvalidated_natural_first_harmonic_fraction": float(
            crossvalidated_harmonic_fraction
        ),
        "natural_harmonic_energy_percentile": float(
            _midrank_percentile(harmonic_energies, natural_energy)
        ),
        "natural_rdm_correlation": float(natural_rdm),
        "natural_rdm_correlation_percentile": float(
            _midrank_percentile(order_rdm_correlations, natural_rdm)
        ),
        "natural_cycle_shorter_percentile": float(
            _midrank_percentile(test_cycle_lengths, natural_length)
        ),
        "train_shortest_cycle_test_shorter_percentile": float(
            _midrank_percentile(test_cycle_lengths, selected_test_length)
        ),
        "train_shortest_cycle": "-".join(
            str(int(digit) + 1) for digit in orders[selected_order_index]
        ),
        "train_test_distance_correlation": _pearson(
            train_pairwise,
            test_pairwise,
        ),
        **{
            f"test_centroid_{key}": value
            for key, value in _centroid_spectrum(test_centroids).items()
            if key != "eigenvalue_fractions"
        },
        "test_centroid_eigenvalue_fractions": _centroid_spectrum(
            test_centroids
        )["eigenvalue_fractions"],
        "train_centroids": train_centroids.float(),
        "test_centroids": test_centroids.float(),
        "harmonic_energy_null": harmonic_energies.float(),
    }


def _orthonormalize(matrix):
    if matrix.size(1) != 2:
        raise ValueError("basis matrix must have two columns")
    basis, _ = torch.linalg.qr(matrix.float(), mode="reduced")
    return basis


def _procrustes_phase_alignment(
    train_centroids,
    test_centroids,
    basis,
    code=None,
):
    if code is None:
        code = periodic_code(dtype=torch.float32)
    train_points = train_centroids.float() @ basis.float()
    test_points = test_centroids.float() @ basis.float()
    train_mean = train_points.mean(0, keepdim=True)
    centered_train = train_points - train_mean
    centered_code = code - code.mean(0, keepdim=True)
    left, singular_values, right = torch.linalg.svd(
        centered_train.T @ centered_code,
        full_matrices=False,
    )
    rotation = left @ right
    scale = singular_values.sum() / centered_train.square().sum().clamp_min(1e-20)
    predicted = (test_points - train_mean) @ rotation * scale
    predicted_unit = F.normalize(predicted, dim=1, eps=1e-12)
    phase_cosine = (predicted_unit * code).sum(1).mean()
    angle_error = torch.atan2(
        predicted_unit[:, 1] * code[:, 0] - predicted_unit[:, 0] * code[:, 1],
        (predicted_unit * code).sum(1),
    ).abs()
    return {
        "phase_cosine": float(phase_cosine),
        "angular_mae_degrees": float(angle_error.mean() * 180 / math.pi),
    }


def random_basis_controls(
    train_values,
    train_labels,
    train_puzzles,
    test_values,
    test_labels,
    test_puzzles,
    output_weight,
    seed,
    random_basis_count=128,
):
    train_centroids, _ = _puzzle_equal_centroids(
        train_values, train_labels, train_puzzles
    )
    test_centroids, _ = _puzzle_equal_centroids(
        test_values, test_labels, test_puzzles
    )
    if train_centroids is None or test_centroids is None:
        return []
    train_centroids -= train_centroids.mean(0, keepdim=True)
    test_centroids -= test_centroids.mean(0, keepdim=True)
    code = periodic_code()
    rows = []

    supervised_basis = _orthonormalize(train_centroids.T @ code)
    rows.append({
        "basis": "train_fitted_natural_fourier",
        **_procrustes_phase_alignment(
            train_centroids,
            test_centroids,
            supervised_basis,
            code,
        ),
    })
    permuted_order = (0, 5, 2, 8, 3, 7, 1, 6, 4)
    permuted_code = periodic_code(permuted_order)
    shuffled_basis = _orthonormalize(train_centroids.T @ permuted_code)
    rows.append({
        "basis": "train_fitted_fixed_random_order",
        "order": "-".join(str(digit + 1) for digit in permuted_order),
        **_procrustes_phase_alignment(
            train_centroids,
            test_centroids,
            shuffled_basis,
            permuted_code,
        ),
    })

    centered_train = train_values.float() - train_values.float().mean(0, keepdim=True)
    covariance = centered_train.T @ centered_train / len(centered_train)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    pca_basis = eigenvectors[:, torch.argsort(eigenvalues, descending=True)[:2]]
    rows.append({
        "basis": "train_sample_pca",
        **_procrustes_phase_alignment(
            train_centroids,
            test_centroids,
            pca_basis,
            code,
        ),
    })

    output_weight = output_weight.float()
    output_weight -= output_weight.mean(0, keepdim=True)
    output_basis = _orthonormalize(output_weight.T @ code)
    rows.append({
        "basis": "output_head_natural_fourier",
        **_procrustes_phase_alignment(
            train_centroids,
            test_centroids,
            output_basis,
            code,
        ),
    })

    generator = torch.Generator().manual_seed(seed)
    for basis_index in range(random_basis_count):
        basis = _orthonormalize(torch.randn(
            train_values.size(1), 2, generator=generator
        ))
        rows.append({
            "basis": "haar_random",
            "basis_index": basis_index,
            **_procrustes_phase_alignment(
                train_centroids,
                test_centroids,
                basis,
                code,
            ),
        })
    return rows


def _trajectory_label_mapping(arrays, mask):
    puzzle = arrays["puzzle"][mask]
    position = arrays["position"][mask]
    labels = arrays["true_digit"][mask]
    keys = puzzle * 81 + position
    unique_keys, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    trajectory_labels = torch.empty(len(unique_keys), dtype=torch.long)
    for key_index in range(len(unique_keys)):
        trajectory_labels[key_index] = labels[(inverse == key_index).nonzero()[0, 0]]
    return puzzle, position, unique_keys, inverse, trajectory_labels


def shuffled_trajectory_labels(arrays, mask, kind, seed):
    puzzle, position, _, inverse, trajectory_labels = _trajectory_label_mapping(
        arrays, mask
    )
    generator = torch.Generator().manual_seed(seed)
    shuffled = trajectory_labels.clone()
    if kind == "puzzle_digit_renaming":
        unique_puzzles = torch.unique(puzzle)
        for puzzle_index in unique_puzzles.tolist():
            permutation = torch.randperm(DIGIT_COUNT, generator=generator)
            trajectory_indices = torch.unique(inverse[puzzle == puzzle_index])
            shuffled[trajectory_indices] = permutation[
                trajectory_labels[trajectory_indices]
            ]
    elif kind == "position_blocked_shuffle":
        trajectory_positions = torch.empty_like(trajectory_labels)
        for index in range(len(trajectory_labels)):
            trajectory_positions[index] = position[(inverse == index).nonzero()[0, 0]]
        for cell_position in range(81):
            chosen = (trajectory_positions == cell_position).nonzero().flatten()
            if len(chosen) > 1:
                order = torch.randperm(len(chosen), generator=generator)
                shuffled[chosen] = trajectory_labels[chosen[order]]
    elif kind == "random_fixed_trajectory_labels":
        # Preserve the exact split-level class counts while assigning a random
        # label that stays fixed for every snapshot of a puzzle-cell strand.
        shuffled = trajectory_labels[
            torch.randperm(len(trajectory_labels), generator=generator)
        ]
    else:
        raise ValueError(f"unknown label null: {kind}")
    return shuffled[inverse]


def weight_geometry(model, model_name):
    matrices = {
        "initial_encoder": model.initial_encoder.weight.detach().float().cpu()[:, 1:10].T,
        "prediction_feedback": model.pred_proj.weight.detach().float().cpu().T,
        "output_head": model.output_head.weight.detach().float().cpu(),
    }
    rows = []
    centered = {}
    for name, matrix in matrices.items():
        centered[name] = matrix - matrix.mean(0, keepdim=True)
        spectrum = _centroid_spectrum(matrix)
        cross_gram = centered[name].double() @ centered[name].double().T
        harmonic_energies = torch.einsum(
            "oij,ij->o",
            CYCLE_CONTROLS.harmonic_kernels,
            cross_gram,
        )
        natural = harmonic_energies[CYCLE_CONTROLS.natural_index]
        rows.append({
            "model": model_name,
            "matrix": name,
            "top2_fraction": spectrum["top2_fraction"],
            "top3_fraction": spectrum["top3_fraction"],
            "effective_rank": spectrum["effective_rank"],
            "pairwise_distance_cv": spectrum["pairwise_distance_cv"],
            "natural_harmonic_fraction": float(
                natural / torch.diagonal(cross_gram).sum().clamp_min(1e-20)
            ),
            "natural_harmonic_energy_percentile": float(
                _midrank_percentile(harmonic_energies, natural)
            ),
            "eigenvalue_fractions": spectrum["eigenvalue_fractions"],
        })
    for first_name, second_name in itertools.combinations(matrices, 2):
        first_gram = centered[first_name] @ centered[first_name].T
        second_gram = centered[second_name] @ centered[second_name].T
        cka = (first_gram * second_gram).sum() / (
            first_gram.norm() * second_gram.norm()
        ).clamp_min(1e-20)
        rows.append({
            "model": model_name,
            "matrix": f"cka:{first_name}:{second_name}",
            "cka": float(cka),
        })
    return rows


def _public_metrics(metrics):
    if metrics is None:
        return None
    excluded = {
        "train_centroids",
        "test_centroids",
        "harmonic_energy_null",
        "test_centroid_eigenvalue_fractions",
    }
    return {
        key: _safe_float(value) if isinstance(value, (float, int)) else value
        for key, value in metrics.items()
        if key not in excluded
    }


def natural_centroid_metrics(
    train_values,
    train_labels,
    train_puzzles,
    test_values,
    test_labels,
    test_puzzles,
):
    train_centroids, _ = _puzzle_equal_centroids(
        train_values, train_labels, train_puzzles
    )
    test_centroids, _ = _puzzle_equal_centroids(
        test_values, test_labels, test_puzzles
    )
    if train_centroids is None or test_centroids is None:
        return None
    train_centroids = train_centroids.double()
    test_centroids = test_centroids.double()
    train_centroids -= train_centroids.mean(0, keepdim=True)
    test_centroids -= test_centroids.mean(0, keepdim=True)
    cross_gram = train_centroids @ test_centroids.T
    natural_kernel = CYCLE_CONTROLS.harmonic_kernels[
        CYCLE_CONTROLS.natural_index
    ]
    harmonic_energy = (natural_kernel * cross_gram).sum()
    total_energy = torch.diagonal(cross_gram).sum()
    test_gram = test_centroids @ test_centroids.T
    test_harmonic_energy = (natural_kernel * test_gram).sum()
    test_total_energy = torch.diagonal(test_gram).sum()
    rows = CYCLE_CONTROLS.triangle_rows
    columns = CYCLE_CONTROLS.triangle_columns
    train_distances = torch.cdist(train_centroids, train_centroids)[rows, columns]
    test_distances = torch.cdist(test_centroids, test_centroids)[rows, columns]
    train_difference = train_centroids[rows] - train_centroids[columns]
    test_difference = test_centroids[rows] - test_centroids[columns]
    cross_distance = (train_difference * test_difference).sum(1)
    natural_ideal = CYCLE_CONTROLS.ideal_rdm_normalized[
        CYCLE_CONTROLS.natural_index
    ]
    cross_distance -= cross_distance.mean()
    cross_distance /= cross_distance.norm().clamp_min(1e-20)
    return {
        "natural_first_harmonic_fraction": float(
            test_harmonic_energy / test_total_energy.clamp_min(1e-20)
        ),
        "crossvalidated_natural_first_harmonic_fraction": float(
            harmonic_energy / total_energy
        ) if total_energy.abs() > 1e-20 else float("nan"),
        "natural_rdm_correlation": float(natural_ideal @ cross_distance),
        "train_test_distance_correlation": _pearson(
            train_distances, test_distances
        ),
    }


def _representation_values(
    representation,
    arrays,
    train_mask,
    test_mask,
    residual_cache,
):
    if representation in arrays["representations"]:
        return (
            arrays["representations"][representation][train_mask],
            arrays["representations"][representation][test_mask],
        )
    cached = residual_cache[representation]
    train_indices = cached["train_indices"]
    test_indices = cached["test_indices"]
    train_selected = train_mask[train_indices]
    test_selected = test_mask[test_indices]
    return (
        cached["train_values"][train_selected],
        cached["test_values"][test_selected],
    )


def _build_residual_cache(arrays, train_base_mask, test_base_mask):
    train_indices = train_base_mask.nonzero().flatten()
    test_indices = test_base_mask.nonzero().flatten()
    train_unit = arrays["representations"]["unit_state"][train_indices]
    test_unit = arrays["representations"]["unit_state"][test_indices]
    position_train, position_test = _position_time_center(
        train_unit,
        test_unit,
        arrays,
        train_base_mask,
        test_base_mask,
    )
    cache = {
        "position_time_residual": {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_values": position_train,
            "test_values": position_test,
        }
    }
    for name, include_true_margin in (
        ("full_confound_residual", False),
        ("supervised_true_margin_residual", True),
    ):
        train_design, statistics = continuous_score_design(
            arrays,
            train_base_mask,
            include_true_margin=include_true_margin,
        )
        test_design, _ = continuous_score_design(
            arrays,
            test_base_mask,
            statistics=statistics,
            include_true_margin=include_true_margin,
        )
        train_residual, test_residual = residualize_train_test(
            position_train,
            position_test,
            train_design,
            test_design,
        )
        cache[name] = {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_values": train_residual,
            "test_values": test_residual,
        }
    return cache


def analyze_model(
    model_name,
    model,
    collected,
    bucket_names,
    fold_count,
    seed,
    random_basis_count,
    label_null_repetitions,
):
    decoder_basis = output_digit_basis(model)
    output_weight = model.output_head.weight.detach().float().cpu()
    arrays = build_model_arrays(collected, decoder_basis)
    folds = stratified_puzzle_folds(bucket_names, fold_count, seed)
    intrinsic_rows = []
    probe_rows = []
    basis_rows = []
    label_null_rows = []
    position_transfer_rows = []
    strata_rows = []
    centroid_plot_data = {}
    projection_plot_data = None

    for fold in folds:
        fold_index = fold["fold"]
        primary_phase_mask = sample_mask(arrays, "all_1_1024", "blank")
        train_base_mask = split_mask(
            arrays, fold["train_puzzles"], primary_phase_mask
        )
        test_base_mask = split_mask(
            arrays, fold["test_puzzles"], primary_phase_mask
        )
        residual_cache = _build_residual_cache(
            arrays, train_base_mask, test_base_mask
        )

        condition_specs = []
        for phase in PHASES:
            for label_key in ("true_digit", "predicted_digit"):
                condition_specs.append((
                    "unit_state", label_key, phase, "blank"
                ))
        for representation in (
            "raw_state",
            "unit_update",
            "output_null_state",
            "position_time_residual",
            "full_confound_residual",
            "supervised_true_margin_residual",
        ):
            for label_key in ("true_digit", "predicted_digit"):
                condition_specs.append((
                    representation,
                    label_key,
                    "all_1_1024",
                    "blank",
                ))
        condition_specs.extend((
            ("unit_state", "true_digit", "all_1_1024", "given"),
            ("unit_state", "true_digit", "early_1_16", "incorrect_blank"),
            ("unit_state", "predicted_digit", "early_1_16", "incorrect_blank"),
        ))

        for representation, label_key, phase, scope in condition_specs:
            condition_mask = sample_mask(arrays, phase, scope)
            train_mask = split_mask(
                arrays, fold["train_puzzles"], condition_mask
            )
            test_mask = split_mask(
                arrays, fold["test_puzzles"], condition_mask
            )
            train_values, test_values = _representation_values(
                representation,
                arrays,
                train_mask,
                test_mask,
                residual_cache,
            )
            train_labels = arrays[label_key][train_mask]
            test_labels = arrays[label_key][test_mask]
            train_puzzles = arrays["puzzle"][train_mask]
            test_puzzles = arrays["puzzle"][test_mask]
            if not len(train_values) or not len(test_values):
                continue
            metrics = intrinsic_cycle_metrics(
                train_values,
                train_labels,
                train_puzzles,
                test_values,
                test_labels,
                test_puzzles,
            )
            if metrics is None:
                continue
            intrinsic_rows.append({
                "model": model_name,
                "fold": fold_index,
                "representation": representation,
                "label": label_key,
                "phase": phase,
                "scope": scope,
                "train_samples": len(train_values),
                "test_samples": len(test_values),
                **_public_metrics(metrics),
            })
            if (
                fold_index == 0
                and phase == "all_1_1024"
                and scope == "blank"
                and label_key == "true_digit"
                and representation in {
                    "unit_state",
                    "position_time_residual",
                    "full_confound_residual",
                    "output_null_state",
                }
            ):
                centroid_plot_data[representation] = {
                    "train": metrics["train_centroids"],
                    "test": metrics["test_centroids"],
                }

        for representation in (
            "raw_state",
            "unit_state",
            "output_null_state",
            "position_time_residual",
            "full_confound_residual",
            "supervised_true_margin_residual",
        ):
            train_values, test_values = _representation_values(
                representation,
                arrays,
                train_base_mask,
                test_base_mask,
                residual_cache,
            )
            for row in probe_metrics(
                train_values,
                test_values,
                arrays,
                train_base_mask,
                test_base_mask,
            ):
                probe_rows.append({
                    "model": model_name,
                    "fold": fold_index,
                    "representation": representation,
                    **row,
                })

        for label_key in ("true_digit", "predicted_digit"):
            position_only = position_only_digit_probe(
                arrays,
                train_base_mask,
                test_base_mask,
                label_key,
            )
            for metric_name, value in position_only.items():
                probe_rows.append({
                    "model": model_name,
                    "fold": fold_index,
                    "representation": "position_and_iteration_only",
                    "target": label_key,
                    "metric": metric_name,
                    "value": value,
                })

        train_residual = residual_cache["full_confound_residual"]["train_values"]
        test_residual = residual_cache["full_confound_residual"]["test_values"]
        fold_basis_rows = random_basis_controls(
            train_residual,
            arrays["true_digit"][train_base_mask],
            arrays["puzzle"][train_base_mask],
            test_residual,
            arrays["true_digit"][test_base_mask],
            arrays["puzzle"][test_base_mask],
            output_weight,
            seed + 1000 * fold_index,
            random_basis_count,
        )
        for row in fold_basis_rows:
            basis_rows.append({
                "model": model_name,
                "fold": fold_index,
                "representation": "full_confound_residual",
                **row,
            })

        actual_null_metrics = natural_centroid_metrics(
            train_residual,
            arrays["true_digit"][train_base_mask],
            arrays["puzzle"][train_base_mask],
            test_residual,
            arrays["true_digit"][test_base_mask],
            arrays["puzzle"][test_base_mask],
        )
        label_null_rows.append({
            "model": model_name,
            "fold": fold_index,
            "null": "actual_labels",
            "repetition": 0,
            **actual_null_metrics,
        })
        for null_kind in (
            "puzzle_digit_renaming",
            "position_blocked_shuffle",
            "random_fixed_trajectory_labels",
        ):
            for repetition in range(label_null_repetitions):
                train_null = shuffled_trajectory_labels(
                    arrays,
                    train_base_mask,
                    null_kind,
                    seed + fold_index * 10000 + repetition * 2,
                )
                test_null = shuffled_trajectory_labels(
                    arrays,
                    test_base_mask,
                    null_kind,
                    seed + fold_index * 10000 + repetition * 2 + 1,
                )
                metrics = natural_centroid_metrics(
                    train_residual,
                    train_null,
                    arrays["puzzle"][train_base_mask],
                    test_residual,
                    test_null,
                    arrays["puzzle"][test_base_mask],
                )
                if metrics is not None:
                    label_null_rows.append({
                        "model": model_name,
                        "fold": fold_index,
                        "null": null_kind,
                        "repetition": repetition,
                        **metrics,
                    })

        for variable in ("confidence", "probability_margin"):
            for bin_index in range(4):
                train_bin = train_base_mask.clone()
                test_bin = test_base_mask.clone()
                train_bin[:] = False
                test_bin[:] = False
                used_iterations = 0
                for time_index in range(1, len(ITERATIONS)):
                    train_time_mask = train_base_mask & (
                        arrays["time_index"] == time_index
                    )
                    test_time_mask = test_base_mask & (
                        arrays["time_index"] == time_index
                    )
                    train_indices = train_time_mask.nonzero().flatten()
                    test_indices = test_time_mask.nonzero().flatten()
                    if not len(train_indices) or not len(test_indices):
                        continue
                    train_scores = arrays[variable][train_indices]
                    test_scores = arrays[variable][test_indices]
                    quantiles = torch.quantile(
                        train_scores.float(),
                        torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0)),
                    )
                    lower = quantiles[bin_index]
                    upper = quantiles[bin_index + 1]
                    test_lower = -torch.inf if bin_index == 0 else lower
                    test_upper = torch.inf if bin_index == 3 else upper
                    if bin_index == 3:
                        train_local = (train_scores >= lower) & (train_scores <= upper)
                        test_local = (
                            (test_scores >= test_lower)
                            & (test_scores <= test_upper)
                        )
                    else:
                        train_local = (train_scores >= lower) & (train_scores < upper)
                        test_local = (
                            (test_scores >= test_lower)
                            & (test_scores < test_upper)
                        )
                    train_bin[train_indices[train_local]] = True
                    test_bin[test_indices[test_local]] = True
                    used_iterations += int(train_local.any())
                train_values = arrays["representations"]["unit_state"][train_bin]
                test_values = arrays["representations"]["unit_state"][test_bin]
                if not len(train_values) or not len(test_values):
                    continue
                for label_key in ("true_digit", "predicted_digit"):
                    if (
                        torch.unique(arrays[label_key][train_bin]).numel()
                        < DIGIT_COUNT
                        or torch.unique(arrays[label_key][test_bin]).numel()
                        < DIGIT_COUNT
                    ):
                        continue
                    metrics = intrinsic_cycle_metrics(
                        train_values,
                        arrays[label_key][train_bin],
                        arrays["puzzle"][train_bin],
                        test_values,
                        arrays[label_key][test_bin],
                        arrays["puzzle"][test_bin],
                    )
                    if metrics is not None:
                        strata_rows.append({
                            "model": model_name,
                            "fold": fold_index,
                            "variable": variable,
                            "bin": bin_index + 1,
                            "label": label_key,
                            "binning": "train quartiles within each iteration",
                            "iterations_with_nonempty_training_bin": used_iterations,
                            "test_samples": int(test_bin.sum()),
                            **_public_metrics(metrics),
                        })

        if fold_index == 0:
            primary_unit_train = arrays["representations"]["unit_state"][train_base_mask]
            primary_unit_test = arrays["representations"]["unit_state"][test_base_mask]
            primary_labels_train = arrays["true_digit"][train_base_mask]
            primary_labels_test = arrays["true_digit"][test_base_mask]
            primary_puzzles_train = arrays["puzzle"][train_base_mask]
            primary_puzzles_test = arrays["puzzle"][test_base_mask]
            train_centroids, _ = _puzzle_equal_centroids(
                primary_unit_train,
                primary_labels_train,
                primary_puzzles_train,
            )
            test_centroids, _ = _puzzle_equal_centroids(
                primary_unit_test,
                primary_labels_test,
                primary_puzzles_test,
            )
            natural_code = periodic_code()
            random_order = (0, 5, 2, 8, 3, 7, 1, 6, 4)
            random_code = periodic_code(random_order)
            natural_basis = _orthonormalize(train_centroids.T @ natural_code)
            random_order_basis = _orthonormalize(train_centroids.T @ random_code)
            covariance = (
                (primary_unit_train - primary_unit_train.mean(0)).T
                @ (primary_unit_train - primary_unit_train.mean(0))
                / len(primary_unit_train)
            )
            _, eigenvectors = torch.linalg.eigh(covariance)
            pca_basis = eigenvectors[:, -2:]
            generator = torch.Generator().manual_seed(seed)
            fixed_random_basis = _orthonormalize(torch.randn(
                primary_unit_train.size(1), 2, generator=generator
            ))
            projection_plot_data = {
                "natural": (
                    train_centroids @ natural_basis,
                    test_centroids @ natural_basis,
                    natural_code,
                ),
                "random_order": (
                    train_centroids @ random_order_basis,
                    test_centroids @ random_order_basis,
                    random_code,
                ),
                "pca": (
                    train_centroids @ pca_basis,
                    test_centroids @ pca_basis,
                    natural_code,
                ),
                "random_basis": (
                    train_centroids @ fixed_random_basis,
                    test_centroids @ fixed_random_basis,
                    natural_code,
                ),
            }

    primary_phase_mask = sample_mask(arrays, "all_1_1024", "blank")
    position_group = (arrays["row"] + 2 * arrays["column"]) % 3
    for group_index in range(3):
        for fold in folds:
            train_mask = split_mask(
                arrays,
                fold["train_puzzles"],
                primary_phase_mask & (position_group != group_index),
            )
            test_mask = split_mask(
                arrays,
                fold["test_puzzles"],
                primary_phase_mask & (position_group == group_index),
            )
            metrics = intrinsic_cycle_metrics(
                arrays["representations"]["unit_state"][train_mask],
                arrays["true_digit"][train_mask],
                arrays["puzzle"][train_mask],
                arrays["representations"]["unit_state"][test_mask],
                arrays["true_digit"][test_mask],
                arrays["puzzle"][test_mask],
            )
            if metrics is not None:
                position_transfer_rows.append({
                    "model": model_name,
                    "position_group": group_index,
                    "train_puzzle_fold": fold["fold"],
                    "train_positions": "other_two_groups",
                    "test_positions": "heldout_group",
                    **_public_metrics(metrics),
                })

    return {
        "intrinsic_rows": intrinsic_rows,
        "probe_rows": probe_rows,
        "basis_rows": basis_rows,
        "label_null_rows": label_null_rows,
        "position_transfer_rows": position_transfer_rows,
        "strata_rows": strata_rows,
        "centroid_plot_data": centroid_plot_data,
        "projection_plot_data": projection_plot_data,
        "folds": folds,
        "arrays": arrays,
    }


def _group_values(rows, filters, value_key):
    values = []
    for row in rows:
        if all(row.get(key) == value for key, value in filters.items()):
            value = row.get(value_key)
            if value is not None and math.isfinite(float(value)):
                values.append(float(value))
    return np.asarray(values, dtype=float)


def _mean_error(values):
    if not len(values):
        return float("nan"), 0.0
    # Cross-validation training folds overlap, so this is descriptive fold SD,
    # not a standard error or confidence interval.
    return float(values.mean()), float(values.std(ddof=0))


def plot_natural_order(intrinsic_rows, output_dir):
    models = list(MODEL_HEALTH)
    phases = list(PHASES)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    x = np.arange(len(phases))
    for model_index, model in enumerate(models):
        filters = {
            "model": model,
            "representation": "unit_state",
            "label": "true_digit",
            "scope": "blank",
        }
        fraction_means = []
        fraction_errors = []
        energy_percentiles = []
        rdm_percentiles = []
        for phase in phases:
            phase_filters = {**filters, "phase": phase}
            mean, error = _mean_error(_group_values(
                intrinsic_rows,
                phase_filters,
                "natural_first_harmonic_fraction",
            ))
            fraction_means.append(mean)
            fraction_errors.append(error)
            energy_percentiles.append(_mean_error(_group_values(
                intrinsic_rows,
                phase_filters,
                "natural_harmonic_energy_percentile",
            ))[0])
            rdm_percentiles.append(_mean_error(_group_values(
                intrinsic_rows,
                phase_filters,
                "natural_rdm_correlation_percentile",
            ))[0])
        axes[0].errorbar(
            x,
            fraction_means,
            yerr=fraction_errors,
            marker="o",
            label=model,
            color=colors[model_index],
        )
        axes[1].plot(
            x,
            energy_percentiles,
            marker="o",
            label=model,
            color=colors[model_index],
        )
        axes[2].plot(
            x,
            rdm_percentiles,
            marker="o",
            label=model,
            color=colors[model_index],
        )
    axes[0].axhline(2 / 8, color="black", linestyle="--", linewidth=1,
                    label="isotropic 8D category: 2/8")
    axes[0].set_ylabel("held-out digit energy in natural first harmonic")
    axes[0].set_title("How much digit geometry is 2D/cyclic?")
    for axis, title in zip(
        axes[1:],
        (
            "Natural order rank among all 20,160 cycles\n(first-harmonic energy; high favors claim)",
            "Natural order rank among all 20,160 cycles\n(circle-distance correlation; high favors claim)",
        ),
    ):
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.axhspan(0.025, 0.975, color="#eeeeee", zorder=-1)
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("exact order percentile")
        axis.set_title(title)
    for axis in axes:
        axis.set_xticks(x, [phase.replace("_", "\n") for phase in phases])
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.suptitle("Held-out blank cells; five rating-stratified puzzle folds")
    figure.savefig(os.path.join(output_dir, "natural_order_checkpoints.png"), dpi=190)
    plt.close(figure)


def plot_confound_and_probes(intrinsic_rows, probe_rows, position_rows, output_dir):
    models = list(MODEL_HEALTH)
    representations = (
        "unit_state",
        "position_time_residual",
        "full_confound_residual",
        "output_null_state",
    )
    short_names = ("unit state", "− position/time", "− all nuisances", "− output span")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    width = 0.18
    x = np.arange(len(representations))
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    for model_index, model in enumerate(models):
        offset = (model_index - 1.5) * width
        energy = []
        order_rank = []
        digit_accuracy = []
        position_accuracy = []
        for representation in representations:
            base = {
                "model": model,
                "representation": representation,
            }
            intrinsic_filter = {
                **base,
                "label": "true_digit",
                "phase": "all_1_1024",
                "scope": "blank",
            }
            energy.append(_mean_error(_group_values(
                intrinsic_rows,
                intrinsic_filter,
                "natural_first_harmonic_fraction",
            ))[0])
            order_rank.append(_mean_error(_group_values(
                intrinsic_rows,
                intrinsic_filter,
                "natural_harmonic_energy_percentile",
            ))[0])
            digit_accuracy.append(_mean_error(_group_values(
                probe_rows,
                {**base, "target": "true_digit", "metric": "balanced_accuracy"},
                "value",
            ))[0])
            position_accuracy.append(_mean_error(_group_values(
                probe_rows,
                {**base, "target": "cell_position", "metric": "balanced_accuracy"},
                "value",
            ))[0])
        axes[0, 0].bar(x + offset, energy, width, color=colors[model_index], label=model)
        axes[0, 1].bar(x + offset, order_rank, width, color=colors[model_index])
        axes[1, 0].bar(x + offset, digit_accuracy, width, color=colors[model_index])
        axes[1, 1].bar(x + offset, position_accuracy, width, color=colors[model_index])
    axes[0, 0].axhline(0.25, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_title("Natural 2D harmonic / total digit energy")
    axes[0, 1].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Natural order exact percentile")
    axes[1, 0].axhline(1 / 9, color="black", linestyle="--", linewidth=1)
    axes[1, 0].set_title("True-digit balanced probe accuracy")
    axes[1, 1].axhline(1 / 81, color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Cell-position balanced probe accuracy")
    for axis in axes.flat:
        axis.set_xticks(x, short_names, rotation=12, ha="right")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Confound and decoder-span controls on held-out puzzles")
    figure.savefig(os.path.join(output_dir, "confounds_and_decoder_span.png"), dpi=190)
    plt.close(figure)

    targets = (
        "iteration_log2",
        "confidence",
        "probability_margin",
        "true_logit_margin",
        "state_norm_log",
    )
    figure, axes = plt.subplots(1, len(models), figsize=(16, 4), constrained_layout=True, sharey=True)
    for axis, model in zip(axes, models):
        values = []
        errors = []
        for target in targets:
            mean, error = _mean_error(_group_values(
                probe_rows,
                {
                    "model": model,
                    "representation": "unit_state",
                    "target": target,
                    "metric": "r2",
                },
                "value",
            ))
            values.append(mean)
            errors.append(error)
        axis.bar(np.arange(len(targets)), values, yerr=errors, color="#4C78A8")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(
            np.arange(len(targets)),
            ("log iteration", "confidence", "prob. margin", "true margin", "log state norm"),
            rotation=35,
            ha="right",
        )
        axis.set_title(model)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("held-out linear-probe $R^2$")
    figure.suptitle("Candidate helix axes are separately linearly readable")
    figure.savefig(os.path.join(output_dir, "axis_probe_r2.png"), dpi=190)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for model_index, model in enumerate(models):
        values = _group_values(
            position_rows,
            {"model": model},
            "natural_harmonic_energy_percentile",
        )
        axis.scatter(
            np.full(len(values), model_index) + np.linspace(-0.08, 0.08, len(values)),
            values,
            color=colors[model_index],
            s=42,
        )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(models)), models)
    axis.set_ylim(-0.03, 1.03)
    axis.set_ylabel("natural-order exact percentile")
    axis.set_title("Jointly held-out puzzles and cell-position groups")
    axis.grid(axis="y", alpha=0.2)
    figure.savefig(os.path.join(output_dir, "heldout_cell_positions.png"), dpi=190)
    plt.close(figure)


def plot_label_and_basis_nulls(label_rows, basis_rows, output_dir):
    models = list(MODEL_HEALTH)
    nulls = (
        "actual_labels",
        "puzzle_digit_renaming",
        "position_blocked_shuffle",
        "random_fixed_trajectory_labels",
    )
    null_labels = (
        "actual",
        "per-puzzle\ndigit rename",
        "within-position\ntrajectory shuffle",
        "random fixed\ntrajectory label",
    )
    figure, axes = plt.subplots(2, len(models), figsize=(17, 8), constrained_layout=True, sharey="row")
    for column, model in enumerate(models):
        distributions = [
            _group_values(label_rows, {"model": model, "null": null},
                          "train_test_distance_correlation")
            for null in nulls
        ]
        axes[0, column].boxplot(distributions, tick_labels=null_labels, showfliers=False)
        axes[0, column].axhline(0, color="black", linewidth=0.8)
        axes[0, column].set_title(model)
        axes[0, column].tick_params(axis="x", rotation=20)
        axes[0, column].grid(axis="y", alpha=0.2)

        basis_types = (
            "train_fitted_natural_fourier",
            "train_fitted_fixed_random_order",
            "train_sample_pca",
            "output_head_natural_fourier",
            "haar_random",
        )
        basis_labels = ("fitted\nnatural", "fitted random\norder", "PCA", "output\nhead", "random\nbasis")
        basis_distributions = [
            _group_values(basis_rows, {"model": model, "basis": basis}, "phase_cosine")
            for basis in basis_types
        ]
        axes[1, column].boxplot(
            basis_distributions,
            tick_labels=basis_labels,
            showfliers=False,
        )
        axes[1, column].axhline(0, color="black", linewidth=0.8)
        axes[1, column].tick_params(axis="x", rotation=20)
        axes[1, column].grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("train/test digit-distance correlation")
    axes[1, 0].set_ylabel("held-out centroid phase cosine")
    figure.suptitle(
        "Destroyed-label and random-basis controls\n"
        "The fitted circle is a readout; its existence is not intrinsic geometry"
    )
    figure.savefig(os.path.join(output_dir, "label_and_basis_nulls.png"), dpi=190)
    plt.close(figure)


def plot_weight_geometry(weight_rows, output_dir):
    models = list(MODEL_HEALTH)
    matrices = ("initial_encoder", "prediction_feedback", "output_head")
    colors = {"initial_encoder": "#4C78A8", "prediction_feedback": "#F58518", "output_head": "#54A24B"}
    figure, axes = plt.subplots(3, len(models), figsize=(16, 10), constrained_layout=True, sharey="row")
    for column, model in enumerate(models):
        for matrix in matrices:
            matches = [
                row for row in weight_rows
                if row.get("model") == model and row.get("matrix") == matrix
            ]
            if not matches:
                continue
            row = matches[0]
            fractions = np.asarray(row["eigenvalue_fractions"][:8], dtype=float)
            axes[0, column].plot(
                np.arange(1, len(fractions) + 1),
                fractions,
                marker="o",
                color=colors[matrix],
                label=matrix,
            )
            axes[1, column].scatter(
                row["effective_rank"],
                row["pairwise_distance_cv"],
                color=colors[matrix],
                s=60,
                label=matrix,
            )
            axes[2, column].scatter(
                matrices.index(matrix),
                row["natural_harmonic_energy_percentile"],
                color=colors[matrix],
                s=60,
            )
        axes[0, column].axhline(1 / 8, color="black", linestyle="--", linewidth=1)
        axes[0, column].set_title(model)
        axes[0, column].set_xticks(np.arange(1, 9))
        axes[0, column].grid(alpha=0.2)
        axes[1, column].axvline(8, color="black", linestyle="--", linewidth=1)
        axes[1, column].set_xlim(0, 8.4)
        axes[1, column].grid(alpha=0.2)
        axes[2, column].axhline(0.5, color="black", linestyle="--", linewidth=1)
        axes[2, column].set_ylim(-0.03, 1.03)
        axes[2, column].set_xticks(range(len(matrices)), ("input", "feedback", "output"))
        axes[2, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel("centered digit eigenvalue fraction")
    axes[1, 0].set_ylabel("pairwise-distance coefficient of variation")
    axes[2, 0].set_ylabel("natural-order exact percentile")
    for axis in axes[1]:
        axis.set_xlabel("effective rank (maximum 8)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Dimension and distance uniformity of model-interface digit vectors")
    figure.savefig(os.path.join(output_dir, "weight_geometry.png"), dpi=190)
    plt.close(figure)


def _align_projection(train_points, test_points, code):
    train_points = train_points.float()
    test_points = test_points.float()
    code = code.float()
    train_mean = train_points.mean(0, keepdim=True)
    centered = train_points - train_mean
    left, singular_values, right = torch.linalg.svd(
        centered.T @ code,
        full_matrices=False,
    )
    rotation = left @ right
    scale = singular_values.sum() / centered.square().sum().clamp_min(1e-20)
    return (test_points - train_mean) @ rotation * scale


def plot_projection_comparison(all_projection_data, output_dir):
    models = list(MODEL_HEALTH)
    keys = ("natural", "random_order", "pca", "random_basis")
    titles = (
        "train-fitted natural order",
        "train-fitted arbitrary order",
        "train PCA + label alignment",
        "fixed random basis + alignment",
    )
    colors = plt.cm.hsv(np.arange(DIGIT_COUNT) / DIGIT_COUNT)
    figure, axes = plt.subplots(len(models), len(keys), figsize=(15, 14), constrained_layout=True)
    for row_index, model in enumerate(models):
        plot_data = all_projection_data.get(model)
        if plot_data is None:
            continue
        for column_index, (key, title) in enumerate(zip(keys, titles)):
            axis = axes[row_index, column_index]
            train_points, test_points, code = plot_data[key]
            points = _align_projection(train_points, test_points, code).numpy()
            order = NATURAL_ORDER
            if key == "random_order":
                order = (0, 5, 2, 8, 3, 7, 1, 6, 4)
            closed = np.asarray([points[digit] for digit in order] + [points[order[0]]])
            axis.plot(closed[:, 0], closed[:, 1], color="#777777", linewidth=1)
            axis.scatter(points[:, 0], points[:, 1], c=colors, s=55, edgecolor="black", linewidth=0.5)
            for digit, point in enumerate(points):
                axis.text(point[0], point[1], str(digit + 1), ha="center", va="center", fontsize=8)
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
            if row_index == 0:
                axis.set_title(title)
            if column_index == 0:
                axis.set_ylabel(model)
    figure.suptitle(
        "Held-out digit centroids: fitted bases draw the requested polygon\n"
        "Blank cells, fold 0; every alignment is fitted on training puzzles"
    )
    figure.savefig(os.path.join(output_dir, "projection_comparison.png"), dpi=190)
    plt.close(figure)


def plot_strata(strata_rows, output_dir):
    models = list(MODEL_HEALTH)
    variables = ("confidence", "probability_margin")
    colors = {"true_digit": "#0072B2", "predicted_digit": "#D55E00"}
    figure, axes = plt.subplots(len(variables), len(models), figsize=(16, 7), constrained_layout=True, sharey=True)
    for row_index, variable in enumerate(variables):
        for column_index, model in enumerate(models):
            axis = axes[row_index, column_index]
            for label in ("true_digit", "predicted_digit"):
                means = []
                errors = []
                for bin_index in range(1, 5):
                    mean, error = _mean_error(_group_values(
                        strata_rows,
                        {"model": model, "variable": variable, "bin": bin_index, "label": label},
                        "natural_harmonic_energy_percentile",
                    ))
                    means.append(mean)
                    errors.append(error)
                axis.errorbar(
                    np.arange(1, 5),
                    means,
                    yerr=errors,
                    marker="o",
                    color=colors[label],
                    label=label.replace("_", " "),
                )
            axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
            axis.set_ylim(-0.03, 1.03)
            axis.set_xticks(np.arange(1, 5))
            axis.set_xlabel(f"{variable.replace('_', ' ')} quartile")
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(model)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("natural-order exact percentile")
    axes[1, 0].set_ylabel("natural-order exact percentile")
    figure.suptitle("Natural cyclic-order rank within iteration-matched confidence and margin strata")
    figure.savefig(os.path.join(output_dir, "confidence_margin_strata.png"), dpi=190)
    plt.close(figure)


def _sample_accuracy_rows(model_name, arrays):
    rows = []
    puzzle_count = int(arrays["puzzle"].max()) + 1
    for time_index, iteration in enumerate(ITERATIONS):
        selected = arrays["time_index"] == time_index
        empty = arrays["is_empty"][selected]
        correct = arrays["predicted_digit"][selected] == arrays["true_digit"][selected]
        puzzle_ids = arrays["puzzle"][selected]
        cell_accuracy = float(correct[empty].float().mean())
        solved = 0
        for puzzle_index in range(puzzle_count):
            puzzle_mask = (puzzle_ids == puzzle_index) & empty
            solved += int(correct[puzzle_mask].all())
        rows.append({
            "model": model_name,
            "iteration": iteration,
            "blank_cell_accuracy": cell_accuracy,
            "puzzles_solved": solved,
            "puzzle_count": puzzle_count,
        })
    return rows


def run(
    output_dir,
    examples_per_bucket=10,
    fold_count=5,
    random_basis_count=128,
    label_null_repetitions=20,
    seed=20260807,
    device="cuda",
    model_configs=DEFAULT_MODELS,
):
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if examples_per_bucket < fold_count:
        raise ValueError(
            "examples_per_bucket must be at least fold_count so every rating "
            "bucket contributes to every held-out fold"
        )
    os.makedirs(output_dir, exist_ok=True)
    torch.set_num_threads(max(1, min(16, os.cpu_count() or 1)))
    resolved_device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    inputs, targets, empty_mask, puzzles, _, bucket_names = _load_balanced_sample(
        examples_per_bucket,
        seed,
    )
    inputs = inputs.to(resolved_device)
    targets = targets.to(resolved_device)
    empty_mask = empty_mask.to(resolved_device)
    started = time.time()
    all_intrinsic = []
    all_probes = []
    all_bases = []
    all_label_nulls = []
    all_position_transfer = []
    all_strata = []
    all_weights = []
    all_accuracy = []
    all_projection_data = {}
    all_folds = {}

    log_path = os.path.join(output_dir, "run.log")
    log_handle = open(log_path, "w")

    def log(message):
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    log(
        f"Helix controls: {len(inputs)} puzzles, {fold_count} folds, "
        f"{len(model_configs)} models, device={resolved_device}"
    )
    for model_index, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        model_started = time.time()
        log(f"MODEL {model_name}: loading {model_config['path']}")
        model = _load_model(model_config, resolved_device)
        all_weights.extend(weight_geometry(model, model_name))
        collected = collect_cell_trajectories(
            model,
            inputs,
            targets,
            empty_mask,
        )
        result = analyze_model(
            model_name,
            model,
            collected,
            bucket_names,
            fold_count,
            seed,
            random_basis_count,
            label_null_repetitions,
        )
        all_intrinsic.extend(result["intrinsic_rows"])
        all_probes.extend(result["probe_rows"])
        all_bases.extend(result["basis_rows"])
        all_label_nulls.extend(result["label_null_rows"])
        all_position_transfer.extend(result["position_transfer_rows"])
        all_strata.extend(result["strata_rows"])
        all_accuracy.extend(_sample_accuracy_rows(model_name, result["arrays"]))
        all_projection_data[model_name] = result["projection_plot_data"]
        all_folds[model_name] = result["folds"]
        log(f"MODEL {model_name}: finished in {time.time() - model_started:.1f}s")
        del model, collected, result
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

        _write_csv(os.path.join(output_dir, "intrinsic_metrics.csv"), all_intrinsic)
        _write_csv(os.path.join(output_dir, "probe_metrics.csv"), all_probes)
        _write_csv(os.path.join(output_dir, "basis_controls.csv"), all_bases)
        _write_csv(os.path.join(output_dir, "label_null_controls.csv"), all_label_nulls)
        _write_csv(os.path.join(output_dir, "position_transfer.csv"), all_position_transfer)
        _write_csv(os.path.join(output_dir, "confidence_margin_strata.csv"), all_strata)
        _write_csv(os.path.join(output_dir, "weight_geometry.csv"), all_weights)
        _write_csv(os.path.join(output_dir, "sample_accuracy.csv"), all_accuracy)

    plot_natural_order(all_intrinsic, output_dir)
    plot_confound_and_probes(
        all_intrinsic,
        all_probes,
        all_position_transfer,
        output_dir,
    )
    plot_label_and_basis_nulls(all_label_nulls, all_bases, output_dir)
    plot_weight_geometry(all_weights, output_dir)
    plot_projection_comparison(all_projection_data, output_dir)
    plot_strata(all_strata, output_dir)

    summary = {
        "config": {
            "iterations": list(ITERATIONS),
            "phases": {key: list(value) for key, value in PHASES.items()},
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(inputs),
            "fold_count": fold_count,
            "random_basis_count": random_basis_count,
            "label_null_repetitions": label_null_repetitions,
            "exact_cycle_orders": len(CYCLE_CONTROLS.orders),
            "seed": seed,
            "device": str(resolved_device),
            "models": list(model_configs),
            "model_health_reference": MODEL_HEALTH,
            "primary_endpoint": (
                "natural first-harmonic energy percentile among all 20,160 "
                "digit cycles, held-out blank cells, full confound residual, "
                "whole-puzzle rating-stratified folds"
            ),
        },
        "sample": {
            "puzzle_sha256": [
                hashlib.sha256(puzzle.encode("ascii")).hexdigest()
                for puzzle in puzzles
            ],
            "buckets": bucket_names,
        },
        "folds": all_folds,
        "row_counts": {
            "intrinsic_metrics": len(all_intrinsic),
            "probe_metrics": len(all_probes),
            "basis_controls": len(all_bases),
            "label_null_controls": len(all_label_nulls),
            "position_transfer": len(all_position_transfer),
            "confidence_margin_strata": len(all_strata),
            "weight_geometry": len(all_weights),
        },
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    log(f"Completed in {summary['elapsed_seconds']:.1f}s")
    log_handle.close()
    return summary


if __name__ == "__main__":
    run(
        os.path.dirname(__file__),
        device="cpu",
        examples_per_bucket=2,
        fold_count=2,
    )
