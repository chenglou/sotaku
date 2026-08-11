"""Held-out linear geometry tests for recurrent Sudoku cell roles."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


ITERATIONS = (0, 1, 4, 16, 128, 512, 1024)
REPRESENTATIONS = (
    "state",
    "unit_state",
    "input_residual",
    "unit_input_residual",
)
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: str
    class_count: int | None
    blank_only: bool
    primary_metric: str


TASKS = (
    TaskSpec("clue", "classification", 2, False, "balanced_accuracy"),
    TaskSpec("row", "classification", 9, True, "balanced_accuracy"),
    TaskSpec("column", "classification", 9, True, "balanced_accuracy"),
    TaskSpec("box", "classification", 9, True, "balanced_accuracy"),
    TaskSpec(
        "candidate_size_categorical",
        "classification",
        9,
        True,
        "balanced_accuracy",
    ),
    TaskSpec(
        "candidate_size_ordinal",
        "regression",
        None,
        True,
        "within_puzzle_spearman",
    ),
)


def stratified_three_way_split(bucket_names, seed):
    """Return discovery, validation, and final puzzle indices per bucket."""

    by_bucket = {}
    for puzzle_index, bucket_name in enumerate(bucket_names):
        by_bucket.setdefault(bucket_name, []).append(puzzle_index)
    generator = np.random.default_rng(seed)
    split_indices = {name: [] for name in ("discovery", "validation", "final")}
    for bucket_name, indices in by_bucket.items():
        if len(indices) < 3 or len(indices) % 3:
            raise ValueError(
                f"bucket {bucket_name!r} needs a positive multiple of three puzzles"
            )
        shuffled = np.asarray(indices, dtype=np.int64)
        generator.shuffle(shuffled)
        per_split = len(shuffled) // 3
        for split_offset, split_name in enumerate(split_indices):
            start = split_offset * per_split
            split_indices[split_name].extend(
                shuffled[start:start + per_split].tolist()
            )
    return {
        split_name: sorted(indices)
        for split_name, indices in split_indices.items()
    }


def compute_cell_roles(puzzles):
    """Compute input roles and legal-candidate counts from the puzzle clues."""

    clue = torch.zeros(len(puzzles), 81, dtype=torch.bool)
    row = torch.arange(81).div(9, rounding_mode="floor").repeat(len(puzzles), 1)
    column = torch.arange(81).remainder(9).repeat(len(puzzles), 1)
    box = ((row // 3) * 3 + column // 3).long()
    input_symbol = torch.zeros(len(puzzles), 81, dtype=torch.long)
    candidate_size = torch.zeros(len(puzzles), 81, dtype=torch.long)

    for puzzle_index, puzzle in enumerate(puzzles):
        if len(puzzle) != 81:
            raise ValueError(f"puzzle {puzzle_index} has length {len(puzzle)}, not 81")
        digits = [0 if character == "." else int(character) for character in puzzle]
        for cell_index, digit in enumerate(digits):
            input_symbol[puzzle_index, cell_index] = digit
            if digit:
                clue[puzzle_index, cell_index] = True
                candidate_size[puzzle_index, cell_index] = 1
                continue
            cell_row = cell_index // 9
            cell_column = cell_index % 9
            box_row = (cell_row // 3) * 3
            box_column = (cell_column // 3) * 3
            used = set(digits[cell_row * 9:(cell_row + 1) * 9])
            used.update(digits[cell_column::9])
            used.update(
                digits[(box_row + row_offset) * 9 + box_column + column_offset]
                for row_offset in range(3)
                for column_offset in range(3)
            )
            used.discard(0)
            candidate_size[puzzle_index, cell_index] = 9 - len(used)

    return {
        "clue": clue.long(),
        "row": row.long(),
        "column": column.long(),
        "box": box.long(),
        "candidate_size": candidate_size.long(),
        "input_symbol": input_symbol.long(),
        "blank": ~clue,
    }


def fit_group_means(values, groups, fit_puzzles, group_count=10):
    """Fit per-input-symbol means using discovery puzzles only."""

    selected_values = values[fit_puzzles].reshape(-1, values.size(-1)).float()
    selected_groups = groups[fit_puzzles].reshape(-1).long()
    means = torch.zeros(group_count, values.size(-1), dtype=torch.float32)
    counts = torch.zeros(group_count, dtype=torch.long)
    for group in range(group_count):
        group_values = selected_values[selected_groups == group]
        if len(group_values) == 0:
            raise ValueError(f"input-symbol group {group} has no discovery samples")
        means[group] = group_values.mean(0)
        counts[group] = len(group_values)
    return means, counts


def build_representations(states, input_symbols, discovery_puzzles):
    means, counts = fit_group_means(
        states,
        input_symbols,
        discovery_puzzles,
    )
    residual = states.float() - means[input_symbols]
    return {
        "state": states.float(),
        "unit_state": F.normalize(states.float(), dim=-1, eps=1e-12),
        "input_residual": residual,
        "unit_input_residual": F.normalize(residual, dim=-1, eps=1e-12),
    }, counts


def task_labels(roles, task_name):
    if task_name == "candidate_size_categorical":
        return roles["candidate_size"] - 1
    if task_name == "candidate_size_ordinal":
        return roles["candidate_size"].float()
    return roles[task_name]


def task_examples(features, roles, task, puzzle_split):
    puzzle_count, cell_count, feature_count = features.shape
    puzzle_ids = torch.arange(puzzle_count)[:, None].expand(-1, cell_count)
    split_mask = torch.zeros(puzzle_count, dtype=torch.bool)
    split_mask[puzzle_split] = True
    mask = split_mask[:, None].expand(-1, cell_count)
    if task.blank_only:
        mask = mask & roles["blank"]
    labels = task_labels(roles, task.name)
    return (
        features[mask].reshape(-1, feature_count),
        labels[mask].reshape(-1),
        puzzle_ids[mask].reshape(-1),
    )


@dataclass
class RidgeReadout:
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    centered_mean: torch.Tensor
    coefficients: torch.Tensor
    intercept: torch.Tensor

    def standardized(self, features):
        return (features.float() - self.feature_mean) / self.feature_scale

    def predict(self, features):
        standardized = self.standardized(features)
        return standardized @ self.coefficients + self.intercept


@dataclass
class RidgeProblem:
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    centered_mean: torch.Tensor
    target_mean: torch.Tensor
    covariance: torch.Tensor
    cross_covariance: torch.Tensor


def _classification_weights(labels, class_count):
    counts = torch.bincount(labels.long(), minlength=class_count).float()
    present_class_count = int((counts > 0).sum().item())
    if present_class_count < 2:
        raise ValueError("classification requires at least two discovery classes")
    weights = counts.sum() / (
        present_class_count * counts[labels.long()].clamp_min(1)
    )
    return weights


def prepare_ridge_problem(
    features,
    labels,
    *,
    kind,
    class_count=None,
    standardize=True,
):
    features = features.float()
    if standardize:
        feature_mean = features.mean(0)
        feature_scale = features.std(0, unbiased=False).clamp_min(1e-5)
    else:
        feature_mean = torch.zeros(features.size(1), dtype=features.dtype)
        feature_scale = torch.ones(features.size(1), dtype=features.dtype)
    standardized = (features - feature_mean) / feature_scale

    if kind == "classification":
        labels = labels.long()
        targets = F.one_hot(labels, class_count).float()
        weights = _classification_weights(labels, class_count)
    elif kind == "regression":
        targets = labels.float().reshape(-1, 1)
        weights = torch.ones(len(labels), dtype=torch.float32)
    else:
        raise ValueError(f"unknown readout kind {kind!r}")

    weight_sum = weights.sum().clamp_min(1e-12)
    centered_mean = (standardized * weights[:, None]).sum(0) / weight_sum
    target_mean = (targets * weights[:, None]).sum(0) / weight_sum
    centered_features = standardized - centered_mean
    centered_targets = targets - target_mean
    weighted_features = centered_features * weights.sqrt()[:, None]
    weighted_targets = centered_targets * weights.sqrt()[:, None]
    covariance = weighted_features.T @ weighted_features / weight_sum
    cross_covariance = weighted_features.T @ weighted_targets / weight_sum
    return RidgeProblem(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        centered_mean=centered_mean,
        target_mean=target_mean,
        covariance=covariance,
        cross_covariance=cross_covariance,
    )


def fit_prepared_ridge(problem, alpha):
    regularized_covariance = problem.covariance.clone()
    regularized_covariance.diagonal().add_(float(alpha))
    coefficients = torch.linalg.solve(
        regularized_covariance,
        problem.cross_covariance,
    )
    intercept = problem.target_mean - problem.centered_mean @ coefficients
    return RidgeReadout(
        feature_mean=problem.feature_mean,
        feature_scale=problem.feature_scale,
        centered_mean=problem.centered_mean,
        coefficients=coefficients,
        intercept=intercept,
    )


def fit_ridge_readout(
    features,
    labels,
    *,
    alpha,
    kind,
    class_count=None,
    standardize=True,
):
    problem = prepare_ridge_problem(
        features,
        labels,
        kind=kind,
        class_count=class_count,
        standardize=standardize,
    )
    return fit_prepared_ridge(problem, alpha)


def _rankdata(values):
    values = values.float().flatten()
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, counts = torch.unique_consecutive(sorted_values, return_counts=True)
    ranks = torch.empty_like(values)
    offset = 0
    for count in counts.tolist():
        average_rank = offset + (count - 1) / 2
        ranks[order[offset:offset + count]] = average_rank
        offset += count
    return ranks


def pearson(first, second):
    first = torch.as_tensor(first, dtype=torch.float64).flatten()
    second = torch.as_tensor(second, dtype=torch.float64).flatten()
    finite = torch.isfinite(first) & torch.isfinite(second)
    first = first[finite]
    second = second[finite]
    if len(first) < 2:
        return 0.0
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if denominator <= 1e-20:
        return 0.0
    return float((first @ second / denominator).item())


def spearman(first, second):
    return pearson(_rankdata(torch.as_tensor(first)), _rankdata(torch.as_tensor(second)))


def classification_metrics(labels, scores, class_count):
    labels = labels.long()
    predictions = scores.argmax(1)
    recalls = []
    per_class = {}
    for class_index in range(class_count):
        class_mask = labels == class_index
        if class_mask.any():
            recall = (predictions[class_mask] == class_index).float().mean().item()
            recalls.append(recall)
            per_class[str(class_index)] = recall
    return {
        "count": len(labels),
        "accuracy": float((predictions == labels).float().mean().item()),
        "balanced_accuracy": float(np.mean(recalls)),
        "per_class_recall": per_class,
        "present_class_count": len(recalls),
    }


def regression_metrics(labels, predictions, puzzle_ids):
    labels = labels.float().flatten()
    predictions = predictions.float().flatten()
    residual_sum = (labels - predictions).square().sum()
    total_sum = (labels - labels.mean()).square().sum().clamp_min(1e-12)
    within_puzzle = []
    for puzzle_id in torch.unique(puzzle_ids):
        mask = puzzle_ids == puzzle_id
        if labels[mask].unique().numel() >= 2:
            within_puzzle.append(spearman(labels[mask], predictions[mask]))
    return {
        "count": len(labels),
        "r_squared": float((1 - residual_sum / total_sum).item()),
        "mae": float((labels - predictions).abs().mean().item()),
        "pearson": pearson(labels, predictions),
        "spearman": spearman(labels, predictions),
        "within_puzzle_spearman": float(np.mean(within_puzzle)),
        "within_puzzle_spearman_median": float(np.median(within_puzzle)),
        "within_puzzle_count": len(within_puzzle),
    }


def evaluate_readout(readout, features, labels, puzzle_ids, task):
    predictions = readout.predict(features)
    if task.kind == "classification":
        return classification_metrics(labels, predictions, task.class_count)
    return regression_metrics(labels, predictions[:, 0], puzzle_ids)


def fit_selected_readout(
    discovery,
    validation,
    final,
    task,
    alphas=RIDGE_ALPHAS,
):
    discovery_features, discovery_labels, _ = discovery
    validation_features, validation_labels, validation_puzzles = validation
    problem = prepare_ridge_problem(
        discovery_features,
        discovery_labels,
        kind=task.kind,
        class_count=task.class_count,
    )
    candidates = []
    for alpha in alphas:
        readout = fit_prepared_ridge(problem, alpha)
        metrics = evaluate_readout(
            readout,
            validation_features,
            validation_labels,
            validation_puzzles,
            task,
        )
        candidates.append((float(metrics[task.primary_metric]), alpha, readout, metrics))
    _, selected_alpha, readout, validation_metrics = max(
        candidates,
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    final_features, final_labels, final_puzzles = final
    final_metrics = evaluate_readout(
        readout,
        final_features,
        final_labels,
        final_puzzles,
        task,
    )
    return {
        "selected_alpha": selected_alpha,
        "validation": validation_metrics,
        "final": final_metrics,
    }, readout


def shuffle_within_puzzles(labels, puzzle_ids, seed):
    shuffled = labels.clone()
    generator = torch.Generator().manual_seed(seed)
    for puzzle_id in torch.unique(puzzle_ids):
        indices = (puzzle_ids == puzzle_id).nonzero().flatten()
        permutation = torch.randperm(len(indices), generator=generator)
        shuffled[indices] = labels[indices[permutation]]
    return shuffled


def _null_summary(values, observed, *, favorable="higher", include_values=True):
    values = np.asarray(values, dtype=np.float64)
    if favorable == "higher":
        p_value = (1 + int(np.sum(values >= observed - 1e-12))) / (len(values) + 1)
    else:
        p_value = (1 + int(np.sum(values <= observed + 1e-12))) / (len(values) + 1)
    summary = {
        "mean": float(values.mean()),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "observed": float(observed),
        "empirical_p_value": float(p_value),
    }
    if include_values:
        summary["values"] = values.tolist()
    return summary


def label_shuffle_control(
    discovery,
    final,
    task,
    *,
    alpha,
    observed,
    repeats=8,
    seed=0,
):
    discovery_features, discovery_labels, discovery_puzzles = discovery
    final_features, final_labels, final_puzzles = final
    values = []
    for repeat in range(repeats):
        shuffled_labels = shuffle_within_puzzles(
            discovery_labels,
            discovery_puzzles,
            seed + repeat,
        )
        readout = fit_ridge_readout(
            discovery_features,
            shuffled_labels,
            alpha=alpha,
            kind=task.kind,
            class_count=task.class_count,
        )
        metrics = evaluate_readout(
            readout,
            final_features,
            final_labels,
            final_puzzles,
            task,
        )
        values.append(metrics[task.primary_metric])
    return _null_summary(values, observed)


def coefficient_rank(readout, task):
    coefficients = readout.coefficients.float()
    if task.kind == "classification":
        coefficients = coefficients - coefficients.mean(1, keepdim=True)
    singular_values = torch.linalg.svdvals(coefficients)
    if len(singular_values) == 0 or singular_values.max() <= 1e-12:
        return 1
    rank = int((singular_values > singular_values.max() * 1e-6).sum().item())
    return max(1, rank)


def random_subspace_control(
    readout,
    discovery,
    final,
    task,
    *,
    alpha,
    observed,
    repeats=16,
    seed=0,
):
    discovery_features, discovery_labels, _ = discovery
    final_features, final_labels, final_puzzles = final
    standardized_discovery = readout.standardized(discovery_features)
    standardized_final = readout.standardized(final_features)
    rank = coefficient_rank(readout, task)
    values = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(repeats):
        random_matrix = torch.randn(
            standardized_discovery.size(1),
            rank,
            generator=generator,
        )
        random_basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
        projected_discovery = standardized_discovery @ random_basis
        projected_final = standardized_final @ random_basis
        random_readout = fit_ridge_readout(
            projected_discovery,
            discovery_labels,
            alpha=alpha,
            kind=task.kind,
            class_count=task.class_count,
            standardize=False,
        )
        metrics = evaluate_readout(
            random_readout,
            projected_final,
            final_labels,
            final_puzzles,
            task,
        )
        values.append(metrics[task.primary_metric])
    summary = _null_summary(values, observed)
    summary["subspace_rank"] = rank
    return summary


def pairwise_centroid_geometry(
    discovery_features,
    discovery_labels,
    final_features,
    final_labels,
    class_count,
    *,
    ordered=False,
    order_permutations=256,
    seed=0,
):
    classes = [
        class_index
        for class_index in range(class_count)
        if (discovery_labels == class_index).any() and (final_labels == class_index).any()
    ]
    if len(classes) < 3:
        return {"class_count": len(classes), "available": False}
    discovery_centroids = torch.stack([
        discovery_features[discovery_labels == class_index].float().mean(0)
        for class_index in classes
    ])
    final_centroids = torch.stack([
        final_features[final_labels == class_index].float().mean(0)
        for class_index in classes
    ])
    rows, columns = torch.triu_indices(len(classes), len(classes), offset=1)
    discovery_distances = torch.cdist(discovery_centroids, discovery_centroids)[rows, columns]
    final_distances = torch.cdist(final_centroids, final_centroids)[rows, columns]
    result = {
        "available": True,
        "classes": classes,
        "discovery_final_distance_correlation": pearson(
            discovery_distances,
            final_distances,
        ),
    }
    if ordered:
        class_values = torch.tensor(classes, dtype=torch.float32)
        ideal = (class_values[rows] - class_values[columns]).abs()
        observed = spearman(final_distances, ideal)
        generator = torch.Generator().manual_seed(seed)
        null = []
        for _ in range(order_permutations):
            shuffled = class_values[torch.randperm(len(classes), generator=generator)]
            shuffled_ideal = (shuffled[rows] - shuffled[columns]).abs()
            null.append(spearman(final_distances, shuffled_ideal))
        result["ordered_distance_spearman"] = observed
        result["ordered_distance_control"] = _null_summary(
            null,
            observed,
            include_values=False,
        )
    return result


def exact_iteration_order_test(iterations, values):
    """Test monotonic progression against every permutation of seven horizons."""

    if len(iterations) != len(values):
        raise ValueError("iterations and values must have the same length")
    if len(values) > 8:
        raise ValueError("exact iteration control is limited to eight points")
    time_ranks = _rankdata(
        torch.log1p(torch.tensor(iterations, dtype=torch.float32))
    ).double().numpy()
    value_ranks = _rankdata(torch.tensor(values, dtype=torch.float32)).double().numpy()
    time_ranks -= time_ranks.mean()
    value_ranks -= value_ranks.mean()
    denominator = np.linalg.norm(time_ranks) * np.linalg.norm(value_ranks)
    if denominator <= 1e-20:
        observed = 0.0
        null = np.zeros(math.factorial(len(values)), dtype=np.float64)
    else:
        permutations = np.asarray(
            list(itertools.permutations(range(len(values)))),
            dtype=np.int64,
        )
        observed = float(time_ranks @ value_ranks / denominator)
        null = np.einsum(
            "ij,j->i",
            value_ranks[permutations],
            time_ranks,
        ) / float(denominator)
    return {
        "observed_spearman": observed,
        "permutation_count": math.factorial(len(values)),
        **_null_summary(null, observed, include_values=False),
    }


def grouped_regression_predictions(labels, predictions):
    labels = labels.long().flatten()
    predictions = predictions.float().flatten()
    rows = []
    for label in sorted(labels.unique().tolist()):
        selected = predictions[labels == label]
        rows.append({
            "candidate_size": int(label),
            "count": len(selected),
            "prediction_mean": float(selected.mean().item()),
            "prediction_std": float(selected.std(unbiased=False).item()),
            "prediction_sem": float(
                selected.std(unbiased=False).item() / math.sqrt(len(selected))
            ),
        })
    return rows
