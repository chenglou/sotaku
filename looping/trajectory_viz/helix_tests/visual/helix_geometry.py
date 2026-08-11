"""Projection and validation utilities for per-cell recurrent geometry.

The supervised projection in this module is deliberately called a periodic
readout. Its target imposes a circle (or helix), so held-out performance shows
linear readability rather than proving that the original feature space is
intrinsically circular.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


DIGIT_COUNT = 9
NATURAL_ORDER = tuple(range(DIGIT_COUNT))
DEFAULT_RIDGE_LAMBDAS = (
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
)


def unit_normalize(values: torch.Tensor) -> torch.Tensor:
    """Normalize each token vector without changing leading dimensions."""

    return F.normalize(values.float(), dim=-1, eps=1e-12)


def decoder_row_basis(output_weight: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis for centered output-head digit directions."""

    centered = output_weight.float() - output_weight.float().mean(0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(centered, full_matrices=False)
    threshold = singular_values.max().clamp_min(1e-12) * 1e-6
    rank = int((singular_values > threshold).sum().item())
    return right[:rank].T.contiguous()


def remove_subspace(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Remove the feature-space span of an orthonormal basis."""

    basis = basis.to(values.device, values.dtype)
    return values - (values @ basis) @ basis.T


def periodic_code(
    order: tuple[int, ...] = NATURAL_ORDER,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Map each digit label to a unit-circle point in the requested cycle."""

    if tuple(sorted(order)) != NATURAL_ORDER:
        raise ValueError("order must contain each zero-based digit exactly once")
    angles = torch.empty(DIGIT_COUNT, dtype=dtype)
    for position, digit in enumerate(order):
        angles[digit] = 2 * math.pi * position / DIGIT_COUNT
    return torch.stack((angles.cos(), angles.sin()), dim=1)


def helix_code(
    order: tuple[int, ...] = NATURAL_ORDER,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a one-turn digit helix with balanced coordinate scales."""

    circle = periodic_code(order, dtype=dtype)
    axial = torch.empty(DIGIT_COUNT, dtype=dtype)
    positions = torch.linspace(-1.0, 1.0, DIGIT_COUNT, dtype=dtype)
    positions = positions / positions.std(unbiased=False)
    for position, digit in enumerate(order):
        axial[digit] = positions[position]
    return torch.cat((circle, axial[:, None]), dim=1)


@lru_cache(maxsize=1)
def unique_cycle_orders() -> tuple[tuple[int, ...], ...]:
    """Enumerate the 20,160 digit cycles modulo rotation and reflection."""

    orders = []
    for tail in itertools.permutations(range(1, DIGIT_COUNT)):
        if tail[0] < tail[-1]:
            orders.append((0, *tail))
    return tuple(orders)


@lru_cache(maxsize=1)
def cycle_tensor() -> torch.Tensor:
    return torch.tensor(unique_cycle_orders(), dtype=torch.long)


def class_centroids(
    values: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    centroids = []
    for digit in range(DIGIT_COUNT):
        selected = values[labels == digit]
        if selected.numel() == 0:
            raise ValueError(f"digit {digit + 1} has no samples")
        centroids.append(selected.float().mean(0))
    return torch.stack(centroids)


def pairwise_distances(values: torch.Tensor) -> torch.Tensor:
    return torch.cdist(values.float(), values.float())


def cycle_lengths(
    distance_matrix: torch.Tensor,
    cycles: torch.Tensor | None = None,
) -> torch.Tensor:
    if cycles is None:
        cycles = cycle_tensor()
    cycles = cycles.to(distance_matrix.device)
    following = cycles.roll(-1, dims=1)
    return distance_matrix[cycles, following].sum(dim=1)


def percentile_at(values: torch.Tensor, observed: float) -> float:
    """Return the fraction of a null distribution at or below an observation."""

    return float((values <= observed + 1e-12).float().mean().item())


def upper_tail_fraction(values: torch.Tensor, observed: float) -> float:
    """Return the inclusive fraction at or above a favorable observation."""

    return float((values >= observed - 1e-12).float().mean().item())


def _upper_triangle(values: torch.Tensor) -> torch.Tensor:
    rows, columns = torch.triu_indices(values.size(0), values.size(1), offset=1)
    return values[rows, columns]


def pearson_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.float().flatten()
    second = second.float().flatten()
    first = first - first.mean()
    second = second - second.mean()
    denominator = first.norm() * second.norm()
    if denominator <= 1e-12:
        return 0.0
    return float((first @ second / denominator).item())


def effective_rank_from_centroids(centroids: torch.Tensor) -> dict:
    centered = centroids.float() - centroids.float().mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    eigenvalues = singular_values.square().clamp_min(0)
    total = eigenvalues.sum().clamp_min(1e-20)
    fractions = eigenvalues / total
    effective_rank = total.square() / eigenvalues.square().sum().clamp_min(1e-20)
    pairwise = _upper_triangle(pairwise_distances(centroids))
    return {
        "eigenvalue_fractions": fractions.tolist(),
        "top2_fraction": float(fractions[:2].sum().item()),
        "top3_fraction": float(fractions[:3].sum().item()),
        "effective_rank": float(effective_rank.item()),
        "pairwise_distance_cv": float(
            pairwise.std(unbiased=False) / pairwise.mean().clamp_min(1e-12)
        ),
    }


def intrinsic_digit_geometry(
    train_values: torch.Tensor,
    train_labels: torch.Tensor,
    test_values: torch.Tensor,
    test_labels: torch.Tensor,
) -> dict:
    """Measure digit-centroid geometry without a label-fitted projection."""

    train_centroids = class_centroids(train_values, train_labels)
    test_centroids = class_centroids(test_values, test_labels)
    train_distances = pairwise_distances(train_centroids)
    test_distances = pairwise_distances(test_centroids)
    cycles = cycle_tensor()
    train_lengths = cycle_lengths(train_distances, cycles)
    test_lengths = cycle_lengths(test_distances, cycles)
    natural_index = unique_cycle_orders().index(NATURAL_ORDER)
    selected_index = int(train_lengths.argmin().item())
    natural_test_length = float(test_lengths[natural_index].item())
    selected_test_length = float(test_lengths[selected_index].item())
    circle_distances = pairwise_distances(periodic_code())
    helix_distances = pairwise_distances(helix_code())
    return {
        "train_centroids": train_centroids.tolist(),
        "test_centroids": test_centroids.tolist(),
        "test_centroid_spectrum": effective_rank_from_centroids(test_centroids),
        "train_test_distance_correlation": pearson_correlation(
            _upper_triangle(train_distances),
            _upper_triangle(test_distances),
        ),
        "natural_circle_distance_correlation": pearson_correlation(
            _upper_triangle(test_distances),
            _upper_triangle(circle_distances),
        ),
        "natural_helix_distance_correlation": pearson_correlation(
            _upper_triangle(test_distances),
            _upper_triangle(helix_distances),
        ),
        "natural_cycle_length": natural_test_length,
        "natural_cycle_shorter_percentile": percentile_at(
            test_lengths,
            natural_test_length,
        ),
        "train_shortest_cycle": [digit + 1 for digit in cycles[selected_index].tolist()],
        "train_shortest_cycle_test_length": selected_test_length,
        "train_shortest_cycle_test_shorter_percentile": percentile_at(
            test_lengths,
            selected_test_length,
        ),
        "cycle_length_null": {
            "minimum": float(test_lengths.min().item()),
            "median": float(test_lengths.median().item()),
            "maximum": float(test_lengths.max().item()),
        },
    }


@dataclass
class PCAProjection:
    mean: torch.Tensor
    basis: torch.Tensor
    eigenvalues: torch.Tensor

    def project(self, values: torch.Tensor) -> torch.Tensor:
        return (values.float() - self.mean) @ self.basis

    def captured_fraction(self, values: torch.Tensor, components: int) -> float:
        centered = values.float() - self.mean
        projected = centered @ self.basis[:, :components]
        return float(
            projected.square().sum()
            / centered.square().sum().clamp_min(1e-20)
        )


def fit_pca(values: torch.Tensor, component_count: int = 16) -> PCAProjection:
    values = values.float()
    mean = values.mean(0, keepdim=True)
    centered = values - mean
    covariance = centered.T @ centered / max(1, centered.size(0) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    count = min(component_count, values.size(1))
    eigenvalues = eigenvalues[order][:count].clamp_min(0)
    basis = eigenvectors[:, order][:, :count]
    # Eigenvector signs are arbitrary. A deterministic largest-loading rule keeps
    # reruns stable without consulting held-out labels.
    largest = basis.abs().argmax(dim=0)
    signs = torch.sign(basis[largest, torch.arange(count)]).clamp(min=-1, max=1)
    signs[signs == 0] = 1
    basis = basis * signs
    return PCAProjection(mean=mean, basis=basis, eigenvalues=eigenvalues)


def pca_metrics(
    projection: PCAProjection,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    test_values: torch.Tensor,
) -> dict:
    total_train_variance = (
        (train_values.float() - projection.mean).square().sum()
        / max(1, train_values.size(0) - 1)
    ).clamp_min(1e-20)
    fractions = projection.eigenvalues / total_train_variance
    return {
        "train_explained_fractions": fractions.tolist(),
        "train_top2_fraction": float(fractions[:2].sum().item()),
        "train_top3_fraction": float(fractions[:3].sum().item()),
        "validation_top2_captured": projection.captured_fraction(
            validation_values, 2
        ),
        "validation_top3_captured": projection.captured_fraction(
            validation_values, 3
        ),
        "test_top2_captured": projection.captured_fraction(test_values, 2),
        "test_top3_captured": projection.captured_fraction(test_values, 3),
    }


def _class_balance_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=DIGIT_COUNT).float()
    if (counts == 0).any():
        raise ValueError("every digit must appear in the fit split")
    weights = counts.sum() / (DIGIT_COUNT * counts[labels])
    return weights / weights.mean()


@dataclass
class RidgeReadout:
    x_mean: torch.Tensor
    x_scale: torch.Tensor
    y_mean: torch.Tensor
    weights: torch.Tensor
    ridge_lambda: float

    def predict(self, values: torch.Tensor) -> torch.Tensor:
        standardized = (values.float() - self.x_mean) / self.x_scale
        return (standardized @ self.weights) + self.y_mean


def fit_ridge_one_hot(
    values: torch.Tensor,
    labels: torch.Tensor,
    ridge_lambda: float,
) -> RidgeReadout:
    values = values.float()
    labels = labels.long()
    targets = F.one_hot(labels, num_classes=DIGIT_COUNT).float()
    sample_weights = _class_balance_weights(labels)
    weight_sum = sample_weights.sum()
    x_mean = (values * sample_weights[:, None]).sum(0, keepdim=True) / weight_sum
    centered = values - x_mean
    x_variance = (
        centered.square() * sample_weights[:, None]
    ).sum(0, keepdim=True) / weight_sum
    x_scale = x_variance.sqrt().clamp_min(1e-5)
    standardized = centered / x_scale
    y_mean = (
        targets * sample_weights[:, None]
    ).sum(0, keepdim=True) / weight_sum
    y_centered = targets - y_mean
    weighted_x = standardized * sample_weights[:, None]
    covariance = standardized.T @ weighted_x / weight_sum
    cross_covariance = standardized.T @ (
        y_centered * sample_weights[:, None]
    ) / weight_sum
    penalty = torch.eye(covariance.size(0), dtype=covariance.dtype)
    weights = torch.linalg.solve(
        covariance + float(ridge_lambda) * penalty,
        cross_covariance,
    )
    return RidgeReadout(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        weights=weights,
        ridge_lambda=float(ridge_lambda),
    )


def periodic_metrics_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    order: tuple[int, ...] = NATURAL_ORDER,
) -> dict:
    code = periodic_code(order, dtype=scores.dtype).to(scores.device)
    predicted = scores.float() @ code
    targets = code[labels.long()]
    predicted_angle = torch.atan2(predicted[:, 1], predicted[:, 0])
    target_angle = torch.atan2(targets[:, 1], targets[:, 0])
    angle_error = torch.atan2(
        torch.sin(predicted_angle - target_angle),
        torch.cos(predicted_angle - target_angle),
    ).abs()
    cosine_alignment = torch.cos(angle_error)
    nearest = torch.cdist(predicted, code).argmin(dim=1)
    residual = (predicted - targets).square().sum()
    target_centered = targets - targets.mean(0, keepdim=True)
    total = target_centered.square().sum().clamp_min(1e-20)
    radii = predicted.norm(dim=1)
    return {
        "mean_cosine_alignment": float(cosine_alignment.mean().item()),
        "angular_mae_degrees": float(
            angle_error.mean().item() * 180.0 / math.pi
        ),
        "sector_accuracy": float((nearest == labels).float().mean().item()),
        "circular_r2": float((1 - residual / total).item()),
        "predicted_radius_mean": float(radii.mean().item()),
        "predicted_radius_cv": float(
            radii.std(unbiased=False) / radii.mean().clamp_min(1e-12)
        ),
    }


def helix_metrics_from_scores(
    scores: torch.Tensor,
    labels: torch.Tensor,
    order: tuple[int, ...] = NATURAL_ORDER,
) -> dict:
    code = helix_code(order, dtype=scores.dtype).to(scores.device)
    predicted = scores.float() @ code
    targets = code[labels.long()]
    residual = (predicted - targets).square().sum(0)
    total = (targets - targets.mean(0, keepdim=True)).square().sum(0).clamp_min(1e-20)
    r2 = 1 - residual / total
    return {
        "coordinate_r2": r2.tolist(),
        "transverse_r2": float(r2[:2].mean().item()),
        "axial_digit_r2": float(r2[2].item()),
        "overall_r2": float((1 - residual.sum() / total.sum()).item()),
    }


def select_ridge_lambda(
    train_values: torch.Tensor,
    train_labels: torch.Tensor,
    validation_values: torch.Tensor,
    validation_labels: torch.Tensor,
    lambdas: tuple[float, ...] = DEFAULT_RIDGE_LAMBDAS,
) -> tuple[float, list[dict]]:
    candidates = []
    for ridge_lambda in lambdas:
        model = fit_ridge_one_hot(train_values, train_labels, ridge_lambda)
        scores = model.predict(validation_values)
        metrics = periodic_metrics_from_scores(scores, validation_labels)
        one_hot = F.one_hot(
            validation_labels.long(), num_classes=DIGIT_COUNT
        ).float()
        candidates.append({
            "lambda": float(ridge_lambda),
            "one_hot_mse": float(F.mse_loss(scores, one_hot).item()),
            "one_hot_accuracy": float(
                (scores.argmax(dim=1) == validation_labels).float().mean().item()
            ),
            **metrics,
        })
    # One-hot regression error is invariant to the ordering assigned to the
    # nine digit labels. Tuning on it avoids favoring the natural cycle before
    # comparing that cycle with the 20,159 alternatives.
    best = min(
        candidates,
        key=lambda item: (
            round(item["one_hot_mse"], 12),
            -item["lambda"],
        ),
    )
    return best["lambda"], candidates


def _centroid_score_alignment(
    score_centroids: torch.Tensor,
    order: tuple[int, ...],
) -> float:
    code = periodic_code(order, dtype=score_centroids.dtype)
    predicted = score_centroids @ code
    normalized = F.normalize(predicted, dim=1, eps=1e-12)
    return float((normalized * code).sum(1).mean().item())


def periodic_cycle_null(
    scores: torch.Tensor,
    labels: torch.Tensor,
    train_shortest_cycle: tuple[int, ...],
) -> dict:
    score_centroids = class_centroids(scores, labels)
    alignments = torch.empty(len(unique_cycle_orders()))
    natural_index = unique_cycle_orders().index(NATURAL_ORDER)
    selected_index = unique_cycle_orders().index(train_shortest_cycle)
    for index, order in enumerate(unique_cycle_orders()):
        alignments[index] = _centroid_score_alignment(score_centroids, order)
    natural = float(alignments[natural_index].item())
    selected = float(alignments[selected_index].item())
    median = float(alignments.median().item())
    return {
        "natural_centroid_alignment": natural,
        "natural_alignment_percentile": percentile_at(alignments, natural),
        "natural_alignment_upper_tail_fraction": upper_tail_fraction(
            alignments, natural
        ),
        "natural_alignment_advantage_over_median": natural - median,
        "train_shortest_centroid_alignment": selected,
        "train_shortest_alignment_percentile": percentile_at(
            alignments, selected
        ),
        "train_shortest_alignment_upper_tail_fraction": upper_tail_fraction(
            alignments, selected
        ),
        "train_shortest_alignment_advantage_over_median": selected - median,
        "alignment_null": {
            "minimum": float(alignments.min().item()),
            "median": median,
            "maximum": float(alignments.max().item()),
        },
        "alignments": alignments.tolist(),
    }


def puzzle_bootstrap_periodic(
    scores: torch.Tensor,
    labels: torch.Tensor,
    puzzle_ids: torch.Tensor,
    bucket_ids: torch.Tensor,
    *,
    seed: int,
    samples: int = 400,
) -> dict:
    unique_puzzles = torch.unique(puzzle_ids).tolist()
    puzzle_buckets = {}
    for puzzle_id in unique_puzzles:
        selected_buckets = torch.unique(bucket_ids[puzzle_ids == puzzle_id])
        if len(selected_buckets) != 1:
            raise ValueError("each puzzle must belong to exactly one rating bucket")
        puzzle_buckets[puzzle_id] = int(selected_buckets.item())
    by_bucket = {}
    for puzzle_id, bucket_id in puzzle_buckets.items():
        by_bucket.setdefault(bucket_id, []).append(puzzle_id)
    code = periodic_code(dtype=scores.dtype).to(scores.device)
    predicted = scores.float() @ code
    targets = code[labels.long()]
    cosine = F.cosine_similarity(predicted, targets, dim=1, eps=1e-12)
    nearest = torch.cdist(predicted, code).argmin(dim=1)
    row_metrics = torch.stack((
        cosine,
        (nearest == labels).float(),
    ), dim=1)
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(samples):
        selected_rows = []
        for puzzles in by_bucket.values():
            draw = torch.randint(
                len(puzzles),
                (len(puzzles),),
                generator=generator,
            )
            for drawn_index in draw.tolist():
                selected_rows.append(
                    torch.nonzero(
                        puzzle_ids == puzzles[drawn_index],
                        as_tuple=False,
                    ).flatten()
                )
        selected_rows = torch.cat(selected_rows)
        estimates.append(row_metrics[selected_rows].mean(0))
    estimates = torch.stack(estimates)
    lower = torch.quantile(estimates, 0.025, dim=0)
    upper = torch.quantile(estimates, 0.975, dim=0)
    headline = row_metrics.mean(0)
    return {
        "puzzle_count": len(unique_puzzles),
        "rating_bucket_count": len(by_bucket),
        "bootstrap_samples": samples,
        "estimator": "blank-cell-weighted metric; puzzles resampled within rating bucket",
        "mean_cosine_alignment": {
            "estimate": float(headline[0].item()),
            "ci95": [float(lower[0].item()), float(upper[0].item())],
        },
        "sector_accuracy": {
            "estimate": float(headline[1].item()),
            "ci95": [float(lower[1].item()), float(upper[1].item())],
        },
    }


def analyze_periodic_readout(
    train_values: torch.Tensor,
    train_labels: torch.Tensor,
    validation_values: torch.Tensor,
    validation_labels: torch.Tensor,
    test_values: torch.Tensor,
    test_labels: torch.Tensor,
    test_puzzle_ids: torch.Tensor,
    test_bucket_ids: torch.Tensor,
    train_shortest_cycle: tuple[int, ...],
    *,
    seed: int,
) -> tuple[dict, RidgeReadout, torch.Tensor]:
    selected_lambda, validation_candidates = select_ridge_lambda(
        train_values,
        train_labels,
        validation_values,
        validation_labels,
    )
    train_model = fit_ridge_one_hot(
        train_values,
        train_labels,
        selected_lambda,
    )
    validation_scores = train_model.predict(validation_values)
    combined_values = torch.cat((train_values, validation_values), dim=0)
    combined_labels = torch.cat((train_labels, validation_labels), dim=0)
    final_model = fit_ridge_one_hot(
        combined_values,
        combined_labels,
        selected_lambda,
    )
    test_scores = final_model.predict(test_values)
    validation_metrics = periodic_metrics_from_scores(
        validation_scores,
        validation_labels,
    )
    test_metrics = periodic_metrics_from_scores(test_scores, test_labels)
    test_selected_cycle_metrics = periodic_metrics_from_scores(
        test_scores,
        test_labels,
        train_shortest_cycle,
    )
    null = periodic_cycle_null(
        test_scores,
        test_labels,
        train_shortest_cycle,
    )
    summary = {
        "selected_lambda": selected_lambda,
        "validation_candidates": validation_candidates,
        "validation_natural_cycle": validation_metrics,
        "test_natural_cycle": test_metrics,
        "test_natural_helix": helix_metrics_from_scores(
            test_scores,
            test_labels,
        ),
        "test_train_shortest_cycle": test_selected_cycle_metrics,
        "cycle_null": null,
        "puzzle_bootstrap": puzzle_bootstrap_periodic(
            test_scores,
            test_labels,
            test_puzzle_ids,
            test_bucket_ids,
            seed=seed,
        ),
    }
    return summary, final_model, test_scores


def per_iteration_periodic_metrics(
    scores: torch.Tensor,
    labels: torch.Tensor,
    iterations: torch.Tensor,
) -> dict:
    result = {}
    for iteration in torch.unique(iterations).tolist():
        mask = iterations == iteration
        result[str(int(iteration))] = periodic_metrics_from_scores(
            scores[mask], labels[mask]
        )
    return result


def stratified_split_indices(
    bucket_names: list[str],
    *,
    seed: int,
    train_fraction: float = 0.5,
    validation_fraction: float = 0.25,
) -> dict[str, list[int]]:
    """Split whole puzzles within each difficulty bucket."""

    grouped: dict[str, list[int]] = {}
    for index, bucket in enumerate(bucket_names):
        grouped.setdefault(bucket, []).append(index)
    generator = np.random.default_rng(seed)
    splits = {"train": [], "validation": [], "test": []}
    for bucket, indices in grouped.items():
        shuffled = np.asarray(indices)[generator.permutation(len(indices))].tolist()
        train_count = int(round(len(indices) * train_fraction))
        validation_count = int(round(len(indices) * validation_fraction))
        train_count = max(1, min(train_count, len(indices) - 2))
        validation_count = max(
            1,
            min(validation_count, len(indices) - train_count - 1),
        )
        splits["train"].extend(shuffled[:train_count])
        splits["validation"].extend(
            shuffled[train_count : train_count + validation_count]
        )
        splits["test"].extend(shuffled[train_count + validation_count :])
    return {name: sorted(indices) for name, indices in splits.items()}


def downsample_indices(
    labels: torch.Tensor,
    iterations: torch.Tensor,
    maximum: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Downsample roughly uniformly over digit and recorded iteration."""

    if labels.numel() <= maximum:
        return torch.arange(labels.numel())
    generator = torch.Generator().manual_seed(seed)
    groups = []
    unique_iterations = torch.unique(iterations)
    per_group = max(1, maximum // (DIGIT_COUNT * len(unique_iterations)))
    for digit in range(DIGIT_COUNT):
        for iteration in unique_iterations.tolist():
            indices = torch.nonzero(
                (labels == digit) & (iterations == iteration),
                as_tuple=False,
            ).flatten()
            if len(indices) > per_group:
                order = torch.randperm(len(indices), generator=generator)[:per_group]
                indices = indices[order]
            groups.append(indices)
    selected = torch.cat(groups)
    if len(selected) < maximum:
        used = torch.zeros(labels.numel(), dtype=torch.bool)
        used[selected] = True
        remaining = torch.nonzero(~used, as_tuple=False).flatten()
        order = torch.randperm(len(remaining), generator=generator)
        selected = torch.cat((selected, remaining[order[: maximum - len(selected)]]))
    return selected[:maximum]
