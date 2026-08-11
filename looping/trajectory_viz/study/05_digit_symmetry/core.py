"""Pure statistical helpers for the digit-symmetry study.

All fitting functions accept complete discovery/validation/final arrays. The
caller is responsible for splitting whole puzzles before constructing those
arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np


DIGIT_COUNT = 9
SPLIT_NAMES = ("discovery", "validation", "final")


def stratified_three_way_split(rating_buckets, *, seed):
    """Split whole puzzles equally within each rating bucket."""

    rating_buckets = np.asarray(rating_buckets)
    if rating_buckets.ndim != 1 or not len(rating_buckets):
        raise ValueError("rating_buckets must be a non-empty vector")
    generator = np.random.default_rng(seed)
    split_by_puzzle = np.empty(len(rating_buckets), dtype=object)
    for bucket in dict.fromkeys(rating_buckets.tolist()):
        indices = np.flatnonzero(rating_buckets == bucket)
        if len(indices) < 3 or len(indices) % 3:
            raise ValueError(
                f"rating bucket {bucket!r} must contain a positive multiple "
                f"of three puzzles, got {len(indices)}"
            )
        shuffled = generator.permutation(indices)
        per_split = len(indices) // 3
        for split_index, split_name in enumerate(SPLIT_NAMES):
            start = split_index * per_split
            split_by_puzzle[shuffled[start : start + per_split]] = split_name
    return split_by_puzzle


def puzzle_equal_weights(puzzle_ids):
    """Give each represented puzzle equal total weight."""

    puzzle_ids = np.asarray(puzzle_ids)
    if puzzle_ids.ndim != 1 or not len(puzzle_ids):
        raise ValueError("puzzle_ids must be a non-empty vector")
    _, inverse, counts = np.unique(
        puzzle_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def puzzle_mean_accuracy(predictions, labels, puzzle_ids):
    """Average cell accuracy within puzzles, then average puzzles."""

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    puzzle_ids = np.asarray(puzzle_ids)
    if predictions.shape != labels.shape or labels.shape != puzzle_ids.shape:
        raise ValueError("predictions, labels, and puzzle_ids must align")
    return float(
        np.mean(
            [
                np.mean(predictions[puzzle_ids == puzzle] == labels[puzzle_ids == puzzle])
                for puzzle in np.unique(puzzle_ids)
            ]
        )
    )


@dataclass(frozen=True)
class RidgeDecoder:
    coefficients: np.ndarray
    intercept: np.ndarray

    def predict_scores(self, values):
        values = np.asarray(values, dtype=np.float64)
        return values @ self.coefficients + self.intercept

    def predict(self, values):
        return self.predict_scores(values).argmax(axis=1)


def fit_ridge_decoder(values, labels, sample_weight, *, alpha):
    """Fit weighted least-squares digit classification with an intercept."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("values and labels have incompatible shapes")
    if weights.shape != labels.shape or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("sample_weight must be aligned and nonnegative")
    if np.any((labels < 0) | (labels >= DIGIT_COUNT)):
        raise ValueError("labels must lie in 0..8")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")

    weights = weights / weights.sum()
    value_mean = np.sum(weights[:, None] * values, axis=0)
    targets = np.eye(DIGIT_COUNT, dtype=np.float64)[labels]
    target_mean = np.sum(weights[:, None] * targets, axis=0)
    centered_values = values - value_mean
    centered_targets = targets - target_mean
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        covariance = centered_values.T @ (weights[:, None] * centered_values)
        cross_covariance = centered_values.T @ (
            weights[:, None] * centered_targets
        )
    if not np.all(np.isfinite(covariance)) or not np.all(
        np.isfinite(cross_covariance)
    ):
        raise FloatingPointError("ridge covariance is non-finite")
    scale = max(float(np.trace(covariance)) / max(values.shape[1], 1), 1e-12)
    regularized = covariance + alpha * scale * np.eye(values.shape[1])
    coefficients = np.linalg.solve(regularized, cross_covariance)
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError("ridge coefficients are non-finite")
    intercept = target_mean - value_mean @ coefficients
    return RidgeDecoder(coefficients=coefficients, intercept=intercept)


def select_ridge_alpha(
    discovery_values,
    discovery_labels,
    discovery_weights,
    validation_values,
    validation_labels,
    validation_puzzle_ids,
    *,
    candidates,
):
    """Choose a predefined ridge value on validation puzzle accuracy."""

    records = []
    for alpha in candidates:
        decoder = fit_ridge_decoder(
            discovery_values,
            discovery_labels,
            discovery_weights,
            alpha=alpha,
        )
        accuracy = puzzle_mean_accuracy(
            decoder.predict(validation_values),
            validation_labels,
            validation_puzzle_ids,
        )
        records.append((float(accuracy), float(alpha), decoder))
    best_accuracy, best_alpha, best_decoder = max(
        records, key=lambda record: (record[0], -record[1])
    )
    return {
        "alpha": best_alpha,
        "validation_accuracy": best_accuracy,
        "decoder": best_decoder,
        "candidates": [
            {"alpha": alpha, "validation_accuracy": accuracy}
            for accuracy, alpha, _ in records
        ],
    }


def category_centroids(values, labels, puzzle_ids):
    """Average each digit within puzzles and then equally across puzzles."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    puzzle_ids = np.asarray(puzzle_ids)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("values and labels have incompatible shapes")
    if puzzle_ids.shape != labels.shape:
        raise ValueError("puzzle_ids must align with labels")

    centroids = []
    for digit in range(DIGIT_COUNT):
        per_puzzle = []
        for puzzle in np.unique(puzzle_ids[labels == digit]):
            mask = (labels == digit) & (puzzle_ids == puzzle)
            per_puzzle.append(values[mask].mean(axis=0))
        if not per_puzzle:
            raise ValueError(f"digit {digit} has no observations")
        centroids.append(np.mean(per_puzzle, axis=0))
    return np.stack(centroids)


def centered_centroids(centroids):
    centroids = np.asarray(centroids, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[0] != DIGIT_COUNT:
        raise ValueError("centroids must have shape [9, features]")
    return centroids - centroids.mean(axis=0, keepdims=True)


def category_subspace(centroids, *, relative_tolerance=1e-8):
    """Return an orthonormal feature basis for centered digit means."""

    centered = centered_centroids(centroids)
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    if not len(singular_values) or singular_values[0] <= 0:
        raise ValueError("digit centroids have no nonconstant component")
    rank = int(np.sum(singular_values > relative_tolerance * singular_values[0]))
    return right_vectors[:rank].T


def random_orthonormal_basis(feature_count, rank, generator):
    if not 0 < rank <= feature_count:
        raise ValueError("rank must lie in 1..feature_count")
    matrix = generator.normal(size=(feature_count, rank))
    basis, _ = np.linalg.qr(matrix, mode="reduced")
    return basis


def subspace_overlap(first_basis, second_basis):
    """Mean squared cosine of principal angles between two subspaces."""

    first_basis = np.asarray(first_basis, dtype=np.float64)
    second_basis = np.asarray(second_basis, dtype=np.float64)
    if first_basis.ndim != 2 or second_basis.ndim != 2:
        raise ValueError("bases must be matrices")
    if first_basis.shape[0] != second_basis.shape[0]:
        raise ValueError("bases must use the same feature coordinates")
    denominator = min(first_basis.shape[1], second_basis.shape[1])
    if denominator == 0:
        raise ValueError("bases must contain at least one direction")
    return float(np.square(first_basis.T @ second_basis).sum() / denominator)


def pairwise_distance_matrix(centroids):
    centered = centered_centroids(centroids)
    differences = centered[:, None, :] - centered[None, :, :]
    return np.linalg.norm(differences, axis=-1)


def _standardized_distance_vector(distance_matrix):
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    if distance_matrix.shape != (DIGIT_COUNT, DIGIT_COUNT):
        raise ValueError("distance matrix must have shape [9, 9]")
    row, column = np.triu_indices(DIGIT_COUNT, 1)
    vector = distance_matrix[row, column]
    standard_deviation = vector.std()
    if standard_deviation <= 1e-12:
        raise ValueError("distance matrix has no off-diagonal variation")
    return (vector - vector.mean()) / standard_deviation


def all_digit_permutations():
    return np.asarray(list(itertools.permutations(range(DIGIT_COUNT))), dtype=np.int16)


def permutation_correlations(first_distances, second_distances, permutations):
    """Correlate one digit RDM with every relabeling of another RDM."""

    first = _standardized_distance_vector(first_distances)
    second = np.asarray(second_distances, dtype=np.float64)
    permutations = np.asarray(permutations, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != DIGIT_COUNT:
        raise ValueError("permutations must have shape [count, 9]")
    row, column = np.triu_indices(DIGIT_COUNT, 1)
    permuted = second[permutations[:, row], permutations[:, column]]
    permuted = (permuted - permuted.mean(axis=1, keepdims=True)) / np.maximum(
        permuted.std(axis=1, keepdims=True), 1e-12
    )
    return np.mean(permuted * first[None, :], axis=1)


def select_rdm_permutation(first_centroids, second_centroids, permutations):
    correlations = permutation_correlations(
        pairwise_distance_matrix(first_centroids),
        pairwise_distance_matrix(second_centroids),
        permutations,
    )
    best_index = int(np.argmax(correlations))
    return permutations[best_index].astype(int), float(correlations[best_index])


def rdm_correlation(first_centroids, second_centroids, permutation=None):
    if permutation is None:
        permutation = np.arange(DIGIT_COUNT)
    second = np.asarray(second_centroids)[np.asarray(permutation, dtype=np.int64)]
    first_vector = _standardized_distance_vector(
        pairwise_distance_matrix(first_centroids)
    )
    second_vector = _standardized_distance_vector(pairwise_distance_matrix(second))
    return float(np.mean(first_vector * second_vector))


def procrustes_similarity(first_centroids, second_centroids, permutation=None):
    """Shape similarity after translation, scaling, and orthogonal rotation."""

    if permutation is None:
        permutation = np.arange(DIGIT_COUNT)
    first = centered_centroids(first_centroids)
    second = centered_centroids(second_centroids)[
        np.asarray(permutation, dtype=np.int64)
    ]
    first /= max(np.linalg.norm(first), 1e-12)
    second /= max(np.linalg.norm(second), 1e-12)
    singular_values = np.linalg.svd(first.T @ second, compute_uv=False)
    return float(np.clip(singular_values.sum(), 0.0, 1.0))


def digit_code(labels, kind, order=None):
    """Encode categories using an ordinal, cyclic, or unconstrained code."""

    labels = np.asarray(labels, dtype=np.int64)
    if np.any((labels < 0) | (labels >= DIGIT_COUNT)):
        raise ValueError("labels must lie in 0..8")
    if order is None:
        order = np.arange(DIGIT_COUNT)
    order = np.asarray(order, dtype=np.int64)
    if sorted(order.tolist()) != list(range(DIGIT_COUNT)):
        raise ValueError("order must be a permutation of 0..8")
    position = np.empty(DIGIT_COUNT, dtype=np.int64)
    position[order] = np.arange(DIGIT_COUNT)
    category_position = position[labels]
    if kind == "ordinal":
        return ((category_position - 4.0) / math.sqrt(20.0 / 3.0))[:, None]
    if kind == "cyclic":
        angle = 2.0 * np.pi * category_position / DIGIT_COUNT
        return np.column_stack((np.cos(angle), np.sin(angle)))
    if kind == "categorical":
        return (labels[:, None] == np.arange(DIGIT_COUNT - 1)).astype(float)
    raise ValueError(f"unknown digit code {kind!r}")


@dataclass(frozen=True)
class MultivariateLinearFit:
    coefficients: np.ndarray
    intercept: np.ndarray

    def predict(self, design):
        design = np.asarray(design, dtype=np.float64)
        return design @ self.coefficients + self.intercept


def fit_multivariate_linear(design, response, sample_weight):
    design = np.asarray(design, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if design.ndim != 2 or response.ndim != 2 or len(design) != len(response):
        raise ValueError("design and response must be aligned matrices")
    if weights.shape != (len(design),) or weights.sum() <= 0:
        raise ValueError("sample_weight must align with design")
    weights = weights / weights.sum()
    design_mean = np.sum(weights[:, None] * design, axis=0)
    response_mean = np.sum(weights[:, None] * response, axis=0)
    centered_design = design - design_mean
    centered_response = response - response_mean
    square_root_weight = np.sqrt(weights)[:, None]
    coefficients = np.linalg.lstsq(
        centered_design * square_root_weight,
        centered_response * square_root_weight,
        rcond=1e-8,
    )[0]
    intercept = response_mean - design_mean @ coefficients
    return MultivariateLinearFit(coefficients=coefficients, intercept=intercept)


def weighted_sse(response, prediction, sample_weight):
    response = np.asarray(response, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if response.shape != prediction.shape or weights.shape != (len(response),):
        raise ValueError("response, prediction, and weights must align")
    return float(np.sum(weights[:, None] * np.square(response - prediction)))


def fit_order_probe(
    discovery_values,
    discovery_labels,
    discovery_weights,
    evaluation_values,
    evaluation_labels,
    evaluation_weights,
    *,
    kind,
    order=None,
):
    design = digit_code(discovery_labels, kind, order)
    fit = fit_multivariate_linear(design, discovery_values, discovery_weights)
    prediction = fit.predict(digit_code(evaluation_labels, kind, order))
    return weighted_sse(evaluation_values, prediction, evaluation_weights)


def effect_fraction(*, base_sse, model_sse, categorical_sse):
    categorical_gain = base_sse - categorical_sse
    if categorical_gain <= np.finfo(float).eps * max(1.0, abs(base_sse)):
        return math.nan
    return float((base_sse - model_sse) / categorical_gain)


def summarize_null(observed, null_values, *, larger_is_better=True):
    null_values = np.asarray(null_values, dtype=np.float64)
    if null_values.ndim != 1 or not len(null_values):
        raise ValueError("null_values must be a non-empty vector")
    if larger_is_better:
        exceedances = np.sum(null_values >= observed)
        percentile = np.mean(null_values <= observed)
    else:
        exceedances = np.sum(null_values <= observed)
        percentile = np.mean(null_values >= observed)
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null_values)),
        "null_q05": float(np.quantile(null_values, 0.05)),
        "null_q50": float(np.quantile(null_values, 0.50)),
        "null_q95": float(np.quantile(null_values, 0.95)),
        "percentile": float(percentile),
        "p_one_sided": float((1 + exceedances) / (1 + len(null_values))),
        "null_values": null_values.tolist(),
    }
