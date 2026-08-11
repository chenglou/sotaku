"""Pure statistical helpers for the held-out decision-margin study.

The functions in this module never load checkpoints or puzzles.  Callers must
split whole puzzles before constructing observation rows.  Every fit uses only
the discovery split, hyperparameters are selected on validation puzzles, and
the final split is accepted only by evaluation helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SPLIT_NAMES = ("discovery", "validation", "final")


def stratified_three_way_split(rating_buckets, *, examples_per_split_bucket, seed):
    """Assign complete puzzles to three equally sized rating-balanced splits."""

    buckets = np.asarray(rating_buckets, dtype=object)
    if buckets.ndim != 1 or not len(buckets):
        raise ValueError("rating_buckets must be a non-empty vector")
    if examples_per_split_bucket <= 0:
        raise ValueError("examples_per_split_bucket must be positive")
    expected = examples_per_split_bucket * len(SPLIT_NAMES)
    generator = np.random.default_rng(seed)
    assignments = np.empty(len(buckets), dtype=object)
    for bucket in dict.fromkeys(buckets.tolist()):
        indices = np.flatnonzero(buckets == bucket)
        if len(indices) != expected:
            raise ValueError(
                f"rating bucket {bucket!r} must contain {expected} puzzles, "
                f"got {len(indices)}"
            )
        shuffled = generator.permutation(indices)
        for split_index, split_name in enumerate(SPLIT_NAMES):
            start = split_index * examples_per_split_bucket
            stop = start + examples_per_split_bucket
            assignments[shuffled[start:stop]] = split_name
    return assignments


def puzzle_equal_weights(puzzle_ids):
    """Give every represented puzzle equal total observation weight."""

    puzzle_ids = np.asarray(puzzle_ids)
    if puzzle_ids.ndim != 1 or not len(puzzle_ids):
        raise ValueError("puzzle_ids must be a non-empty vector")
    _, inverse, counts = np.unique(
        puzzle_ids, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def _validated_rows(features, target, iterations, weights):
    features = np.asarray(features, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    iterations = np.asarray(iterations, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    row_count = len(features)
    if features.ndim != 2 or target.shape != (row_count,):
        raise ValueError("features and target must be aligned rows")
    if iterations.shape != target.shape or weights.shape != target.shape:
        raise ValueError("iterations and weights must align with target")
    if not np.all(np.isfinite(features)) or not np.all(np.isfinite(target)):
        raise ValueError("features and target must be finite")
    if np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError("weights must be finite, nonnegative, and nonzero")
    return features, target, iterations, weights / weights.sum()


def _iteration_indices(observed_iterations, known_iterations):
    lookup = {int(value): index for index, value in enumerate(known_iterations)}
    try:
        return np.asarray(
            [lookup[int(value)] for value in observed_iterations], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"unknown iteration {error.args[0]}") from error


def _weighted_time_means(values, iteration_indices, weights, time_count):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty((time_count,) + values.shape[1:], dtype=np.float64)
    for time_index in range(time_count):
        selected = iteration_indices == time_index
        if not np.any(selected):
            raise ValueError(f"iteration index {time_index} has no rows")
        selected_weights = weights[selected]
        selected_weights = selected_weights / selected_weights.sum()
        result[time_index] = np.tensordot(
            selected_weights, values[selected], axes=(0, 0)
        )
    return result


@dataclass(frozen=True)
class OrderedProbe:
    """One supervised hidden-state axis after removing iteration means."""

    iteration_values: np.ndarray
    feature_time_means: np.ndarray
    feature_scales: np.ndarray
    target_time_means: np.ndarray
    coefficient: np.ndarray
    alpha: float

    def _time_indices(self, iterations):
        return _iteration_indices(iterations, self.iteration_values)

    def transformed_features(self, features, iterations):
        features = np.asarray(features, dtype=np.float64)
        time_indices = self._time_indices(iterations)
        return (
            features - self.feature_time_means[time_indices]
        ) / self.feature_scales

    def coordinate(self, features, iterations):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            coordinate = (
                self.transformed_features(features, iterations) @ self.coefficient
            )
        if not np.all(np.isfinite(coordinate)):
            raise FloatingPointError("ordered-probe coordinate is not finite")
        return coordinate

    def baseline(self, iterations):
        return self.target_time_means[self._time_indices(iterations)]

    def predict(self, features, iterations):
        return self.baseline(iterations) + self.coordinate(features, iterations)

    @property
    def raw_axis(self):
        return self.coefficient / self.feature_scales


def fit_ordered_probe(features, target, iterations, weights, *, alpha):
    """Fit one ridge-regression coordinate after iteration residualization."""

    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    features, target, iterations, weights = _validated_rows(
        features, target, iterations, weights
    )
    iteration_values = np.unique(iterations)
    time_indices = _iteration_indices(iterations, iteration_values)
    feature_means = _weighted_time_means(
        features, time_indices, weights, len(iteration_values)
    )
    target_means = _weighted_time_means(
        target, time_indices, weights, len(iteration_values)
    )
    centered_features = features - feature_means[time_indices]
    centered_target = target - target_means[time_indices]
    scales = np.sqrt(np.sum(weights[:, None] * centered_features**2, axis=0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = centered_features / scales
    # Accelerate-backed NumPy can report stale LAPACK floating-point flags on
    # a later finite matmul, so verify results explicitly after suppressing
    # those spurious warnings.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        covariance = standardized.T @ (weights[:, None] * standardized)
    covariance_scale = max(
        float(np.trace(covariance)) / max(features.shape[1], 1), 1e-12
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cross_covariance = standardized.T @ (weights * centered_target)
    if not np.all(np.isfinite(covariance)) or not np.all(
        np.isfinite(cross_covariance)
    ):
        raise FloatingPointError("ordered-probe normal equations are not finite")
    coefficient = np.linalg.solve(
        covariance + alpha * covariance_scale * np.eye(features.shape[1]),
        cross_covariance,
    )
    return OrderedProbe(
        iteration_values=iteration_values,
        feature_time_means=feature_means,
        feature_scales=scales,
        target_time_means=target_means,
        coefficient=coefficient,
        alpha=float(alpha),
    )


def _weighted_correlation(first, second, weights):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    centered_first = first - np.sum(weights * first)
    centered_second = second - np.sum(weights * second)
    denominator = math.sqrt(
        float(np.sum(weights * centered_first**2))
        * float(np.sum(weights * centered_second**2))
    )
    if denominator <= 1e-15:
        return math.nan
    return float(np.sum(weights * centered_first * centered_second) / denominator)


def _rankdata(values):
    """Average ranks for ties, matching the usual Spearman definition."""

    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def continuous_metrics(target, prediction, baseline, weights):
    """Return puzzle-weighted regression effects against an iteration baseline."""

    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (target.shape == prediction.shape == baseline.shape == weights.shape):
        raise ValueError("target, prediction, baseline, and weights must align")
    weights = weights / weights.sum()
    sse = float(np.sum(weights * (target - prediction) ** 2))
    baseline_sse = float(np.sum(weights * (target - baseline) ** 2))
    target_mean = float(np.sum(weights * target))
    null_sse = float(np.sum(weights * (target - target_mean) ** 2))
    partial_r2 = (
        1.0 - sse / baseline_sse if baseline_sse > 1e-15 else math.nan
    )
    r2 = 1.0 - sse / null_sse if null_sse > 1e-15 else math.nan
    return {
        "partial_r2_over_iteration": float(partial_r2),
        "r2": float(r2),
        "rmse": math.sqrt(max(sse, 0.0)),
        "iteration_baseline_rmse": math.sqrt(max(baseline_sse, 0.0)),
        "pearson": _weighted_correlation(target, prediction, weights),
        "spearman": _weighted_correlation(
            _rankdata(target), _rankdata(prediction), weights
        ),
        "sse": sse,
        "iteration_baseline_sse": baseline_sse,
    }


def evaluate_ordered_probe(probe, features, target, iterations, weights):
    prediction = probe.predict(features, iterations)
    baseline = probe.baseline(iterations)
    result = continuous_metrics(target, prediction, baseline, weights)
    result["prediction"] = prediction
    result["baseline"] = baseline
    result["coordinate"] = probe.coordinate(features, iterations)
    return result


def select_ordered_alpha(
    discovery_features,
    discovery_target,
    discovery_iterations,
    discovery_weights,
    validation_features,
    validation_target,
    validation_iterations,
    validation_weights,
    *,
    candidates,
):
    """Select a predefined ridge value using validation partial R-squared."""

    records = []
    for alpha in candidates:
        probe = fit_ordered_probe(
            discovery_features,
            discovery_target,
            discovery_iterations,
            discovery_weights,
            alpha=alpha,
        )
        metrics = evaluate_ordered_probe(
            probe,
            validation_features,
            validation_target,
            validation_iterations,
            validation_weights,
        )
        score = metrics["partial_r2_over_iteration"]
        records.append((float(score), float(alpha), probe, metrics))
    score, alpha, probe, metrics = max(
        records,
        key=lambda record: (
            -math.inf if not np.isfinite(record[0]) else record[0],
            -record[1],
        ),
    )
    return {
        "alpha": alpha,
        "validation_metrics": {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, np.ndarray)
        },
        "candidates": [
            {
                "alpha": candidate_alpha,
                "partial_r2_over_iteration": candidate_score,
            }
            for candidate_score, candidate_alpha, _, _ in records
        ],
        "probe": probe,
    }


def weighted_quantile(values, quantiles, weights):
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.asarray(quantiles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("values and weights must be aligned vectors")
    if np.any((quantiles <= 0) | (quantiles >= 1)):
        raise ValueError("quantiles must lie strictly between zero and one")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= sorted_weights.sum()
    return np.interp(quantiles, cumulative, sorted_values)


def make_bin_thresholds(values, weights, *, bin_count):
    if bin_count < 2:
        raise ValueError("bin_count must be at least two")
    thresholds = weighted_quantile(
        values,
        np.arange(1, bin_count, dtype=np.float64) / bin_count,
        weights,
    )
    if np.any(np.diff(thresholds) <= 1e-12):
        raise ValueError("discovery target does not support distinct quantile bins")
    return thresholds


def assign_bins(values, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.ndim != 1 or np.any(np.diff(thresholds) <= 0):
        raise ValueError("thresholds must be a strictly increasing vector")
    return np.searchsorted(thresholds, np.asarray(values), side="right")


@dataclass(frozen=True)
class CategoricalProbe:
    """Unconstrained ridge decoder for discovery-defined margin bins."""

    iteration_values: np.ndarray
    feature_time_means: np.ndarray
    feature_scales: np.ndarray
    class_time_means: np.ndarray
    coefficients: np.ndarray
    thresholds: np.ndarray
    alpha: float

    def _time_indices(self, iterations):
        return _iteration_indices(iterations, self.iteration_values)

    def predict_scores(self, features, iterations):
        features = np.asarray(features, dtype=np.float64)
        time_indices = self._time_indices(iterations)
        standardized = (
            features - self.feature_time_means[time_indices]
        ) / self.feature_scales
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            scores = (
                self.class_time_means[time_indices]
                + standardized @ self.coefficients
            )
        if not np.all(np.isfinite(scores)):
            raise FloatingPointError("categorical-probe scores are not finite")
        return scores

    def predict(self, features, iterations):
        return self.predict_scores(features, iterations).argmax(axis=1)

    def baseline_predict(self, iterations):
        time_indices = self._time_indices(iterations)
        return self.class_time_means[time_indices].argmax(axis=1)


def fit_categorical_probe(
    features, target, iterations, weights, *, thresholds, alpha
):
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")
    features, target, iterations, weights = _validated_rows(
        features, target, iterations, weights
    )
    labels = assign_bins(target, thresholds)
    class_count = len(thresholds) + 1
    one_hot = np.eye(class_count, dtype=np.float64)[labels]
    iteration_values = np.unique(iterations)
    time_indices = _iteration_indices(iterations, iteration_values)
    feature_means = _weighted_time_means(
        features, time_indices, weights, len(iteration_values)
    )
    class_means = _weighted_time_means(
        one_hot, time_indices, weights, len(iteration_values)
    )
    centered_features = features - feature_means[time_indices]
    centered_classes = one_hot - class_means[time_indices]
    scales = np.sqrt(np.sum(weights[:, None] * centered_features**2, axis=0))
    scales = np.where(scales > 1e-12, scales, 1.0)
    standardized = centered_features / scales
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        covariance = standardized.T @ (weights[:, None] * standardized)
    covariance_scale = max(
        float(np.trace(covariance)) / max(features.shape[1], 1), 1e-12
    )
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        cross_covariance = standardized.T @ (
            weights[:, None] * centered_classes
        )
    if not np.all(np.isfinite(covariance)) or not np.all(
        np.isfinite(cross_covariance)
    ):
        raise FloatingPointError("categorical-probe normal equations are not finite")
    coefficients = np.linalg.solve(
        covariance + alpha * covariance_scale * np.eye(features.shape[1]),
        cross_covariance,
    )
    return CategoricalProbe(
        iteration_values=iteration_values,
        feature_time_means=feature_means,
        feature_scales=scales,
        class_time_means=class_means,
        coefficients=coefficients,
        thresholds=np.asarray(thresholds, dtype=np.float64),
        alpha=float(alpha),
    )


def classification_metrics(labels, predictions, baseline_predictions, weights):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    baseline_predictions = np.asarray(baseline_predictions, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if not (
        labels.shape == predictions.shape == baseline_predictions.shape == weights.shape
    ):
        raise ValueError("classification arrays must align")
    weights = weights / weights.sum()
    class_count = int(max(labels.max(), predictions.max(), baseline_predictions.max())) + 1
    accuracy = float(np.sum(weights * (labels == predictions)))
    baseline_accuracy = float(np.sum(weights * (labels == baseline_predictions)))
    recalls = []
    confusion = np.zeros((class_count, class_count), dtype=np.float64)
    for label in range(class_count):
        selected = labels == label
        mass = weights[selected].sum()
        recalls.append(
            float(np.sum(weights[selected] * (predictions[selected] == label)) / mass)
            if mass > 0
            else math.nan
        )
        for predicted in range(class_count):
            confusion[label, predicted] = weights[
                selected & (predictions == predicted)
            ].sum()
    return {
        "accuracy": accuracy,
        "iteration_baseline_accuracy": baseline_accuracy,
        "accuracy_gain": accuracy - baseline_accuracy,
        "macro_recall": float(np.nanmean(recalls)),
        "confusion_weighted": confusion,
    }


def evaluate_categorical_probe(probe, features, target, iterations, weights):
    labels = assign_bins(target, probe.thresholds)
    predictions = probe.predict(features, iterations)
    baseline_predictions = probe.baseline_predict(iterations)
    result = classification_metrics(
        labels, predictions, baseline_predictions, weights
    )
    result["labels"] = labels
    result["predictions"] = predictions
    result["baseline_predictions"] = baseline_predictions
    return result


def select_categorical_alpha(
    discovery_features,
    discovery_target,
    discovery_iterations,
    discovery_weights,
    validation_features,
    validation_target,
    validation_iterations,
    validation_weights,
    *,
    thresholds,
    candidates,
):
    records = []
    for alpha in candidates:
        probe = fit_categorical_probe(
            discovery_features,
            discovery_target,
            discovery_iterations,
            discovery_weights,
            thresholds=thresholds,
            alpha=alpha,
        )
        metrics = evaluate_categorical_probe(
            probe,
            validation_features,
            validation_target,
            validation_iterations,
            validation_weights,
        )
        records.append((metrics["accuracy"], float(alpha), probe, metrics))
    accuracy, alpha, probe, metrics = max(
        records, key=lambda record: (record[0], -record[1])
    )
    return {
        "alpha": alpha,
        "validation_metrics": {
            key: value
            for key, value in metrics.items()
            if not isinstance(value, np.ndarray)
        },
        "candidates": [
            {"alpha": candidate_alpha, "accuracy": candidate_accuracy}
            for candidate_accuracy, candidate_alpha, _, _ in records
        ],
        "probe": probe,
    }


def ordered_bin_metrics(probe, thresholds, features, target, iterations, weights):
    labels = assign_bins(target, thresholds)
    predictions = assign_bins(probe.predict(features, iterations), thresholds)
    baseline = assign_bins(probe.baseline(iterations), thresholds)
    return classification_metrics(labels, predictions, baseline, weights)


def ordered_fraction_of_categorical_gain(ordered_metrics, categorical_metrics):
    denominator = categorical_metrics["accuracy_gain"]
    if denominator <= 1e-12:
        return math.nan
    return float(ordered_metrics["accuracy_gain"] / denominator)


def shuffle_within_iteration(values, iterations, generator):
    """Shuffle labels while preserving every iteration's marginal distribution."""

    values = np.asarray(values)
    iterations = np.asarray(iterations)
    if values.shape[0] != iterations.shape[0]:
        raise ValueError("values and iterations must have the same row count")
    shuffled = values.copy()
    for iteration in np.unique(iterations):
        selected = np.flatnonzero(iterations == iteration)
        shuffled[selected] = values[generator.permutation(selected)]
    return shuffled


def random_rank_one_probe(reference_probe, features, target, iterations, weights, generator):
    """Fit the best scalar calibration inside one random one-dimensional subspace."""

    features, target, iterations, weights = _validated_rows(
        features, target, iterations, weights
    )
    standardized = reference_probe.transformed_features(features, iterations)
    centered_target = target - reference_probe.baseline(iterations)
    direction = generator.normal(size=features.shape[1])
    direction /= max(np.linalg.norm(direction), 1e-15)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        coordinate = standardized @ direction
    if not np.all(np.isfinite(coordinate)):
        raise FloatingPointError("random-subspace coordinate is not finite")
    denominator = float(np.sum(weights * coordinate**2))
    slope = (
        float(np.sum(weights * coordinate * centered_target) / denominator)
        if denominator > 1e-15
        else 0.0
    )
    return OrderedProbe(
        iteration_values=reference_probe.iteration_values.copy(),
        feature_time_means=reference_probe.feature_time_means.copy(),
        feature_scales=reference_probe.feature_scales.copy(),
        target_time_means=reference_probe.target_time_means.copy(),
        coefficient=direction * slope,
        alpha=math.nan,
    )


def summarize_null(observed, null_values, *, larger_is_better=True):
    null_values = np.asarray(null_values, dtype=np.float64)
    finite = np.isfinite(null_values)
    null_values = null_values[finite]
    if not len(null_values):
        raise ValueError("null_values must contain at least one finite value")
    exceedances = int(
        np.sum(null_values >= observed)
        if larger_is_better
        else np.sum(null_values <= observed)
    )
    return {
        "observed": float(observed),
        "null_mean": float(np.mean(null_values)),
        "null_q05": float(np.quantile(null_values, 0.05)),
        "null_q50": float(np.quantile(null_values, 0.50)),
        "null_q95": float(np.quantile(null_values, 0.95)),
        "exceedances": exceedances,
        "p_one_sided": float((1 + exceedances) / (1 + len(null_values))),
        "null_values": null_values.tolist(),
    }


@dataclass(frozen=True)
class CalibratedAxis:
    """A transferred raw axis with target-validation scale and time baseline."""

    raw_axis: np.ndarray
    iteration_values: np.ndarray
    coordinate_time_means: np.ndarray
    target_time_means: np.ndarray
    slope: float
    unconstrained_slope: float

    def _time_indices(self, iterations):
        return _iteration_indices(iterations, self.iteration_values)

    def coordinate(self, features):
        return np.asarray(features, dtype=np.float64) @ self.raw_axis

    def baseline(self, iterations):
        return self.target_time_means[self._time_indices(iterations)]

    def predict(self, features, iterations):
        time_indices = self._time_indices(iterations)
        centered_coordinate = (
            self.coordinate(features) - self.coordinate_time_means[time_indices]
        )
        return self.target_time_means[time_indices] + self.slope * centered_coordinate


def fit_transferred_axis(
    raw_axis,
    validation_features,
    validation_target,
    validation_iterations,
    validation_weights,
    *,
    preserve_orientation=True,
):
    """Calibrate a source axis on target validation rows without rotating it."""

    features, target, iterations, weights = _validated_rows(
        validation_features,
        validation_target,
        validation_iterations,
        validation_weights,
    )
    raw_axis = np.asarray(raw_axis, dtype=np.float64)
    if raw_axis.shape != (features.shape[1],) or np.linalg.norm(raw_axis) <= 1e-15:
        raise ValueError("raw_axis must be a nonzero feature vector")
    raw_axis = raw_axis / np.linalg.norm(raw_axis)
    coordinate = features @ raw_axis
    iteration_values = np.unique(iterations)
    time_indices = _iteration_indices(iterations, iteration_values)
    coordinate_means = _weighted_time_means(
        coordinate, time_indices, weights, len(iteration_values)
    )
    target_means = _weighted_time_means(
        target, time_indices, weights, len(iteration_values)
    )
    centered_coordinate = coordinate - coordinate_means[time_indices]
    centered_target = target - target_means[time_indices]
    denominator = float(np.sum(weights * centered_coordinate**2))
    unconstrained_slope = (
        float(np.sum(weights * centered_coordinate * centered_target) / denominator)
        if denominator > 1e-15
        else 0.0
    )
    slope = max(0.0, unconstrained_slope) if preserve_orientation else unconstrained_slope
    return CalibratedAxis(
        raw_axis=raw_axis,
        iteration_values=iteration_values,
        coordinate_time_means=coordinate_means,
        target_time_means=target_means,
        slope=slope,
        unconstrained_slope=unconstrained_slope,
    )


def evaluate_transferred_axis(axis, features, target, iterations, weights):
    prediction = axis.predict(features, iterations)
    baseline = axis.baseline(iterations)
    result = continuous_metrics(target, prediction, baseline, weights)
    result["prediction"] = prediction
    result["baseline"] = baseline
    result["coordinate"] = axis.coordinate(features)
    result["calibration_slope"] = axis.slope
    result["unconstrained_calibration_slope"] = axis.unconstrained_slope
    result["orientation_preserved"] = bool(axis.unconstrained_slope > 0)
    return result


def aggregate_rows_by_puzzle_iteration(values, puzzle_ids, iterations):
    """Mean arbitrary cell rows into a dense [puzzle, iteration] matrix."""

    values = np.asarray(values, dtype=np.float64)
    puzzle_ids = np.asarray(puzzle_ids)
    iterations = np.asarray(iterations, dtype=np.int64)
    if not (values.shape == puzzle_ids.shape == iterations.shape):
        raise ValueError("values, puzzle_ids, and iterations must align")
    unique_puzzles = np.unique(puzzle_ids)
    unique_iterations = np.unique(iterations)
    result = np.empty((len(unique_puzzles), len(unique_iterations)), dtype=np.float64)
    for puzzle_index, puzzle in enumerate(unique_puzzles):
        for time_index, iteration in enumerate(unique_iterations):
            selected = (puzzle_ids == puzzle) & (iterations == iteration)
            if not np.any(selected):
                raise ValueError("every puzzle must have every iteration")
            result[puzzle_index, time_index] = values[selected].mean()
    return unique_puzzles, unique_iterations, result


def _linear_slopes(values, time):
    values = np.asarray(values, dtype=np.float64)
    centered_time = np.asarray(time, dtype=np.float64)
    centered_time = centered_time - centered_time.mean()
    denominator = float(np.sum(centered_time**2))
    if denominator <= 1e-15:
        raise ValueError("time must contain at least two distinct values")
    return (values - values.mean(axis=1, keepdims=True)) @ centered_time / denominator


def _delta_alignment(true_values, coordinate_values):
    true_delta = np.diff(true_values, axis=1).reshape(-1)
    coordinate_delta = np.diff(coordinate_values, axis=1).reshape(-1)
    correlation = _weighted_correlation(
        true_delta,
        coordinate_delta,
        np.ones(len(true_delta), dtype=np.float64),
    )
    denominator = np.linalg.norm(true_delta) * np.linalg.norm(coordinate_delta)
    cosine = (
        float(true_delta @ coordinate_delta / denominator)
        if denominator > 1e-15
        else math.nan
    )
    return correlation, cosine


def _stratified_bootstrap(values, bucket_names, generator, statistic, repetitions):
    values = tuple(np.asarray(value) for value in values)
    bucket_names = np.asarray(bucket_names, dtype=object)
    bucket_order = tuple(dict.fromkeys(bucket_names.tolist()))
    samples = []
    for _ in range(repetitions):
        indices = []
        for bucket in bucket_order:
            group = np.flatnonzero(bucket_names == bucket)
            indices.extend(generator.choice(group, size=len(group), replace=True))
        samples.append(statistic(*(value[indices] for value in values)))
    samples = np.asarray(samples, dtype=np.float64)
    samples = samples[np.isfinite(samples)]
    if not len(samples):
        return {"ci_low": math.nan, "ci_high": math.nan}
    return {
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
    }


def deep_temporal_metrics(
    true_values,
    coordinate_values,
    iteration_values,
    bucket_names,
    *,
    minimum_iteration=128,
    permutation_repetitions=199,
    bootstrap_repetitions=500,
    seed=0,
):
    """Summarize held-out deep motion and an independent iteration-shuffle null."""

    true_values = np.asarray(true_values, dtype=np.float64)
    coordinate_values = np.asarray(coordinate_values, dtype=np.float64)
    iteration_values = np.asarray(iteration_values, dtype=np.int64)
    bucket_names = np.asarray(bucket_names, dtype=object)
    if true_values.shape != coordinate_values.shape or true_values.ndim != 2:
        raise ValueError("true and coordinate values must be aligned matrices")
    if true_values.shape[1] != len(iteration_values):
        raise ValueError("iteration_values must align with matrix columns")
    if true_values.shape[0] != len(bucket_names):
        raise ValueError("bucket_names must align with matrix rows")
    selected = iteration_values >= minimum_iteration
    if selected.sum() < 3:
        raise ValueError("deep temporal analysis requires at least three snapshots")
    true_deep = true_values[:, selected]
    coordinate_deep = coordinate_values[:, selected]
    deep_iterations = iteration_values[selected]
    log_time = np.log2(deep_iterations.astype(np.float64))
    true_slopes = _linear_slopes(true_deep, log_time)
    coordinate_slopes = _linear_slopes(coordinate_deep, log_time)
    alignment, cosine = _delta_alignment(true_deep, coordinate_deep)
    generator = np.random.default_rng(seed)

    shuffled_alignments = []
    shuffled_coordinate_slopes = []
    for _ in range(permutation_repetitions):
        shuffled = np.stack(
            [row[generator.permutation(len(row))] for row in coordinate_deep]
        )
        shuffled_alignments.append(_delta_alignment(true_deep, shuffled)[0])
        shuffled_coordinate_slopes.append(float(_linear_slopes(shuffled, log_time).mean()))

    slope_correlation = _weighted_correlation(
        true_slopes,
        coordinate_slopes,
        np.ones(len(true_slopes), dtype=np.float64),
    )
    bootstrap_generator = np.random.default_rng(seed + 1)
    true_slope_interval = _stratified_bootstrap(
        (true_slopes,),
        bucket_names,
        bootstrap_generator,
        lambda values: float(np.mean(values)),
        bootstrap_repetitions,
    )
    coordinate_slope_interval = _stratified_bootstrap(
        (coordinate_slopes,),
        bucket_names,
        bootstrap_generator,
        lambda values: float(np.mean(values)),
        bootstrap_repetitions,
    )
    alignment_interval = _stratified_bootstrap(
        (true_deep, coordinate_deep),
        bucket_names,
        bootstrap_generator,
        lambda first, second: _delta_alignment(first, second)[0],
        bootstrap_repetitions,
    )
    return {
        "minimum_iteration": int(minimum_iteration),
        "iterations": deep_iterations.tolist(),
        "mean_true_slope_per_log2_iteration": float(true_slopes.mean()),
        "mean_true_slope_ci": true_slope_interval,
        "fraction_puzzles_with_positive_true_slope": float(np.mean(true_slopes > 0)),
        "mean_coordinate_slope_per_log2_iteration": float(coordinate_slopes.mean()),
        "mean_coordinate_slope_ci": coordinate_slope_interval,
        "fraction_puzzles_with_positive_coordinate_slope": float(
            np.mean(coordinate_slopes > 0)
        ),
        "slope_spearman_across_puzzles": _weighted_correlation(
            _rankdata(true_slopes),
            _rankdata(coordinate_slopes),
            np.ones(len(true_slopes), dtype=np.float64),
        ),
        "slope_pearson_across_puzzles": slope_correlation,
        "deep_delta_alignment_pearson": alignment,
        "deep_delta_alignment_cosine": cosine,
        "deep_delta_alignment_ci": alignment_interval,
        "shuffled_iteration_alignment": summarize_null(
            alignment, shuffled_alignments, larger_is_better=True
        ),
        "shuffled_iteration_coordinate_slope": summarize_null(
            float(coordinate_slopes.mean()),
            shuffled_coordinate_slopes,
            larger_is_better=True,
        ),
        "per_puzzle_true_slopes": true_slopes.tolist(),
        "per_puzzle_coordinate_slopes": coordinate_slopes.tolist(),
    }
