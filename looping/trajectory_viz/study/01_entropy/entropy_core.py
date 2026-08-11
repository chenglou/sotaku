"""Statistical helpers for the held-out uncertainty-coordinate study."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


TARGET_SPECS = {
    "entropy": {
        "label": "normalized prediction entropy",
        "expected_direction": -1,
    },
    "top_two_gap": {
        "label": "top-two probability gap",
        "expected_direction": 1,
    },
    "residual_uncertainty": {
        "label": "one minus maximum probability",
        "expected_direction": -1,
    },
}


@dataclass(frozen=True)
class ThreeWaySplit:
    discovery: torch.Tensor
    validation: torch.Tensor
    final: torch.Tensor


@dataclass(frozen=True)
class FeaturePreprocessor:
    feature_mean: torch.Tensor
    feature_scale: torch.Tensor
    iteration_feature_means: torch.Tensor


@dataclass(frozen=True)
class LinearProbe:
    preprocessor: FeaturePreprocessor
    iteration_target_means: torch.Tensor
    global_target_mean: torch.Tensor
    coefficient: torch.Tensor
    alpha: float


def predictive_targets(logits):
    """Return three label-free uncertainty statistics from digit logits."""

    if logits.ndim < 2 or logits.size(-1) != 9:
        raise ValueError("logits must end with nine digit classes")
    probabilities = F.softmax(logits.float(), dim=-1)
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    entropy = -(probabilities * log_probabilities).sum(dim=-1) / math.log(9)
    top_two = probabilities.topk(2, dim=-1).values
    return {
        "entropy": entropy,
        "top_two_gap": top_two[..., 0] - top_two[..., 1],
        "residual_uncertainty": 1.0 - top_two[..., 0],
    }


def rating_balanced_three_way_split(rating_buckets, seed):
    """Split whole puzzles equally within every rating bucket."""

    rating_buckets = tuple(rating_buckets)
    if not rating_buckets:
        raise ValueError("rating_buckets must not be empty")
    generator = torch.Generator().manual_seed(seed)
    groups = {name: [] for name in dict.fromkeys(rating_buckets)}
    for puzzle_index, bucket in enumerate(rating_buckets):
        groups[bucket].append(puzzle_index)

    split_indices = {name: [] for name in ("discovery", "validation", "final")}
    for bucket, values in groups.items():
        if len(values) < 3 or len(values) % 3:
            raise ValueError(
                f"rating bucket {bucket!r} must contain a multiple of three puzzles"
            )
        indices = torch.tensor(values, dtype=torch.long)
        indices = indices[torch.randperm(len(indices), generator=generator)]
        width = len(indices) // 3
        split_indices["discovery"].extend(indices[:width].tolist())
        split_indices["validation"].extend(indices[width : 2 * width].tolist())
        split_indices["final"].extend(indices[2 * width :].tolist())

    result = ThreeWaySplit(
        discovery=torch.tensor(split_indices["discovery"], dtype=torch.long),
        validation=torch.tensor(split_indices["validation"], dtype=torch.long),
        final=torch.tensor(split_indices["final"], dtype=torch.long),
    )
    all_indices = torch.cat((result.discovery, result.validation, result.final))
    if len(torch.unique(all_indices)) != len(rating_buckets):
        raise RuntimeError("three-way split lost or duplicated a puzzle")
    return result

def puzzle_equal_weights(puzzle_indices):
    """Give every represented puzzle equal total observation weight."""

    puzzle_indices = torch.as_tensor(puzzle_indices, dtype=torch.long)
    if puzzle_indices.ndim != 1 or not puzzle_indices.numel():
        raise ValueError("puzzle_indices must be a non-empty vector")
    _, inverse, counts = torch.unique(
        puzzle_indices, sorted=True, return_inverse=True, return_counts=True
    )
    weights = counts[inverse].double().reciprocal()
    return weights / weights.sum()


def weighted_mean(values, weights, dim=0):
    weights = weights.to(device=values.device, dtype=values.dtype)
    if values.size(dim) != weights.numel():
        raise ValueError("weights do not align with values")
    shape = [1] * values.ndim
    shape[dim] = weights.numel()
    normalized = weights / weights.sum().clamp_min(1e-24)
    return (values * normalized.reshape(shape)).sum(dim=dim)


def fit_feature_preprocessor(features, iteration_indices, weights, time_count):
    """Fit discovery-only scaling and per-iteration feature means."""

    features = features.double()
    iteration_indices = torch.as_tensor(iteration_indices, dtype=torch.long)
    weights = weights.double()
    if features.ndim != 2 or len(features) != len(iteration_indices):
        raise ValueError("features and iteration_indices are misaligned")
    if iteration_indices.min() < 0 or iteration_indices.max() >= time_count:
        raise ValueError("iteration index is outside the declared time count")

    feature_mean = weighted_mean(features, weights)
    variance = weighted_mean((features - feature_mean).square(), weights)
    feature_scale = variance.sqrt().clamp_min(1e-8)
    standardized = (features - feature_mean) / feature_scale
    iteration_means = []
    for iteration_index in range(time_count):
        selected = iteration_indices == iteration_index
        if not selected.any():
            raise ValueError(f"iteration {iteration_index} has no observations")
        iteration_means.append(
            weighted_mean(standardized[selected], weights[selected])
        )
    return FeaturePreprocessor(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        iteration_feature_means=torch.stack(iteration_means),
    )


def transform_features(features, iteration_indices, preprocessor):
    standardized = (
        features.double() - preprocessor.feature_mean
    ) / preprocessor.feature_scale
    residual = standardized - preprocessor.iteration_feature_means[
        iteration_indices.long()
    ]
    return standardized, residual


def fit_iteration_target_means(targets, iteration_indices, weights, time_count):
    targets = targets.double()
    means = []
    for iteration_index in range(time_count):
        selected = iteration_indices == iteration_index
        if not selected.any():
            raise ValueError(f"iteration {iteration_index} has no targets")
        means.append(weighted_mean(targets[selected], weights[selected]))
    return torch.stack(means)


def fit_ridge_coefficients(features, targets, weights, alpha):
    """Fit one or more ridge targets with a shared feature matrix."""

    features = features.double()
    targets = targets.double()
    weights = weights.to(dtype=torch.float64, device=features.device)
    if targets.ndim == 1:
        targets = targets[:, None]
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must be matrices")
    if len(features) != len(targets) or len(features) != len(weights):
        raise ValueError("ridge inputs are misaligned")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    normalized_weights = weights / weights.sum().clamp_min(1e-24)
    weighted_features = features * normalized_weights[:, None]
    gram = features.T @ weighted_features
    gram = gram + (alpha + 1e-10) * torch.eye(
        features.size(1), dtype=features.dtype, device=features.device
    )
    cross = features.T @ (targets * normalized_weights[:, None])
    return torch.linalg.solve(gram, cross)


def fit_probe(
    features,
    targets,
    iteration_indices,
    puzzle_indices,
    *,
    time_count,
    alpha,
    preprocessor=None,
):
    weights = puzzle_equal_weights(puzzle_indices)
    if preprocessor is None:
        preprocessor = fit_feature_preprocessor(
            features, iteration_indices, weights, time_count
        )
    _, residual_features = transform_features(
        features, iteration_indices, preprocessor
    )
    iteration_target_means = fit_iteration_target_means(
        targets, iteration_indices, weights, time_count
    )
    residual_targets = targets.double() - iteration_target_means[
        iteration_indices.long()
    ]
    coefficient = fit_ridge_coefficients(
        residual_features, residual_targets, weights, alpha
    ).squeeze(1)
    return LinearProbe(
        preprocessor=preprocessor,
        iteration_target_means=iteration_target_means,
        global_target_mean=weighted_mean(targets.double(), weights),
        coefficient=coefficient,
        alpha=float(alpha),
    )


def weighted_squared_error(targets, predictions, weights):
    errors = targets.double() - predictions.double()
    weights = weights.to(device=errors.device, dtype=errors.dtype)
    if errors.ndim == 1:
        return (errors.square() * weights).sum() / weights.sum()
    return (errors.square() * weights[:, None]).sum(dim=0) / weights.sum()


def weighted_correlation(left, right, weights):
    left = left.double()
    right = right.double()
    weights = weights.to(dtype=torch.float64, device=left.device)
    left_centered = left - weighted_mean(left, weights)
    right_centered = right - weighted_mean(right, weights)
    covariance = weighted_mean(left_centered * right_centered, weights)
    denominator = (
        weighted_mean(left_centered.square(), weights)
        * weighted_mean(right_centered.square(), weights)
    ).sqrt()
    if denominator <= 1e-15:
        return None
    return float((covariance / denominator).item())


def evaluate_probe(probe, features, targets, iteration_indices, puzzle_indices):
    weights = puzzle_equal_weights(puzzle_indices)
    _, residual_features = transform_features(
        features, iteration_indices, probe.preprocessor
    )
    baseline = probe.iteration_target_means[iteration_indices.long()]
    within_prediction = residual_features @ probe.coefficient
    predictions = baseline + within_prediction
    full_sse = weighted_squared_error(targets, predictions, weights)
    iteration_sse = weighted_squared_error(targets, baseline, weights)
    global_baseline = torch.full_like(
        targets.double(), float(probe.global_target_mean)
    )
    global_sse = weighted_squared_error(targets, global_baseline, weights)
    target_residual = targets.double() - baseline

    per_iteration = []
    for iteration_index in range(len(probe.iteration_target_means)):
        selected = iteration_indices == iteration_index
        selected_weights = weights[selected]
        selected_full_sse = weighted_squared_error(
            targets[selected], predictions[selected], selected_weights
        )
        selected_baseline_sse = weighted_squared_error(
            targets[selected], baseline[selected], selected_weights
        )
        if selected_baseline_sse <= 1e-20:
            partial_r2 = None
        else:
            partial_r2 = float(1.0 - selected_full_sse / selected_baseline_sse)
        per_iteration.append(
            {
                "partial_r2_over_discovery_iteration_mean": partial_r2,
                "within_iteration_correlation": weighted_correlation(
                    target_residual[selected],
                    within_prediction[selected],
                    selected_weights,
                ),
            }
        )

    return {
        "partial_r2_over_iteration": float(1.0 - full_sse / iteration_sse),
        "pooled_r2": float(1.0 - full_sse / global_sse),
        "within_iteration_correlation": weighted_correlation(
            target_residual, within_prediction, weights
        ),
        "per_iteration": per_iteration,
    }


def select_ridge_alpha(
    discovery,
    validation,
    *,
    time_count,
    alphas,
):
    """Choose ridge strength on validation without refitting the final probe."""

    discovery_weights = puzzle_equal_weights(discovery[3])
    preprocessor = fit_feature_preprocessor(
        discovery[0], discovery[2], discovery_weights, time_count
    )
    candidates = []
    for alpha in alphas:
        probe = fit_probe(
            *discovery,
            time_count=time_count,
            alpha=alpha,
            preprocessor=preprocessor,
        )
        metrics = evaluate_probe(probe, *validation)
        candidates.append(
            {
                "alpha": float(alpha),
                "validation_partial_r2": metrics["partial_r2_over_iteration"],
                "probe": probe,
            }
        )
    best = max(
        candidates,
        key=lambda candidate: (
            candidate["validation_partial_r2"], -candidate["alpha"]
        ),
    )
    return best["probe"], [
        {key: value for key, value in candidate.items() if key != "probe"}
        for candidate in candidates
    ]


def shuffled_within_iteration(
    targets,
    iteration_indices,
    *,
    count,
    seed,
):
    """Shuffle targets across cells and puzzles within each snapshot."""

    if count <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator().manual_seed(seed)
    targets = targets.double()
    shuffled = targets[:, None].repeat(1, count)
    for iteration_index in torch.unique(iteration_indices.long()):
        selected = iteration_indices == iteration_index
        values = targets[selected]
        for shuffle_index in range(count):
            order = torch.randperm(len(values), generator=generator)
            shuffled[selected, shuffle_index] = values[order]
    return shuffled


def evaluate_coefficient_matrix(
    coefficients,
    preprocessor,
    iteration_target_means,
    features,
    targets,
    iteration_indices,
    puzzle_indices,
):
    """Evaluate several coefficient vectors against the same true targets."""

    weights = puzzle_equal_weights(puzzle_indices)
    _, residual_features = transform_features(
        features, iteration_indices, preprocessor
    )
    baseline = iteration_target_means[iteration_indices.long()]
    predictions = baseline[:, None] + residual_features @ coefficients
    full_sse = weighted_squared_error(targets[:, None], predictions, weights)
    baseline_sse = weighted_squared_error(targets, baseline, weights)
    return 1.0 - full_sse / baseline_sse


def fit_shuffled_target_controls(
    probe,
    discovery,
    final,
    *,
    count,
    seed,
):
    discovery_features, discovery_targets, discovery_iterations, discovery_puzzles = (
        discovery
    )
    weights = puzzle_equal_weights(discovery_puzzles)
    _, residual_features = transform_features(
        discovery_features, discovery_iterations, probe.preprocessor
    )
    shuffled_targets = shuffled_within_iteration(
        discovery_targets,
        discovery_iterations,
        count=count,
        seed=seed,
    )
    shuffled_residuals = shuffled_targets - probe.iteration_target_means[
        discovery_iterations.long(), None
    ]
    coefficients = fit_ridge_coefficients(
        residual_features, shuffled_residuals, weights, probe.alpha
    )
    return evaluate_coefficient_matrix(
        coefficients,
        probe.preprocessor,
        probe.iteration_target_means,
        *final,
    )


def random_axis_controls(
    probe,
    discovery,
    final,
    *,
    count,
    seed,
):
    """Compare the fitted axis with random one-dimensional subspaces."""

    if count <= 0 or count > probe.coefficient.numel():
        raise ValueError("count must be between one and the feature dimension")
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(
        probe.coefficient.numel(), count, generator=generator, dtype=torch.float64
    )
    random_axes = torch.linalg.qr(random_matrix, mode="reduced").Q

    discovery_features, discovery_targets, discovery_iterations, discovery_puzzles = (
        discovery
    )
    discovery_weights = puzzle_equal_weights(discovery_puzzles)
    _, discovery_residual_features = transform_features(
        discovery_features, discovery_iterations, probe.preprocessor
    )
    discovery_scores = discovery_residual_features @ random_axes
    discovery_target_residual = discovery_targets.double() - (
        probe.iteration_target_means[discovery_iterations.long()]
    )
    normalized_weights = discovery_weights / discovery_weights.sum()
    numerator = (
        discovery_scores
        * discovery_target_residual[:, None]
        * normalized_weights[:, None]
    ).sum(dim=0)
    denominator = (
        discovery_scores.square() * normalized_weights[:, None]
    ).sum(dim=0).clamp_min(1e-24)
    coefficients = random_axes * (numerator / denominator)[None, :]

    return evaluate_coefficient_matrix(
        coefficients,
        probe.preprocessor,
        probe.iteration_target_means,
        *final,
    )


def monotonicity_metrics(paths, puzzle_indices, expected_direction, tolerance=1e-10):
    """Measure adjacent and all-pairs movement in an expected direction."""

    paths = paths.double()
    if paths.ndim != 2 or paths.size(1) < 2:
        raise ValueError("paths must have shape [cells, at least two snapshots]")
    if expected_direction not in (-1, 1):
        raise ValueError("expected_direction must be -1 or 1")
    weights = puzzle_equal_weights(puzzle_indices)

    def concordance(delta):
        signed = expected_direction * delta
        return torch.where(
            signed > tolerance,
            torch.ones_like(signed),
            torch.where(
                signed < -tolerance,
                torch.zeros_like(signed),
                torch.full_like(signed, 0.5),
            ),
        )

    adjacent = concordance(paths[:, 1:] - paths[:, :-1]).mean(dim=1)
    earlier, later = torch.triu_indices(paths.size(1), paths.size(1), offset=1)
    all_pairs = concordance(paths[:, later] - paths[:, earlier]).mean(dim=1)
    net = concordance(paths[:, -1] - paths[:, 0])
    return {
        "adjacent_direction_fraction": float(weighted_mean(adjacent, weights)),
        "all_pairs_direction_fraction": float(weighted_mean(all_pairs, weights)),
        "net_direction_fraction": float(weighted_mean(net, weights)),
    }


def shuffled_time_controls(
    paths,
    puzzle_indices,
    expected_direction,
    *,
    count,
    seed,
):
    if count <= 0:
        raise ValueError("count must be positive")
    generator = torch.Generator().manual_seed(seed)
    metrics = []
    for _ in range(count):
        order = torch.randperm(paths.size(1), generator=generator)
        metrics.append(
            monotonicity_metrics(paths[:, order], puzzle_indices, expected_direction)
        )
    return {
        key: torch.tensor([metric[key] for metric in metrics], dtype=torch.float64)
        for key in metrics[0]
    }


def summarize_null(values, observed=None):
    values = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not values.numel():
        raise ValueError("null distribution must not be empty")
    result = {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "values": values.tolist(),
    }
    if observed is not None:
        result["plus_one_upper_tail_p"] = float(
            (1 + (values >= observed).sum().item()) / (len(values) + 1)
        )
    return result
