"""Core statistics for the recurrent-trajectory adversarial audit.

The functions in this module are NumPy-only so the selection logic, null
controls, and acceptance checks can be tested without loading a checkpoint.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


REPRESENTATION_NAMES = (
    "raw_state",
    "normalized_state",
    "raw_update",
    "normalized_update",
)
PLANES = ((0, 1, 2), (0, 2, 1), (1, 2, 0))


def load_json(path):
    with open(path) as input_file:
        return json.load(input_file)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def puzzle_sha256(puzzle):
    return hashlib.sha256(puzzle.encode("utf-8")).hexdigest()


def make_three_way_split(rating_buckets, *, seed, puzzles_per_bucket_per_split):
    """Split whole puzzles within rating buckets before fitting anything."""

    rating_buckets = tuple(rating_buckets)
    if not rating_buckets:
        raise ValueError("rating_buckets must not be empty")
    if puzzles_per_bucket_per_split <= 0:
        raise ValueError("puzzles_per_bucket_per_split must be positive")
    required = 3 * puzzles_per_bucket_per_split
    generator = np.random.default_rng(seed)
    result = {name: [] for name in ("discovery", "validation", "holdout")}
    for bucket in dict.fromkeys(rating_buckets):
        indices = np.array(
            [index for index, value in enumerate(rating_buckets) if value == bucket],
            dtype=np.int64,
        )
        if len(indices) != required:
            raise ValueError(
                f"rating bucket {bucket!r} has {len(indices)} puzzles; expected "
                f"exactly {required}"
            )
        generator.shuffle(indices)
        width = puzzles_per_bucket_per_split
        result["discovery"].extend(indices[:width].tolist())
        result["validation"].extend(indices[width : 2 * width].tolist())
        result["holdout"].extend(indices[2 * width :].tolist())
    arrays = {name: np.array(values, dtype=np.int64) for name, values in result.items()}
    combined = np.concatenate(tuple(arrays.values()))
    if len(np.unique(combined)) != len(combined):
        raise RuntimeError("the three puzzle splits overlap")
    return arrays


def make_development_resplits(
    rating_buckets,
    development_indices,
    *,
    seed,
    repeats,
):
    """Make repeated balanced discovery/validation splits inside development."""

    development_indices = np.asarray(development_indices, dtype=np.int64)
    development_set = set(development_indices.tolist())
    bucket_order = tuple(dict.fromkeys(rating_buckets))
    result = []
    for repeat_index in range(repeats):
        generator = np.random.default_rng(seed + 1009 * repeat_index)
        discovery = []
        validation = []
        for bucket in bucket_order:
            indices = np.array(
                [
                    index
                    for index, value in enumerate(rating_buckets)
                    if value == bucket and index in development_set
                ],
                dtype=np.int64,
            )
            if len(indices) % 2:
                raise ValueError("each development bucket must have even size")
            generator.shuffle(indices)
            midpoint = len(indices) // 2
            discovery.extend(indices[:midpoint].tolist())
            validation.extend(indices[midpoint:].tolist())
        result.append(
            {
                "discovery": np.array(discovery, dtype=np.int64),
                "validation": np.array(validation, dtype=np.int64),
            }
        )
    return tuple(result)


def board_l2_normalize(values, *, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.maximum(norms, eps)


def build_representations(states, updates):
    """Return raw and board-normalized state/update paths.

    Inputs may have cell and feature dimensions or may already be flattened.
    States are anchored at iteration zero. Normalized states are normalized
    before anchoring so iteration zero remains a well-defined zero vector.
    """

    states = np.asarray(states, dtype=np.float32)
    updates = np.asarray(updates, dtype=np.float32)
    if states.shape[:2] != updates.shape[:2] or states.ndim < 3:
        raise ValueError("states and updates must align on puzzle and time")
    state_flat = states.reshape(states.shape[0], states.shape[1], -1)
    update_flat = updates.reshape(updates.shape[0], updates.shape[1], -1)
    normalized_state = board_l2_normalize(state_flat)
    return {
        "raw_state": state_flat - state_flat[:, :1],
        "normalized_state": normalized_state - normalized_state[:, :1],
        "raw_update": update_flat,
        "normalized_update": board_l2_normalize(update_flat),
    }


def temporal_center(values):
    values = np.asarray(values)
    if values.ndim != 3:
        raise ValueError("trajectory values must have shape [puzzle, time, feature]")
    return values.astype(np.float64) - values.astype(np.float64).mean(
        axis=1, keepdims=True
    )


def fit_pca_numpy(values, rank):
    """Fit an exact PCA basis to temporally centered puzzle trajectories."""

    centered = temporal_center(values)
    matrix = centered.reshape(-1, centered.shape[-1]).astype(np.float64)
    mean = matrix.mean(axis=0)
    matrix -= mean
    maximum_rank = min(matrix.shape)
    if rank <= 0 or rank > maximum_rank:
        raise ValueError(f"rank must be in [1, {maximum_rank}]")
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    total_energy = float(np.square(matrix).sum())
    return {
        "mean": mean.astype(np.float32),
        "basis": right_vectors[:rank].T.astype(np.float32),
        "singular_values": singular_values[:rank].astype(np.float32),
        "training_energy": total_energy,
    }


def project(values, mean, basis):
    values = np.asarray(values, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if values.ndim != 3 or mean.shape != (values.shape[-1],):
        raise ValueError("mean does not match the trajectory feature dimension")
    if basis.ndim != 2 or basis.shape[0] != values.shape[-1]:
        raise ValueError("basis does not match the trajectory feature dimension")
    # Apple's Accelerate backend can leave stale floating-point status flags
    # around large BLAS calls. Validate the result directly instead of treating
    # those flags as evidence of a non-finite projection.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        coordinates = (values - mean) @ basis
    if not np.isfinite(coordinates).all():
        raise FloatingPointError("projection produced non-finite coordinates")
    return coordinates - coordinates.mean(axis=1, keepdims=True)


def dynamic_variance_fraction(values, coordinates):
    centered_values = temporal_center(values).astype(np.float64)
    centered_coordinates = temporal_center(coordinates).astype(np.float64)
    denominator = float(np.square(centered_values).sum())
    if denominator <= 1e-24:
        return 0.0
    return float(np.square(centered_coordinates).sum() / denominator)


def random_orthonormal_coefficients(ambient_rank, output_rank, count, *, seed):
    if not 0 < output_rank <= ambient_rank:
        raise ValueError("output_rank must be positive and no larger than ambient_rank")
    generator = np.random.default_rng(seed)
    coefficients = []
    for _ in range(count):
        gaussian = generator.normal(size=(ambient_rank, output_rank))
        orthonormal, _ = np.linalg.qr(gaussian, mode="reduced")
        coefficients.append(orthonormal.astype(np.float32))
    return np.stack(coefficients)


def local_pca_coordinates(values, rank=3):
    """Fit PCA separately to each full path; this is a circular control."""

    centered = temporal_center(values).astype(np.float64)
    puzzle_count, time_count, _ = centered.shape
    if not 0 < rank < time_count:
        raise ValueError("rank must be smaller than the number of snapshots")
    output = np.zeros((puzzle_count, time_count, rank), dtype=np.float32)
    for puzzle_index, trajectory in enumerate(centered):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gram = trajectory @ trajectory.T
        if not np.isfinite(gram).all():
            raise FloatingPointError("local PCA Gram matrix is non-finite")
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][:rank]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        output[puzzle_index] = (
            eigenvectors[:, order] * np.sqrt(eigenvalues)[None, :]
        ).astype(np.float32)
    return output


def _safe_correlation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _rank_values(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def continuity_components(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    differences = np.diff(trajectory, axis=0)
    mean_step_distance = float(np.linalg.norm(differences, axis=1).mean())
    gram = trajectory @ trajectory.T
    squared_norms = np.diag(gram)
    squared_distances = np.maximum(
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * gram,
        0.0,
    )
    upper = np.sqrt(squared_distances[np.triu_indices(len(trajectory), k=1)])
    typical_distance = float(np.median(upper)) if len(upper) else 0.0
    score = math.log((typical_distance + 1e-12) / (mean_step_distance + 1e-12))
    return {
        "score": score,
        "mean_step_distance": mean_step_distance,
        "median_pairwise_distance": typical_distance,
    }


def arc_components(trajectory):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    steps = np.diff(trajectory, axis=0)
    step_lengths = np.linalg.norm(steps, axis=1)
    path_length = float(step_lengths.sum())
    chord = trajectory[-1] - trajectory[0]
    chord_length = float(np.linalg.norm(chord))
    efficiency = chord_length / max(path_length, 1e-12)
    valid = step_lengths > 1e-12
    tangent_cosines = []
    for index in range(len(steps) - 1):
        if valid[index] and valid[index + 1]:
            tangent_cosines.append(
                float(
                    np.dot(steps[index], steps[index + 1])
                    / (step_lengths[index] * step_lengths[index + 1])
                )
            )
    mean_tangent_cosine = float(np.mean(tangent_cosines)) if tangent_cosines else -1.0
    tangent_smoothness = float(np.clip((mean_tangent_cosine + 1.0) / 2.0, 0.0, 1.0))
    if chord_length > 1e-12:
        unit_chord = chord / chord_length
        relative = trajectory - trajectory[0]
        perpendicular = relative - np.outer(relative @ unit_chord, unit_chord)
        deviation_ratio = float(np.linalg.norm(perpendicular, axis=1).max() / chord_length)
    else:
        deviation_ratio = 0.0
    bend_strength = float(1.0 - math.exp(-4.0 * deviation_ratio))
    score = float(max(efficiency * tangent_smoothness * bend_strength, 0.0) ** (1.0 / 3.0))
    return {
        "score": score,
        "path_efficiency": efficiency,
        "mean_tangent_cosine": mean_tangent_cosine,
        "chord_deviation_ratio": deviation_ratio,
    }


def cyclic_components(trajectory, plane=(0, 1, 2)):
    trajectory = np.asarray(trajectory, dtype=np.float64)
    plane_x, plane_y, axial_index = plane
    if trajectory.ndim != 2 or trajectory.shape[1] <= max(plane):
        raise ValueError("trajectory has too few projected dimensions")
    transverse = trajectory[:, [plane_x, plane_y]]
    transverse = transverse - transverse.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(transverse, axis=1)
    median_radius = float(np.median(radii))
    radius_cv = float(radii.std() / max(radii.mean(), 1e-12))
    radius_consistency = float(math.exp(-min(radius_cv, 20.0)))
    angles = np.unwrap(np.arctan2(transverse[:, 1], transverse[:, 0]))
    time = np.linspace(0.0, 1.0, len(trajectory))
    angular_correlation = _safe_correlation(time, angles)
    angular_linearity_r2 = angular_correlation * angular_correlation
    net_turns = float(abs(angles[-1] - angles[0]) / (2.0 * math.pi))
    turn_factor = min(net_turns, 1.0)
    transverse_closure_distance = float(np.linalg.norm(transverse[-1] - transverse[0]))
    closure = float(math.exp(-transverse_closure_distance / max(2.0 * median_radius, 1e-12)))
    axial = trajectory[:, axial_index]
    axial_spearman = abs(_safe_correlation(_rank_values(time), _rank_values(axial)))
    shared = angular_linearity_r2 * radius_consistency * turn_factor
    return {
        "loop_score": float(shared * closure),
        "helix_score": float(shared * axial_spearman),
        "angular_linearity_r2": angular_linearity_r2,
        "net_turns": net_turns,
        "radius_cv": radius_cv,
        "radius_consistency": radius_consistency,
        "transverse_closure": closure,
        "absolute_axial_spearman": axial_spearman,
    }


def metric_components(trajectory, metric, plane=(0, 1, 2)):
    if metric == "continuity":
        return continuity_components(trajectory)
    if metric == "arc":
        return arc_components(trajectory)
    cyclic = cyclic_components(trajectory, plane)
    if metric == "loop":
        return {"score": cyclic["loop_score"], **cyclic}
    if metric == "helix":
        return {"score": cyclic["helix_score"], **cyclic}
    raise ValueError(f"unknown metric: {metric}")


def per_puzzle_metrics(coordinates, metric, plane=(0, 1, 2)):
    coordinates = np.asarray(coordinates)
    if coordinates.ndim != 3:
        raise ValueError("coordinates must have shape [puzzle, time, feature]")
    rows = [metric_components(path, metric, plane) for path in coordinates]
    keys = rows[0].keys()
    return {key: np.array([row[key] for row in rows], dtype=np.float64) for key in keys}


def aggregate_metric(coordinates, metric, plane=(0, 1, 2)):
    return float(np.median(per_puzzle_metrics(coordinates, metric, plane)["score"]))


def select_plane(coordinates, metric):
    if metric not in ("loop", "helix"):
        return PLANES[0], aggregate_metric(coordinates, metric, PLANES[0])
    scores = [(plane, aggregate_metric(coordinates, metric, plane)) for plane in PLANES]
    return max(scores, key=lambda item: item[1])


def bootstrap_median_interval(values, *, repetitions, seed, confidence=0.95):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("values must be a non-empty vector")
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = values[generator.integers(0, len(values), size=len(values))]
        estimates[index] = np.median(sample)
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(estimates, tail)), float(np.quantile(estimates, 1.0 - tail))]


def time_shuffle_test(
    coordinates,
    metric,
    *,
    plane=(0, 1, 2),
    permutations,
    bootstraps,
    seed,
):
    """Compare ordered paths with independently shuffled time within puzzles."""

    coordinates = np.asarray(coordinates)
    if metric == "continuity":
        return _continuity_time_shuffle_test(
            coordinates,
            permutations=permutations,
            bootstraps=bootstraps,
            seed=seed,
        )
    observed_components = per_puzzle_metrics(coordinates, metric, plane)
    observed = observed_components["score"]
    generator = np.random.default_rng(seed)
    null_scores = np.empty((permutations, len(coordinates)), dtype=np.float64)
    for permutation_index in range(permutations):
        shuffled = np.empty_like(coordinates)
        for puzzle_index in range(len(coordinates)):
            order = generator.permutation(coordinates.shape[1])
            shuffled[puzzle_index] = coordinates[puzzle_index, order]
        null_scores[permutation_index] = per_puzzle_metrics(
            shuffled, metric, plane
        )["score"]
    observed_aggregate = float(np.median(observed))
    null_aggregates = np.median(null_scores, axis=1)
    p_value = float(
        (1 + np.count_nonzero(null_aggregates >= observed_aggregate))
        / (permutations + 1)
    )
    null_per_puzzle = np.median(null_scores, axis=0)
    per_puzzle_effect = observed - null_per_puzzle
    effect = float(observed_aggregate - np.median(null_aggregates))
    null_standard_deviation = float(null_aggregates.std(ddof=1))
    standardized_effect = (
        effect / null_standard_deviation if null_standard_deviation > 1e-15 else math.inf
    )
    component_medians = {
        key: float(np.median(values))
        for key, values in observed_components.items()
        if key != "score"
    }
    return {
        "metric": metric,
        "plane": list(plane),
        "observed_median": observed_aggregate,
        "null_median": float(np.median(null_aggregates)),
        "null_95_percent_interval": [
            float(np.quantile(null_aggregates, 0.025)),
            float(np.quantile(null_aggregates, 0.975)),
        ],
        "ordered_minus_shuffle_effect": effect,
        "standardized_effect": float(standardized_effect),
        "time_shuffle_p": p_value,
        "puzzle_bootstrap_95_percent_effect_interval": bootstrap_median_interval(
            per_puzzle_effect,
            repetitions=bootstraps,
            seed=seed + 7919,
        ),
        "component_medians": component_medians,
        "per_puzzle_observed": observed.tolist(),
        "per_puzzle_null_median": null_per_puzzle.tolist(),
    }


def _continuity_time_shuffle_test(coordinates, *, permutations, bootstraps, seed):
    """Continuity permutation test that reuses high-dimensional distances."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    puzzle_count, time_count, _ = coordinates.shape
    distance_matrices = []
    typical_distances = []
    observed_scores = []
    observed_steps = []
    for trajectory in coordinates:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            gram = trajectory @ trajectory.T
        if not np.isfinite(gram).all():
            raise FloatingPointError("continuity distance matrix is non-finite")
        squared_norms = np.diag(gram)
        distances = np.sqrt(
            np.maximum(
                squared_norms[:, None] + squared_norms[None, :] - 2.0 * gram,
                0.0,
            )
        )
        typical = float(np.median(distances[np.triu_indices(time_count, k=1)]))
        mean_step = float(np.diag(distances, k=1).mean())
        distance_matrices.append(distances)
        typical_distances.append(typical)
        observed_steps.append(mean_step)
        observed_scores.append(math.log((typical + 1e-12) / (mean_step + 1e-12)))
    observed_scores = np.asarray(observed_scores)
    generator = np.random.default_rng(seed)
    null_scores = np.empty((permutations, puzzle_count), dtype=np.float64)
    for permutation_index in range(permutations):
        for puzzle_index, distances in enumerate(distance_matrices):
            order = generator.permutation(time_count)
            mean_step = float(distances[order[:-1], order[1:]].mean())
            null_scores[permutation_index, puzzle_index] = math.log(
                (typical_distances[puzzle_index] + 1e-12) / (mean_step + 1e-12)
            )
    observed_aggregate = float(np.median(observed_scores))
    null_aggregates = np.median(null_scores, axis=1)
    null_per_puzzle = np.median(null_scores, axis=0)
    per_puzzle_effect = observed_scores - null_per_puzzle
    effect = observed_aggregate - float(np.median(null_aggregates))
    null_standard_deviation = float(null_aggregates.std(ddof=1))
    return {
        "metric": "continuity",
        "plane": None,
        "observed_median": observed_aggregate,
        "null_median": float(np.median(null_aggregates)),
        "null_95_percent_interval": [
            float(np.quantile(null_aggregates, 0.025)),
            float(np.quantile(null_aggregates, 0.975)),
        ],
        "ordered_minus_shuffle_effect": float(effect),
        "standardized_effect": float(
            effect / null_standard_deviation
            if null_standard_deviation > 1e-15
            else math.inf
        ),
        "time_shuffle_p": float(
            (1 + np.count_nonzero(null_aggregates >= observed_aggregate))
            / (permutations + 1)
        ),
        "puzzle_bootstrap_95_percent_effect_interval": bootstrap_median_interval(
            per_puzzle_effect,
            repetitions=bootstraps,
            seed=seed + 7919,
        ),
        "component_medians": {
            "mean_step_distance": float(np.median(observed_steps)),
            "median_pairwise_distance": float(np.median(typical_distances)),
        },
        "per_puzzle_observed": observed_scores.tolist(),
        "per_puzzle_null_median": null_per_puzzle.tolist(),
    }


def benjamini_hochberg(p_values):
    p_values = np.asarray(p_values, dtype=np.float64)
    if p_values.ndim != 1 or np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be a vector in [0, 1]")
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def percentile_against_controls(observed, controls):
    controls = np.asarray(controls, dtype=np.float64)
    return float((1 + np.count_nonzero(controls <= observed)) / (len(controls) + 1))


def _smooth_independent_features(generator, time_count, feature_count, passes=4):
    values = generator.normal(size=(time_count, feature_count))
    for _ in range(passes):
        padded = np.pad(values, ((1, 1), (0, 0)), mode="edge")
        values = (padded[:-2] + 2.0 * padded[1:-1] + padded[2:]) / 4.0
    return values


def generate_null_trajectory(kind, *, time_count, feature_count, generator):
    """Generate a task-free high-dimensional temporal process."""

    if kind == "random_walk":
        values = np.cumsum(generator.normal(size=(time_count, feature_count)), axis=0)
    elif kind == "brownian_bridge":
        values = np.cumsum(generator.normal(size=(time_count, feature_count)), axis=0)
        time = np.linspace(0.0, 1.0, time_count)[:, None]
        values = values - time * values[-1]
    elif kind == "smooth_process":
        values = _smooth_independent_features(
            generator, time_count, feature_count, passes=6
        )
    elif kind == "decay_mixture":
        time = np.linspace(0.0, 1.0, time_count)[:, None]
        rates = generator.uniform(0.5, 8.0, size=(1, feature_count))
        signs = generator.choice([-1.0, 1.0], size=(1, feature_count))
        values = signs * np.exp(-rates * time)
        values += 0.015 * generator.normal(size=values.shape)
    else:
        raise ValueError(f"unknown synthetic null kind: {kind}")
    values = values - values[0]
    scale = np.sqrt(np.mean(np.square(values)))
    return (values / max(scale, 1e-12)).astype(np.float32)


def best_axes_from_local_pca(local_coordinates, metric):
    """Select axes after seeing the full path, intentionally favoring appearance."""

    rank = local_coordinates.shape[1]
    best = None
    for axes in itertools.combinations(range(rank), 3):
        candidate = local_coordinates[:, axes]
        if metric in ("loop", "helix"):
            planes = PLANES
        else:
            planes = (PLANES[0],)
        for plane in planes:
            components = metric_components(candidate, metric, plane)
            score = components["score"]
            if best is None or score > best["score"]:
                best = {
                    "score": float(score),
                    "axes": list(axes),
                    "plane": list(plane),
                    "coordinates": candidate.astype(np.float32),
                    "components": {key: float(value) for key, value in components.items()},
                }
    return best


def synthetic_selection_experiment(
    *,
    seed,
    candidates=256,
    time_count=33,
    feature_count=64,
    local_rank=8,
):
    """Quantify how local PCA and axis selection manufacture attractive nulls."""

    generator = np.random.default_rng(seed)
    specifications = {
        "arc": "decay_mixture",
        "loop": "brownian_bridge",
        "helix": "smooth_process",
    }
    output = {}
    for metric, kind in specifications.items():
        honest_scores = []
        local_first3_scores = []
        selected_scores = []
        gallery_best = None
        for candidate_index in range(candidates):
            trajectory = generate_null_trajectory(
                kind,
                time_count=time_count,
                feature_count=feature_count,
                generator=generator,
            )
            honest = trajectory[:, :3]
            honest_plane, honest_score = select_plane(honest[None], metric)
            local = local_pca_coordinates(trajectory[None], rank=local_rank)[0]
            first3_plane, first3_score = select_plane(local[None, :, :3], metric)
            selected = best_axes_from_local_pca(local, metric)
            honest_scores.append(honest_score)
            local_first3_scores.append(first3_score)
            selected_scores.append(selected["score"])
            if gallery_best is None or selected["score"] > gallery_best["score"]:
                gallery_best = {
                    **selected,
                    "candidate_index": candidate_index,
                    "kind": kind,
                    "honest_score": float(honest_score),
                    "honest_plane": list(honest_plane),
                    "local_first3_score": float(first3_score),
                    "local_first3_plane": list(first3_plane),
                }
        honest_scores = np.asarray(honest_scores)
        local_first3_scores = np.asarray(local_first3_scores)
        selected_scores = np.asarray(selected_scores)
        output[metric] = {
            "null_kind": kind,
            "candidate_count": candidates,
            "honest_fixed_axes": _distribution_summary(honest_scores),
            "local_pca_first3": _distribution_summary(local_first3_scores),
            "local_pca_selected_axes": _distribution_summary(selected_scores),
            "median_selection_inflation": float(
                np.median(selected_scores - honest_scores)
            ),
            "gallery": gallery_best,
        }
    return output


def _distribution_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w") as output_file:
        json.dump(json_ready(value), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(path)
