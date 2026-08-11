"""Analyze whether recurrent hidden states encode Sudoku constraints."""

import argparse
import csv
import functools
import html
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# NumPy 2.2 on the local Accelerate backend emits false matmul overflow warnings
# for finite results. Explicit finite checks below still reject real failures.
def _quiet_accelerate_matmul(function):
    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            return function(*args, **kwargs)
    return wrapped


PROBE_SNAPSHOTS = (16, 128, 1024)
RIDGE_ALPHAS = (0.01, 1.0, 100.0, 10_000.0)
CONTROL_REPEATS = 16
TREND_PERMUTATIONS = 2_000
RANDOM_SEED = 20260811

EXPECTED_CONFLICT_TARGETS = (
    "expected_row_conflicts",
    "expected_column_conflicts",
    "expected_box_conflicts",
    "expected_unique_conflicts",
)
COUNT_TARGETS = (
    "hard_row_conflicts",
    "hard_column_conflicts",
    "hard_box_conflicts",
    "hard_unique_conflicts",
    "row_candidate_count",
    "column_candidate_count",
    "box_candidate_count",
    "joint_candidate_count",
    "static_joint_candidate_count",
)
NUMERIC_TARGETS = EXPECTED_CONFLICT_TARGETS + COUNT_TARGETS
CATEGORICAL_TARGETS = COUNT_TARGETS + ("target_candidate_legal",)
CLASS_VALUES = {
    "hard_row_conflicts": tuple(range(9)),
    "hard_column_conflicts": tuple(range(9)),
    "hard_box_conflicts": tuple(range(9)),
    "hard_unique_conflicts": tuple(range(21)),
    "row_candidate_count": tuple(range(10)),
    "column_candidate_count": tuple(range(10)),
    "box_candidate_count": tuple(range(10)),
    "joint_candidate_count": tuple(range(10)),
    "static_joint_candidate_count": tuple(range(10)),
    "target_candidate_legal": (0, 1),
}
PRIMARY_DYNAMICS = {
    "expected_unique_conflicts": -1,
    "joint_candidate_deviation": -1,
    "target_candidate_legal_fraction": 1,
    "prediction_accuracy": 1,
}
MODEL_LABELS = {
    "stable_plain": "Stable plain",
    "collapsed_plain": "Collapsed plain",
    "late_state_ce": "Late-state CE",
    "combined_margin": "Combined",
}
MODEL_COLORS = {
    "stable_plain": "#237a57",
    "collapsed_plain": "#c33c32",
    "late_state_ce": "#2868a9",
    "combined_margin": "#8a5b9e",
}


def build_peer_matrices():
    """Return row, column, box, and unique-peer adjacency matrices."""
    matrices = {
        name: np.zeros((81, 81), dtype=np.float32)
        for name in ("row", "column", "box")
    }
    for cell in range(81):
        row, column = divmod(cell, 9)
        row_cells = {row * 9 + other for other in range(9)} - {cell}
        column_cells = {other * 9 + column for other in range(9)} - {cell}
        box_row = (row // 3) * 3
        box_column = (column // 3) * 3
        box_cells = {
            (box_row + row_offset) * 9 + box_column + column_offset
            for row_offset in range(3)
            for column_offset in range(3)
        } - {cell}
        for peer in row_cells:
            matrices["row"][cell, peer] = 1.0
        for peer in column_cells:
            matrices["column"][cell, peer] = 1.0
        for peer in box_cells:
            matrices["box"][cell, peer] = 1.0
    matrices["unique"] = np.maximum.reduce(
        [matrices["row"], matrices["column"], matrices["box"]]
    )
    return matrices


def _softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def derive_constraint_quantities(logits, inputs, targets):
    """Derive per-cell Sudoku quantities from predictions and fixed clues."""
    logits = np.asarray(logits, dtype=np.float32)
    inputs = np.asarray(inputs)
    targets = np.asarray(targets, dtype=np.int64)
    if logits.ndim != 4 or logits.shape[2:] != (81, 9):
        raise ValueError("logits must have shape [puzzles, snapshots, 81, 9]")
    if inputs.shape != (logits.shape[0], 81, 10):
        raise ValueError("inputs must have shape [puzzles, 81, 10]")
    if targets.shape != (logits.shape[0], 81):
        raise ValueError("targets must have shape [puzzles, 81]")

    encoded_values = inputs.argmax(axis=-1)
    empty_mask = encoded_values == 0
    clue_digits = np.clip(encoded_values - 1, 0, 8)
    digit_eye = np.eye(9, dtype=np.float32)
    clue_one_hot = digit_eye[clue_digits] * (~empty_mask)[..., None]

    probabilities = _softmax(logits)
    board_probabilities = np.where(
        (~empty_mask)[:, None, :, None],
        clue_one_hot[:, None, :, :],
        probabilities,
    )
    hard_digits = board_probabilities.argmax(axis=-1)
    hard_one_hot = digit_eye[hard_digits]
    peer_matrices = build_peer_matrices()
    quantities = {}
    peer_hard_counts = {}

    for group_name, peer_matrix in peer_matrices.items():
        probability_mass = np.einsum(
            "ij,ntjd->ntid",
            peer_matrix,
            board_probabilities,
            optimize=True,
        )
        quantities[f"expected_{group_name}_conflicts"] = (
            board_probabilities * probability_mass
        ).sum(axis=-1)
        hard_counts = np.einsum(
            "ij,ntjd->ntid",
            peer_matrix,
            hard_one_hot,
            optimize=True,
        )
        peer_hard_counts[group_name] = hard_counts
        quantities[f"hard_{group_name}_conflicts"] = np.take_along_axis(
            hard_counts,
            hard_digits[..., None],
            axis=-1,
        )[..., 0]
        candidate_name = "joint" if group_name == "unique" else group_name
        quantities[f"{candidate_name}_candidate_count"] = (
            hard_counts == 0
        ).sum(axis=-1)

    static_peer_counts = np.einsum(
        "ij,njd->nid",
        peer_matrices["unique"],
        clue_one_hot,
        optimize=True,
    )
    static_candidate_count = (static_peer_counts == 0).sum(axis=-1)
    quantities["static_joint_candidate_count"] = np.broadcast_to(
        static_candidate_count[:, None, :],
        hard_digits.shape,
    ).copy()
    target_peer_counts = np.take_along_axis(
        peer_hard_counts["unique"],
        targets[:, None, :, None],
        axis=-1,
    )[..., 0]
    quantities["target_candidate_legal"] = (target_peer_counts == 0).astype(
        np.int64
    )
    quantities["prediction_correct"] = (
        hard_digits == targets[:, None, :]
    ).astype(np.int64)
    quantities["hard_predictions"] = hard_digits
    quantities["empty_mask"] = empty_mask
    return quantities


def _r2_score(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    denominator = np.square(targets - targets.mean()).sum()
    if denominator <= 1e-12:
        return None
    return float(1.0 - np.square(targets - predictions).sum() / denominator)


def _require_finite(name, *arrays):
    if any(not np.isfinite(np.asarray(array)).all() for array in arrays):
        raise ValueError(f"{name} contains non-finite values")


def _numeric_scores(targets, predictions, baseline_value):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    _require_finite("numeric score input", targets, predictions, baseline_value)
    baseline = np.full_like(targets, baseline_value)
    mae = float(np.abs(targets - predictions).mean())
    baseline_mae = float(np.abs(targets - baseline).mean())
    return {
        "r2": _r2_score(targets, predictions),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "mae_improvement": (
            float(1.0 - mae / baseline_mae)
            if baseline_mae > 1e-12
            else None
        ),
    }


def _categorical_scores(targets, predictions, classes, baseline_class):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    _require_finite("categorical score input", targets, predictions)
    supported_classes = [value for value in classes if np.any(targets == value)]
    recalls = [
        float((predictions[targets == value] == value).mean())
        for value in supported_classes
    ]
    baseline_predictions = np.full_like(targets, baseline_class)
    baseline_recalls = [
        float((baseline_predictions[targets == value] == value).mean())
        for value in supported_classes
    ]
    return {
        "accuracy": float((predictions == targets).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "baseline_accuracy": float((baseline_predictions == targets).mean()),
        "baseline_balanced_accuracy": float(np.mean(baseline_recalls)),
        "supported_classes": [int(value) for value in supported_classes],
    }


def _standardize_features(discovery, validation, final):
    _require_finite("probe features", discovery, validation, final)
    mean = discovery.mean(axis=0)
    scale = discovery.std(axis=0)
    scale[scale < 1e-6] = 1.0
    standardized = tuple((values - mean) / scale for values in (
        discovery,
        validation,
        final,
    ))
    _require_finite("standardized probe features", *standardized)
    return standardized


def _ridge_operators(features, alphas=RIDGE_ALPHAS):
    features = np.asarray(features, dtype=np.float64)
    _require_finite("ridge features", features)
    gram = features.T @ features
    identity = np.eye(features.shape[1], dtype=np.float64)
    operators = {
        alpha: np.linalg.solve(gram + alpha * identity, features.T)
        for alpha in alphas
    }
    _require_finite("ridge operators", *operators.values())
    return operators


def _safe_selection_score(value):
    return -math.inf if value is None or not np.isfinite(value) else value


def _optional_mean(values):
    return float(np.mean(values)) if values else None


def _optional_maximum(values):
    return float(np.max(values)) if values else None


def _stable_seed(*parts):
    value = RANDOM_SEED
    for part in parts:
        for character in str(part):
            value = (value * 131 + ord(character)) % (2**32)
    return value


@_quiet_accelerate_matmul
def fit_numeric_probe(
    discovery_features,
    validation_features,
    final_features,
    discovery_targets,
    validation_targets,
    final_targets,
    *,
    control_key,
):
    operators = _ridge_operators(discovery_features)
    target_mean = float(np.mean(discovery_targets))
    centered_targets = discovery_targets - target_mean
    candidates = []
    for alpha, operator in operators.items():
        coefficients = operator @ centered_targets
        predictions = validation_features @ coefficients + target_mean
        scores = _numeric_scores(validation_targets, predictions, target_mean)
        candidates.append((alpha, coefficients, scores))
    alpha, coefficients, validation_scores = max(
        candidates,
        key=lambda candidate: _safe_selection_score(candidate[2]["r2"]),
    )
    final_predictions = final_features @ coefficients + target_mean
    final_scores = _numeric_scores(
        final_targets,
        final_predictions,
        target_mean,
    )

    generator = np.random.default_rng(_stable_seed(control_key, "numeric"))
    shuffled_scores = []
    for _ in range(CONTROL_REPEATS):
        shuffled_targets = discovery_targets[
            generator.permutation(len(discovery_targets))
        ]
        shuffled_coefficients = operators[alpha] @ (
            shuffled_targets - shuffled_targets.mean()
        )
        shuffled_predictions = (
            final_features @ shuffled_coefficients + shuffled_targets.mean()
        )
        shuffled_scores.append(
            _numeric_scores(
                final_targets,
                shuffled_predictions,
                shuffled_targets.mean(),
            )["r2"]
        )

    random_subspace_scores = []
    for _ in range(CONTROL_REPEATS):
        random_direction = generator.normal(size=(discovery_features.shape[1], 1))
        random_direction /= np.linalg.norm(random_direction)
        projected = (
            discovery_features @ random_direction,
            validation_features @ random_direction,
            final_features @ random_direction,
        )
        projected_operators = _ridge_operators(projected[0])
        projected_candidates = []
        for projected_alpha, operator in projected_operators.items():
            projected_coefficients = operator @ centered_targets
            predictions = projected[1] @ projected_coefficients + target_mean
            score = _r2_score(validation_targets, predictions)
            projected_candidates.append(
                (projected_alpha, projected_coefficients, score)
            )
        _, projected_coefficients, _ = max(
            projected_candidates,
            key=lambda candidate: _safe_selection_score(candidate[2]),
        )
        random_predictions = projected[2] @ projected_coefficients + target_mean
        random_subspace_scores.append(
            _r2_score(final_targets, random_predictions)
        )

    finite_shuffled = [value for value in shuffled_scores if value is not None]
    finite_random = [
        value for value in random_subspace_scores if value is not None
    ]
    return {
        "selected_alpha": alpha,
        "validation": validation_scores,
        "final": final_scores,
        "label_shuffle": {
            "repeats": CONTROL_REPEATS,
            "mean_r2": _optional_mean(finite_shuffled),
            "maximum_r2": _optional_maximum(finite_shuffled),
        },
        "matched_rank_random_subspace": {
            "rank": 1,
            "repeats": CONTROL_REPEATS,
            "mean_r2": _optional_mean(finite_random),
            "maximum_r2": _optional_maximum(finite_random),
        },
    }, final_predictions


def _one_hot(values, classes):
    class_to_index = {value: index for index, value in enumerate(classes)}
    encoded = np.zeros((len(values), len(classes)), dtype=np.float64)
    for row, value in enumerate(values):
        if int(value) not in class_to_index:
            raise ValueError(f"unexpected class value: {value}")
        encoded[row, class_to_index[int(value)]] = 1.0
    return encoded


def _decode_scores(scores, classes):
    indices = np.asarray(scores).argmax(axis=1)
    return np.asarray(classes)[indices]


@_quiet_accelerate_matmul
def fit_categorical_probe(
    discovery_features,
    validation_features,
    final_features,
    discovery_targets,
    validation_targets,
    final_targets,
    classes,
    *,
    control_key,
):
    operators = _ridge_operators(discovery_features)
    encoded_targets = _one_hot(discovery_targets, classes)
    target_mean = encoded_targets.mean(axis=0)
    centered_targets = encoded_targets - target_mean
    values, counts = np.unique(discovery_targets, return_counts=True)
    baseline_class = int(values[np.argmax(counts)])
    candidates = []
    for alpha, operator in operators.items():
        coefficients = operator @ centered_targets
        predictions = _decode_scores(
            validation_features @ coefficients + target_mean,
            classes,
        )
        scores = _categorical_scores(
            validation_targets,
            predictions,
            classes,
            baseline_class,
        )
        candidates.append((alpha, coefficients, scores))
    alpha, coefficients, validation_scores = max(
        candidates,
        key=lambda candidate: candidate[2]["balanced_accuracy"],
    )
    final_predictions = _decode_scores(
        final_features @ coefficients + target_mean,
        classes,
    )
    final_scores = _categorical_scores(
        final_targets,
        final_predictions,
        classes,
        baseline_class,
    )
    learned_rank = max(1, int(np.linalg.matrix_rank(coefficients, tol=1e-7)))

    generator = np.random.default_rng(_stable_seed(control_key, "categorical"))
    shuffled_scores = []
    for _ in range(CONTROL_REPEATS):
        shuffled_targets = discovery_targets[
            generator.permutation(len(discovery_targets))
        ]
        shuffled_encoded = _one_hot(shuffled_targets, classes)
        shuffled_mean = shuffled_encoded.mean(axis=0)
        shuffled_coefficients = operators[alpha] @ (
            shuffled_encoded - shuffled_mean
        )
        shuffled_predictions = _decode_scores(
            final_features @ shuffled_coefficients + shuffled_mean,
            classes,
        )
        shuffled_scores.append(
            _categorical_scores(
                final_targets,
                shuffled_predictions,
                classes,
                baseline_class,
            )["balanced_accuracy"]
        )

    random_subspace_scores = []
    rank = min(learned_rank, discovery_features.shape[1])
    for _ in range(CONTROL_REPEATS):
        random_matrix = generator.normal(
            size=(discovery_features.shape[1], rank)
        )
        random_basis, _ = np.linalg.qr(random_matrix)
        projected = (
            discovery_features @ random_basis,
            validation_features @ random_basis,
            final_features @ random_basis,
        )
        projected_operators = _ridge_operators(projected[0])
        projected_candidates = []
        for projected_alpha, operator in projected_operators.items():
            projected_coefficients = operator @ centered_targets
            predictions = _decode_scores(
                projected[1] @ projected_coefficients + target_mean,
                classes,
            )
            score = _categorical_scores(
                validation_targets,
                predictions,
                classes,
                baseline_class,
            )["balanced_accuracy"]
            projected_candidates.append(
                (projected_alpha, projected_coefficients, score)
            )
        _, projected_coefficients, _ = max(
            projected_candidates,
            key=lambda candidate: candidate[2],
        )
        random_predictions = _decode_scores(
            projected[2] @ projected_coefficients + target_mean,
            classes,
        )
        random_subspace_scores.append(
            _categorical_scores(
                final_targets,
                random_predictions,
                classes,
                baseline_class,
            )["balanced_accuracy"]
        )

    return {
        "selected_alpha": alpha,
        "validation": validation_scores,
        "final": final_scores,
        "label_shuffle": {
            "repeats": CONTROL_REPEATS,
            "mean_balanced_accuracy": float(np.mean(shuffled_scores)),
            "maximum_balanced_accuracy": float(np.max(shuffled_scores)),
        },
        "matched_rank_random_subspace": {
            "rank": rank,
            "repeats": CONTROL_REPEATS,
            "mean_balanced_accuracy": float(
                np.mean(random_subspace_scores)
            ),
            "maximum_balanced_accuracy": float(
                np.max(random_subspace_scores)
            ),
        },
    }, final_predictions


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return np.sum(values * weights[:, None], axis=0)


def _fit_weighted_standardizer(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    mean = _weighted_mean(values, weights)
    variance = _weighted_mean(np.square(values - mean), weights)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    return {"mean": mean, "scale": scale}


def _apply_weighted_standardizer(values, standardizer):
    values = np.asarray(values, dtype=np.float64)
    result = (values - standardizer["mean"]) / standardizer["scale"]
    _require_finite("weighted standardized features", result)
    return result


def _weighted_sse(targets, predictions, weights):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    return float(np.sum(weights * np.square(targets - predictions)))


def _weighted_r2(targets, predictions, weights):
    weights = np.asarray(weights, dtype=np.float64)
    mean = float(np.sum(weights * targets) / weights.sum())
    denominator = _weighted_sse(
        targets, np.full_like(targets, mean), weights
    )
    if denominator <= 1e-12:
        return None
    return float(1.0 - _weighted_sse(targets, predictions, weights) / denominator)


def _fit_weighted_ridge_scalar(design, targets, weights, alpha):
    design = np.asarray(design, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    covariance = design.T @ (weights[:, None] * design)
    cross_covariance = design.T @ (weights * targets)
    penalty = np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    penalized_scale = max(
        float(np.diag(covariance)[1:].mean())
        if design.shape[1] > 1 else 1.0,
        1e-12,
    )
    coefficients = np.linalg.solve(
        covariance + alpha * penalized_scale * penalty,
        cross_covariance,
    )
    _require_finite("weighted ridge coefficients", coefficients)
    return coefficients


def _select_weighted_ridge(
    discovery_design,
    validation_design,
    discovery_targets,
    validation_targets,
    discovery_weights,
    validation_weights,
    *,
    offset_validation=None,
):
    if offset_validation is None:
        offset_validation = np.zeros_like(validation_targets, dtype=np.float64)
    candidates = []
    for alpha in RIDGE_ALPHAS:
        coefficients = _fit_weighted_ridge_scalar(
            discovery_design,
            discovery_targets,
            discovery_weights,
            alpha,
        )
        predictions = offset_validation + validation_design @ coefficients
        candidates.append((
            _weighted_sse(
                validation_targets, predictions, validation_weights
            ),
            alpha,
            coefficients,
        ))
    _, alpha, coefficients = min(candidates, key=lambda item: item[0])
    return float(alpha), coefficients


def _equal_puzzle_time_weights(puzzle_ids, time_slots):
    puzzle_ids = np.asarray(puzzle_ids, dtype=np.int64)
    time_slots = np.asarray(time_slots, dtype=np.int64)
    pairs = np.column_stack((puzzle_ids, time_slots))
    _, inverse, counts = np.unique(
        pairs, axis=0, return_inverse=True, return_counts=True
    )
    weights = 1.0 / counts[inverse]
    return weights / weights.sum()


def build_temporal_probe_datasets(
    states,
    logits,
    quantities,
    empty_mask,
    split_names,
    snapshots,
    mode,
):
    """Build whole-puzzle splits for current-state and future-change probes.

    The current-state nuisance model contains exact sampled iteration,
    confidence, top-two probability margin, and entropy.  The future model also
    contains current expected/hard conflicts, candidate deviation, and
    correctness.  State coefficients therefore measure held-out gain beyond
    iteration, certainty, and the current constraint status itself.
    """

    if mode == "current_constraint":
        time_indices = [snapshots.index(value) for value in PROBE_SNAPSHOTS]
    elif mode == "future_improvement":
        time_indices = list(range(len(snapshots) - 1))
    else:
        raise ValueError(f"unknown temporal probe mode {mode!r}")

    probabilities = _softmax(logits)
    sorted_probabilities = np.sort(probabilities, axis=-1)
    confidence = sorted_probabilities[..., -1]
    probability_margin = (
        sorted_probabilities[..., -1] - sorted_probabilities[..., -2]
    )
    entropy = -np.sum(
        probabilities * np.log(np.maximum(probabilities, 1e-30)), axis=-1
    )
    split_names = np.asarray(split_names)
    datasets = {}

    for split_name in ("discovery", "validation", "final"):
        puzzle_indices = np.flatnonzero(split_names == split_name)
        state_rows = []
        nuisance_rows = []
        target_rows = []
        puzzle_rows = []
        time_rows = []
        cell_rows = []
        for time_slot, time_index in enumerate(time_indices):
            time_code = np.zeros(len(time_indices), dtype=np.float64)
            time_code[time_slot] = 1.0
            for puzzle_index in puzzle_indices:
                cells = np.flatnonzero(empty_mask[puzzle_index])
                row_count = len(cells)
                state_rows.append(states[puzzle_index, time_index, cells])
                continuous = [
                    confidence[puzzle_index, time_index, cells],
                    probability_margin[puzzle_index, time_index, cells],
                    entropy[puzzle_index, time_index, cells],
                ]
                if mode == "future_improvement":
                    continuous.extend([
                        quantities["expected_unique_conflicts"][
                            puzzle_index, time_index, cells
                        ],
                        quantities["hard_unique_conflicts"][
                            puzzle_index, time_index, cells
                        ],
                        np.abs(
                            quantities["joint_candidate_count"][
                                puzzle_index, time_index, cells
                            ] - 1
                        ),
                        quantities["prediction_correct"][
                            puzzle_index, time_index, cells
                        ],
                    ])
                    target = (
                        quantities["expected_unique_conflicts"][
                            puzzle_index, time_index, cells
                        ]
                        - quantities["expected_unique_conflicts"][
                            puzzle_index, time_index + 1, cells
                        ]
                    )
                else:
                    target = quantities["expected_unique_conflicts"][
                        puzzle_index, time_index, cells
                    ]
                nuisance_rows.append(np.column_stack(
                    [np.broadcast_to(time_code, (row_count, len(time_code)))]
                    + [np.asarray(value)[:, None] for value in continuous]
                ))
                target_rows.append(target)
                puzzle_rows.append(np.full(row_count, puzzle_index))
                time_rows.append(np.full(row_count, time_slot))
                cell_rows.append(cells)
        puzzle_ids = np.concatenate(puzzle_rows).astype(np.int64)
        time_slots = np.concatenate(time_rows).astype(np.int64)
        datasets[split_name] = {
            "states": np.concatenate(state_rows).astype(np.float64),
            "nuisance": np.concatenate(nuisance_rows).astype(np.float64),
            "targets": np.concatenate(target_rows).astype(np.float64),
            "puzzle_ids": puzzle_ids,
            "time_slots": time_slots,
            "cell_ids": np.concatenate(cell_rows).astype(np.int64),
            "weights": _equal_puzzle_time_weights(puzzle_ids, time_slots),
        }
    return datasets


def _permute_time_rows(values, dataset, permutation):
    permutation = np.asarray(permutation, dtype=np.int64)
    if sorted(permutation.tolist()) != list(range(len(permutation))):
        raise ValueError("permutation must contain every time slot")
    keys = {
        (int(puzzle), int(time), int(cell)): index
        for index, (puzzle, time, cell) in enumerate(zip(
            dataset["puzzle_ids"],
            dataset["time_slots"],
            dataset["cell_ids"],
        ))
    }
    source_indices = [
        keys[(int(puzzle), int(permutation[time]), int(cell))]
        for puzzle, time, cell in zip(
            dataset["puzzle_ids"],
            dataset["time_slots"],
            dataset["cell_ids"],
        )
    ]
    return np.asarray(values)[source_indices]


def _residual_probe_scores(dataset, base_predictions, full_predictions):
    targets = dataset["targets"]
    weights = dataset["weights"]
    target_mean = float(np.sum(weights * targets) / weights.sum())
    null_predictions = np.full_like(targets, target_mean)
    null_sse = _weighted_sse(targets, null_predictions, weights)
    base_sse = _weighted_sse(targets, base_predictions, weights)
    full_sse = _weighted_sse(targets, full_predictions, weights)
    return {
        "baseline_r2": (
            float(1.0 - base_sse / null_sse) if null_sse > 1e-12 else None
        ),
        "full_r2": (
            float(1.0 - full_sse / null_sse) if null_sse > 1e-12 else None
        ),
        "state_partial_r2": (
            float(1.0 - full_sse / base_sse) if base_sse > 1e-12 else None
        ),
        "baseline_sse": base_sse,
        "full_sse": full_sse,
    }


def _upper_control_p(observed, controls):
    controls = [value for value in controls if value is not None]
    if observed is None or not controls:
        return None
    return float(
        (1 + sum(value >= observed for value in controls))
        / (len(controls) + 1)
    )


@_quiet_accelerate_matmul
def fit_residual_temporal_probe(datasets, *, control_key):
    """Fit nuisance then state-residual probes without touching final labels."""

    discovery = datasets["discovery"]
    validation = datasets["validation"]
    final = datasets["final"]
    nuisance_standardizer = _fit_weighted_standardizer(
        discovery["nuisance"], discovery["weights"]
    )
    nuisance_design = {}
    for name, dataset in datasets.items():
        standardized = _apply_weighted_standardizer(
            dataset["nuisance"], nuisance_standardizer
        )
        nuisance_design[name] = np.column_stack(
            (np.ones(len(standardized)), standardized)
        )
    base_alpha, base_coefficients = _select_weighted_ridge(
        nuisance_design["discovery"],
        nuisance_design["validation"],
        discovery["targets"],
        validation["targets"],
        discovery["weights"],
        validation["weights"],
    )
    base_predictions = {
        name: nuisance_design[name] @ base_coefficients
        for name in datasets
    }
    residual_targets = {
        name: datasets[name]["targets"] - base_predictions[name]
        for name in datasets
    }

    state_standardizer = _fit_weighted_standardizer(
        discovery["states"], discovery["weights"]
    )
    standardized_states = {
        name: _apply_weighted_standardizer(
            dataset["states"], state_standardizer
        )
        for name, dataset in datasets.items()
    }
    state_design = {
        name: np.column_stack((np.ones(len(values)), values))
        for name, values in standardized_states.items()
    }
    state_alpha, state_coefficients = _select_weighted_ridge(
        state_design["discovery"],
        state_design["validation"],
        residual_targets["discovery"],
        validation["targets"],
        discovery["weights"],
        validation["weights"],
        offset_validation=base_predictions["validation"],
    )
    full_predictions = {
        name: base_predictions[name] + state_design[name] @ state_coefficients
        for name in datasets
    }
    final_scores = _residual_probe_scores(
        final, base_predictions["final"], full_predictions["final"]
    )

    generator = np.random.default_rng(_stable_seed(control_key, "residual"))
    label_shuffle_scores = []
    for _ in range(CONTROL_REPEATS):
        shuffled = residual_targets["discovery"].copy()
        for time_slot in np.unique(discovery["time_slots"]):
            indices = np.flatnonzero(discovery["time_slots"] == time_slot)
            shuffled[indices] = shuffled[indices][generator.permutation(len(indices))]
        coefficients = _fit_weighted_ridge_scalar(
            state_design["discovery"],
            shuffled,
            discovery["weights"],
            state_alpha,
        )
        predictions = base_predictions["final"] + state_design["final"] @ coefficients
        label_shuffle_scores.append(_residual_probe_scores(
            final, base_predictions["final"], predictions
        )["state_partial_r2"])

    random_subspace_scores = []
    for _ in range(CONTROL_REPEATS):
        direction = generator.normal(size=(standardized_states["discovery"].shape[1], 1))
        direction /= np.linalg.norm(direction)
        projected_design = {
            name: np.column_stack((np.ones(len(values)), values @ direction))
            for name, values in standardized_states.items()
        }
        _, coefficients = _select_weighted_ridge(
            projected_design["discovery"],
            projected_design["validation"],
            residual_targets["discovery"],
            validation["targets"],
            discovery["weights"],
            validation["weights"],
            offset_validation=base_predictions["validation"],
        )
        predictions = (
            base_predictions["final"]
            + projected_design["final"] @ coefficients
        )
        random_subspace_scores.append(_residual_probe_scores(
            final, base_predictions["final"], predictions
        )["state_partial_r2"])

    iteration_shuffle_scores = []
    time_count = int(discovery["time_slots"].max()) + 1
    for _ in range(CONTROL_REPEATS):
        permutation = generator.permutation(time_count)
        if np.array_equal(permutation, np.arange(time_count)):
            permutation = np.roll(permutation, 1)
        shuffled_states = {
            name: _permute_time_rows(
                standardized_states[name], dataset, permutation
            )
            for name, dataset in datasets.items()
        }
        shuffled_design = {
            name: np.column_stack((np.ones(len(values)), values))
            for name, values in shuffled_states.items()
        }
        coefficients = _fit_weighted_ridge_scalar(
            shuffled_design["discovery"],
            residual_targets["discovery"],
            discovery["weights"],
            state_alpha,
        )
        predictions = (
            base_predictions["final"]
            + shuffled_design["final"] @ coefficients
        )
        iteration_shuffle_scores.append(_residual_probe_scores(
            final, base_predictions["final"], predictions
        )["state_partial_r2"])

    return {
        "selected_base_alpha": base_alpha,
        "selected_state_alpha": state_alpha,
        "validation": _residual_probe_scores(
            validation,
            base_predictions["validation"],
            full_predictions["validation"],
        ),
        "final": final_scores,
        "label_shuffle": {
            "repeats": CONTROL_REPEATS,
            "mean_partial_r2": _optional_mean([
                value for value in label_shuffle_scores if value is not None
            ]),
            "maximum_partial_r2": _optional_maximum([
                value for value in label_shuffle_scores if value is not None
            ]),
            "one_sided_p": _upper_control_p(
                final_scores["state_partial_r2"], label_shuffle_scores
            ),
        },
        "matched_rank_random_subspace": {
            "rank": 1,
            "repeats": CONTROL_REPEATS,
            "mean_partial_r2": _optional_mean([
                value for value in random_subspace_scores if value is not None
            ]),
            "maximum_partial_r2": _optional_maximum([
                value for value in random_subspace_scores if value is not None
            ]),
            "one_sided_p": _upper_control_p(
                final_scores["state_partial_r2"], random_subspace_scores
            ),
        },
        "shuffled_iteration": {
            "repeats": CONTROL_REPEATS,
            "mean_partial_r2": _optional_mean([
                value for value in iteration_shuffle_scores if value is not None
            ]),
            "maximum_partial_r2": _optional_maximum([
                value for value in iteration_shuffle_scores if value is not None
            ]),
            "one_sided_p": _upper_control_p(
                final_scores["state_partial_r2"], iteration_shuffle_scores
            ),
        },
    }, {
        "nuisance_standardizer": nuisance_standardizer,
        "state_standardizer": state_standardizer,
        "base_coefficients": base_coefficients,
        "state_coefficients": state_coefficients,
        "mode": control_key[-1],
    }


@_quiet_accelerate_matmul
def evaluate_cross_checkpoint_probe(cache, target_dataset):
    """Apply a source checkpoint's nuisance and state probe without refitting."""

    nuisance = _apply_weighted_standardizer(
        target_dataset["nuisance"], cache["nuisance_standardizer"]
    )
    state = _apply_weighted_standardizer(
        target_dataset["states"], cache["state_standardizer"]
    )
    base_design = np.column_stack((np.ones(len(nuisance)), nuisance))
    state_design = np.column_stack((np.ones(len(state)), state))
    base_predictions = base_design @ cache["base_coefficients"]
    full_predictions = (
        base_predictions + state_design @ cache["state_coefficients"]
    )
    return _residual_probe_scores(
        target_dataset, base_predictions, full_predictions
    )


def _average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman_correlation(first, second):
    first_ranks = _average_ranks(first)
    second_ranks = _average_ranks(second)
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = np.linalg.norm(first_centered) * np.linalg.norm(second_centered)
    if denominator <= 1e-12:
        return 0.0
    return float(first_centered @ second_centered / denominator)


def iteration_shuffle_test(iterations, values, desired_direction, seed):
    """Compare the ordered trend with shuffled iteration labels."""
    iterations = np.asarray(iterations)
    values = np.asarray(values, dtype=np.float64)
    observed = desired_direction * spearman_correlation(iterations, values)
    generator = np.random.default_rng(seed)
    null = np.asarray([
        desired_direction * spearman_correlation(
            iterations,
            values[generator.permutation(len(values))],
        )
        for _ in range(TREND_PERMUTATIONS)
    ])
    return {
        "desired_direction": "increase" if desired_direction > 0 else "decrease",
        "ordered_spearman": float(observed),
        "shuffle_repeats": TREND_PERMUTATIONS,
        "shuffle_mean": float(null.mean()),
        "shuffle_standard_deviation": float(null.std()),
        "one_sided_p": float((1 + np.sum(null >= observed)) / (len(null) + 1)),
    }


def _bootstrap_mean_interval(values, seed, repeats=2_000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.choice(values, size=(repeats, len(values)), replace=True)
    means = samples.mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "puzzles": int(len(values)),
    }


def summarize_dynamics(quantities, split_names, snapshots):
    empty_mask = quantities["empty_mask"]
    split_names = np.asarray(split_names)
    summaries = {}
    per_puzzle = {}
    for split_name in ("discovery", "validation", "final"):
        puzzle_indices = np.flatnonzero(split_names == split_name)
        split_summary = []
        split_per_puzzle = {}
        for snapshot_index, snapshot in enumerate(snapshots):
            mask = empty_mask[puzzle_indices]

            def masked_mean(name):
                values = quantities[name][puzzle_indices, snapshot_index]
                return float(values[mask].mean())

            hard_unique = quantities["hard_unique_conflicts"][
                puzzle_indices, snapshot_index
            ]
            predictions_correct = quantities["prediction_correct"][
                puzzle_indices, snapshot_index
            ]
            candidate_count = quantities["joint_candidate_count"][
                puzzle_indices, snapshot_index
            ]
            target_legal = quantities["target_candidate_legal"][
                puzzle_indices, snapshot_index
            ]
            valid_boards = np.asarray([
                np.all(hard_unique[index][mask[index]] == 0)
                for index in range(len(puzzle_indices))
            ])
            solved_boards = np.asarray([
                np.all(predictions_correct[index][mask[index]] == 1)
                for index in range(len(puzzle_indices))
            ])
            record = {
                "snapshot": int(snapshot),
                "expected_row_conflicts": masked_mean(
                    "expected_row_conflicts"
                ),
                "expected_column_conflicts": masked_mean(
                    "expected_column_conflicts"
                ),
                "expected_box_conflicts": masked_mean(
                    "expected_box_conflicts"
                ),
                "expected_unique_conflicts": masked_mean(
                    "expected_unique_conflicts"
                ),
                "hard_unique_conflicts": masked_mean(
                    "hard_unique_conflicts"
                ),
                "joint_candidate_deviation": float(
                    np.abs(candidate_count[mask] - 1).mean()
                ),
                "target_candidate_legal_fraction": float(
                    target_legal[mask].mean()
                ),
                "prediction_accuracy": float(
                    predictions_correct[mask].mean()
                ),
                "valid_board_fraction": float(valid_boards.mean()),
                "solved_board_fraction": float(solved_boards.mean()),
            }
            split_summary.append(record)

            def puzzle_means(values):
                return np.asarray([
                    values[index][mask[index]].mean()
                    for index in range(len(puzzle_indices))
                ])

            split_per_puzzle[int(snapshot)] = {
                "expected_unique_conflicts": puzzle_means(
                    quantities["expected_unique_conflicts"][
                        puzzle_indices, snapshot_index
                    ]
                ),
                "joint_candidate_deviation": puzzle_means(
                    np.abs(candidate_count - 1)
                ),
                "target_candidate_legal_fraction": puzzle_means(target_legal),
                "prediction_accuracy": puzzle_means(predictions_correct),
            }
        summaries[split_name] = split_summary
        per_puzzle[split_name] = split_per_puzzle
    return summaries, per_puzzle


def analyze_model_probes(
    states,
    quantities,
    empty_mask,
    split_names,
    snapshots,
    model_name,
):
    split_names = np.asarray(split_names)
    results = []
    calibration = {}
    for snapshot in PROBE_SNAPSHOTS:
        snapshot_index = snapshots.index(snapshot)
        split_features = {}
        split_targets = {}
        for split_name in ("discovery", "validation", "final"):
            puzzle_mask = split_names == split_name
            cell_mask = empty_mask[puzzle_mask]
            split_features[split_name] = states[
                puzzle_mask, snapshot_index
            ][cell_mask].astype(np.float64)
            split_targets[split_name] = {
                target_name: quantities[target_name][
                    puzzle_mask, snapshot_index
                ][cell_mask]
                for target_name in set(NUMERIC_TARGETS + CATEGORICAL_TARGETS)
            }
        standardized = _standardize_features(
            split_features["discovery"],
            split_features["validation"],
            split_features["final"],
        )

        for target_name in NUMERIC_TARGETS:
            probe, predictions = fit_numeric_probe(
                *standardized,
                split_targets["discovery"][target_name].astype(np.float64),
                split_targets["validation"][target_name].astype(np.float64),
                split_targets["final"][target_name].astype(np.float64),
                control_key=(model_name, snapshot, target_name),
            )
            results.append({
                "model": model_name,
                "snapshot": snapshot,
                "target": target_name,
                "probe_type": "numeric",
                **probe,
            })
            if target_name == "expected_unique_conflicts" and snapshot == 16:
                calibration[target_name] = {
                    "targets": split_targets["final"][target_name].astype(
                        np.float64
                    ),
                    "predictions": predictions,
                }

        for target_name in CATEGORICAL_TARGETS:
            probe, predictions = fit_categorical_probe(
                *standardized,
                split_targets["discovery"][target_name].astype(np.int64),
                split_targets["validation"][target_name].astype(np.int64),
                split_targets["final"][target_name].astype(np.int64),
                CLASS_VALUES[target_name],
                control_key=(model_name, snapshot, target_name),
            )
            results.append({
                "model": model_name,
                "snapshot": snapshot,
                "target": target_name,
                "probe_type": "categorical",
                **probe,
            })
    return results, calibration


def _paired_changes(per_puzzle, model_name):
    comparisons = ((16, 128), (128, 1024), (16, 1024))
    records = []
    for metric_name, desired_direction in PRIMARY_DYNAMICS.items():
        for start, end in comparisons:
            raw_change = (
                per_puzzle[end][metric_name] - per_puzzle[start][metric_name]
            )
            improvement = desired_direction * raw_change
            records.append({
                "model": model_name,
                "metric": metric_name,
                "start": start,
                "end": end,
                "positive_means_improvement": True,
                **_bootstrap_mean_interval(
                    improvement,
                    _stable_seed(model_name, metric_name, start, end),
                ),
            })
    return records


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_csv(path, records, columns):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column) for column in columns})


def _save_figure(figure, path):
    figure.savefig(path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_dynamics(dynamics, snapshots, output_path):
    panels = (
        ("expected_unique_conflicts", "Expected peer conflicts", False),
        ("joint_candidate_deviation", "|candidate count - 1|", False),
        ("target_candidate_legal_fraction", "Correct digit remains legal", True),
        ("prediction_accuracy", "Cell accuracy", True),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, (metric_name, title, fraction) in zip(axes.flat, panels):
        for model_name, model_dynamics in dynamics.items():
            records = model_dynamics["final"]
            axis.plot(
                snapshots,
                [record[metric_name] for record in records],
                marker="o",
                linewidth=2,
                markersize=4,
                color=MODEL_COLORS.get(model_name),
                label=MODEL_LABELS.get(model_name, model_name),
            )
        axis.set_xscale("symlog", linthresh=1)
        axis.set_title(title)
        axis.set_xlabel("Recurrent iterations")
        if fraction:
            axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle("Constraint behavior on 20 final-holdout puzzles", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_constraint_components(dynamics, snapshots, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    component_styles = {
        "expected_row_conflicts": ("Row", "#2a6f97"),
        "expected_column_conflicts": ("Column", "#c05a32"),
        "expected_box_conflicts": ("Box", "#6a8e3a"),
    }
    for axis, (model_name, model_dynamics) in zip(axes.flat, dynamics.items()):
        records = model_dynamics["final"]
        for metric_name, (label, color) in component_styles.items():
            axis.plot(
                snapshots,
                [record[metric_name] for record in records],
                marker="o",
                markersize=3,
                color=color,
                label=label,
            )
        axis.set_xscale("symlog", linthresh=1)
        axis.set_title(MODEL_LABELS.get(model_name, model_name))
        axis.set_xlabel("Recurrent iterations")
        axis.set_ylabel("Expected same-digit peers")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.suptitle("Row, column, and box conflict components", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output_path)


def _find_probe(probe_records, model, snapshot, target, probe_type):
    return next(
        record
        for record in probe_records
        if record["model"] == model
        and record["snapshot"] == snapshot
        and record["target"] == target
        and record["probe_type"] == probe_type
    )


def plot_numeric_probes(probe_records, output_path):
    targets = (
        "expected_unique_conflicts",
        "hard_unique_conflicts",
        "joint_candidate_count",
        "static_joint_candidate_count",
    )
    target_labels = {
        "expected_unique_conflicts": "Expected conflict",
        "hard_unique_conflicts": "Hard conflict count",
        "joint_candidate_count": "Dynamic candidates",
        "static_joint_candidate_count": "Clue candidates",
    }
    model_names = list(MODEL_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    lower_display_limit = -0.5

    def display_value(value):
        if value is None or not np.isfinite(value):
            return np.nan
        return max(float(value), lower_display_limit)

    for axis, model_name in zip(axes.flat, model_names):
        for target_name in targets:
            records = [
                _find_probe(
                    probe_records,
                    model_name,
                    snapshot,
                    target_name,
                    "numeric",
                )
                for snapshot in PROBE_SNAPSHOTS
            ]
            axis.plot(
                PROBE_SNAPSHOTS,
                [display_value(record["final"]["r2"]) for record in records],
                marker="o",
                linewidth=2,
                label=target_labels[target_name],
            )
            axis.plot(
                PROBE_SNAPSHOTS,
                [
                    display_value(record["label_shuffle"]["mean_r2"])
                    for record in records
                ],
                linestyle=":",
                alpha=0.45,
            )
        axis.axhline(0, color="#555", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_ylim(lower_display_limit - 0.03, 1.03)
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Recurrent iterations")
        axis.set_ylabel("Final-holdout R²")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Linear readout of constraint quantities (dotted: shuffled labels; $R^2 < -0.5$ clipped)",
        fontsize=14,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_categorical_probes(probe_records, output_path):
    targets = (
        "hard_unique_conflicts",
        "joint_candidate_count",
        "static_joint_candidate_count",
    )
    target_labels = {
        "hard_unique_conflicts": "Hard conflict class",
        "joint_candidate_count": "Dynamic candidate class",
        "static_joint_candidate_count": "Clue candidate class",
    }
    model_names = list(MODEL_LABELS)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for axis, model_name in zip(axes.flat, model_names):
        for target_name in targets:
            records = [
                _find_probe(
                    probe_records,
                    model_name,
                    snapshot,
                    target_name,
                    "categorical",
                )
                for snapshot in PROBE_SNAPSHOTS
            ]
            axis.plot(
                PROBE_SNAPSHOTS,
                [
                    record["final"]["balanced_accuracy"]
                    if len(record["final"]["supported_classes"]) > 1
                    else np.nan
                    for record in records
                ],
                marker="o",
                linewidth=2,
                label=target_labels[target_name],
            )
            axis.plot(
                PROBE_SNAPSHOTS,
                [
                    record["label_shuffle"]["mean_balanced_accuracy"]
                    if len(record["final"]["supported_classes"]) > 1
                    else np.nan for record in records
                ],
                linestyle=":",
                alpha=0.45,
            )
        axis.set_xscale("log")
        axis.set_ylim(0, 1.03)
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Recurrent iterations")
        axis.set_ylabel("Balanced accuracy")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Unconstrained categorical readout (dotted: shuffled labels)",
        fontsize=14,
    )
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_calibration(calibrations, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, (model_name, calibration) in zip(axes.flat, calibrations.items()):
        values = calibration["expected_unique_conflicts"]
        targets = np.asarray(values["targets"])
        predictions = np.asarray(values["predictions"])
        axis.hexbin(
            targets,
            predictions,
            gridsize=28,
            mincnt=1,
            cmap="viridis",
        )
        lower = min(targets.min(), predictions.min())
        upper = max(targets.max(), predictions.max())
        axis.plot([lower, upper], [lower, upper], color="#c33c32", linewidth=1)
        axis.set_title(MODEL_LABELS.get(model_name, model_name))
        axis.set_xlabel("Measured expected conflicts")
        axis.set_ylabel("Linear-probe prediction")
        axis.grid(alpha=0.18)
    figure.suptitle("Final-holdout calibration at iteration 16", fontsize=14)
    figure.tight_layout()
    _save_figure(figure, output_path)


def plot_residual_probes(records, output_path):
    models = list(MODEL_LABELS)
    modes = ("current_constraint", "future_improvement")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    x = np.arange(len(models))
    for axis, mode in zip(axes, modes):
        selected = {
            record["model"]: record
            for record in records if record["mode"] == mode
        }
        observed = [
            selected[model]["final"]["state_partial_r2"] for model in models
        ]
        bars = axis.bar(
            x,
            observed,
            color=[MODEL_COLORS[model] for model in models],
            alpha=0.86,
            label="state gain",
        )
        for index, model in enumerate(models):
            record = selected[model]
            controls = (
                ("label_shuffle", "x", "shuffled labels"),
                ("matched_rank_random_subspace", "o", "random rank 1"),
                ("shuffled_iteration", "s", "shuffled iteration"),
            )
            for control_index, (key, marker, label) in enumerate(controls):
                axis.scatter(
                    index + (control_index - 1) * 0.08,
                    record[key]["maximum_partial_r2"],
                    marker=marker,
                    s=35,
                    color="black" if control_index == 0 else "#666666",
                    zorder=3,
                    label=label if index == 0 else None,
                )
        axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_xticks(x, [MODEL_LABELS[model] for model in models], rotation=20)
        axis.set_title(
            "Current expected conflicts" if mode == "current_constraint"
            else "Next-snapshot conflict improvement"
        )
        axis.set_ylabel("Final-holdout partial $R^2$ beyond nuisance")
        axis.legend(fontsize=8, loc="best")
        for bar in bars:
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )
    figure.suptitle(
        "Constraint geometry beyond iteration, confidence, margin, entropy, and current status"
    )
    _save_figure(figure, output_path)


def plot_cross_checkpoint_transfer(records, output_path):
    models = list(MODEL_LABELS)
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    image = None
    for axis, mode in zip(
        axes, ("current_constraint", "future_improvement")
    ):
        matrix = np.full((len(models), len(models)), np.nan)
        for record in records:
            if record["mode"] != mode:
                continue
            row = models.index(record["source_model"])
            column = models.index(record["target_model"])
            value = record["state_partial_r2"]
            matrix[row, column] = np.nan if value is None else value
        limit = 1.0
        image = axis.imshow(
            matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="equal"
        )
        axis.set_xticks(
            range(len(models)), [MODEL_LABELS[model] for model in models],
            rotation=35, ha="right",
        )
        axis.set_yticks(
            range(len(models)), [MODEL_LABELS[model] for model in models]
        )
        axis.set_xlabel("Target checkpoint (final puzzles)")
        axis.set_ylabel("Source checkpoint (discovery fit)")
        axis.set_title(
            "Current conflicts" if mode == "current_constraint"
            else "Future improvement"
        )
        for row in range(len(models)):
            for column in range(len(models)):
                value = matrix[row, column]
                label = "n/a" if not np.isfinite(value) else f"{value:.2f}"
                axis.text(
                    column, row, label, ha="center", va="center", fontsize=8,
                    color=(
                        "white" if np.isfinite(value)
                        and abs(np.clip(value, -limit, limit)) > limit * 0.55
                        else "black"
                    ),
                )
        figure.colorbar(image, ax=axis, shrink=0.72, label="partial $R^2$")
    figure.suptitle("Direct cross-checkpoint transfer without target refitting")
    _save_figure(figure, output_path)


def write_html(output_path, metrics, artifact_names):
    final_rows = []
    for model_name, split_dynamics in metrics["dynamics"].items():
        by_snapshot = {
            record["snapshot"]: record
            for record in split_dynamics["final"]
        }
        for snapshot in (16, 128, 1024):
            record = by_snapshot[snapshot]
            final_rows.append(
                "<tr>"
                f"<td>{html.escape(MODEL_LABELS.get(model_name, model_name))}</td>"
                f"<td>{snapshot}</td>"
                f"<td>{record['expected_unique_conflicts']:.3f}</td>"
                f"<td>{record['joint_candidate_deviation']:.3f}</td>"
                f"<td>{record['target_candidate_legal_fraction']:.3f}</td>"
                f"<td>{record['prediction_accuracy']:.3f}</td>"
                "</tr>"
            )
    figures = "".join(
        f'<section><h2>{html.escape(name)}</h2><img src="{html.escape(name)}"></section>'
        for name in artifact_names
    )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Sotaku Sudoku-constraint geometry</title>
<style>
body {{ font: 15px system-ui, sans-serif; margin: 28px; color: #1b1b1b; background: #f5f5f2; }}
main {{ max-width: 1200px; margin: auto; }}
table {{ border-collapse: collapse; background: white; }}
th, td {{ border: 1px solid #ccc; padding: 6px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
section {{ margin: 32px 0; }}
img {{ max-width: 100%; background: white; border: 1px solid #ccc; }}
h1, h2 {{ letter-spacing: 0; }}
</style>
<main>
<h1>Sotaku Sudoku-constraint geometry</h1>
<p>Projections are fit on 20 discovery puzzles, selected on 20 validation puzzles, and reported on 20 final-holdout puzzles. Dotted probe lines are shuffled-label controls.</p>
<table><thead><tr><th>Model</th><th>Iteration</th><th>Expected conflicts</th><th>|Candidates - 1|</th><th>Correct digit legal</th><th>Cell accuracy</th></tr></thead><tbody>{''.join(final_rows)}</tbody></table>
{figures}
</main>"""
    with open(output_path, "w") as handle:
        handle.write(document)


def analyze(payload_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    snapshots = [int(value) for value in payload["config"]["snapshots"]]
    for required_snapshot in PROBE_SNAPSHOTS:
        if required_snapshot not in snapshots:
            raise ValueError(f"missing probe snapshot {required_snapshot}")
    sample = payload["sample"]
    inputs = sample["inputs"].numpy()
    targets = sample["targets"].numpy()
    empty_mask = sample["empty_mask"].numpy().astype(bool)
    split_names = sample["split_names"]

    metrics = {
        "protocol": {
            "discovery_puzzles": split_names.count("discovery"),
            "validation_puzzles": split_names.count("validation"),
            "final_holdout_puzzles": split_names.count("final"),
            "snapshots": snapshots,
            "probe_snapshots": list(PROBE_SNAPSHOTS),
            "ridge_alphas": list(RIDGE_ALPHAS),
            "label_shuffle_repeats": CONTROL_REPEATS,
            "random_subspace_repeats": CONTROL_REPEATS,
            "iteration_shuffle_repeats": TREND_PERMUTATIONS,
            "seed": payload["config"]["seed"],
            "split_rule": (
                "Within each balanced rating bucket: first 4 sampled puzzles "
                "discovery, next 4 validation, final 4 final holdout."
            ),
        },
        "models": {
            model_name: model_payload["model_config"]
            for model_name, model_payload in payload["models"].items()
        },
        "dynamics": {},
        "trend_controls": [],
        "paired_changes": [],
        "probes": [],
        "residual_probes": [],
        "cross_checkpoint_transfer": [],
    }
    calibrations = {}
    temporal_datasets = {}
    temporal_caches = {}

    for model_name, model_payload in payload["models"].items():
        logits = model_payload["logits"].numpy()
        states = model_payload["states"].numpy()
        quantities = derive_constraint_quantities(logits, inputs, targets)
        dynamics, per_puzzle = summarize_dynamics(
            quantities,
            split_names,
            snapshots,
        )
        metrics["dynamics"][model_name] = dynamics
        final_records = dynamics["final"]
        for metric_name, desired_direction in PRIMARY_DYNAMICS.items():
            values = [record[metric_name] for record in final_records]
            metrics["trend_controls"].append({
                "model": model_name,
                "metric": metric_name,
                **iteration_shuffle_test(
                    snapshots,
                    values,
                    desired_direction,
                    _stable_seed(model_name, metric_name, "trend"),
                ),
            })
        metrics["paired_changes"].extend(
            _paired_changes(per_puzzle["final"], model_name)
        )
        probe_results, calibration = analyze_model_probes(
            states,
            quantities,
            empty_mask,
            split_names,
            snapshots,
            model_name,
        )
        metrics["probes"].extend(probe_results)
        calibrations[model_name] = calibration

        temporal_datasets[model_name] = {}
        temporal_caches[model_name] = {}
        for mode in ("current_constraint", "future_improvement"):
            datasets = build_temporal_probe_datasets(
                states,
                logits,
                quantities,
                empty_mask,
                split_names,
                snapshots,
                mode,
            )
            probe, cache = fit_residual_temporal_probe(
                datasets,
                control_key=(model_name, mode),
            )
            temporal_datasets[model_name][mode] = datasets
            temporal_caches[model_name][mode] = cache
            metrics["residual_probes"].append({
                "model": model_name,
                "mode": mode,
                "target": (
                    "expected_unique_conflicts"
                    if mode == "current_constraint"
                    else "next_snapshot_expected_conflict_reduction"
                ),
                "nuisance": (
                    "exact sampled iteration, maximum probability, top-two "
                    "probability margin, and entropy"
                    + (
                        ", plus current expected/hard conflicts, candidate "
                        "deviation, and correctness"
                        if mode == "future_improvement" else ""
                    )
                ),
                **probe,
            })

    for source_model in payload["models"]:
        for target_model in payload["models"]:
            for mode in ("current_constraint", "future_improvement"):
                transfer = evaluate_cross_checkpoint_probe(
                    temporal_caches[source_model][mode],
                    temporal_datasets[target_model][mode]["final"],
                )
                metrics["cross_checkpoint_transfer"].append({
                    "source_model": source_model,
                    "target_model": target_model,
                    "mode": mode,
                    **transfer,
                })

    metrics = _json_ready(metrics)
    with open(output_dir / "metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")

    dynamics_rows = []
    for model_name, model_dynamics in metrics["dynamics"].items():
        for split_name, records in model_dynamics.items():
            for record in records:
                dynamics_rows.append({
                    "model": model_name,
                    "split": split_name,
                    **record,
                })
    _write_csv(
        output_dir / "dynamics.csv",
        dynamics_rows,
        (
            "model", "split", "snapshot", "expected_row_conflicts",
            "expected_column_conflicts", "expected_box_conflicts",
            "expected_unique_conflicts", "hard_unique_conflicts",
            "joint_candidate_deviation", "target_candidate_legal_fraction",
            "prediction_accuracy", "valid_board_fraction",
            "solved_board_fraction",
        ),
    )

    probe_rows = []
    for record in metrics["probes"]:
        row = {
            "model": record["model"],
            "snapshot": record["snapshot"],
            "target": record["target"],
            "probe_type": record["probe_type"],
            "selected_alpha": record["selected_alpha"],
        }
        if record["probe_type"] == "numeric":
            row.update({
                "validation_score": record["validation"]["r2"],
                "final_score": record["final"]["r2"],
                "final_secondary_score": record["final"]["mae_improvement"],
                "label_shuffle_mean": record["label_shuffle"]["mean_r2"],
                "label_shuffle_max": record["label_shuffle"]["maximum_r2"],
                "random_subspace_rank": record[
                    "matched_rank_random_subspace"
                ]["rank"],
                "random_subspace_mean": record[
                    "matched_rank_random_subspace"
                ]["mean_r2"],
                "random_subspace_max": record[
                    "matched_rank_random_subspace"
                ]["maximum_r2"],
            })
        else:
            row.update({
                "validation_score": record["validation"][
                    "balanced_accuracy"
                ],
                "final_score": record["final"]["balanced_accuracy"],
                "final_secondary_score": record["final"]["accuracy"],
                "label_shuffle_mean": record["label_shuffle"][
                    "mean_balanced_accuracy"
                ],
                "label_shuffle_max": record["label_shuffle"][
                    "maximum_balanced_accuracy"
                ],
                "random_subspace_rank": record[
                    "matched_rank_random_subspace"
                ]["rank"],
                "random_subspace_mean": record[
                    "matched_rank_random_subspace"
                ]["mean_balanced_accuracy"],
                "random_subspace_max": record[
                    "matched_rank_random_subspace"
                ]["maximum_balanced_accuracy"],
            })
        probe_rows.append(row)
    _write_csv(
        output_dir / "probe_metrics.csv",
        probe_rows,
        (
            "model", "snapshot", "target", "probe_type", "selected_alpha",
            "validation_score", "final_score", "final_secondary_score",
            "label_shuffle_mean", "label_shuffle_max",
            "random_subspace_rank", "random_subspace_mean",
            "random_subspace_max",
        ),
    )
    _write_csv(
        output_dir / "trend_controls.csv",
        metrics["trend_controls"],
        (
            "model", "metric", "desired_direction", "ordered_spearman",
            "shuffle_repeats", "shuffle_mean", "shuffle_standard_deviation",
            "one_sided_p",
        ),
    )
    _write_csv(
        output_dir / "paired_changes.csv",
        metrics["paired_changes"],
        (
            "model", "metric", "start", "end",
            "positive_means_improvement", "mean", "ci95_low", "ci95_high",
            "puzzles",
        ),
    )
    residual_rows = []
    for record in metrics["residual_probes"]:
        residual_rows.append({
            "model": record["model"],
            "mode": record["mode"],
            "target": record["target"],
            "selected_base_alpha": record["selected_base_alpha"],
            "selected_state_alpha": record["selected_state_alpha"],
            "baseline_r2": record["final"]["baseline_r2"],
            "full_r2": record["final"]["full_r2"],
            "state_partial_r2": record["final"]["state_partial_r2"],
            "label_shuffle_max": record["label_shuffle"]["maximum_partial_r2"],
            "random_rank1_max": record[
                "matched_rank_random_subspace"
            ]["maximum_partial_r2"],
            "iteration_shuffle_max": record[
                "shuffled_iteration"
            ]["maximum_partial_r2"],
        })
    _write_csv(
        output_dir / "residual_probes.csv",
        residual_rows,
        (
            "model", "mode", "target", "selected_base_alpha",
            "selected_state_alpha", "baseline_r2", "full_r2",
            "state_partial_r2", "label_shuffle_max", "random_rank1_max",
            "iteration_shuffle_max",
        ),
    )
    _write_csv(
        output_dir / "cross_checkpoint_transfer.csv",
        metrics["cross_checkpoint_transfer"],
        (
            "source_model", "target_model", "mode", "baseline_r2",
            "full_r2", "state_partial_r2", "baseline_sse", "full_sse",
        ),
    )

    artifact_names = (
        "constraint_dynamics.png",
        "constraint_components.png",
        "numeric_probe_transfer.png",
        "categorical_probe_transfer.png",
        "probe_calibration_16.png",
        "future_improvement_prediction.png",
        "cross_checkpoint_transfer.png",
    )
    plot_dynamics(
        metrics["dynamics"],
        snapshots,
        output_dir / artifact_names[0],
    )
    plot_constraint_components(
        metrics["dynamics"],
        snapshots,
        output_dir / artifact_names[1],
    )
    plot_numeric_probes(metrics["probes"], output_dir / artifact_names[2])
    plot_categorical_probes(metrics["probes"], output_dir / artifact_names[3])
    plot_calibration(calibrations, output_dir / artifact_names[4])
    plot_residual_probes(
        metrics["residual_probes"], output_dir / artifact_names[5]
    )
    plot_cross_checkpoint_transfer(
        metrics["cross_checkpoint_transfer"], output_dir / artifact_names[6]
    )
    write_html(output_dir / "index.html", metrics, artifact_names)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("payload_path")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent))
    arguments = parser.parse_args()
    analyze(arguments.payload_path, arguments.output_dir)


if __name__ == "__main__":
    main()
