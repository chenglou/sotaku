"""Statistical tests for digit geometry in recurrent Sudoku representations.

The response is an ``[observation, feature]`` matrix, normally a flattened
collection of hidden states or updates.  All splits keep complete puzzles in
one fold.  Digit models are compared with the same nuisance (``base_design``)
model on observations from held-out puzzles.

The four digit designs are fixed a priori:

* ``linear``: the values 1,...,9, centered at 5 and scaled to unit population
  variance over the nine values;
* ``cyclic``: cosine and sine at period 9;
* ``helix``: the union of the linear and cyclic columns;
* ``categorical``: eight treatment columns with digit 9 as the reference.

No held-out partial R2 is clipped.  A negative value therefore records a
genuine out-of-sample loss rather than being silently changed to zero.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


DIGIT_MODEL_NAMES = ("linear", "cyclic", "helix", "categorical")
_LINEAR_SCALE = math.sqrt(20.0 / 3.0)
DEFAULT_OLS_RCOND = 1e-8


def _as_1d(values: Any, name: str, length: int | None = None) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if length is not None and len(array) != length:
        raise ValueError(f"{name} has length {len(array)}, expected {length}")
    return array


def _as_float_matrix(values: Any, name: str, rows: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix, got shape {array.shape}")
    if rows is not None and len(array) != rows:
        raise ValueError(f"{name} has {len(array)} rows, expected {rows}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _stable_unique(values: Sequence[Any]) -> tuple[list[Any], np.ndarray]:
    """Return first-occurrence-ordered values and their integer inverse."""
    unique: list[Any] = []
    index_by_value: dict[Any, int] = {}
    inverse = np.empty(len(values), dtype=np.int64)
    for row, raw_value in enumerate(values):
        value = _python_scalar(raw_value)
        try:
            index = index_by_value.get(value)
        except TypeError as error:
            raise ValueError("puzzle IDs and strata must be hashable") from error
        if index is None:
            index = len(unique)
            unique.append(value)
            index_by_value[value] = index
        inverse[row] = index
    return unique, inverse


def _digit_indices(digits: Any, digit_offset: int) -> np.ndarray:
    raw = _as_1d(digits, "digits")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError("digits must be numeric integer labels")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.round(numeric)):
        raise ValueError("digits must contain finite integers")
    indices = numeric.astype(np.int64) - int(digit_offset)
    if np.any((indices < 0) | (indices >= 9)):
        low = int(indices.min()) if len(indices) else None
        high = int(indices.max()) if len(indices) else None
        raise ValueError(
            f"digits minus digit_offset must be in 0..8, got range {low}..{high}"
        )
    return indices


def digit_code_matrix(
    digits: Any,
    model: str,
    *,
    digit_offset: int = 0,
) -> np.ndarray:
    """Construct one of the prespecified digit-code matrices.

    ``digit_offset=0`` is appropriate for model targets 0,...,8.  Pass
    ``digit_offset=1`` when the input contains the human-readable digits
    1,...,9.  The resulting matrices are identical for equivalent labels.
    """
    indices = _digit_indices(digits, digit_offset)
    digit_values = indices.astype(np.float64) + 1.0
    linear = ((digit_values - 5.0) / _LINEAR_SCALE)[:, None]
    angles = 2.0 * np.pi * indices.astype(np.float64) / 9.0
    cyclic = np.column_stack((np.cos(angles), np.sin(angles)))
    if model == "linear":
        return linear
    if model == "cyclic":
        return cyclic
    if model == "helix":
        return np.column_stack((linear, cyclic))
    if model == "categorical":
        # Digit 9 (index 8) is the reference.  With an intercept these eight
        # columns span all digit-specific means without a redundant column.
        return np.equal(indices[:, None], np.arange(8)[None, :]).astype(np.float64)
    raise ValueError(f"unknown digit model {model!r}; expected one of {DIGIT_MODEL_NAMES}")


@dataclass(frozen=True)
class WeightedOLSFit:
    """A weighted multivariate ordinary least-squares fit."""

    coefficients: np.ndarray
    rank: int
    singular_values: np.ndarray

    def predict(self, design: Any) -> np.ndarray:
        matrix = _as_float_matrix(design, "design")
        if matrix.shape[1] != self.coefficients.shape[0]:
            raise ValueError(
                f"design has {matrix.shape[1]} columns, expected "
                f"{self.coefficients.shape[0]}"
            )
        # Apple Accelerate can leave floating-point status flags set after the
        # preceding LAPACK solve, which makes a later, otherwise finite matmul
        # emit spurious overflow/divide warnings.  Suppress those flags here,
        # then validate the actual prediction instead of hiding a real failure.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            prediction = matrix @ self.coefficients
        if not np.all(np.isfinite(prediction)):
            raise FloatingPointError("weighted OLS prediction is non-finite")
        return prediction


def fit_weighted_multivariate_ols(
    design: Any,
    response: Any,
    sample_weight: Any | None = None,
    *,
    rcond: float | None = DEFAULT_OLS_RCOND,
) -> WeightedOLSFit:
    """Fit multivariate OLS after multiplying rows by square-root weights.

    The conservative default rank threshold removes numerically redundant
    nuisance columns, e.g. a nearly constant confidence term alongside an
    intercept after a model has converged.
    """
    design_matrix = _as_float_matrix(design, "design")
    response_matrix = _as_float_matrix(response, "response", len(design_matrix))
    if sample_weight is None:
        weights = np.ones(len(design_matrix), dtype=np.float64)
    else:
        weights = np.asarray(
            _as_1d(sample_weight, "sample_weight", len(design_matrix)),
            dtype=np.float64,
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must contain finite, nonnegative values")
    if not np.any(weights > 0):
        raise ValueError("sample_weight must contain at least one positive value")
    square_root_weight = np.sqrt(weights)[:, None]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design_matrix * square_root_weight,
        response_matrix * square_root_weight,
        rcond=rcond,
    )
    if not np.all(np.isfinite(coefficients)) or not np.all(
        np.isfinite(singular_values)
    ):
        raise FloatingPointError("weighted OLS fit is non-finite")
    return WeightedOLSFit(
        coefficients=coefficients,
        rank=int(rank),
        singular_values=singular_values,
    )


def weighted_sse(response: Any, prediction: Any, sample_weight: Any | None = None) -> float:
    response_matrix = _as_float_matrix(response, "response")
    prediction_matrix = _as_float_matrix(prediction, "prediction", len(response_matrix))
    if prediction_matrix.shape != response_matrix.shape:
        raise ValueError(
            f"prediction shape {prediction_matrix.shape} does not match response "
            f"shape {response_matrix.shape}"
        )
    if sample_weight is None:
        weights = np.ones(len(response_matrix), dtype=np.float64)
    else:
        weights = np.asarray(
            _as_1d(sample_weight, "sample_weight", len(response_matrix)),
            dtype=np.float64,
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must contain finite, nonnegative values")
    return float(np.sum(weights[:, None] * np.square(response_matrix - prediction_matrix)))


def _puzzle_level_values(
    puzzle_ids: np.ndarray,
    values: Any | None,
    name: str,
    *,
    default: Any,
) -> tuple[list[Any], np.ndarray, list[Any]]:
    unique_puzzles, inverse = _stable_unique(puzzle_ids)
    if values is None:
        return unique_puzzles, inverse, [default for _ in unique_puzzles]
    if isinstance(values, Mapping):
        missing = [puzzle for puzzle in unique_puzzles if puzzle not in values]
        if missing:
            raise ValueError(f"{name} is missing puzzle IDs, including {missing[0]!r}")
        return unique_puzzles, inverse, [_python_scalar(values[p]) for p in unique_puzzles]
    value_array = _as_1d(values, name)
    if len(value_array) == len(puzzle_ids):
        per_puzzle: list[Any] = [None] * len(unique_puzzles)
        seen = np.zeros(len(unique_puzzles), dtype=bool)
        for row, puzzle_index in enumerate(inverse):
            value = _python_scalar(value_array[row])
            if seen[puzzle_index] and value != per_puzzle[puzzle_index]:
                raise ValueError(
                    f"{name} must be constant within puzzle {unique_puzzles[puzzle_index]!r}"
                )
            per_puzzle[puzzle_index] = value
            seen[puzzle_index] = True
        return unique_puzzles, inverse, per_puzzle
    if len(value_array) == len(unique_puzzles):
        return unique_puzzles, inverse, [_python_scalar(value) for value in value_array]
    raise ValueError(
        f"{name} has length {len(value_array)}; expected one value per row "
        f"({len(puzzle_ids)}) or puzzle ({len(unique_puzzles)})"
    )


def make_stratified_puzzle_folds(
    puzzle_ids: Any,
    strata: Any | None = None,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> np.ndarray:
    """Assign rows to balanced folds while keeping each puzzle intact.

    The returned array is row-aligned.  Within each stratum, puzzle counts can
    differ across folds by at most one; ties are resolved to keep total fold
    sizes balanced and then randomly using ``seed``.
    """
    puzzle_array = _as_1d(puzzle_ids, "puzzle_ids")
    unique_puzzles, inverse, puzzle_strata = _puzzle_level_values(
        puzzle_array, strata, "strata", default="__all__"
    )
    if not isinstance(n_folds, (int, np.integer)) or n_folds < 2:
        raise ValueError("n_folds must be an integer of at least 2")
    if len(unique_puzzles) < n_folds:
        raise ValueError(
            f"need at least n_folds={n_folds} puzzles, got {len(unique_puzzles)}"
        )

    stratum_names, stratum_inverse = _stable_unique(puzzle_strata)
    generator = np.random.default_rng(seed)
    fold_total = np.zeros(n_folds, dtype=np.int64)
    fold_by_puzzle = np.full(len(unique_puzzles), -1, dtype=np.int64)
    for stratum_index, _ in enumerate(stratum_names):
        members = np.flatnonzero(stratum_inverse == stratum_index)
        members = generator.permutation(members)
        stratum_counts = np.zeros(n_folds, dtype=np.int64)
        tie_order = generator.permutation(n_folds)
        tie_rank = np.empty(n_folds, dtype=np.int64)
        tie_rank[tie_order] = np.arange(n_folds)
        for puzzle_index in members:
            eligible = np.flatnonzero(stratum_counts == stratum_counts.min())
            totals = fold_total[eligible]
            eligible = eligible[totals == totals.min()]
            chosen = int(eligible[np.argmin(tie_rank[eligible])])
            fold_by_puzzle[puzzle_index] = chosen
            fold_total[chosen] += 1
            stratum_counts[chosen] += 1

    if np.any(fold_total == 0):
        # This should only be reachable for adversarial tie patterns.  Keeping
        # it explicit is safer than returning a fold with no held-out puzzle.
        raise RuntimeError(f"internal fold allocation failure: sizes={fold_total.tolist()}")
    return fold_by_puzzle[inverse]


def _validated_fold_ids(
    puzzle_ids: np.ndarray,
    fold_ids: Any,
) -> tuple[np.ndarray, list[Any], np.ndarray, np.ndarray]:
    unique_puzzles, inverse, puzzle_folds = _puzzle_level_values(
        puzzle_ids, fold_ids, "fold_ids", default=None
    )
    try:
        puzzle_fold_array = np.asarray(puzzle_folds, dtype=np.int64)
    except (TypeError, ValueError) as error:
        raise ValueError("fold_ids must be integer-valued") from error
    if any(raw != int(encoded) for raw, encoded in zip(puzzle_folds, puzzle_fold_array)):
        raise ValueError("fold_ids must be integer-valued")
    observed = np.unique(puzzle_fold_array)
    expected = np.arange(len(observed), dtype=np.int64)
    if not np.array_equal(observed, expected) or len(observed) < 2:
        raise ValueError(
            "fold_ids must contain contiguous values 0,...,n_folds-1 and at least two folds"
        )
    return puzzle_fold_array[inverse], unique_puzzles, inverse, puzzle_fold_array


def _safe_partial_r2(full_sse: float, reduced_sse: float) -> float:
    if not np.isfinite(full_sse) or not np.isfinite(reduced_sse) or reduced_sse <= 0:
        return math.nan
    return 1.0 - full_sse / reduced_sse


def _gain_fraction(model_sse: float, categorical_sse: float, base_sse: float) -> float:
    categorical_gain = base_sse - categorical_sse
    tolerance = np.finfo(np.float64).eps * max(1.0, abs(base_sse), abs(categorical_sse))
    if categorical_gain <= tolerance:
        return math.nan
    return (base_sse - model_sse) / categorical_gain


def _axis_descriptors(helix_coefficients: np.ndarray) -> dict[str, float | int]:
    if helix_coefficients.ndim != 2 or helix_coefficients.shape[0] != 3:
        raise ValueError("helix coefficient block must have shape [3, response_dimension]")
    linear_axis, cosine_axis, sine_axis = helix_coefficients
    cyclic_block = np.stack((cosine_axis, sine_axis), axis=0)
    singular_values = np.linalg.svd(cyclic_block, compute_uv=False)
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    smallest = float(singular_values[1]) if len(singular_values) > 1 else 0.0
    cyclic_ratio = smallest / largest if largest > 0 else math.nan
    cosine_norm = float(np.linalg.norm(cosine_axis))
    sine_norm = float(np.linalg.norm(sine_axis))
    if cosine_norm > 0 and sine_norm > 0:
        axis_cosine = abs(float(np.dot(cosine_axis, sine_axis) / (cosine_norm * sine_norm)))
    else:
        axis_cosine = math.nan

    linear_norm = float(np.linalg.norm(linear_axis))
    if linear_norm > 0 and largest > 0:
        _, singular, right_vectors = np.linalg.svd(cyclic_block, full_matrices=False)
        tolerance = np.finfo(np.float64).eps * max(cyclic_block.shape) * singular[0]
        basis = right_vectors[singular > tolerance].T
        projected = basis @ (basis.T @ linear_axis) if basis.size else np.zeros_like(linear_axis)
        # An energy fraction is comparable across response representations and
        # matches the corresponding learned-parameter descriptor.
        leakage = float(np.dot(projected, projected) / (linear_norm * linear_norm))
    else:
        leakage = math.nan
    return {
        "cyclic_singular_value_ratio": cyclic_ratio,
        "cyclic_axis_absolute_cosine": axis_cosine,
        "linear_axis_leakage_into_cyclic_span": leakage,
        "response_dimension": int(helix_coefficients.shape[1]),
    }


def _row_space_basis(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float64)
    _, singular_values, right_vectors = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular_values) or singular_values[0] == 0:
        return np.zeros((matrix.shape[1], 0), dtype=np.float64)
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    return right_vectors[singular_values > tolerance].T


def _subspace_stability(blocks: Sequence[np.ndarray]) -> dict[str, Any]:
    bases = [_row_space_basis(block) for block in blocks]
    pairwise: list[dict[str, Any]] = []
    scores: list[float] = []
    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            left_basis, right_basis = bases[left], bases[right]
            denominator = max(left_basis.shape[1], right_basis.shape[1])
            if denominator == 0:
                score = math.nan
            else:
                # Mean squared cosine of principal angles, penalizing a rank
                # mismatch through the larger subspace dimension.
                score = float(np.square(left_basis.T @ right_basis).sum() / denominator)
            pairwise.append({"fold_a": left, "fold_b": right, "overlap": score})
            if np.isfinite(score):
                scores.append(score)
    return {
        "definition": "mean squared principal-angle cosine; rank mismatch penalized",
        "mean": float(np.mean(scores)) if scores else math.nan,
        "minimum": float(np.min(scores)) if scores else math.nan,
        "maximum": float(np.max(scores)) if scores else math.nan,
        "pairwise": pairwise,
    }


def _metric_summary(sse_by_model: Mapping[str, float]) -> dict[str, Any]:
    base_sse = float(sse_by_model["base"])
    null_sse = float(sse_by_model["null"])
    partial: dict[str, float] = {}
    for model in DIGIT_MODEL_NAMES:
        if model in sse_by_model:
            partial[f"{model}_over_base"] = _safe_partial_r2(
                float(sse_by_model[model]), base_sse
            )
    if "linear" in sse_by_model and "helix" in sse_by_model:
        partial["helix_over_linear"] = _safe_partial_r2(
            float(sse_by_model["helix"]), float(sse_by_model["linear"])
        )
    if "cyclic" in sse_by_model and "helix" in sse_by_model:
        partial["helix_over_cyclic"] = _safe_partial_r2(
            float(sse_by_model["helix"]), float(sse_by_model["cyclic"])
        )
    if "helix" in sse_by_model and "categorical" in sse_by_model:
        partial["categorical_over_helix"] = _safe_partial_r2(
            float(sse_by_model["categorical"]), float(sse_by_model["helix"])
        )
    result: dict[str, Any] = {
        "partial_r2": partial,
        "r2_over_null": {
            name: _safe_partial_r2(float(sse), null_sse)
            for name, sse in sse_by_model.items()
            if name != "null"
        },
    }
    if "helix" in sse_by_model and "categorical" in sse_by_model:
        result["helix_categorical_gain_fraction"] = _gain_fraction(
            float(sse_by_model["helix"]),
            float(sse_by_model["categorical"]),
            base_sse,
        )
        result["heldout_gain"] = {
            "helix": base_sse - float(sse_by_model["helix"]),
            "categorical": base_sse - float(sse_by_model["categorical"]),
        }
    return result


def _prepare_inputs(
    response: Any,
    digits: Any,
    puzzle_ids: Any,
    base_design: Any | None,
    sample_weight: Any | None,
    *,
    add_intercept: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    response_matrix = _as_float_matrix(response, "response")
    row_count = len(response_matrix)
    digit_array = _as_1d(digits, "digits", row_count)
    puzzle_array = _as_1d(puzzle_ids, "puzzle_ids", row_count)
    if base_design is None:
        supplied_base = np.empty((row_count, 0), dtype=np.float64)
    else:
        supplied_base = _as_float_matrix(base_design, "base_design", row_count)
    if add_intercept:
        base_matrix = np.column_stack((np.ones(row_count), supplied_base))
    else:
        base_matrix = supplied_base
    if base_matrix.shape[1] == 0:
        raise ValueError("the base design is empty; enable add_intercept or supply columns")
    if sample_weight is None:
        weights = np.ones(row_count, dtype=np.float64)
    else:
        weights = np.asarray(
            _as_1d(sample_weight, "sample_weight", row_count), dtype=np.float64
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weight must contain finite, nonnegative values")
    if not np.any(weights > 0):
        raise ValueError("sample_weight must contain at least one positive value")
    return response_matrix, digit_array, puzzle_array, base_matrix, weights


def _cross_validate_core(
    response: np.ndarray,
    digits: np.ndarray,
    puzzle_ids: np.ndarray,
    base_design: np.ndarray,
    sample_weight: np.ndarray,
    fold_ids: np.ndarray,
    *,
    digit_offset: int,
    models: Sequence[str],
    strata: Any | None,
    collect_coefficients: bool,
) -> dict[str, Any]:
    requested_models = tuple(dict.fromkeys(models))
    invalid = [model for model in requested_models if model not in DIGIT_MODEL_NAMES]
    if invalid:
        raise ValueError(f"unknown digit models: {invalid}")
    designs = {
        model: np.column_stack(
            (base_design, digit_code_matrix(digits, model, digit_offset=digit_offset))
        )
        for model in requested_models
    }
    null_design = np.ones((len(response), 1), dtype=np.float64)
    model_names = ("null", "base") + requested_models
    sse_by_model = {name: 0.0 for name in model_names}
    folds: list[dict[str, Any]] = []
    fold_helix_blocks: list[np.ndarray] = []

    unique_puzzles, puzzle_inverse, puzzle_strata = _puzzle_level_values(
        puzzle_ids, strata, "strata", default="__all__"
    )
    _, _, _, puzzle_fold_ids = _validated_fold_ids(puzzle_ids, fold_ids)
    puzzle_sse = {
        name: np.zeros(len(unique_puzzles), dtype=np.float64) for name in model_names
    }
    puzzle_weight = np.zeros(len(unique_puzzles), dtype=np.float64)
    puzzle_rows = np.zeros(len(unique_puzzles), dtype=np.int64)
    for puzzle_index in range(len(unique_puzzles)):
        chosen = puzzle_inverse == puzzle_index
        puzzle_weight[puzzle_index] = sample_weight[chosen].sum()
        puzzle_rows[puzzle_index] = int(chosen.sum())

    for fold in range(int(fold_ids.max()) + 1):
        test = fold_ids == fold
        train = ~test
        if not np.any(test) or not np.any(train):
            raise ValueError(f"fold {fold} has an empty train or test set")
        if sample_weight[train].sum() <= 0 or sample_weight[test].sum() <= 0:
            raise ValueError(f"fold {fold} has no positive train or test weight")
        fold_result: dict[str, Any] = {
            "fold": fold,
            "train_puzzles": int(len(np.unique(puzzle_inverse[train]))),
            "test_puzzles": int(len(np.unique(puzzle_inverse[test]))),
            "train_rows": int(train.sum()),
            "test_rows": int(test.sum()),
            "test_weight": float(sample_weight[test].sum()),
            "models": {},
        }
        fit_designs = {"null": null_design, "base": base_design, **designs}
        for model_name in model_names:
            design = fit_designs[model_name]
            fit = fit_weighted_multivariate_ols(
                design[train], response[train], sample_weight[train]
            )
            prediction = fit.predict(design[test])
            squared_error_by_row = np.square(response[test] - prediction).sum(axis=1)
            fold_sse = float(np.dot(sample_weight[test], squared_error_by_row))
            sse_by_model[model_name] += fold_sse
            fold_model: dict[str, Any] = {
                "sse": fold_sse,
                "rank": fit.rank,
                "design_columns": int(design.shape[1]),
            }
            if model_name == "helix" and collect_coefficients:
                block = fit.coefficients[-3:].copy()
                fold_helix_blocks.append(block)
                fold_model["coefficient_descriptors"] = _axis_descriptors(block)
            fold_result["models"][model_name] = fold_model
            test_rows = np.flatnonzero(test)
            np.add.at(
                puzzle_sse[model_name],
                puzzle_inverse[test_rows],
                sample_weight[test_rows] * squared_error_by_row,
            )
        fold_result.update(_metric_summary({
            name: fold_result["models"][name]["sse"] for name in model_names
        }))
        folds.append(fold_result)

    per_puzzle = []
    for puzzle_index, puzzle in enumerate(unique_puzzles):
        per_puzzle.append({
            "puzzle_id": _python_scalar(puzzle),
            "stratum": _python_scalar(puzzle_strata[puzzle_index]),
            "fold": int(puzzle_fold_ids[puzzle_index]),
            "rows": int(puzzle_rows[puzzle_index]),
            "weight": float(puzzle_weight[puzzle_index]),
            "sse": {
                name: float(puzzle_sse[name][puzzle_index]) for name in model_names
            },
        })

    result: dict[str, Any] = {
        "models": {
            name: {"sse": float(sse_by_model[name])} for name in model_names
        },
        "folds": folds,
        "per_puzzle": per_puzzle,
        **_metric_summary(sse_by_model),
    }
    if fold_helix_blocks:
        result["coefficient_geometry"] = {
            "fold_descriptors": [
                {"fold": fold, **_axis_descriptors(block)}
                for fold, block in enumerate(fold_helix_blocks)
            ],
            "cyclic_subspace_stability": _subspace_stability(
                [block[1:] for block in fold_helix_blocks]
            ),
            "helix_subspace_stability": _subspace_stability(fold_helix_blocks),
        }
    return result


def bootstrap_geometry_intervals(
    per_puzzle: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Cluster bootstrap held-out SSE metrics, stratified by puzzle stratum.

    The fitted fold models stay fixed.  Each replicate samples complete puzzle
    residual blocks with replacement within each stratum.  This estimates
    uncertainty over puzzles without pretending that cells or iterations are
    independent observations.
    """
    if not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")
    if not per_puzzle:
        raise ValueError("per_puzzle must not be empty")
    model_names = tuple(per_puzzle[0]["sse"].keys())
    if "base" not in model_names or "null" not in model_names:
        raise ValueError("per_puzzle SSE must include null and base models")
    for record in per_puzzle:
        if set(record["sse"].keys()) != set(model_names):
            raise ValueError("all per-puzzle records must contain the same SSE models")

    stratum_values = [record.get("stratum", "__all__") for record in per_puzzle]
    _, stratum_inverse = _stable_unique(stratum_values)
    groups = [
        np.flatnonzero(stratum_inverse == stratum_index)
        for stratum_index in range(int(stratum_inverse.max()) + 1)
    ]
    sse_matrix = np.asarray(
        [[float(record["sse"][name]) for name in model_names] for record in per_puzzle],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(sse_matrix)) or np.any(sse_matrix < 0):
        raise ValueError("per-puzzle SSE values must be finite and nonnegative")
    generator = np.random.default_rng(seed)
    sampled_metrics: dict[str, list[float]] = {}
    for _ in range(replicates):
        sampled_indices = np.concatenate([
            generator.choice(group, size=len(group), replace=True) for group in groups
        ])
        totals = sse_matrix[sampled_indices].sum(axis=0)
        summary = _metric_summary(dict(zip(model_names, totals)))
        for name, value in summary["partial_r2"].items():
            sampled_metrics.setdefault(f"partial_r2.{name}", []).append(value)
        if "helix_categorical_gain_fraction" in summary:
            sampled_metrics.setdefault("helix_categorical_gain_fraction", []).append(
                summary["helix_categorical_gain_fraction"]
            )

    alpha = (1.0 - confidence_level) / 2.0
    intervals: dict[str, Any] = {}
    for name, values in sampled_metrics.items():
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            intervals[name] = {
                "lower": math.nan,
                "median": math.nan,
                "upper": math.nan,
                "finite_replicates": 0,
            }
        else:
            lower, median, upper = np.quantile(finite, [alpha, 0.5, 1.0 - alpha])
            intervals[name] = {
                "lower": float(lower),
                "median": float(median),
                "upper": float(upper),
                "finite_replicates": int(len(finite)),
            }
    return {
        "method": (
            "conditional puzzle-residual percentile bootstrap stratified by "
            "puzzle stratum; held-out probe fits and folds remain fixed"
        ),
        "inference_note": (
            "Descriptive conditional interval only: overlapping cross-validation "
            "training sets make puzzle residual blocks dependent, and probes are "
            "not refitted."
        ),
        "replicates": int(replicates),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
        "intervals": intervals,
    }


def evaluate_digit_geometry(
    response: Any,
    digits: Any,
    puzzle_ids: Any,
    *,
    base_design: Any | None = None,
    sample_weight: Any | None = None,
    strata: Any | None = None,
    n_folds: int = 5,
    fold_ids: Any | None = None,
    seed: int = 0,
    digit_offset: int = 0,
    models: Sequence[str] = DIGIT_MODEL_NAMES,
    add_intercept: bool = True,
    bootstrap_replicates: int = 0,
    bootstrap_confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Fit and evaluate the nested digit-geometry models on held-out puzzles."""
    response_matrix, digit_array, puzzle_array, base_matrix, weights = _prepare_inputs(
        response, digits, puzzle_ids, base_design, sample_weight,
        add_intercept=add_intercept,
    )
    # Validate labels even when an empty model list was requested.
    _digit_indices(digit_array, digit_offset)
    if fold_ids is None:
        row_folds = make_stratified_puzzle_folds(
            puzzle_array, strata, n_folds=n_folds, seed=seed
        )
    else:
        row_folds, _, _, _ = _validated_fold_ids(puzzle_array, fold_ids)
    result = _cross_validate_core(
        response_matrix,
        digit_array,
        puzzle_array,
        base_matrix,
        weights,
        row_folds,
        digit_offset=digit_offset,
        models=models,
        strata=strata,
        collect_coefficients=True,
    )
    result["config"] = {
        "n_rows": int(len(response_matrix)),
        "response_dimension": int(response_matrix.shape[1]),
        "n_puzzles": int(len(_stable_unique(puzzle_array)[0])),
        "n_folds": int(row_folds.max()) + 1,
        "seed": int(seed),
        "digit_offset": int(digit_offset),
        "models": list(dict.fromkeys(models)),
        "base_columns_including_intercept": int(base_matrix.shape[1]),
        "ols_relative_singular_value_cutoff": DEFAULT_OLS_RCOND,
        "weighted": sample_weight is not None,
    }
    if bootstrap_replicates:
        result["bootstrap"] = bootstrap_geometry_intervals(
            result["per_puzzle"],
            replicates=bootstrap_replicates,
            confidence_level=bootstrap_confidence_level,
            seed=seed + 1,
        )
    return result


def _permutation_statistic(result: Mapping[str, Any], statistic: str) -> float:
    if statistic == "helix_partial_r2":
        return float(result["partial_r2"]["helix_over_base"])
    if statistic == "cyclic_partial_r2":
        return float(result["partial_r2"]["cyclic_over_base"])
    if statistic == "linear_partial_r2":
        return float(result["partial_r2"]["linear_over_base"])
    if statistic == "cyclic_given_linear_partial_r2":
        return float(result["partial_r2"]["helix_over_linear"])
    if statistic == "linear_given_cyclic_partial_r2":
        return float(result["partial_r2"]["helix_over_cyclic"])
    if statistic == "helix_categorical_gain_fraction":
        return float(result["helix_categorical_gain_fraction"])
    raise ValueError(
        "statistic must be helix_partial_r2, cyclic_partial_r2, "
        "linear_partial_r2, cyclic_given_linear_partial_r2, "
        "linear_given_cyclic_partial_r2, or helix_categorical_gain_fraction"
    )


@dataclass(frozen=True)
class _FWLFoldStatistics:
    """Nine-group sufficient statistics for one train/test puzzle fold."""

    train_group_cross: np.ndarray
    train_group_response: np.ndarray
    test_group_cross: np.ndarray
    test_group_response: np.ndarray
    test_base_sse: float


def _prepare_fwl_fold_statistics(
    response: np.ndarray,
    digit_indices: np.ndarray,
    base_design: np.ndarray,
    sample_weight: np.ndarray,
    fold_ids: np.ndarray,
) -> list[_FWLFoldStatistics]:
    """Compress every fold to cross-products of nine residualized groups.

    Let G be the nine-column digit indicator matrix and B the nuisance design.
    On each training split, both G and Y are residualized against B.  A digit
    code C then has residualized design ``G_residual @ C``.  Its train normal
    equations and held-out SSE depend only on 9x9, 9xresponse_dimension, and
    scalar sufficient statistics.  The original rows never need to be touched
    again during the permutation loop.
    """
    groups = np.equal(
        digit_indices[:, None], np.arange(9, dtype=np.int64)[None, :]
    ).astype(np.float64)
    fold_statistics: list[_FWLFoldStatistics] = []
    for fold in range(int(fold_ids.max()) + 1):
        test = fold_ids == fold
        train = ~test
        combined_response = np.column_stack((response[train], groups[train]))
        base_fit = fit_weighted_multivariate_ols(
            base_design[train], combined_response, sample_weight[train]
        )
        response_dimension = response.shape[1]
        response_coefficients = base_fit.coefficients[:, :response_dimension]
        group_coefficients = base_fit.coefficients[:, response_dimension:]

        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            train_response_residual = (
                response[train] - base_design[train] @ response_coefficients
            )
            train_group_residual = (
                groups[train] - base_design[train] @ group_coefficients
            )
        train_weight = sample_weight[train, None]
        # Some Accelerate-backed NumPy builds leave stale floating-point flags
        # after LAPACK calls and emit spurious matmul warnings even when every
        # operand and result is small and finite.  Check the resulting
        # sufficient statistics explicitly instead.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            train_group_cross = train_group_residual.T @ (
                train_weight * train_group_residual
            )
            train_group_response = train_group_residual.T @ (
                train_weight * train_response_residual
            )

        # Predictions below use the training projection of each group column.
        # H_test @ C is consequently the test digit design after accounting
        # for the base-coefficient adjustment in the joint [B, G C] fit.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            test_response_residual = (
                response[test] - base_design[test] @ response_coefficients
            )
            test_group_residual = (
                groups[test] - base_design[test] @ group_coefficients
            )
        test_weight = sample_weight[test, None]
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            test_group_cross = test_group_residual.T @ (
                test_weight * test_group_residual
            )
            test_group_response = test_group_residual.T @ (
                test_weight * test_response_residual
            )
        test_base_sse = float(
            np.sum(test_weight * np.square(test_response_residual))
        )
        sufficient_values = (
            train_group_cross,
            train_group_response,
            test_group_cross,
            test_group_response,
            np.asarray(test_base_sse),
        )
        if not all(np.all(np.isfinite(value)) for value in sufficient_values):
            raise FloatingPointError("a fold sufficient statistic is non-finite")
        fold_statistics.append(_FWLFoldStatistics(
            train_group_cross=(train_group_cross + train_group_cross.T) / 2.0,
            train_group_response=train_group_response,
            test_group_cross=(test_group_cross + test_group_cross.T) / 2.0,
            test_group_response=test_group_response,
            test_base_sse=test_base_sse,
        ))
    return fold_statistics


def _solve_small_normal_equations(
    cross_product: np.ndarray,
    cross_response: np.ndarray,
) -> np.ndarray:
    """Solve a 1--8 dimensional positive-semidefinite normal equation."""
    symmetric = (cross_product + cross_product.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = float(np.max(eigenvalues, initial=0.0))
    if largest <= 0:
        return np.zeros_like(cross_response)
    # These are normal equations, so the design-matrix relative singular-value
    # cutoff is squared when applied to Gram-matrix eigenvalues.
    keep = eigenvalues > (DEFAULT_OLS_RCOND ** 2) * largest
    if not np.any(keep):
        return np.zeros_like(cross_response)
    kept_vectors = eigenvectors[:, keep]
    return kept_vectors @ (
        (kept_vectors.T @ cross_response) / eigenvalues[keep, None]
    )


def _digit_model_sse_from_fwl(
    fold_statistics: Sequence[_FWLFoldStatistics],
    code_by_original_digit: np.ndarray,
) -> float:
    code = _as_float_matrix(code_by_original_digit, "code_by_original_digit")
    if code.shape[0] != 9:
        raise ValueError("code_by_original_digit must have nine rows")
    total_sse = 0.0
    for fold in fold_statistics:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            train_cross = code.T @ fold.train_group_cross @ code
            train_response = code.T @ fold.train_group_response
        digit_coefficients = _solve_small_normal_equations(
            train_cross, train_response
        )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            test_cross = code.T @ fold.test_group_cross @ code
            test_response = code.T @ fold.test_group_response
        if not all(np.all(np.isfinite(value)) for value in (
            train_cross, train_response, digit_coefficients,
            test_cross, test_response,
        )):
            raise FloatingPointError("a coded FWL statistic is non-finite")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            linear_term = float(np.sum(digit_coefficients * test_response))
            quadratic_term = float(np.sum(
                digit_coefficients * (test_cross @ digit_coefficients)
            ))
        fold_sse = fold.test_base_sse - 2.0 * linear_term + quadratic_term
        tolerance = np.finfo(np.float64).eps * max(1.0, fold.test_base_sse) * 100
        if fold_sse < -tolerance:
            raise FloatingPointError(
                f"FWL held-out SSE became materially negative ({fold_sse})"
            )
        total_sse += max(0.0, fold_sse)
    return total_sse


def _models_for_statistic(statistic: str) -> tuple[str, ...]:
    if statistic == "helix_categorical_gain_fraction":
        return ("helix",)
    if statistic == "cyclic_given_linear_partial_r2":
        return ("linear", "helix")
    if statistic == "linear_given_cyclic_partial_r2":
        return ("cyclic", "helix")
    suffix = "_partial_r2"
    if statistic.endswith(suffix):
        model = statistic[:-len(suffix)]
        if model in ("linear", "cyclic", "helix"):
            return (model,)
    # Reuse the public error text.
    _permutation_statistic({}, statistic)
    raise AssertionError("unreachable")


def _statistic_from_sse(
    statistic: str,
    model_sse: Mapping[str, float],
    base_sse: float,
    categorical_sse: float | None,
) -> float:
    if statistic == "helix_categorical_gain_fraction":
        if categorical_sse is None:
            raise ValueError("categorical SSE is required for the gain fraction")
        return _gain_fraction(model_sse["helix"], categorical_sse, base_sse)
    if statistic == "cyclic_given_linear_partial_r2":
        return _safe_partial_r2(model_sse["helix"], model_sse["linear"])
    if statistic == "linear_given_cyclic_partial_r2":
        return _safe_partial_r2(model_sse["helix"], model_sse["cyclic"])
    models = _models_for_statistic(statistic)
    return _safe_partial_r2(model_sse[models[0]], base_sse)


def global_digit_order_permutation_test(
    response: Any,
    digits: Any,
    puzzle_ids: Any,
    *,
    base_design: Any | None = None,
    sample_weight: Any | None = None,
    strata: Any | None = None,
    n_folds: int = 5,
    fold_ids: Any | None = None,
    permutations: int = 999,
    statistic: str | Sequence[str] = "helix_partial_r2",
    seed: int = 0,
    digit_offset: int = 0,
    add_intercept: bool = True,
) -> dict[str, Any]:
    """Monte Carlo test of the prespecified numeric order of the nine digits.

    Each null replicate samples one permutation of the nine labels and applies
    that same mapping to every observation and every fold.  Folds never change
    between the observed and permuted fits.  The one-sided p-value is exactly
    ``(1 + number(null >= observed)) / (permutations + 1)``.

    ``statistic`` may be one name or a sequence.  Requesting linear, cyclic,
    and helix statistics together reuses the same global permutations and the
    same fold sufficient statistics.  A single string retains the original
    top-level ``observed``, ``null_values``, and ``p_value`` fields.
    """
    if not isinstance(permutations, (int, np.integer)) or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    response_matrix, digit_array, puzzle_array, base_matrix, weights = _prepare_inputs(
        response, digits, puzzle_ids, base_design, sample_weight,
        add_intercept=add_intercept,
    )
    indices = _digit_indices(digit_array, digit_offset)
    if fold_ids is None:
        row_folds = make_stratified_puzzle_folds(
            puzzle_array, strata, n_folds=n_folds, seed=seed
        )
    else:
        row_folds, _, _, _ = _validated_fold_ids(puzzle_array, fold_ids)

    if isinstance(statistic, str):
        requested_statistics = (statistic,)
        single_statistic = True
    else:
        requested_statistics = tuple(dict.fromkeys(statistic))
        single_statistic = False
        if not requested_statistics:
            raise ValueError("statistic sequence must not be empty")
        if not all(isinstance(name, str) for name in requested_statistics):
            raise ValueError("every statistic name must be a string")
    statistic_models = {
        name: _models_for_statistic(name) for name in requested_statistics
    }
    required_models = set(
        model
        for models in statistic_models.values()
        for model in models
    )

    fold_statistics = _prepare_fwl_fold_statistics(
        response_matrix, indices, base_matrix, weights, row_folds
    )
    base_sse = float(sum(fold.test_base_sse for fold in fold_statistics))
    identity = np.arange(9, dtype=np.int64)
    observed_model_sse = {
        model: _digit_model_sse_from_fwl(
            fold_statistics, digit_code_matrix(identity, model)
        )
        for model in required_models
    }
    need_categorical = "helix_categorical_gain_fraction" in requested_statistics
    categorical_sse = (
        _digit_model_sse_from_fwl(
            fold_statistics, digit_code_matrix(identity, "categorical")
        )
        if need_categorical else None
    )
    observed = {
        name: _statistic_from_sse(
            name, observed_model_sse, base_sse,
            categorical_sse,
        )
        for name in requested_statistics
    }
    if not all(np.isfinite(value) for value in observed.values()):
        raise ValueError("an observed permutation statistic is not finite")

    generator = np.random.default_rng(seed + 1)
    null_values: dict[str, list[float]] = {
        name: [] for name in requested_statistics
    }
    label_permutations: list[list[int]] = []
    for _ in range(permutations):
        mapping = generator.permutation(9)
        permuted_model_sse = {
            model: _digit_model_sse_from_fwl(
                fold_statistics, digit_code_matrix(mapping, model)
            )
            for model in required_models
        }
        for name in requested_statistics:
            null_values[name].append(_statistic_from_sse(
                name,
                permuted_model_sse,
                base_sse,
                categorical_sse,
            ))
        # Store the mapping in the caller's label convention.
        label_permutations.append((mapping + digit_offset).tolist())
    statistic_results: dict[str, Any] = {}
    for name in requested_statistics:
        null_array = np.asarray(null_values[name], dtype=np.float64)
        if not np.all(np.isfinite(null_array)):
            raise ValueError(f"a permuted {name} statistic is not finite")
        exceedances = int(np.count_nonzero(null_array >= observed[name]))
        statistic_results[name] = {
            "observed": float(observed[name]),
            "exceedances": exceedances,
            "p_value": float((1 + exceedances) / (permutations + 1)),
            "null_values": null_array.tolist(),
        }

    result: dict[str, Any] = {
        "method": "global nine-label order permutation; one mapping shared by all folds",
        "alternative": "observed statistic is larger",
        "requested_statistics": list(requested_statistics),
        "permutations": int(permutations),
        "seed": int(seed),
        "label_permutations": label_permutations,
        "statistics": statistic_results,
        "sufficient_statistics": {
            "method": "Frisch-Waugh-Lovell residualization with nine digit-group sums",
            "folds": len(fold_statistics),
            "base_sse": base_sse,
            "categorical_sse": categorical_sse,
        },
    }
    if single_statistic:
        name = requested_statistics[0]
        result["statistic"] = name
        result.update(statistic_results[name])
    return result


__all__ = [
    "DIGIT_MODEL_NAMES",
    "WeightedOLSFit",
    "bootstrap_geometry_intervals",
    "digit_code_matrix",
    "evaluate_digit_geometry",
    "fit_weighted_multivariate_ols",
    "global_digit_order_permutation_test",
    "make_stratified_puzzle_folds",
    "weighted_sse",
]
