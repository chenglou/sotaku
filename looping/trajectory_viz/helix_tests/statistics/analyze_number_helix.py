"""Held-out linear and cyclic tests of Sotaku's per-cell digit geometry.

The confirmatory response is the direction of each cell state after removing
the board-wide mean at the same puzzle and iteration.  Complete puzzles are
assigned to rating-stratified folds.  Linear, one-turn cyclic, combined helix,
and unrestricted categorical digit models are fitted on training puzzles and
scored on held-out puzzles.  The analysis also repeats the fits outside the
output-head contrast space and on recurrent update directions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence

import numpy as np
import torch

from looping.eval_loop_diagnostics import _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS
from looping.trajectory_viz.helix_tests.statistics.parameter_geometry import (
    analyze_model_parameter_geometry,
)
from looping.trajectory_viz.helix_tests.statistics.statistics_core import (
    evaluate_digit_geometry,
    global_digit_order_permutation_test,
)
from looping.trajectory_viz.helix_tests.statistics.trajectory_data import (
    SNAPSHOT_ITERATIONS,
    build_cell_table,
    build_response_representations,
    collect_trajectory,
    load_balanced_trajectory_sample,
    rating_stratified_fold_ids,
)


PRIMARY_REPRESENTATION = "cell_centered_unit_hidden_direction"
OUTPUT_NULL_REPRESENTATION = "hidden_direction_outside_output_contrast"
UPDATE_REPRESENTATION = "cell_centered_unit_update_direction"
REPRESENTATIONS = (
    PRIMARY_REPRESENTATION,
    OUTPUT_NULL_REPRESENTATION,
    UPDATE_REPRESENTATION,
)
DIGIT_LABELS = ("true_digit", "predicted_digit")
MODEL_DISPLAY_NAMES = {
    "stable_plain": "stable plain",
    "collapsed_plain": "collapsed plain",
    "late_state_ce": "late-state CE",
    "combined_margin": "combined margin",
}
DEFAULT_BOOTSTRAPS = 2000
CERTAINTY_BIN_EDGES = {
    "max_confidence": (0.5, 0.8, 0.95, 0.99),
    "decision_margin": (0.5, 1.0, 2.0, 4.0, 8.0),
    "true_probability": (0.5, 0.8, 0.95, 0.99),
    "target_margin": (-8.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 8.0),
}


def _json_ready(value):
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, torch.Tensor):
        return _json_ready(value.detach().cpu().tolist())
    if isinstance(value, (np.floating, np.integer)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(_json_ready(value), handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def _atomic_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(_json_ready(rows))
    os.replace(temporary_path, path)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_hot_drop_reference(values, category_count):
    values = np.asarray(values, dtype=np.int64)
    if np.any((values < 0) | (values >= category_count)):
        raise ValueError("categorical value is outside its declared range")
    if category_count <= 1:
        return np.empty((len(values), 0), dtype=np.float64)
    return np.equal(
        values[:, None], np.arange(1, category_count)[None, :]
    ).astype(np.float64)


def _fixed_bin_one_hot(values, edges):
    """Encode a continuous nuisance variable using prespecified bins.

    Full one-hot coding is intentional.  Each complete bin group is redundant
    with the fitted intercept, and the exact redundancy is safely removed by
    the SVD solve.  In contrast, fitting a slope to an almost constant
    confidence value can extrapolate numerical noise on held-out puzzles after
    the model has converged.
    """

    values = np.asarray(values, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("certainty values must be a finite vector")
    if edges.ndim != 1 or not np.all(np.isfinite(edges)):
        raise ValueError("certainty bin edges must be a finite vector")
    if len(edges) and np.any(np.diff(edges) <= 0):
        raise ValueError("certainty bin edges must be strictly increasing")
    bin_index = np.digitize(values, edges)
    return np.equal(
        bin_index[:, None], np.arange(len(edges) + 1)[None, :]
    ).astype(np.float64)


def _iteration_indices(iterations):
    lookup = {iteration: index for index, iteration in enumerate(SNAPSHOT_ITERATIONS)}
    try:
        return np.asarray([lookup[int(value)] for value in iterations], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unexpected trajectory iteration {error.args[0]}") from error


def build_nuisance_designs(metadata, blank_counts_by_puzzle, *, pooled):
    """Build the label-free primary design and two certainty sensitivities.

    The first three models add, in order: rating bucket and blank count;
    iteration fixed effects; and exact cell-position fixed effects.  The
    ``decision_adjusted`` sensitivity then adds fixed bins of label-free max
    confidence and top-one/top-two margin.  ``target_adjusted`` additionally
    uses fixed bins of true-class probability and true-vs-best-wrong margin, so
    it is explicitly target-informed and is never the confirmatory baseline.
    Fixed bins avoid unstable held-out extrapolation when confidence is almost
    constant after convergence.  The pooled base already includes exact
    iteration fixed effects; the bin effects are additive rather than expanded
    into hundreds of sparsely occupied iteration-by-bin interactions.
    """

    puzzle = np.asarray(metadata["puzzle_index"], dtype=np.int64)
    cell = np.asarray(metadata["cell_index"], dtype=np.int64)
    bucket = np.asarray(metadata["rating_bucket_index"], dtype=np.int64)
    iteration_index = _iteration_indices(metadata["iteration"])
    blank_counts = np.asarray(blank_counts_by_puzzle, dtype=np.float64)
    if puzzle.max(initial=-1) >= len(blank_counts):
        raise ValueError("blank-count array is not aligned with puzzle IDs")

    bucket_count = int(bucket.max(initial=0)) + 1
    context = np.column_stack(
        (
            _one_hot_drop_reference(bucket, bucket_count),
            blank_counts[puzzle] / 81.0,
        )
    )
    if pooled:
        iteration = np.column_stack(
            (
                context,
                _one_hot_drop_reference(
                    iteration_index, len(SNAPSHOT_ITERATIONS)
                ),
            )
        )
    else:
        iteration = context
    position = np.column_stack(
        (iteration, _one_hot_drop_reference(cell, 81))
    )

    certainty_columns = {
        "max_confidence": _fixed_bin_one_hot(
            metadata["max_confidence"], CERTAINTY_BIN_EDGES["max_confidence"]
        ),
        "decision_margin": _fixed_bin_one_hot(
            metadata["top1_top2_margin"], CERTAINTY_BIN_EDGES["decision_margin"]
        ),
        "true_probability": _fixed_bin_one_hot(
            metadata["true_probability"], CERTAINTY_BIN_EDGES["true_probability"]
        ),
        "target_margin": _fixed_bin_one_hot(
            metadata["target_minus_max_wrong_logit_margin"],
            CERTAINTY_BIN_EDGES["target_margin"],
        ),
    }
    for name, columns in certainty_columns.items():
        if not np.allclose(columns.sum(axis=1), 1.0):
            raise RuntimeError(f"internal certainty-bin encoding failure for {name}")

    decision_certainty = np.column_stack(
        (certainty_columns["max_confidence"], certainty_columns["decision_margin"])
    )
    target_certainty = np.column_stack(
        (certainty_columns["true_probability"], certainty_columns["target_margin"])
    )
    decision_adjusted = np.column_stack((position, decision_certainty))
    target_adjusted = np.column_stack(
        (decision_adjusted, target_certainty)
    )
    return {
        "context": context,
        "iteration": iteration,
        "position": position,
        "decision_adjusted": decision_adjusted,
        "target_adjusted": target_adjusted,
    }


def equal_puzzle_weights(puzzle_ids):
    puzzle_ids = np.asarray(puzzle_ids, dtype=np.int64)
    if puzzle_ids.ndim != 1 or not len(puzzle_ids):
        raise ValueError("puzzle_ids must be a non-empty vector")
    _, inverse, counts = np.unique(
        puzzle_ids, return_inverse=True, return_counts=True
    )
    return 1.0 / (len(counts) * counts[inverse])


def nested_control_contributions(
    response,
    digits,
    puzzle_ids,
    strata,
    fold_ids,
    weights,
    designs,
    *,
    seed,
):
    """Score iteration, cell position, and certainty as nested controls."""

    base_sse = {}
    design_order = (
        "context",
        "iteration",
        "position",
        "decision_adjusted",
        "target_adjusted",
    )
    for index, name in enumerate(design_order):
        result = evaluate_digit_geometry(
            response,
            digits,
            puzzle_ids,
            base_design=designs[name],
            sample_weight=weights,
            strata=strata,
            fold_ids=fold_ids,
            seed=seed + index,
            models=(),
        )
        base_sse[name] = result["models"]["base"]["sse"]

    def partial(full, reduced):
        return 1.0 - full / reduced if reduced > 0 else math.nan

    return {
        "sse": base_sse,
        "partial_r2": {
            "iteration_over_context": partial(
                base_sse["iteration"], base_sse["context"]
            ),
            "cell_position_over_iteration": partial(
                base_sse["position"], base_sse["iteration"]
            ),
            "confidence_decision_margin_over_position": partial(
                base_sse["decision_adjusted"], base_sse["position"]
            ),
            "target_probability_margin_over_decision": partial(
                base_sse["target_adjusted"], base_sse["decision_adjusted"]
            ),
        },
    }


def puzzle_gain_summary(per_puzzle, model_name):
    """Describe complete-puzzle held-out SSE gains without an independence test."""
    improvement = np.asarray(
        [
            record["sse"]["base"] - record["sse"][model_name]
            for record in per_puzzle
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(improvement)):
        raise ValueError("per-puzzle held-out SSE improvements must be finite")
    return {
        "method": "descriptive complete-puzzle held-out SSE gains",
        "note": (
            "No p-value: cross-validated puzzle losses share overlapping "
            "training fits and are not independent sign-flip units."
        ),
        "model": model_name,
        "total_sse_gain": float(improvement.sum()),
        "median_sse_gain": float(np.median(improvement)),
        "p10_sse_gain": float(np.quantile(improvement, 0.1)),
        "p90_sse_gain": float(np.quantile(improvement, 0.9)),
        "positive_puzzle_fraction": float(np.mean(improvement > 0)),
        "puzzle_count": int(len(improvement)),
    }


def _strip_permutation_payload(test, null_store, key):
    test = dict(test)
    if "null_values" in test:
        null_store[key] = np.asarray(test.pop("null_values"), dtype=np.float64)
    test.pop("label_permutations", None)
    return test


def _strip_parameter_permutations(result, null_store, model_name):
    result = dict(result)
    result.pop("global_digit_permutations", None)
    for source_name, source in result["sources"].items():
        for fit_name, fit in source["fits"].items():
            permutation = fit["permutation_test"]
            null = permutation.pop("null_r2")
            null_store[
                f"parameter__{model_name}__{source_name}__{fit_name}"
            ] = np.asarray(null, dtype=np.float64)
    return result


def _pooled_digit_fit(
    response,
    digits,
    metadata,
    base_design,
    *,
    bootstrap_replicates,
    seed,
):
    puzzle_ids = np.asarray(metadata["puzzle_index"], dtype=np.int64)
    strata = np.asarray(metadata["rating_bucket_index"], dtype=np.int64)
    fold_ids = np.asarray(metadata["fold_index"], dtype=np.int64)
    weights = equal_puzzle_weights(puzzle_ids)
    result = evaluate_digit_geometry(
        response,
        digits,
        puzzle_ids,
        base_design=base_design,
        sample_weight=weights,
        strata=strata,
        fold_ids=fold_ids,
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    result["puzzle_gain_summary"] = {
        model_name: puzzle_gain_summary(result["per_puzzle"], model_name)
        for model_name in ("linear", "cyclic", "helix", "categorical")
    }
    return result


def _per_iteration_fits(
    response,
    metadata,
    blank_counts,
    *,
    seed,
):
    results = {}
    iterations = np.asarray(metadata["iteration"], dtype=np.int64)
    for time_index, iteration in enumerate(SNAPSHOT_ITERATIONS):
        selected = iterations == iteration
        selected_metadata = {
            key: np.asarray(value)[selected] for key, value in metadata.items()
        }
        designs = build_nuisance_designs(
            selected_metadata, blank_counts, pooled=False
        )
        puzzle_ids = selected_metadata["puzzle_index"].astype(np.int64)
        weights = equal_puzzle_weights(puzzle_ids)
        labels = {}
        for label_index, label_name in enumerate(DIGIT_LABELS):
            labels[label_name] = evaluate_digit_geometry(
                response[selected],
                selected_metadata[label_name],
                puzzle_ids,
                base_design=designs["position"],
                sample_weight=weights,
                strata=selected_metadata["rating_bucket_index"],
                fold_ids=selected_metadata["fold_index"],
                seed=seed + time_index * 10 + label_index,
            )
        labels["cell_count"] = int(selected.sum())
        correct = np.asarray(selected_metadata["correct"], dtype=np.float64)
        labels["correct_fraction"] = float(np.mean([
            correct[puzzle_ids == puzzle].mean()
            for puzzle in np.unique(puzzle_ids)
        ]))
        results[str(iteration)] = labels
    return results


def _wrong_only_fits(
    response,
    metadata,
    blank_counts,
    *,
    seed,
):
    selected = ~np.asarray(metadata["correct"], dtype=bool)
    if selected.sum() < 100:
        return {"rows": int(selected.sum()), "status": "too few wrong rows"}
    wrong_metadata = {
        key: np.asarray(value)[selected] for key, value in metadata.items()
    }
    designs = build_nuisance_designs(wrong_metadata, blank_counts, pooled=True)
    puzzle_ids = wrong_metadata["puzzle_index"].astype(np.int64)
    weights = equal_puzzle_weights(puzzle_ids)
    result = {"rows": int(selected.sum()), "labels": {}}
    for label_index, label_name in enumerate(DIGIT_LABELS):
        result["labels"][label_name] = evaluate_digit_geometry(
            response[selected],
            wrong_metadata[label_name],
            puzzle_ids,
            base_design=designs["position"],
            sample_weight=weights,
            strata=wrong_metadata["rating_bucket_index"],
            fold_ids=wrong_metadata["fold_index"],
            seed=seed + label_index,
            bootstrap_replicates=500,
        )
    return result


def _iteration_sample_statistics(metadata):
    iteration_values = np.asarray(metadata["iteration"], dtype=np.int64)
    puzzle_values = np.asarray(metadata["puzzle_index"], dtype=np.int64)

    def equal_puzzle_mean(values, selected):
        values = np.asarray(values, dtype=np.float64)[selected]
        puzzles = puzzle_values[selected]
        return float(np.mean([
            values[puzzles == puzzle].mean() for puzzle in np.unique(puzzles)
        ]))

    def equal_puzzle_tail(values, selected, quantile):
        values = np.asarray(values, dtype=np.float64)[selected]
        puzzles = puzzle_values[selected]
        return float(np.mean([
            np.quantile(values[puzzles == puzzle], quantile)
            for puzzle in np.unique(puzzles)
        ]))

    records = {}
    for iteration in SNAPSHOT_ITERATIONS:
        selected = iteration_values == iteration
        records[str(iteration)] = {
            "blank_cells": int(selected.sum()),
            "correct_fraction": equal_puzzle_mean(
                metadata["correct"], selected
            ),
            "mean_max_confidence": equal_puzzle_mean(
                metadata["max_confidence"], selected
            ),
            "mean_true_probability": equal_puzzle_mean(
                metadata["true_probability"], selected
            ),
            "target_margin_mean": equal_puzzle_mean(
                metadata["target_minus_max_wrong_logit_margin"], selected
            ),
            "target_margin_p10": equal_puzzle_tail(
                metadata["target_minus_max_wrong_logit_margin"], selected, 0.1
            ),
            "decision_margin_mean": equal_puzzle_mean(
                metadata["top1_top2_margin"], selected
            ),
            "decision_margin_p10": equal_puzzle_tail(
                metadata["top1_top2_margin"], selected, 0.1
            ),
        }
    return records


def _holm_adjust(p_value_records):
    """Attach Holm family-wise adjusted p-values to mutable records."""

    finite = [
        (float(record["p_value"]), index, record)
        for index, record in enumerate(p_value_records)
        if record.get("p_value") is not None
        and math.isfinite(float(record["p_value"]))
    ]
    finite.sort(key=lambda item: item[0])
    adjusted = []
    running = 0.0
    for rank, (p_value, _, _) in enumerate(finite):
        running = max(running, p_value * (len(finite) - rank))
        adjusted.append(min(1.0, running))
    for (_, _, record), adjusted_p in zip(finite, adjusted):
        record["p_value_holm"] = float(adjusted_p)


def _collect_csv_rows(summary):
    pooled_rows = []
    sensitivity_rows = []
    iteration_rows = []
    control_rows = []
    parameter_rows = []
    wrong_rows = []
    for model_name, model in summary["models"].items():
        for representation, controls in model["control_contributions"].items():
            control_rows.append(
                {
                    "model": model_name,
                    "representation": representation,
                    **controls["partial_r2"],
                }
            )
        for representation, representation_result in model["pooled"].items():
            for label_name, fit in representation_result.items():
                row = {
                    "model": model_name,
                    "representation": representation,
                    "digit_label": label_name,
                    "rows": fit["config"]["n_rows"],
                    "puzzles": fit["config"]["n_puzzles"],
                    **fit["partial_r2"],
                    "helix_categorical_gain_fraction": fit.get(
                        "helix_categorical_gain_fraction"
                    ),
                }
                for statistic, permutation in fit.get(
                    "natural_order_permutation", {}
                ).items():
                    row[f"{statistic}_permutation_p"] = permutation["p_value"]
                    row[f"{statistic}_permutation_p_holm"] = permutation.get(
                        "p_value_holm"
                    )
                pooled_rows.append(row)
        for representation, representation_result in model[
            "certainty_sensitivity"
        ].items():
            for label_name, design_results in representation_result.items():
                for design_name, fit in design_results.items():
                    sensitivity_rows.append(
                        {
                            "model": model_name,
                            "representation": representation,
                            "digit_label": label_name,
                            "nuisance_design": design_name,
                            "rows": fit["config"]["n_rows"],
                            "puzzles": fit["config"]["n_puzzles"],
                            **fit["partial_r2"],
                            "helix_categorical_gain_fraction": fit.get(
                                "helix_categorical_gain_fraction"
                            ),
                        }
                    )
        for iteration, iteration_result in model["by_iteration"].items():
            for label_name in DIGIT_LABELS:
                fit = iteration_result[label_name]
                iteration_rows.append(
                    {
                        "model": model_name,
                        "iteration": int(iteration),
                        "digit_label": label_name,
                        "rows": fit["config"]["n_rows"],
                        "correct_fraction": iteration_result["correct_fraction"],
                        **fit["partial_r2"],
                        "helix_categorical_gain_fraction": fit.get(
                            "helix_categorical_gain_fraction"
                        ),
                    }
                )
        for source_name, source in model["parameter_geometry"]["sources"].items():
            for fit_name, fit in source["fits"].items():
                parameter_rows.append(
                    {
                        "model": model_name,
                        "parameter_source": source_name,
                        "fit": fit_name,
                        "r2": fit["r2"],
                        "permutation_p": fit["permutation_test"]["p_value"],
                        "permutation_p_holm": fit["permutation_test"].get(
                            "p_value_holm"
                        ),
                        **source["axis_geometry"],
                    }
                )
        for representation, wrong in model["wrong_only"].items():
            if "labels" not in wrong:
                wrong_rows.append(
                    {
                        "model": model_name,
                        "representation": representation,
                        "rows": wrong.get("rows", 0),
                        "status": wrong.get("status"),
                    }
                )
                continue
            for label_name, fit in wrong["labels"].items():
                wrong_rows.append(
                    {
                        "model": model_name,
                        "representation": representation,
                        "digit_label": label_name,
                        "rows": wrong["rows"],
                        "puzzles": fit["config"]["n_puzzles"],
                        **fit["partial_r2"],
                        "helix_categorical_gain_fraction": fit.get(
                            "helix_categorical_gain_fraction"
                        ),
                    }
                )
    return (
        pooled_rows,
        sensitivity_rows,
        iteration_rows,
        control_rows,
        parameter_rows,
        wrong_rows,
    )


def run(
    output_dir,
    *,
    examples_per_bucket=20,
    permutations=9999,
    seed=20260807,
    device="cuda",
    model_configs=DEFAULT_MODELS,
    bootstrap_replicates=DEFAULT_BOOTSTRAPS,
):
    """Run the complete held-out analysis and write compact artifacts."""

    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    log_path = os.path.join(output_dir, "run.log")
    log_file = open(log_path, "w")

    def log(message=""):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    started_at = time.time()
    sample = load_balanced_trajectory_sample(examples_per_bucket, seed)
    fold_id_by_puzzle = rating_stratified_fold_ids(
        sample.rating_buckets, n_splits=5, seed=seed
    )
    device_sample = sample.to(resolved_device)
    blank_counts = sample.originally_blank.sum(dim=1).numpy()
    summary = {
        "config": {
            "examples_per_bucket": examples_per_bucket,
            "sample_size": sample.puzzle_count,
            "rating_buckets": list(dict.fromkeys(sample.rating_buckets)),
            "folds": 5,
            "fold_assignment": "whole puzzles, shuffled within every rating bucket",
            "iterations": list(SNAPSHOT_ITERATIONS),
            "permutations": permutations,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_scope": (
                "conditional descriptive puzzle-residual intervals; folds and "
                "probe fits remain fixed"
            ),
            "seed": seed,
            "device": str(resolved_device),
            "software": {
                "numpy": np.__version__,
                "torch": str(torch.__version__),
            },
            "primary_cells": "originally blank cells only",
            "weighting": (
                "equal total weight per puzzle in fits; accuracy, confidence, "
                "and margin summaries average within puzzle first"
            ),
            "primary_response": PRIMARY_REPRESENTATION,
            "primary_nuisance_design": (
                "rating bucket + blank fraction + iteration fixed effects + "
                "exact cell-position fixed effects"
            ),
            "certainty_sensitivities": {
                "decision_adjusted": (
                    "primary nuisance design + fixed bins of max confidence and "
                    "top-one/top-two decision margin"
                ),
                "target_adjusted": (
                    "decision_adjusted + fixed bins of true-class probability and "
                    "true-vs-best-wrong target margin; "
                    "target-informed"
                ),
                "fixed_bin_edges": CERTAINTY_BIN_EDGES,
            },
            "response_normalization": (
                "cell-centered directions normalized per cell; raw norm and "
                "output-null energy summaries are stored separately"
            ),
            "digit_codes": {
                "linear": "centered digits 1..9, unit variance",
                "cyclic": "cos/sin of one fixed turn over digits 1..9",
                "helix": "linear plus cyclic (3 degrees of freedom)",
                "categorical": "unrestricted digit identity (8 degrees of freedom)",
            },
            "target_margin": "true logit minus maximum wrong-class logit",
            "decision_margin": "largest logit minus second-largest logit",
            "model_configs": list(model_configs),
        },
        "sample": {
            "puzzle_sha256": [
                hashlib.sha256(puzzle.encode("ascii")).hexdigest()
                for puzzle in sample.puzzles
            ],
            "puzzles": list(sample.puzzles),
            "rating_buckets": list(sample.rating_buckets),
            "fold_id_by_puzzle": fold_id_by_puzzle.tolist(),
            "blank_counts": blank_counts.tolist(),
        },
        "models": {},
    }
    null_store = {}
    natural_order_records = []
    parameter_permutation_records = []
    log(
        f"Number geometry: {sample.puzzle_count} puzzles, "
        f"{len(model_configs)} models, {permutations} digit-order permutations"
    )

    for model_index, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        model_started_at = time.time()
        log(f"\nMODEL {model_name}: {model_config['path']}")
        checkpoint_metadata = {
            "path": model_config["path"],
            "size_bytes": os.path.getsize(model_config["path"]),
            "sha256": _file_sha256(model_config["path"]),
        }
        model = _load_model(model_config, resolved_device)
        trajectory = collect_trajectory(
            model,
            device_sample.inputs,
            device_sample.targets,
            device_sample.originally_blank,
            iterations=SNAPSHOT_ITERATIONS,
            output_device="cpu",
        )
        responses = build_response_representations(
            trajectory, model.output_head
        )
        table = build_cell_table(
            trajectory,
            sample.rating_buckets,
            fold_id_by_puzzle=fold_id_by_puzzle,
        )
        metadata_torch = table.flatten_primary()
        metadata = {
            key: value.detach().cpu().numpy()
            for key, value in metadata_torch.items()
            if key != "sample_weight"
        }
        response_tensors = responses.flatten(table.primary_mask)
        response_arrays = {
            name: values.detach().cpu().numpy().astype(np.float64, copy=False)
            for name, values in response_tensors.items()
        }
        designs = build_nuisance_designs(
            metadata, blank_counts, pooled=True
        )
        puzzle_ids = metadata["puzzle_index"].astype(np.int64)
        strata = metadata["rating_bucket_index"].astype(np.int64)
        row_fold_ids = metadata["fold_index"].astype(np.int64)
        weights = equal_puzzle_weights(puzzle_ids)

        parameter_geometry = analyze_model_parameter_geometry(
            model,
            permutations=permutations,
            seed=seed + model_index * 1000,
        )
        parameter_geometry = _strip_parameter_permutations(
            parameter_geometry, null_store, model_name
        )
        for source in parameter_geometry["sources"].values():
            for fit_name, fit in source["fits"].items():
                if fit_name == "categorical":
                    continue
                parameter_permutation_records.append(
                    fit["permutation_test"]
                )

        model_result = {
            "display_name": MODEL_DISPLAY_NAMES.get(model_name, model_name),
            "checkpoint": checkpoint_metadata,
            "sample_statistics": _iteration_sample_statistics(metadata),
            "output_contrast_rank": int(responses.output_contrast_basis.size(0)),
            "response_scale_statistics": {
                str(iteration): {
                    key: float(value[time_index].item())
                    for key, value in responses.scale_statistics.items()
                }
                for time_index, iteration in enumerate(SNAPSHOT_ITERATIONS)
            },
            "parameter_geometry": parameter_geometry,
            "control_contributions": {},
            "pooled": {},
            "certainty_sensitivity": {},
            "by_iteration": {},
            "wrong_only": {},
        }
        for representation_index, representation in enumerate(REPRESENTATIONS):
            log(f"  pooled {representation}")
            response = response_arrays[representation]
            model_result["control_contributions"][representation] = (
                nested_control_contributions(
                    response,
                    metadata["true_digit"],
                    puzzle_ids,
                    strata,
                    row_fold_ids,
                    weights,
                    designs,
                    seed=seed + model_index * 10000 + representation_index * 100,
                )
            )
            representation_result = {}
            sensitivity_result = {}
            for label_index, label_name in enumerate(DIGIT_LABELS):
                fit_seed = (
                    seed
                    + model_index * 10000
                    + representation_index * 1000
                    + label_index * 100
                )
                fit = _pooled_digit_fit(
                    response,
                    metadata[label_name],
                    metadata,
                    designs["position"],
                    bootstrap_replicates=bootstrap_replicates,
                    seed=fit_seed,
                )
                sensitivity_result[label_name] = {
                    design_name: evaluate_digit_geometry(
                        response,
                        metadata[label_name],
                        puzzle_ids,
                        base_design=designs[design_name],
                        sample_weight=weights,
                        strata=strata,
                        fold_ids=row_fold_ids,
                        seed=fit_seed + 20 + design_index,
                        bootstrap_replicates=0,
                    )
                    for design_index, design_name in enumerate(
                        ("decision_adjusted", "target_adjusted")
                    )
                }
                if representation != UPDATE_REPRESENTATION:
                    fit["natural_order_permutation"] = {}
                    statistics = (
                        "linear_partial_r2",
                        "cyclic_partial_r2",
                        "helix_partial_r2",
                        "cyclic_given_linear_partial_r2",
                        "linear_given_cyclic_partial_r2",
                    )
                    permutation_bundle = global_digit_order_permutation_test(
                        response,
                        metadata[label_name],
                        puzzle_ids,
                        base_design=designs["position"],
                        sample_weight=weights,
                        strata=strata,
                        fold_ids=row_fold_ids,
                        permutations=permutations,
                        statistic=statistics,
                        seed=fit_seed,
                    )
                    for statistic in statistics:
                        key = (
                            f"order__{model_name}__{representation}__"
                            f"{label_name}__{statistic}"
                        )
                        statistic_result = _strip_permutation_payload(
                            permutation_bundle["statistics"][statistic],
                            null_store,
                            key,
                        )
                        statistic_result.update(
                            {
                                "method": permutation_bundle["method"],
                                "alternative": permutation_bundle["alternative"],
                                "statistic": statistic,
                                "permutations": permutation_bundle["permutations"],
                                "seed": permutation_bundle["seed"],
                            }
                        )
                        fit["natural_order_permutation"][statistic] = (
                            statistic_result
                        )
                        natural_order_records.append(statistic_result)
                representation_result[label_name] = fit
            model_result["pooled"][representation] = representation_result
            model_result["certainty_sensitivity"][representation] = sensitivity_result

        log("  per-iteration primary-state fits")
        model_result["by_iteration"] = _per_iteration_fits(
            response_arrays[PRIMARY_REPRESENTATION],
            metadata,
            blank_counts,
            seed=seed + model_index * 10000 + 7000,
        )
        for representation in (
            PRIMARY_REPRESENTATION,
            OUTPUT_NULL_REPRESENTATION,
        ):
            model_result["wrong_only"][representation] = _wrong_only_fits(
                response_arrays[representation],
                metadata,
                blank_counts,
                seed=seed + model_index * 10000 + 8000,
            )

        model_result["elapsed_seconds"] = time.time() - model_started_at
        summary["models"][model_name] = model_result
        log(f"  done in {model_result['elapsed_seconds']:.1f}s")
        del model, trajectory, responses, table, response_tensors, response_arrays
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    _holm_adjust(natural_order_records)
    _holm_adjust(parameter_permutation_records)
    summary["multiple_testing"] = {
        "method": "Holm family-wise adjusted p-values within each prespecified family",
        "families": {
            "natural_digit_order": len(natural_order_records),
            "learned_parameter_digit_order": len(parameter_permutation_records),
        },
    }
    summary["elapsed_seconds"] = time.time() - started_at

    (
        pooled_rows,
        sensitivity_rows,
        iteration_rows,
        control_rows,
        parameter_rows,
        wrong_rows,
    ) = _collect_csv_rows(summary)
    _atomic_json(os.path.join(output_dir, "results.json"), summary)
    _atomic_csv(os.path.join(output_dir, "pooled_fits.csv"), pooled_rows)
    _atomic_csv(
        os.path.join(output_dir, "certainty_sensitivity.csv"), sensitivity_rows
    )
    _atomic_csv(os.path.join(output_dir, "iteration_fits.csv"), iteration_rows)
    _atomic_csv(os.path.join(output_dir, "nuisance_controls.csv"), control_rows)
    _atomic_csv(os.path.join(output_dir, "parameter_geometry.csv"), parameter_rows)
    _atomic_csv(os.path.join(output_dir, "wrong_only_fits.csv"), wrong_rows)
    np.savez_compressed(
        os.path.join(output_dir, "permutation_nulls.npz"), **null_store
    )

    from looping.trajectory_viz.helix_tests.statistics.plot_statistics import (
        render_all_plots,
    )

    render_all_plots(summary, null_store, output_dir)
    log(f"\nTotal time: {summary['elapsed_seconds']:.1f}s")
    log_file.close()
    return _json_ready(summary)


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu", examples_per_bucket=5, permutations=99)
