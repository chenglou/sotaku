"""Fit fixed early-warning models and evaluate untouched final puzzles."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import OrderedDict, defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sotaku-arm10-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core import (
    DIFFICULTY_FEATURES,
    GEOMETRY_FEATURES,
    OUTPUT_FEATURES,
    SHUFFLED_GEOMETRY_FEATURES,
    SPLIT_NAMES,
    Standardizer,
    apply_platt,
    average_precision,
    calibration_bins,
    fit_ridge_logistic,
    minimum_events_for_auc,
    minimum_non_events_for_auc,
    predict_logit,
    probability_metrics,
    roc_auc,
    stratified_permutation,
)


BASELINE_FEATURES = DIFFICULTY_FEATURES + OUTPUT_FEATURES
MODEL_SPECS = OrderedDict(
    (
        ("prevalence", ((), False)),
        ("difficulty", (DIFFICULTY_FEATURES, False)),
        ("early_output", (OUTPUT_FEATURES, False)),
        ("difficulty_output", (BASELINE_FEATURES, False)),
        ("geometry", (GEOMETRY_FEATURES, False)),
        ("shuffled_iteration_geometry", (SHUFFLED_GEOMETRY_FEATURES, False)),
        ("difficulty_output_geometry", (BASELINE_FEATURES + GEOMETRY_FEATURES, False)),
        (
            "difficulty_output_shuffled_iteration_geometry",
            (BASELINE_FEATURES + SHUFFLED_GEOMETRY_FEATURES, False),
        ),
        ("checkpoint_identity", ((), True)),
        ("checkpoint_difficulty_output", (BASELINE_FEATURES, True)),
        (
            "checkpoint_difficulty_output_geometry",
            (BASELINE_FEATURES + GEOMETRY_FEATURES, True),
        ),
        (
            "checkpoint_difficulty_output_shuffled_iteration_geometry",
            (BASELINE_FEATURES + SHUFFLED_GEOMETRY_FEATURES, True),
        ),
    )
)
PRIMARY_BASE_MODEL = "checkpoint_difficulty_output"
PRIMARY_FULL_MODEL = "checkpoint_difficulty_output_geometry"
SHUFFLED_FULL_MODEL = "checkpoint_difficulty_output_shuffled_iteration_geometry"


def _atomic_json(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary_path, path)


def _validate_payload(payload):
    if payload.get("format_version") != 1 or not payload.get("rows"):
        raise ValueError("unsupported or empty collection")
    manifest = payload.get("feature_manifest", {})
    if manifest.get("maximum_iteration") != 128:
        raise ValueError("feature manifest does not enforce iteration 128")
    if manifest.get("uses_targets") or manifest.get("uses_fitted_projection"):
        raise ValueError("collection violates the leakage controls")
    split_by_puzzle = {}
    for row in payload["rows"]:
        previous = split_by_puzzle.setdefault(row["puzzle_hash"], row["split"])
        if previous != row["split"]:
            raise ValueError("one puzzle appears in multiple splits")
        for feature in BASELINE_FEATURES + GEOMETRY_FEATURES + SHUFFLED_GEOMETRY_FEATURES:
            if feature not in row or not math.isfinite(float(row[feature])):
                raise ValueError(f"missing or invalid feature {feature!r}")


def _eligible_rows(payload, split=None, exclude_checkpoint=None, only_checkpoint=None):
    result = []
    for row in payload["rows"]:
        if not row["eligible"]:
            continue
        if split is not None and row["split"] != split:
            continue
        if exclude_checkpoint is not None and row["checkpoint"] == exclude_checkpoint:
            continue
        if only_checkpoint is not None and row["checkpoint"] != only_checkpoint:
            continue
        result.append(row)
    return result


def _checkpoint_levels(rows):
    return tuple(dict.fromkeys(row["checkpoint"] for row in rows))


def _feature_matrix(rows, numerical_features, checkpoint_levels=()):
    names = list(numerical_features)
    names.extend(f"checkpoint={name}" for name in checkpoint_levels[1:])
    matrix = []
    for row in rows:
        values = [float(row[name]) for name in numerical_features]
        values.extend(float(row["checkpoint"] == name) for name in checkpoint_levels[1:])
        matrix.append(values)
    return np.asarray(matrix, dtype=np.float64).reshape(len(rows), len(names)), names


def _labels(rows):
    return np.asarray([int(row["collapse"]) for row in rows], dtype=int)


def _fit_offset_calibration(raw_logits, labels):
    """Fallback calibration that adjusts prevalence without reversing ranks."""

    raw_logits = np.asarray(raw_logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    offset = 0.0
    for _ in range(100):
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(raw_logits + offset, -40, 40)))
        gradient = np.sum(probabilities - labels)
        hessian = np.sum(probabilities * (1 - probabilities)) + 1e-8
        step = gradient / hessian
        offset -= step
        if abs(step) < 1e-9:
            break
    return np.asarray([offset, 1.0])


def prepare_design(discovery_rows, validation_rows, final_rows, numerical_features, include_checkpoint):
    levels = _checkpoint_levels(discovery_rows) if include_checkpoint else ()
    discovery_matrix, names = _feature_matrix(discovery_rows, numerical_features, levels)
    validation_matrix, _ = _feature_matrix(validation_rows, numerical_features, levels)
    final_matrix, _ = _feature_matrix(final_rows, numerical_features, levels)
    standardizer = Standardizer.fit(discovery_matrix)
    return {
        "names": names,
        "levels": levels,
        "standardizer": standardizer,
        "discovery": standardizer.transform(discovery_matrix),
        "validation": standardizer.transform(validation_matrix),
        "final": standardizer.transform(final_matrix),
    }


def fit_prepared(design, discovery_labels, validation_labels):
    coefficients = fit_ridge_logistic(design["discovery"], discovery_labels, l2=1.0)
    validation_logits = predict_logit(coefficients, design["validation"])
    # Validation adjusts calibration-in-the-large while preserving the ranking
    # learned on discovery.  This one-parameter calibration is appropriate for
    # the small validation split and cannot reverse a model by chance.
    calibration = _fit_offset_calibration(validation_logits, validation_labels)
    calibration_method = "validation_logit_offset"
    final_logits = predict_logit(coefficients, design["final"])
    final_probabilities = apply_platt(final_logits, calibration)
    return {
        "coefficients": coefficients,
        "calibration": calibration,
        "calibration_method": calibration_method,
        "final_logits": final_logits,
        "final_probabilities": final_probabilities,
    }


def _bootstrap_metrics(rows, labels, probabilities, repetitions, generator):
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["puzzle_hash"]].append(index)
    puzzle_hashes = list(groups)
    samples = defaultdict(list)
    for _ in range(repetitions):
        selected_hashes = generator.choice(puzzle_hashes, size=len(puzzle_hashes), replace=True)
        indices = np.concatenate([np.asarray(groups[value], dtype=int) for value in selected_hashes])
        metrics = probability_metrics(labels[indices], probabilities[indices])
        for name in ("auroc", "average_precision", "brier", "log_loss"):
            if metrics[name] is not None and math.isfinite(metrics[name]):
                samples[name].append(metrics[name])
    result = {}
    for name in ("auroc", "average_precision", "brier", "log_loss"):
        values = samples[name]
        result[name] = {
            "low": float(np.quantile(values, 0.025)) if values else None,
            "high": float(np.quantile(values, 0.975)) if values else None,
            "valid_repetitions": len(values),
        }
    return result


def _fit_model_set(discovery_rows, validation_rows, final_rows, bootstrap_repetitions, generator):
    discovery_labels = _labels(discovery_rows)
    validation_labels = _labels(validation_rows)
    final_labels = _labels(final_rows)
    fitted = {}
    for model_name, (features, include_checkpoint) in MODEL_SPECS.items():
        design = prepare_design(
            discovery_rows,
            validation_rows,
            final_rows,
            features,
            include_checkpoint,
        )
        fit = fit_prepared(design, discovery_labels, validation_labels)
        metrics = probability_metrics(final_labels, fit["final_probabilities"])
        metrics["bootstrap_95"] = _bootstrap_metrics(
            final_rows,
            final_labels,
            fit["final_probabilities"],
            bootstrap_repetitions,
            generator,
        )
        fitted[model_name] = {
            "features": design["names"],
            "coefficients_standardized": {
                "intercept": float(fit["coefficients"][0]),
                **{
                    name: float(value)
                    for name, value in zip(design["names"], fit["coefficients"][1:])
                },
            },
            "calibration": {
                "method": fit["calibration_method"],
                "intercept": float(fit["calibration"][0]),
                "slope": float(fit["calibration"][1]),
                "validation_n": int(len(validation_labels)),
                "validation_events": int(validation_labels.sum()),
            },
            "metrics": metrics,
            "calibration_bins": calibration_bins(final_labels, fit["final_probabilities"]),
            "probabilities": fit["final_probabilities"],
            "design": design,
        }
    return fitted


def _conditional_permutation_test(
    discovery_rows,
    validation_rows,
    final_rows,
    base_fit,
    full_fit,
    repetitions,
    generator,
):
    observed_labels = {
        "discovery": _labels(discovery_rows),
        "validation": _labels(validation_rows),
        "final": _labels(final_rows),
    }
    rows_by_split = {
        "discovery": discovery_rows,
        "validation": validation_rows,
        "final": final_rows,
    }
    strata = {
        split: np.asarray(
            [f"{row['checkpoint']}|{row['rating_bucket']}" for row in rows],
            dtype=object,
        )
        for split, rows in rows_by_split.items()
    }

    def statistic(base_probabilities, full_probabilities, final_labels):
        base_metrics = probability_metrics(final_labels, base_probabilities)
        full_metrics = probability_metrics(final_labels, full_probabilities)
        return {
            "log_loss_improvement": base_metrics["log_loss"] - full_metrics["log_loss"],
            "auroc_improvement": (
                full_metrics["auroc"] - base_metrics["auroc"]
                if full_metrics["auroc"] is not None and base_metrics["auroc"] is not None
                else None
            ),
        }

    observed = statistic(
        base_fit["probabilities"],
        full_fit["probabilities"],
        observed_labels["final"],
    )
    null_log_loss = []
    null_auroc = []
    for _ in range(repetitions):
        permuted = {
            split: stratified_permutation(observed_labels[split], strata[split], generator)
            for split in SPLIT_NAMES
        }
        base = fit_prepared(
            base_fit["design"],
            permuted["discovery"],
            permuted["validation"],
        )
        full = fit_prepared(
            full_fit["design"],
            permuted["discovery"],
            permuted["validation"],
        )
        value = statistic(base["final_probabilities"], full["final_probabilities"], permuted["final"])
        null_log_loss.append(value["log_loss_improvement"])
        if value["auroc_improvement"] is not None:
            null_auroc.append(value["auroc_improvement"])

    def summarize(values, observed_value):
        values = np.asarray(values, dtype=float)
        return {
            "observed": observed_value,
            "p_one_sided": float((1 + np.sum(values >= observed_value)) / (1 + len(values))),
            "null_q025": float(np.quantile(values, 0.025)),
            "null_median": float(np.median(values)),
            "null_q975": float(np.quantile(values, 0.975)),
            "null_values": values.tolist(),
        }

    result = {
        "null": "collapse labels shuffled within checkpoint x rating bucket, independently in each split",
        "repetitions": repetitions,
        "log_loss_improvement": summarize(null_log_loss, observed["log_loss_improvement"]),
    }
    if observed["auroc_improvement"] is not None and null_auroc:
        result["auroc_improvement"] = summarize(null_auroc, observed["auroc_improvement"])
    return result


def _leave_one_checkpoint_out(payload, bootstrap_repetitions, permutation_repetitions, generator):
    checkpoints = [model["name"] for model in payload["config"]["models"]]
    result = {}
    for target in checkpoints:
        discovery_rows = _eligible_rows(payload, "discovery", exclude_checkpoint=target)
        validation_rows = _eligible_rows(payload, "validation", exclude_checkpoint=target)
        final_rows = _eligible_rows(payload, "final", only_checkpoint=target)
        target_labels = _labels(final_rows)
        models = {}
        for name, features in (
            ("difficulty_output", BASELINE_FEATURES),
            ("difficulty_output_geometry", BASELINE_FEATURES + GEOMETRY_FEATURES),
        ):
            design = prepare_design(discovery_rows, validation_rows, final_rows, features, False)
            fit = fit_prepared(design, _labels(discovery_rows), _labels(validation_rows))
            metrics = probability_metrics(target_labels, fit["final_probabilities"])
            metrics["bootstrap_95"] = _bootstrap_metrics(
                final_rows,
                target_labels,
                fit["final_probabilities"],
                bootstrap_repetitions,
                generator,
            )
            models[name] = {
                "metrics": metrics,
                "probabilities": fit["final_probabilities"],
                "calibration": {
                    "method": fit["calibration_method"],
                    "intercept": float(fit["calibration"][0]),
                    "slope": float(fit["calibration"][1]),
                },
            }
        observed_improvement = (
            models["difficulty_output"]["metrics"]["log_loss"]
            - models["difficulty_output_geometry"]["metrics"]["log_loss"]
        )
        strata = np.asarray([row["rating_bucket"] for row in final_rows], dtype=object)
        null = []
        if target_labels.sum() and target_labels.sum() < len(target_labels):
            for _ in range(permutation_repetitions):
                permuted = stratified_permutation(target_labels, strata, generator)
                base_loss = probability_metrics(permuted, models["difficulty_output"]["probabilities"])["log_loss"]
                full_loss = probability_metrics(permuted, models["difficulty_output_geometry"]["probabilities"])["log_loss"]
                null.append(base_loss - full_loss)
        result[target] = {
            "source_checkpoints": [name for name in checkpoints if name != target],
            "final_n": int(len(target_labels)),
            "final_events": int(target_labels.sum()),
            "models": {
                name: {key: value for key, value in model.items() if key != "probabilities"}
                for name, model in models.items()
            },
            "geometry_log_loss_improvement": observed_improvement,
            "permutation_p_one_sided": (
                float((1 + np.sum(np.asarray(null) >= observed_improvement)) / (1 + len(null)))
                if null else None
            ),
            "permutation_repetitions": len(null),
            "probabilities": {
                name: model["probabilities"] for name, model in models.items()
            },
        }
    return result


def _event_counts(payload):
    counts = {}
    for checkpoint in [model["name"] for model in payload["config"]["models"]]:
        counts[checkpoint] = {}
        for split in SPLIT_NAMES:
            rows = [
                row for row in payload["rows"]
                if row["checkpoint"] == checkpoint and row["split"] == split
            ]
            eligible = [row for row in rows if row["eligible"]]
            counts[checkpoint][split] = {
                "sampled": len(rows),
                "eligible": len(eligible),
                "events": int(sum(row["collapse"] for row in eligible)),
                "event_rate": (
                    float(np.mean([row["collapse"] for row in eligible]))
                    if eligible else None
                ),
            }
    return counts


def _save_plots(output_dir, metrics, pooled_fits, final_rows, final_labels, transfer):
    artifact_names = []
    colors = {
        "stable_plain": "#2a9d8f",
        "collapsed_plain": "#e76f51",
        "late_state_ce": "#457b9d",
        "combined_margin": "#7b2cbf",
    }

    figure, axis = plt.subplots(figsize=(9, 4.8))
    checkpoint_names = list(metrics["event_counts"])
    x = np.arange(len(checkpoint_names))
    offsets = (-0.22, 0, 0.22)
    for offset, split in zip(offsets, SPLIT_NAMES):
        rates = [metrics["event_counts"][name][split]["event_rate"] for name in checkpoint_names]
        axis.bar(x + offset, rates, width=0.2, label=split)
        for position, rate, name in zip(x + offset, rates, checkpoint_names):
            count = metrics["event_counts"][name][split]["events"]
            axis.text(position, rate + 0.025, str(count), ha="center", va="bottom", fontsize=8)
    axis.set_xticks(x, [name.replace("_", "\n") for name in checkpoint_names])
    axis.set_ylabel("Collapse rate among puzzles solved at 128")
    axis.set_ylim(0, 1.08)
    axis.set_title("Late-collapse events are concentrated in one checkpoint")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = os.path.join(output_dir, "event_rates.png")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    artifact_names.append(os.path.basename(path))

    selected_models = (
        "difficulty_output",
        "geometry",
        "shuffled_iteration_geometry",
        "difficulty_output_geometry",
        "checkpoint_identity",
        PRIMARY_BASE_MODEL,
        SHUFFLED_FULL_MODEL,
        PRIMARY_FULL_MODEL,
    )
    display_names = {
        "difficulty_output": "difficulty + output",
        "geometry": "ordered geometry",
        "shuffled_iteration_geometry": "shuffled-order geometry",
        "difficulty_output_geometry": "baseline + ordered geometry",
        "checkpoint_identity": "checkpoint identity",
        PRIMARY_BASE_MODEL: "checkpoint + baseline",
        SHUFFLED_FULL_MODEL: "checkpoint + baseline + shuffled geometry",
        PRIMARY_FULL_MODEL: "checkpoint + baseline + ordered geometry",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, metric_name, title in (
        (axes[0], "auroc", "AUROC"),
        (axes[1], "average_precision", "Average precision"),
    ):
        values = [metrics["pooled_models"][name][metric_name] for name in selected_models]
        positions = np.arange(len(selected_models))
        axis.barh(positions, values, color="#4c78a8")
        axis.set_yticks(positions, [display_names[name] for name in selected_models])
        axis.invert_yaxis()
        axis.set_xlim(0, 1)
        axis.set_title(title)
        reference = 0.5 if metric_name == "auroc" else metrics["pooled_final_prevalence"]
        axis.axvline(reference, color="#777", linestyle=":", alpha=0.7)
        axis.grid(axis="x", alpha=0.25)
    figure.suptitle("Pooled final-puzzle performance (checkpoint-confounded)")
    figure.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    artifact_names.append(os.path.basename(path))

    figure, axis = plt.subplots(figsize=(6.4, 5.4))
    axis.plot([0, 1], [0, 1], color="#777", linestyle="--", label="perfect calibration")
    for model_name, color in ((PRIMARY_BASE_MODEL, "#e76f51"), (PRIMARY_FULL_MODEL, "#2a9d8f")):
        bins = pooled_fits[model_name]["calibration_bins"]
        axis.plot(
            [item["mean_probability"] for item in bins],
            [item["event_rate"] for item in bins],
            marker="o",
            color=color,
            label=model_name.replace("_", " "),
        )
    axis.set_xlabel("Calibrated predicted probability")
    axis.set_ylabel("Observed collapse rate")
    axis.set_xlim(-0.02, 1.02)
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Pooled final reliability (checkpoint-confounded)")
    axis.legend(frameon=False)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    path = os.path.join(output_dir, "calibration.png")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    artifact_names.append(os.path.basename(path))

    coefficient_map = pooled_fits[PRIMARY_FULL_MODEL]["coefficients_standardized"]
    coefficient_values = [coefficient_map[name] for name in GEOMETRY_FEATURES]
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    positions = np.arange(len(GEOMETRY_FEATURES))
    axis.barh(
        positions,
        coefficient_values,
        color=["#d1495b" if value > 0 else "#2a9d8f" for value in coefficient_values],
    )
    axis.set_yticks(positions, [name.replace("_", " ") for name in GEOMETRY_FEATURES])
    axis.invert_yaxis()
    axis.axvline(0, color="#333", linewidth=0.8)
    axis.set_xlabel("Discovery-fit coefficient per 1 SD (log odds)")
    axis.set_title("Discovery coefficients; descriptive under checkpoint separation")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    path = os.path.join(output_dir, "geometry_coefficients.png")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    artifact_names.append(os.path.basename(path))

    figure, axis = plt.subplots(figsize=(9, 4.8))
    checkpoint_names = list(transfer)
    positions = np.arange(len(checkpoint_names))
    width = 0.32
    for offset, model_name, color in (
        (-width / 2, "difficulty_output", "#e76f51"),
        (width / 2, "difficulty_output_geometry", "#2a9d8f"),
    ):
        values = [transfer[name]["models"][model_name]["metrics"]["auroc"] for name in checkpoint_names]
        display = [value if value is not None else 0 for value in values]
        axis.bar(positions + offset, display, width=width, color=color, label=model_name.replace("_", " "))
    for position, name in zip(positions, checkpoint_names):
        axis.text(
            position,
            1.02,
            f"{transfer[name]['final_events']} events",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xticks(positions, [name.replace("_", "\n") for name in checkpoint_names])
    axis.set_ylabel("Leave-one-checkpoint-out AUROC")
    axis.set_ylim(0, 1.12)
    axis.set_title("Checkpoint transfer; missing bars have one outcome class")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path = os.path.join(output_dir, "checkpoint_transfer.png")
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    artifact_names.append(os.path.basename(path))
    return artifact_names


def analyze(collection_path, output_dir, *, permutation_repetitions=500, bootstrap_repetitions=500, seed=20260811):
    with open(collection_path) as handle:
        payload = json.load(handle)
    _validate_payload(payload)
    os.makedirs(output_dir, exist_ok=True)
    bootstrap_generator = np.random.default_rng(seed + 1)
    permutation_generator = np.random.default_rng(seed + 2)
    transfer_generator = np.random.default_rng(seed + 3)
    discovery_rows = _eligible_rows(payload, "discovery")
    validation_rows = _eligible_rows(payload, "validation")
    final_rows = _eligible_rows(payload, "final")
    final_labels = _labels(final_rows)
    pooled_fits = _fit_model_set(
        discovery_rows,
        validation_rows,
        final_rows,
        bootstrap_repetitions,
        bootstrap_generator,
    )
    permutation = _conditional_permutation_test(
        discovery_rows,
        validation_rows,
        final_rows,
        pooled_fits[PRIMARY_BASE_MODEL],
        pooled_fits[PRIMARY_FULL_MODEL],
        permutation_repetitions,
        permutation_generator,
    )
    transfer = _leave_one_checkpoint_out(
        payload,
        bootstrap_repetitions,
        permutation_repetitions,
        transfer_generator,
    )
    event_counts = _event_counts(payload)
    final_negative_count = int(len(final_labels) - final_labels.sum())
    metrics = {
        "format_version": 1,
        "analysis_config": {
            "collection": os.path.basename(collection_path),
            "seed": seed,
            "ridge_penalty": 1.0,
            "permutation_repetitions": permutation_repetitions,
            "bootstrap_repetitions": bootstrap_repetitions,
            "feature_selection": "none; all feature groups were fixed before collection",
            "probability_calibration": "one-parameter logit-offset calibration on validation puzzles",
        },
        "leakage_audit": {
            "puzzles_split_before_model_inference": True,
            "whole_puzzle_splits": True,
            "maximum_feature_iteration": payload["feature_manifest"]["maximum_iteration"],
            "targets_available_to_feature_extractor": payload["feature_manifest"]["uses_targets"],
            "fitted_projection_used": payload["feature_manifest"]["uses_fitted_projection"],
            "final_labels_used_for_fitting_or_calibration": False,
            "label_defined_from_iteration": payload["config"]["late_outcome_iteration"],
        },
        "event_counts": event_counts,
        "pooled_final_n": len(final_rows),
        "pooled_final_events": int(final_labels.sum()),
        "pooled_final_prevalence": float(final_labels.mean()),
        "pooled_models": {
            name: fit["metrics"] for name, fit in pooled_fits.items()
        },
        "model_details": {
            name: {
                "features": fit["features"],
                "coefficients_standardized": fit["coefficients_standardized"],
                "calibration": fit["calibration"],
                "calibration_bins": fit["calibration_bins"],
            }
            for name, fit in pooled_fits.items()
        },
        "primary_conditional_permutation": permutation,
        "iteration_order_control": {
            "ordered_model": PRIMARY_FULL_MODEL,
            "shuffled_model": SHUFFLED_FULL_MODEL,
            "ordered_minus_shuffled_auroc": (
                pooled_fits[PRIMARY_FULL_MODEL]["metrics"]["auroc"]
                - pooled_fits[SHUFFLED_FULL_MODEL]["metrics"]["auroc"]
            ),
            "shuffled_minus_ordered_log_loss": (
                pooled_fits[SHUFFLED_FULL_MODEL]["metrics"]["log_loss"]
                - pooled_fits[PRIMARY_FULL_MODEL]["metrics"]["log_loss"]
            ),
            "ordered_metrics": pooled_fits[PRIMARY_FULL_MODEL]["metrics"],
            "shuffled_metrics": pooled_fits[SHUFFLED_FULL_MODEL]["metrics"],
            "shuffle": "fixed label-independent within-puzzle permutation of snapshots 64,80,96,112,128",
        },
        "leave_one_checkpoint_out": {
            target: {
                key: value for key, value in result.items() if key != "probabilities"
            }
            for target, result in transfer.items()
        },
        "sample_size_limits": {
            "canonical_checkpoint_count": len(payload["config"]["models"]),
            "independently_collapsed_checkpoint_families": int(
                sum(
                    event_counts[name]["final"]["event_rate"] is not None
                    and event_counts[name]["final"]["event_rate"] > 0.20
                    for name in event_counts
                )
            ),
            "minimum_events_for_auc_0_70_lower_95_above_0_5_given_final_negatives": minimum_events_for_auc(
                0.70,
                max(final_negative_count, 2),
            ),
            "minimum_non_events_for_auc_0_70_lower_95_above_0_5_given_final_events": minimum_non_events_for_auc(
                0.70,
                max(int(final_labels.sum()), 2),
            ),
            "checkpoints_with_at_least_10_events_and_10_non_events_on_final": int(
                sum(
                    event_counts[name]["final"]["events"] >= 10
                    and (
                        event_counts[name]["final"]["eligible"]
                        - event_counts[name]["final"]["events"]
                    ) >= 10
                    for name in event_counts
                )
            ),
            "checkpoint_specific_calibration_rule_of_thumb_events": 100,
            "note": "Puzzle count cannot replace independent checkpoint families for a checkpoint-level transfer claim.",
        },
    }

    predictions = []
    for index, row in enumerate(final_rows):
        item = {
            "puzzle_hash": row["puzzle_hash"],
            "checkpoint": row["checkpoint"],
            "rating_bucket": row["rating_bucket"],
            "collapse": int(row["collapse"]),
            "pooled_probabilities": {
                name: float(fit["probabilities"][index])
                for name, fit in pooled_fits.items()
            },
        }
        target_indices = [
            target_index
            for target_index, target_row in enumerate(
                _eligible_rows(payload, "final", only_checkpoint=row["checkpoint"])
            )
            if target_row["puzzle_hash"] == row["puzzle_hash"]
        ]
        if target_indices:
            target_index = target_indices[0]
            item["leave_one_checkpoint_out_probabilities"] = {
                name: float(probabilities[target_index])
                for name, probabilities in transfer[row["checkpoint"]]["probabilities"].items()
            }
        predictions.append(item)

    artifact_names = _save_plots(
        output_dir,
        metrics,
        pooled_fits,
        final_rows,
        final_labels,
        transfer,
    )
    metrics["artifacts"] = artifact_names + ["index.html", "final_predictions.json"]
    _atomic_json(os.path.join(output_dir, "metrics.json"), metrics)
    _atomic_json(os.path.join(output_dir, "final_predictions.json"), predictions)
    cards = "".join(
        f'<section><h2>{name}</h2><img src="{name}" alt="{name}"></section>'
        for name in artifact_names
    )
    html = f"""<!doctype html><meta charset="utf-8"><title>Sotaku early warning</title>
<style>body{{font:15px system-ui;max-width:1200px;margin:28px auto;padding:0 20px;color:#202020}}img{{max-width:100%;border:1px solid #ddd}}section{{margin:32px 0}}code{{background:#f3f3f3;padding:2px 4px}}</style>
<h1>Iteration-128 early warning for iteration-1024 collapse</h1>
<p>All displayed results use untouched final puzzles. Features are target-free and end at iteration 128. Numerical results and uncertainty intervals are in <code>metrics.json</code>.</p>{cards}"""
    with open(os.path.join(output_dir, "index.html"), "w") as handle:
        handle.write(html)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--permutations", type=int, default=500)
    parser.add_argument("--bootstraps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260811)
    arguments = parser.parse_args()
    analyze(
        arguments.collection,
        arguments.output_dir,
        permutation_repetitions=arguments.permutations,
        bootstrap_repetitions=arguments.bootstraps,
        seed=arguments.seed,
    )


if __name__ == "__main__":
    main()
