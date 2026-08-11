"""Run the held-out digit-symmetry analysis and render its artifacts."""

from __future__ import annotations

from collections import Counter
import hashlib
import html
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from core import (
    DIGIT_COUNT,
    SPLIT_NAMES,
    all_digit_permutations,
    category_centroids,
    category_subspace,
    effect_fraction,
    fit_multivariate_linear,
    fit_order_probe,
    fit_ridge_decoder,
    pairwise_distance_matrix,
    permutation_correlations,
    procrustes_similarity,
    puzzle_equal_weights,
    puzzle_mean_accuracy,
    random_orthonormal_basis,
    rdm_correlation,
    select_rdm_permutation,
    select_ridge_alpha,
    stratified_three_way_split,
    subspace_overlap,
    summarize_null,
    weighted_sse,
)
from looping.eval_loop_diagnostics import _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS
from looping.trajectory_viz.helix_tests.statistics.trajectory_data import (
    build_response_representations,
    collect_trajectory,
    load_balanced_trajectory_sample,
)


ITERATIONS = (16, 128, 512, 1024)
RIDGE_CANDIDATES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_EXAMPLES_PER_BUCKET = 12
DEFAULT_SEED = 20260811
DEFAULT_SHUFFLE_COUNT = 199
DEFAULT_RANDOM_SUBSPACE_COUNT = 99


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _split_observations(values, targets, blank_mask, puzzle_split, split_name):
    """Flatten originally blank cells while retaining whole-puzzle IDs."""

    puzzle_indices = np.flatnonzero(puzzle_split == split_name)
    selected_values = values[puzzle_indices]
    selected_targets = targets[puzzle_indices]
    selected_blank = blank_mask[puzzle_indices]
    expanded_puzzles = np.broadcast_to(
        puzzle_indices[:, None], selected_blank.shape
    )
    return {
        "values": selected_values[selected_blank].astype(np.float64),
        "labels": selected_targets[selected_blank].astype(np.int64),
        "puzzle_ids": expanded_puzzles[selected_blank].astype(np.int64),
    }


def _model_output_accuracy(logits, targets, blank_mask, puzzle_split, split_name):
    puzzle_indices = np.flatnonzero(puzzle_split == split_name)
    selected_blank = blank_mask[puzzle_indices]
    predictions = logits[puzzle_indices].argmax(axis=-1)[selected_blank]
    labels = targets[puzzle_indices][selected_blank]
    puzzle_ids = np.broadcast_to(
        puzzle_indices[:, None], selected_blank.shape
    )[selected_blank]
    return puzzle_mean_accuracy(predictions, labels, puzzle_ids)


def _decoder_metrics(observations, *, generator, shuffle_count, random_count):
    discovery = observations["discovery"]
    validation = observations["validation"]
    final = observations["final"]
    discovery_weights = puzzle_equal_weights(discovery["puzzle_ids"])

    full_selection = select_ridge_alpha(
        discovery["values"],
        discovery["labels"],
        discovery_weights,
        validation["values"],
        validation["labels"],
        validation["puzzle_ids"],
        candidates=RIDGE_CANDIDATES,
    )
    full_decoder = full_selection.pop("decoder")
    final_accuracy = puzzle_mean_accuracy(
        full_decoder.predict(final["values"]),
        final["labels"],
        final["puzzle_ids"],
    )

    discovery_centroids = category_centroids(
        discovery["values"], discovery["labels"], discovery["puzzle_ids"]
    )
    digit_basis = category_subspace(discovery_centroids)
    projected = {
        split_name: {
            **split,
            "values": split["values"] @ digit_basis,
        }
        for split_name, split in observations.items()
    }
    category_selection = select_ridge_alpha(
        projected["discovery"]["values"],
        projected["discovery"]["labels"],
        discovery_weights,
        projected["validation"]["values"],
        projected["validation"]["labels"],
        projected["validation"]["puzzle_ids"],
        candidates=RIDGE_CANDIDATES,
    )
    category_decoder = category_selection.pop("decoder")
    category_final_accuracy = puzzle_mean_accuracy(
        category_decoder.predict(projected["final"]["values"]),
        projected["final"]["labels"],
        projected["final"]["puzzle_ids"],
    )

    label_shuffle_accuracies = []
    for _ in range(shuffle_count):
        shuffled_labels = generator.permutation(discovery["labels"])
        decoder = fit_ridge_decoder(
            discovery["values"],
            shuffled_labels,
            discovery_weights,
            alpha=full_selection["alpha"],
        )
        label_shuffle_accuracies.append(
            puzzle_mean_accuracy(
                decoder.predict(final["values"]),
                final["labels"],
                final["puzzle_ids"],
            )
        )

    random_subspace_accuracies = []
    for _ in range(random_count):
        random_basis = random_orthonormal_basis(
            discovery["values"].shape[1], digit_basis.shape[1], generator
        )
        decoder = fit_ridge_decoder(
            discovery["values"] @ random_basis,
            discovery["labels"],
            discovery_weights,
            alpha=category_selection["alpha"],
        )
        random_subspace_accuracies.append(
            puzzle_mean_accuracy(
                decoder.predict(final["values"] @ random_basis),
                final["labels"],
                final["puzzle_ids"],
            )
        )

    return {
        "full_hidden": {
            **full_selection,
            "final_accuracy": final_accuracy,
            "label_shuffle_control": summarize_null(
                final_accuracy, label_shuffle_accuracies
            ),
        },
        "discovery_category_subspace": {
            **category_selection,
            "rank": int(digit_basis.shape[1]),
            "final_accuracy": category_final_accuracy,
            "random_subspace_control": summarize_null(
                category_final_accuracy, random_subspace_accuracies
            ),
        },
    }


def _evaluation_baseline(discovery_values, discovery_weights, evaluation_values):
    mean = np.sum(discovery_weights[:, None] * discovery_values, axis=0)
    return np.broadcast_to(mean, evaluation_values.shape)


def _order_probe_metrics(observations, *, shuffled_orders):
    discovery = observations["discovery"]
    discovery_weights = puzzle_equal_weights(discovery["puzzle_ids"])

    split_statistics = {}
    for split_name in ("validation", "final"):
        evaluation = observations[split_name]
        evaluation_weights = puzzle_equal_weights(evaluation["puzzle_ids"])
        baseline = _evaluation_baseline(
            discovery["values"], discovery_weights, evaluation["values"]
        )
        base_sse = weighted_sse(
            evaluation["values"], baseline, evaluation_weights
        )
        categorical_sse = fit_order_probe(
            discovery["values"],
            discovery["labels"],
            discovery_weights,
            evaluation["values"],
            evaluation["labels"],
            evaluation_weights,
            kind="categorical",
        )
        split_statistics[split_name] = {
            "weights": evaluation_weights,
            "base_sse": base_sse,
            "categorical_sse": categorical_sse,
            "categorical_partial_r2": 1.0 - categorical_sse / base_sse,
        }

    result = {
        "validation_categorical_partial_r2": split_statistics["validation"][
            "categorical_partial_r2"
        ],
        "final_categorical_partial_r2": split_statistics["final"][
            "categorical_partial_r2"
        ],
    }
    natural_order = np.arange(DIGIT_COUNT)
    for kind in ("ordinal", "cyclic"):
        split_fractions = {"validation": [], "final": []}
        natural = {}
        for split_name in ("validation", "final"):
            evaluation = observations[split_name]
            statistics = split_statistics[split_name]
            natural_sse = fit_order_probe(
                discovery["values"],
                discovery["labels"],
                discovery_weights,
                evaluation["values"],
                evaluation["labels"],
                statistics["weights"],
                kind=kind,
                order=natural_order,
            )
            natural[split_name] = effect_fraction(
                base_sse=statistics["base_sse"],
                model_sse=natural_sse,
                categorical_sse=statistics["categorical_sse"],
            )
            for order in shuffled_orders:
                shuffled_sse = fit_order_probe(
                    discovery["values"],
                    discovery["labels"],
                    discovery_weights,
                    evaluation["values"],
                    evaluation["labels"],
                    statistics["weights"],
                    kind=kind,
                    order=order,
                )
                split_fractions[split_name].append(
                    effect_fraction(
                        base_sse=statistics["base_sse"],
                        model_sse=shuffled_sse,
                        categorical_sse=statistics["categorical_sse"],
                    )
                )

        best_validation_index = int(np.nanargmax(split_fractions["validation"]))
        result[kind] = {
            "natural_validation_effect_fraction": natural["validation"],
            "natural_final_effect_fraction": natural["final"],
            "natural_order_final_control": summarize_null(
                natural["final"], split_fractions["final"]
            ),
            "best_validation_shuffled_order": shuffled_orders[
                best_validation_index
            ].tolist(),
            "best_validation_shuffled_effect_fraction": split_fractions[
                "validation"
            ][best_validation_index],
            "selected_shuffle_final_effect_fraction": split_fractions["final"][
                best_validation_index
            ],
        }
    return result


def _random_subspace_control(first_basis, second_rank, *, generator, count):
    null = []
    for _ in range(count):
        random_basis = random_orthonormal_basis(
            first_basis.shape[0], second_rank, generator
        )
        null.append(subspace_overlap(first_basis, random_basis))
    return null


def _alignment_record(
    source_by_split,
    target_by_split,
    *,
    all_permutations,
    control_permutations,
    generator,
    random_subspace_count,
):
    """Select a digit permutation on discovery and evaluate it unchanged."""

    selected, discovery_best = select_rdm_permutation(
        source_by_split["discovery"],
        target_by_split["discovery"],
        all_permutations,
    )
    split_metrics = {}
    for split_name in SPLIT_NAMES:
        source = source_by_split[split_name]
        target = target_by_split[split_name]
        split_metrics[split_name] = {
            "identity_rdm_correlation": rdm_correlation(source, target),
            "selected_rdm_correlation": rdm_correlation(
                source, target, selected
            ),
            "identity_procrustes_similarity": procrustes_similarity(
                source, target
            ),
            "selected_procrustes_similarity": procrustes_similarity(
                source, target, selected
            ),
        }

    final_source = source_by_split["final"]
    final_target = target_by_split["final"]
    final_control = permutation_correlations(
        pairwise_distance_matrix(final_source),
        pairwise_distance_matrix(final_target),
        control_permutations,
    )
    oracle_permutation, oracle_correlation = select_rdm_permutation(
        final_source, final_target, all_permutations
    )
    first_basis = category_subspace(final_source)
    second_basis = category_subspace(final_target)
    overlap = subspace_overlap(first_basis, second_basis)
    random_overlap = _random_subspace_control(
        first_basis,
        second_basis.shape[1],
        generator=generator,
        count=random_subspace_count,
    )
    return {
        "permutation_selection_split": "discovery",
        "selected_permutation": selected.tolist(),
        "selected_is_identity": bool(np.array_equal(selected, np.arange(9))),
        "discovery_oracle_rdm_correlation": discovery_best,
        "splits": split_metrics,
        "final_selected_permutation_control": summarize_null(
            split_metrics["final"]["selected_rdm_correlation"], final_control
        ),
        "final_oracle_permutation": oracle_permutation.tolist(),
        "final_oracle_rdm_correlation": oracle_correlation,
        "final_subspace_overlap": summarize_null(overlap, random_overlap),
    }


def _puzzle_transfer_record(
    centroids_by_split,
    *,
    all_permutations,
    control_permutations,
    generator,
    random_subspace_count,
):
    """Choose a template relabeling on validation, then test final puzzles."""

    discovery = centroids_by_split["discovery"]
    validation = centroids_by_split["validation"]
    final = centroids_by_split["final"]
    selected, validation_best = select_rdm_permutation(
        discovery, validation, all_permutations
    )
    selected_final = rdm_correlation(discovery, final, selected)
    final_controls = permutation_correlations(
        pairwise_distance_matrix(discovery),
        pairwise_distance_matrix(final),
        control_permutations,
    )
    oracle_permutation, oracle_correlation = select_rdm_permutation(
        discovery, final, all_permutations
    )
    discovery_basis = category_subspace(discovery)
    final_basis = category_subspace(final)
    overlap = subspace_overlap(discovery_basis, final_basis)
    random_overlap = _random_subspace_control(
        discovery_basis,
        final_basis.shape[1],
        generator=generator,
        count=random_subspace_count,
    )
    return {
        "permutation_selection_split": "validation",
        "selected_permutation": selected.tolist(),
        "selected_is_identity": bool(np.array_equal(selected, np.arange(9))),
        "validation_oracle_rdm_correlation": validation_best,
        "final_identity_rdm_correlation": rdm_correlation(discovery, final),
        "final_selected_rdm_correlation": selected_final,
        "final_identity_procrustes_similarity": procrustes_similarity(
            discovery, final
        ),
        "final_selected_procrustes_similarity": procrustes_similarity(
            discovery, final, selected
        ),
        "final_selected_permutation_control": summarize_null(
            selected_final, final_controls
        ),
        "final_oracle_permutation": oracle_permutation.tolist(),
        "final_oracle_rdm_correlation": oracle_correlation,
        "final_subspace_overlap": summarize_null(overlap, random_overlap),
    }


def _make_shuffled_orders(generator, count):
    natural = tuple(range(DIGIT_COUNT))
    orders = set()
    while len(orders) < count:
        candidate = tuple(generator.permutation(DIGIT_COUNT).tolist())
        if candidate != natural:
            orders.add(candidate)
    return np.asarray(sorted(orders), dtype=np.int64)


def _annotated_heatmap(axis, values, row_labels, column_labels, title, *, fmt):
    image = axis.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axis.set_xticks(range(len(column_labels)), column_labels)
    axis.set_yticks(range(len(row_labels)), row_labels)
    axis.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if values[row, column] < 0.55 else "black"
            axis.text(
                column,
                row,
                format(values[row, column], fmt),
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )
    return image


def _render_decoder_plot(metrics, output_dir):
    model_names = list(metrics["models"])
    iteration_names = [str(iteration) for iteration in ITERATIONS]
    full = np.array(
        [
            [
                metrics["models"][model][iteration]["decoder"]["full_hidden"][
                    "final_accuracy"
                ]
                for iteration in iteration_names
            ]
            for model in model_names
        ]
    )
    category = np.array(
        [
            [
                metrics["models"][model][iteration]["decoder"][
                    "discovery_category_subspace"
                ]["final_accuracy"]
                for iteration in iteration_names
            ]
            for model in model_names
        ]
    )
    output = np.array(
        [
            [
                metrics["models"][model][iteration]["model_output_final_accuracy"]
                for iteration in iteration_names
            ]
            for model in model_names
        ]
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (full, category, output),
        (
            "Linear decoder from all 128 features",
            "Decoder from discovery digit subspace",
            "Model output accuracy",
        ),
    ):
        image = _annotated_heatmap(
            axis, values, model_names, iteration_names, title, fmt=".2f"
        )
        axis.set_xlabel("Recurrent iteration")
    figure.colorbar(image, ax=axes, label="Final puzzle-mean accuracy", shrink=0.8)
    figure.suptitle("Digit information transfers to unseen puzzles")
    figure.savefig(os.path.join(output_dir, "decoder_transfer.png"), dpi=180)
    plt.close(figure)


def _render_order_plot(metrics, output_dir):
    model_names = list(metrics["models"])
    x = np.arange(len(model_names))
    width = 0.35
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for offset, kind, color in (
        (-width / 2, "ordinal", "#3B82F6"),
        (width / 2, "cyclic", "#E4572E"),
    ):
        records = [
            metrics["models"][model]["1024"]["order_probes"][kind][
                "natural_order_final_control"
            ]
            for model in model_names
        ]
        values = [record["observed"] for record in records]
        low = [record["null_q05"] for record in records]
        high = [record["null_q95"] for record in records]
        axis.bar(x + offset, values, width, color=color, label=kind.capitalize())
        axis.vlines(x + offset, low, high, color="black", linewidth=2)
    axis.axhline(0, color="#555555", linewidth=1)
    axis.set_xticks(x, model_names)
    axis.set_ylabel("Fraction of categorical representation effect")
    axis.set_title(
        "Natural 1–9 order at iteration 1024; black lines are shuffled-order 5–95%"
    )
    axis.legend()
    figure.savefig(os.path.join(output_dir, "order_controls.png"), dpi=180)
    plt.close(figure)


def _render_alignment_plot(metrics, output_dir):
    model_names = list(metrics["models"])
    targets = (128, 512, 1024)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for model in model_names:
        values = [
            metrics["alignment"]["across_iterations"][model][f"16_to_{target}"][
                "splits"
            ]["final"]["selected_rdm_correlation"]
            for target in targets
        ]
        axes[0].plot(targets, values, marker="o", label=model)
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(targets, [str(value) for value in targets])
    axes[0].set_ylim(-0.25, 1.03)
    axes[0].set_xlabel("Target iteration (source is 16)")
    axes[0].set_ylabel("Final RDM correlation")
    axes[0].set_title("Digit geometry across iterations")
    axes[0].legend(fontsize=8)

    comparison_names = [name for name in model_names if name != "stable_plain"]
    identity = []
    selected = []
    for model in comparison_names:
        record = metrics["alignment"]["across_checkpoints"][
            f"stable_plain_to_{model}"
        ]["1024"]["splits"]["final"]
        identity.append(record["identity_rdm_correlation"])
        selected.append(record["selected_rdm_correlation"])
    positions = np.arange(len(comparison_names))
    axes[1].bar(positions - 0.18, identity, 0.36, label="Known digit labels")
    axes[1].bar(positions + 0.18, selected, 0.36, label="Discovery-selected permutation")
    axes[1].set_xticks(positions, comparison_names, rotation=12)
    axes[1].set_ylim(-0.25, 1.03)
    axes[1].set_ylabel("Final RDM correlation")
    axes[1].set_title("Checkpoint geometry at iteration 1024")
    axes[1].legend(fontsize=8)
    figure.savefig(os.path.join(output_dir, "geometry_alignment.png"), dpi=180)
    plt.close(figure)


def _render_rdm_plot(metrics, centroids, output_dir):
    model_names = list(metrics["models"])
    figure, axes = plt.subplots(1, len(model_names), figsize=(15, 3.8), constrained_layout=True)
    for axis, model in zip(axes, model_names):
        distances = pairwise_distance_matrix(centroids[model][1024]["final"])
        scale = np.median(distances[distances > 0])
        image = axis.imshow(distances / scale, cmap="magma", vmin=0, vmax=2)
        axis.set_xticks(range(9), range(1, 10), fontsize=7)
        axis.set_yticks(range(9), range(1, 10), fontsize=7)
        axis.set_title(model)
        axis.set_xlabel("Digit label (categorical)")
    axes[0].set_ylabel("Digit label (categorical)")
    figure.colorbar(image, ax=axes, label="Distance / median distance", shrink=0.8)
    figure.suptitle("Final-puzzle digit-centroid distances at iteration 1024")
    figure.savefig(os.path.join(output_dir, "digit_distance_matrices.png"), dpi=180)
    plt.close(figure)


def _render_html(metrics, output_dir):
    rows = []
    for model, iteration_records in metrics["models"].items():
        for iteration, record in iteration_records.items():
            decoder = record["decoder"]["full_hidden"]["final_accuracy"]
            ordinal = record["order_probes"]["ordinal"][
                "natural_order_final_control"
            ]
            cyclic = record["order_probes"]["cyclic"][
                "natural_order_final_control"
            ]
            rows.append(
                "<tr>"
                f"<td>{html.escape(model)}</td><td>{iteration}</td>"
                f"<td>{decoder:.3f}</td>"
                f"<td>{ordinal['observed']:.3f}</td><td>{ordinal['percentile']:.3f}</td>"
                f"<td>{cyclic['observed']:.3f}</td><td>{cyclic['percentile']:.3f}</td>"
                "</tr>"
            )
    document = f"""<!doctype html>
<meta charset="utf-8">
<title>Sotaku digit-symmetry study</title>
<style>
body {{ font: 15px system-ui; max-width: 1100px; margin: 32px auto; color: #202124; }}
img {{ width: 100%; margin: 16px 0 28px; border: 1px solid #ddd; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 7px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
<h1>Digit symmetry and representation alignment</h1>
<p>All reported numbers below evaluate projections fitted on discovery puzzles on the final puzzle split.</p>
<img src="decoder_transfer.png" alt="Decoder transfer heatmaps">
<img src="order_controls.png" alt="Natural versus shuffled digit orders">
<img src="geometry_alignment.png" alt="Geometry transfer">
<img src="digit_distance_matrices.png" alt="Digit distance matrices">
<h2>Held-out summary</h2>
<table><thead><tr><th>Checkpoint</th><th>Iteration</th><th>Decoder accuracy</th><th>Ordinal fraction</th><th>Ordinal percentile</th><th>Cyclic fraction</th><th>Cyclic percentile</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
"""
    with open(os.path.join(output_dir, "index.html"), "w") as file:
        file.write(document)


def run(
    *,
    output_dir,
    examples_per_bucket=DEFAULT_EXAMPLES_PER_BUCKET,
    seed=DEFAULT_SEED,
    shuffle_count=DEFAULT_SHUFFLE_COUNT,
    random_subspace_count=DEFAULT_RANDOM_SUBSPACE_COUNT,
    device="cuda",
):
    """Execute the prespecified study on all four checkpoints."""

    if examples_per_bucket < 12 or examples_per_bucket % 3:
        raise ValueError(
            "examples_per_bucket must be a multiple of three and at least 12"
        )
    if shuffle_count < 19 or random_subspace_count < 19:
        raise ValueError("control counts must each be at least 19")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    log_file = open(log_path, "w")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    started = time.time()
    resolved_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    log(
        f"config seed={seed} examples_per_bucket={examples_per_bucket} "
        f"iterations={ITERATIONS} shuffle_count={shuffle_count} "
        f"random_subspace_count={random_subspace_count} device={resolved_device}"
    )
    log("models=" + ",".join(model["name"] for model in DEFAULT_MODELS))

    sample = load_balanced_trajectory_sample(examples_per_bucket, seed)
    puzzle_split = stratified_three_way_split(sample.rating_buckets, seed=seed + 1)
    split_counts = Counter(puzzle_split.tolist())
    log(f"split_counts={dict(split_counts)}")
    generator = np.random.default_rng(seed + 2)
    shuffled_orders = _make_shuffled_orders(generator, shuffle_count)
    all_permutations = all_digit_permutations()
    control_indices = generator.choice(
        len(all_permutations), size=shuffle_count, replace=False
    )
    control_permutations = all_permutations[control_indices]

    targets = sample.targets.numpy()
    blank_mask = sample.originally_blank.numpy()
    metrics = {
        "config": {
            "seed": seed,
            "examples_per_bucket": examples_per_bucket,
            "puzzle_count": sample.puzzle_count,
            "iterations": list(ITERATIONS),
            "ridge_candidates": list(RIDGE_CANDIDATES),
            "shuffle_count": shuffle_count,
            "random_subspace_count": random_subspace_count,
            "representation": "cell-centered unit hidden-state direction",
            "cells": "originally blank cells only",
            "models": list(DEFAULT_MODELS),
            "device": str(resolved_device),
        },
        "data_split": {
            split_name: {
                "puzzle_indices": np.flatnonzero(puzzle_split == split_name).tolist(),
                "bucket_counts": dict(
                    Counter(
                        np.asarray(sample.rating_buckets)[
                            puzzle_split == split_name
                        ].tolist()
                    )
                ),
                "puzzle_sha256": [
                    hashlib.sha256(sample.puzzles[index].encode()).hexdigest()
                    for index in np.flatnonzero(puzzle_split == split_name)
                ],
            }
            for split_name in SPLIT_NAMES
        },
        "models": {},
        "alignment": {
            "across_puzzle_splits": {},
            "across_iterations": {},
            "across_checkpoints": {},
        },
    }
    centroids = {}

    for model_index, model_config in enumerate(DEFAULT_MODELS):
        model_name = model_config["name"]
        log(f"loading model={model_name} path={model_config['path']}")
        model = _load_model(model_config, resolved_device)
        device_sample = sample.to(resolved_device)
        trajectory = collect_trajectory(
            model,
            device_sample.inputs,
            device_sample.targets,
            device_sample.originally_blank,
            iterations=ITERATIONS,
            output_device="cpu",
        )
        responses = build_response_representations(trajectory, model.output_head)
        hidden_directions = (
            responses.cell_centered_unit_hidden_direction.numpy()
        )
        logits = trajectory.logits.numpy()
        del model, trajectory, responses, device_sample
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

        metrics["models"][model_name] = {}
        centroids[model_name] = {}
        for time_index, iteration in enumerate(ITERATIONS):
            log(f"analyzing model={model_name} iteration={iteration}")
            observations = {
                split_name: _split_observations(
                    hidden_directions[:, time_index],
                    targets,
                    blank_mask,
                    puzzle_split,
                    split_name,
                )
                for split_name in SPLIT_NAMES
            }
            centroids[model_name][iteration] = {
                split_name: category_centroids(
                    split["values"], split["labels"], split["puzzle_ids"]
                )
                for split_name, split in observations.items()
            }
            decoder = _decoder_metrics(
                observations,
                generator=np.random.default_rng(seed + 1000 * model_index + iteration),
                shuffle_count=shuffle_count,
                random_count=random_subspace_count,
            )
            order_probes = _order_probe_metrics(
                observations, shuffled_orders=shuffled_orders
            )
            iteration_key = str(iteration)
            metrics["models"][model_name][iteration_key] = {
                "model_output_final_accuracy": _model_output_accuracy(
                    logits[:, time_index],
                    targets,
                    blank_mask,
                    puzzle_split,
                    "final",
                ),
                "decoder": decoder,
                "order_probes": order_probes,
            }
            metrics["alignment"]["across_puzzle_splits"].setdefault(
                model_name, {}
            )[iteration_key] = _puzzle_transfer_record(
                centroids[model_name][iteration],
                all_permutations=all_permutations,
                control_permutations=control_permutations,
                generator=np.random.default_rng(
                    seed + 2000 * model_index + iteration
                ),
                random_subspace_count=random_subspace_count,
            )

        metrics["alignment"]["across_iterations"][model_name] = {}
        for target_iteration in ITERATIONS[1:]:
            key = f"{ITERATIONS[0]}_to_{target_iteration}"
            metrics["alignment"]["across_iterations"][model_name][key] = (
                _alignment_record(
                    centroids[model_name][ITERATIONS[0]],
                    centroids[model_name][target_iteration],
                    all_permutations=all_permutations,
                    control_permutations=control_permutations,
                    generator=np.random.default_rng(
                        seed + 3000 * model_index + target_iteration
                    ),
                    random_subspace_count=random_subspace_count,
                )
            )

    reference_name = DEFAULT_MODELS[0]["name"]
    for model_index, model_config in enumerate(DEFAULT_MODELS[1:], start=1):
        model_name = model_config["name"]
        comparison_name = f"{reference_name}_to_{model_name}"
        metrics["alignment"]["across_checkpoints"][comparison_name] = {}
        for iteration in ITERATIONS:
            metrics["alignment"]["across_checkpoints"][comparison_name][
                str(iteration)
            ] = _alignment_record(
                centroids[reference_name][iteration],
                centroids[model_name][iteration],
                all_permutations=all_permutations,
                control_permutations=control_permutations,
                generator=np.random.default_rng(
                    seed + 4000 * model_index + iteration
                ),
                random_subspace_count=random_subspace_count,
            )

    metrics["runtime_seconds"] = time.time() - started
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as file:
        json.dump(_json_ready(metrics), file, indent=2, allow_nan=False)
    _render_decoder_plot(metrics, output_dir)
    _render_order_plot(metrics, output_dir)
    _render_alignment_plot(metrics, output_dir)
    _render_rdm_plot(metrics, centroids, output_dir)
    _render_html(metrics, output_dir)
    log(f"completed runtime_seconds={metrics['runtime_seconds']:.1f}")
    log_file.close()
    return _json_ready(metrics)


if __name__ == "__main__":
    run(output_dir=os.path.dirname(__file__), device="cpu")
