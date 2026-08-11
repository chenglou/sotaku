"""Held-out cell- and board-level probes of output decision margin.

The analysis implements ARM 02 of ``study/PROTOCOL.md``.  It samples and
splits complete puzzles before fitting, selects the response scale and ridge
strengths on validation puzzles, touches the final split once, and writes only
aggregate metrics and inspectable figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS

from core import (
    aggregate_rows_by_puzzle_iteration,
    assign_bins,
    continuous_metrics,
    deep_temporal_metrics,
    evaluate_categorical_probe,
    evaluate_ordered_probe,
    evaluate_transferred_axis,
    fit_ordered_probe,
    fit_transferred_axis,
    make_bin_thresholds,
    ordered_bin_metrics,
    ordered_fraction_of_categorical_gain,
    puzzle_equal_weights,
    random_rank_one_probe,
    select_categorical_alpha,
    select_ordered_alpha,
    shuffle_within_iteration,
    stratified_three_way_split,
    summarize_null,
)


SNAPSHOTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
RIDGE_CANDIDATES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
TARGET_NAMES = ("top1_top2_logit_margin", "top1_top2_probability_margin")
LEVEL_NAMES = ("cell", "board")
MODEL_LABELS = {
    "stable_plain": "stable plain",
    "collapsed_plain": "collapsed plain",
    "late_state_ce": "late-state CE",
    "combined_margin": "combined margin",
}
MODEL_COLORS = {
    "stable_plain": "#2878B5",
    "collapsed_plain": "#C84343",
    "late_state_ce": "#4A9B68",
    "combined_margin": "#8D61B5",
}


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(_json_ready(value), handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def _scalar_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def collect_snapshots(model, inputs, snapshots=SNAPSHOTS):
    """Collect hidden states and logits after the requested recurrent steps."""

    if not snapshots or tuple(sorted(set(snapshots))) != tuple(snapshots):
        raise ValueError("snapshots must be strictly increasing")
    if snapshots[0] < 0:
        raise ValueError("snapshots must be nonnegative")
    snapshot_set = set(snapshots)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    states = []
    logits = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            hidden = model.initial_encoder(inputs)
            predictions = torch.zeros(
                inputs.size(0), 81, 9, device=inputs.device, dtype=hidden.dtype
            )
            for iteration in range(snapshots[-1] + 1):
                current_logits = model.output_head(hidden)
                if iteration in snapshot_set:
                    states.append(hidden.detach().float().cpu())
                    logits.append(current_logits.detach().float().cpu())
                if iteration == snapshots[-1]:
                    break
                hidden = model.recurrent_step(
                    hidden, predictions, rope_cos, rope_sin
                )
                predictions = F.softmax(model.output_head(hidden), dim=-1)
    finally:
        model.train(was_training)
    return torch.stack(states, dim=1), torch.stack(logits, dim=1)


def _margin_tensors(logits):
    top_logits = logits.topk(2, dim=-1).values
    probabilities = F.softmax(logits, dim=-1)
    top_probabilities = probabilities.topk(2, dim=-1).values
    return {
        "top1_top2_logit_margin": top_logits[..., 0] - top_logits[..., 1],
        "top1_top2_probability_margin": (
            top_probabilities[..., 0] - top_probabilities[..., 1]
        ),
    }


def prepare_model_rows(
    states,
    logits,
    empty_mask,
    split_by_puzzle,
    rating_buckets,
    snapshots=SNAPSHOTS,
):
    """Construct fixed cell and board rows from one checkpoint's snapshots."""

    if states.ndim != 4 or states.shape[:3] != logits.shape[:3]:
        raise ValueError("states and logits must align as [puzzle,time,cell,...]")
    puzzle_count, time_count, cell_count, _ = states.shape
    if cell_count != 81 or logits.size(-1) != 9:
        raise ValueError("expected 81 cells and nine output logits")
    if tuple(snapshots) != tuple(sorted(set(snapshots))) or time_count != len(snapshots):
        raise ValueError("snapshots do not align with collected tensors")
    empty_mask = empty_mask.bool().cpu()
    if empty_mask.shape != (puzzle_count, cell_count):
        raise ValueError("empty_mask must have shape [puzzle,81]")

    puzzle_grid = torch.arange(puzzle_count).view(-1, 1, 1).expand(
        puzzle_count, time_count, cell_count
    )
    iteration_grid = torch.tensor(snapshots).view(1, -1, 1).expand_as(puzzle_grid)
    blank_grid = empty_mask[:, None, :].expand_as(puzzle_grid)
    split_array = np.asarray(split_by_puzzle, dtype=object)
    bucket_array = np.asarray(rating_buckets, dtype=object)
    margins = _margin_tensors(logits)

    cell_puzzles = puzzle_grid[blank_grid].numpy()
    cell_iterations = iteration_grid[blank_grid].numpy()
    cell_rows = {
        "features": states[blank_grid].numpy(),
        "puzzle_ids": cell_puzzles,
        "iterations": cell_iterations,
        "splits": split_array[cell_puzzles],
        "buckets": bucket_array[cell_puzzles],
        "targets": {
            name: values[blank_grid].numpy()
            for name, values in margins.items()
        },
    }

    blank_float = empty_mask.float()
    blank_counts = blank_float.sum(dim=1)
    if torch.any(blank_counts == 0):
        raise ValueError("every puzzle must contain an originally blank cell")
    board_features = (
        states * blank_float[:, None, :, None]
    ).sum(dim=2) / blank_counts[:, None, None]
    board_puzzles = torch.arange(puzzle_count).view(-1, 1).expand(
        puzzle_count, time_count
    )
    board_iterations = torch.tensor(snapshots).view(1, -1).expand_as(board_puzzles)
    flattened_board_puzzles = board_puzzles.reshape(-1).numpy()
    board_rows = {
        "features": board_features.reshape(-1, states.size(-1)).numpy(),
        "puzzle_ids": flattened_board_puzzles,
        "iterations": board_iterations.reshape(-1).numpy(),
        "splits": split_array[flattened_board_puzzles],
        "buckets": bucket_array[flattened_board_puzzles],
        "targets": {
            name: (
                (values * blank_float[:, None, :]).sum(dim=2)
                / blank_counts[:, None]
            ).reshape(-1).numpy()
            for name, values in margins.items()
        },
    }

    return {
        "cell": cell_rows,
        "board": board_rows,
    }


def _split_rows(rows, split_name, target_name):
    selected = rows["splits"] == split_name
    puzzle_ids = rows["puzzle_ids"][selected]
    if not len(puzzle_ids):
        raise ValueError(f"split {split_name!r} has no observations")
    return {
        "features": rows["features"][selected],
        "target": rows["targets"][target_name][selected],
        "iterations": rows["iterations"][selected],
        "puzzle_ids": puzzle_ids,
        "buckets": rows["buckets"][selected],
        "weights": puzzle_equal_weights(puzzle_ids),
    }


def _fit_target_candidate(model_rows, target_name):
    fits = {}
    validation_scores = []
    for level_name in LEVEL_NAMES:
        discovery = _split_rows(model_rows[level_name], "discovery", target_name)
        validation = _split_rows(model_rows[level_name], "validation", target_name)
        selection = select_ordered_alpha(
            discovery["features"],
            discovery["target"],
            discovery["iterations"],
            discovery["weights"],
            validation["features"],
            validation["target"],
            validation["iterations"],
            validation["weights"],
            candidates=RIDGE_CANDIDATES,
        )
        fits[level_name] = selection
        score = selection["validation_metrics"]["partial_r2_over_iteration"]
        if np.isfinite(score):
            validation_scores.append(score)
    return fits, validation_scores


def _cluster_bootstrap_final(
    target,
    prediction,
    baseline,
    puzzle_ids,
    buckets,
    *,
    labels=None,
    ordered_bins=None,
    categorical_bins=None,
    repetitions,
    seed,
):
    """Bootstrap final effects by puzzle within each rating bucket."""

    target = np.asarray(target)
    prediction = np.asarray(prediction)
    baseline = np.asarray(baseline)
    puzzle_ids = np.asarray(puzzle_ids)
    buckets = np.asarray(buckets, dtype=object)
    unique_puzzles = np.unique(puzzle_ids)
    puzzle_buckets = np.asarray(
        [buckets[np.flatnonzero(puzzle_ids == puzzle)[0]] for puzzle in unique_puzzles],
        dtype=object,
    )
    per_puzzle = []
    for puzzle in unique_puzzles:
        selected = puzzle_ids == puzzle
        record = {
            "sse": float(np.mean((target[selected] - prediction[selected]) ** 2)),
            "baseline_sse": float(np.mean((target[selected] - baseline[selected]) ** 2)),
        }
        if labels is not None:
            record.update(
                {
                    "ordered_accuracy": float(
                        np.mean(labels[selected] == ordered_bins[selected])
                    ),
                    "categorical_accuracy": float(
                        np.mean(labels[selected] == categorical_bins[selected])
                    ),
                }
            )
        per_puzzle.append(record)

    def effects(indices):
        records = [per_puzzle[index] for index in indices]
        sse = np.mean([record["sse"] for record in records])
        baseline_sse = np.mean([record["baseline_sse"] for record in records])
        result = {
            "partial_r2_over_iteration": (
                1.0 - sse / baseline_sse if baseline_sse > 1e-15 else math.nan
            )
        }
        if labels is not None:
            result["ordered_bin_accuracy"] = float(
                np.mean([record["ordered_accuracy"] for record in records])
            )
            result["categorical_bin_accuracy"] = float(
                np.mean([record["categorical_accuracy"] for record in records])
            )
        return result

    generator = np.random.default_rng(seed)
    samples = []
    bucket_order = tuple(dict.fromkeys(puzzle_buckets.tolist()))
    for _ in range(repetitions):
        sampled = []
        for bucket in bucket_order:
            group = np.flatnonzero(puzzle_buckets == bucket)
            sampled.extend(generator.choice(group, size=len(group), replace=True))
        samples.append(effects(sampled))
    observed = effects(np.arange(len(unique_puzzles)))
    intervals = {}
    for metric_name in observed:
        values = np.asarray([sample[metric_name] for sample in samples])
        values = values[np.isfinite(values)]
        intervals[metric_name] = {
            "estimate": observed[metric_name],
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
        }
    return intervals


def _run_controls(
    discovery,
    final,
    probe,
    observed_partial_r2,
    *,
    repetitions,
    seed,
):
    generator = np.random.default_rng(seed)
    shuffled_label_effects = []
    random_rank_effects = []
    for _ in range(repetitions):
        shuffled_target = shuffle_within_iteration(
            discovery["target"], discovery["iterations"], generator
        )
        shuffled_probe = fit_ordered_probe(
            discovery["features"],
            shuffled_target,
            discovery["iterations"],
            discovery["weights"],
            alpha=probe.alpha,
        )
        shuffled_metrics = evaluate_ordered_probe(
            shuffled_probe,
            final["features"],
            final["target"],
            final["iterations"],
            final["weights"],
        )
        shuffled_label_effects.append(
            shuffled_metrics["partial_r2_over_iteration"]
        )

        random_probe = random_rank_one_probe(
            probe,
            discovery["features"],
            discovery["target"],
            discovery["iterations"],
            discovery["weights"],
            generator,
        )
        random_metrics = evaluate_ordered_probe(
            random_probe,
            final["features"],
            final["target"],
            final["iterations"],
            final["weights"],
        )
        random_rank_effects.append(random_metrics["partial_r2_over_iteration"])
    return {
        "shuffled_labels_within_iteration": summarize_null(
            observed_partial_r2,
            shuffled_label_effects,
            larger_is_better=True,
        ),
        "random_matched_rank_one_subspace": summarize_null(
            observed_partial_r2,
            random_rank_effects,
            larger_is_better=True,
        ),
    }


def _mean_curve(values, puzzle_ids, iterations):
    puzzles, iteration_values, matrix = aggregate_rows_by_puzzle_iteration(
        values, puzzle_ids, iterations
    )
    return {
        "puzzle_ids": puzzles,
        "iterations": iteration_values,
        "matrix": matrix,
        "mean": matrix.mean(axis=0),
        "sem": matrix.std(axis=0, ddof=1) / math.sqrt(len(matrix)),
    }


def _plot_probe_performance(metrics, output_dir):
    models = list(metrics["within_checkpoint"])
    positions = np.arange(len(models))
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for column, level_name in enumerate(LEVEL_NAMES):
        partial = [
            metrics["within_checkpoint"][model][level_name]["final"]
            ["ordered_continuous"]["partial_r2_over_iteration"]
            for model in models
        ]
        axes[0, column].bar(
            positions,
            partial,
            color=[MODEL_COLORS[model] for model in models],
            alpha=0.9,
        )
        axes[0, column].axhline(0, color="#444444", linewidth=0.8)
        axes[0, column].set_title(f"{level_name.capitalize()} ordered coordinate")
        axes[0, column].set_ylabel("Final partial $R^2$ over iteration")
        axes[0, column].set_xticks(positions, [MODEL_LABELS[m] for m in models], rotation=18)

        ordered_gain = [
            metrics["within_checkpoint"][model][level_name]["final"]
            ["ordered_bins"]["accuracy_gain"]
            for model in models
        ]
        categorical_gain = [
            metrics["within_checkpoint"][model][level_name]["final"]
            ["categorical_bins"]["accuracy_gain"]
            for model in models
        ]
        width = 0.36
        axes[1, column].bar(
            positions - width / 2,
            ordered_gain,
            width,
            label="ordered scalar",
            color="#4C78A8",
        )
        axes[1, column].bar(
            positions + width / 2,
            categorical_gain,
            width,
            label="unconstrained categorical",
            color="#F58518",
        )
        axes[1, column].axhline(0, color="#444444", linewidth=0.8)
        axes[1, column].set_title(f"{level_name.capitalize()} five-bin decoding")
        axes[1, column].set_ylabel("Accuracy gain over iteration baseline")
        axes[1, column].set_xticks(positions, [MODEL_LABELS[m] for m in models], rotation=18)
        axes[1, column].legend(frameon=False, fontsize=9)
    figure.suptitle(
        f"Held-out margin probes: {metrics['selection']['selected_target_label']}"
    )
    figure.savefig(os.path.join(output_dir, "probe_performance.png"), dpi=190)
    plt.close(figure)


def _plot_controls(metrics, output_dir):
    models = list(metrics["within_checkpoint"])
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for axis, level_name in zip(axes, LEVEL_NAMES):
        positions = np.arange(len(models))
        for control_index, (control_name, label, color) in enumerate(
            (
                ("shuffled_labels_within_iteration", "shuffled labels", "#E45756"),
                ("random_matched_rank_one_subspace", "random rank-1", "#72B7B2"),
            )
        ):
            offset = (control_index - 0.5) * 0.18
            medians = []
            lower = []
            upper = []
            for model in models:
                summary = metrics["controls"][model][level_name][control_name]
                medians.append(summary["null_q50"])
                lower.append(summary["null_q50"] - summary["null_q05"])
                upper.append(summary["null_q95"] - summary["null_q50"])
            axis.errorbar(
                positions + offset,
                medians,
                yerr=np.asarray([lower, upper]),
                fmt="o",
                capsize=3,
                color=color,
                label=label,
            )
        observed = [
            metrics["within_checkpoint"][model][level_name]["final"]
            ["ordered_continuous"]["partial_r2_over_iteration"]
            for model in models
        ]
        axis.scatter(
            positions,
            observed,
            marker="*",
            s=120,
            color="#222222",
            label="observed",
            zorder=4,
        )
        axis.axhline(0, color="#777777", linewidth=0.8)
        axis.set_title(f"{level_name.capitalize()} matched controls")
        axis.set_ylabel("Final partial $R^2$")
        axis.set_xticks(positions, [MODEL_LABELS[m] for m in models], rotation=18)
        axis.legend(frameon=False, fontsize=9)
    figure.savefig(os.path.join(output_dir, "probe_controls.png"), dpi=190)
    plt.close(figure)


def _plot_transfer(metrics, output_dir):
    models = list(metrics["within_checkpoint"])
    figure, axes = plt.subplots(1, 2, figsize=(11.8, 5.1), constrained_layout=True)
    image = None
    for axis, level_name in zip(axes, LEVEL_NAMES):
        matrix = np.asarray(
            [
                [
                    metrics["cross_checkpoint_transfer"][level_name][source][target]
                    ["partial_r2_over_iteration"]
                    for target in models
                ]
                for source in models
            ]
        )
        image = axis.imshow(matrix, cmap="RdBu", vmin=-0.5, vmax=1.0)
        for row in range(len(models)):
            for column in range(len(models)):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if abs(value) > 0.42 else "black",
                )
        axis.set_title(f"{level_name.capitalize()} transferred axis")
        axis.set_xlabel("target checkpoint (validation-calibrated → final)")
        axis.set_ylabel("source checkpoint")
        axis.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models], rotation=25, ha="right")
        axis.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    figure.colorbar(image, ax=axes, label="Final partial $R^2$", shrink=0.8)
    figure.savefig(os.path.join(output_dir, "cross_checkpoint_transfer.png"), dpi=190)
    plt.close(figure)


def _plot_dynamics(metrics, curves, output_dir):
    models = list(metrics["within_checkpoint"])
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for row, level_name in enumerate(LEVEL_NAMES):
        for model in models:
            data = curves[model][level_name]
            iterations = data["iterations"]
            x = np.log2(np.asarray(iterations) + 1)
            true_mean = data["true_mean"]
            true_sem = data["true_sem"]
            coordinate_mean = data["coordinate_mean"]
            coordinate_sem = data["coordinate_sem"]
            deep_index = int(np.flatnonzero(np.asarray(iterations) == 128)[0])
            coordinate_mean = coordinate_mean - coordinate_mean[deep_index]
            axes[row, 0].plot(
                x,
                true_mean,
                marker="o",
                markersize=3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
            axes[row, 0].fill_between(
                x,
                true_mean - 1.96 * true_sem,
                true_mean + 1.96 * true_sem,
                color=MODEL_COLORS[model],
                alpha=0.12,
            )
            axes[row, 1].plot(
                x,
                coordinate_mean,
                marker="o",
                markersize=3,
                color=MODEL_COLORS[model],
                label=MODEL_LABELS[model],
            )
            axes[row, 1].fill_between(
                x,
                coordinate_mean - 1.96 * coordinate_sem,
                coordinate_mean + 1.96 * coordinate_sem,
                color=MODEL_COLORS[model],
                alpha=0.12,
            )
        axes[row, 0].axvline(np.log2(129), color="#777777", linestyle="--", linewidth=0.8)
        axes[row, 1].axvline(np.log2(129), color="#777777", linestyle="--", linewidth=0.8)
        axes[row, 0].set_title(f"{level_name.capitalize()} observed margin")
        axes[row, 1].set_title(f"{level_name.capitalize()} raw probe-axis motion from iter 128")
        axes[row, 0].set_ylabel("Mean margin")
        axes[row, 1].set_ylabel("Coordinate change")
        for axis in axes[row]:
            axis.set_xlabel("Iteration")
            axis.set_xticks(np.log2(np.asarray(SNAPSHOTS) + 1), [str(v) for v in SNAPSHOTS], rotation=35)
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9, ncol=2)
    figure.suptitle("Final-puzzle margin and learned-axis trajectories")
    figure.savefig(os.path.join(output_dir, "deep_margin_dynamics.png"), dpi=190)
    plt.close(figure)


def _generated_report(metrics):
    selected = metrics["selection"]["selected_target_label"]
    rows = []
    for model_name, result in metrics["within_checkpoint"].items():
        label = MODEL_LABELS.get(model_name, model_name)
        for level_name in LEVEL_NAMES:
            final = result[level_name]["final"]
            rows.append(
                "| "
                f"{label} | {level_name} | "
                f"{final['ordered_continuous']['partial_r2_over_iteration']:.3f} | "
                f"{final['ordered_bins']['accuracy']:.3f} | "
                f"{final['categorical_bins']['accuracy']:.3f} |"
            )
    table = "\n".join(rows)
    return f"""# ARM 02: output decision margin\n\n## Hypothesis\n\nRecurrent hidden states contain a shared ordered coordinate for the top-1 minus top-2 output margin. Healthy deep-horizon motion should preserve or increase useful margin, while collapsed motion should lose or reverse it.\n\n## Protocol\n\nThe analysis fixed and split {metrics['sample']['puzzle_count']} complete puzzles before fitting: 20 discovery, 20 validation, and 20 final puzzles, with four puzzles from each of five rating buckets in every split. Discovery fitted axes and quantile bins; validation selected between logit and probability margin and chose ridge strengths; the final split was evaluated once. The selected response was **{selected}**. Originally blank cells remained the fixed cell mask at every iteration.\n\nBoth cell hidden states and the mean hidden state over originally blank cells were tested. A rank-one ordered regression was compared with a five-bin unconstrained categorical ridge decoder. Controls used iteration-matched shuffled labels, random matched-rank one-dimensional subspaces, and shuffled deep-iteration order. Axes were also transferred between all four checkpoints, with only a scalar, orientation-preserving calibration on target validation puzzles.\n\n## Held-out results\n\n| Checkpoint | Level | Ordered partial R² | Ordered-bin accuracy | Categorical accuracy |\n|---|---:|---:|---:|---:|\n{table}\n\nSee `metrics.json` for confidence intervals, null distributions, cross-checkpoint transfer, and deep temporal effects. The four PNGs show the same final-split results.\n\n## Limitations\n\nThe final split contains 20 puzzles, so checkpoint comparisons have wide uncertainty. Top-1/top-2 margin is computed from the model's own linear output head, making some decodability expected; the temporal controls and cross-checkpoint transfer are more discriminating than raw probe fit. Board features and targets are means over originally blank cells and do not directly represent the weakest cell.\n\n## Verdict\n\nThe numerical verdict should be based on the held-out effects, null distributions, and transfer matrix together; a high within-checkpoint fit alone is insufficient evidence for a shared coordinate.\n"""


def run(
    output_dir,
    *,
    examples_per_split_bucket=4,
    label_and_projection_repetitions=99,
    iteration_permutations=199,
    bootstrap_repetitions=500,
    seed=20260811,
    device="cuda",
    model_configs=DEFAULT_MODELS,
):
    """Run the complete pre-registered analysis and write aggregate artifacts."""

    if examples_per_split_bucket < 4:
        raise ValueError("the protocol requires at least 20 puzzles per split")
    if label_and_projection_repetitions < 19:
        raise ValueError("at least 19 label/projection controls are required")
    if iteration_permutations < 19:
        raise ValueError("at least 19 iteration permutations are required")
    if bootstrap_repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions are required")
    if not re.fullmatch(r"[A-Za-z0-9_./-]+", output_dir):
        raise ValueError(f"unsafe output directory: {output_dir!r}")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "run.log")
    started_at = time.time()
    resolved_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )

    examples_per_bucket = examples_per_split_bucket * 3
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    del solutions
    split_by_puzzle = stratified_three_way_split(
        bucket_names,
        examples_per_split_bucket=examples_per_split_bucket,
        seed=seed + 17,
    )
    split_counts = {
        split_name: int(np.sum(split_by_puzzle == split_name))
        for split_name in ("discovery", "validation", "final")
    }
    bucket_split_counts = {
        str(bucket): {
            split_name: int(
                np.sum(
                    (np.asarray(bucket_names, dtype=object) == bucket)
                    & (split_by_puzzle == split_name)
                )
            )
            for split_name in split_counts
        }
        for bucket in dict.fromkeys(bucket_names)
    }

    metrics = {
        "schema_version": 1,
        "hypothesis": (
            "Recurrent hidden states contain an ordered top-1/top-2 margin "
            "coordinate; healthy deep motion preserves it and collapsed motion "
            "loses or misorients it."
        ),
        "config": {
            "seed": seed,
            "device": str(resolved_device),
            "snapshots": list(SNAPSHOTS),
            "ridge_candidates": list(RIDGE_CANDIDATES),
            "target_candidates": list(TARGET_NAMES),
            "margin_bin_count": 5,
            "deep_minimum_iteration": 128,
            "label_and_projection_repetitions": label_and_projection_repetitions,
            "iteration_permutations": iteration_permutations,
            "bootstrap_repetitions": bootstrap_repetitions,
            "models": list(model_configs),
        },
        "sample": {
            "puzzle_count": len(puzzles),
            "examples_per_split_bucket": examples_per_split_bucket,
            "split_counts": split_counts,
            "bucket_split_counts": bucket_split_counts,
            "puzzle_hashes": [
                hashlib.sha256(puzzle.encode("ascii")).hexdigest()[:16]
                for puzzle in puzzles
            ],
            "split_by_puzzle": split_by_puzzle.tolist(),
            "rating_bucket_by_puzzle": list(bucket_names),
            "fixed_primary_mask": "originally blank cells at every snapshot",
        },
        "selection": {},
        "within_checkpoint": {},
        "controls": {},
        "cross_checkpoint_transfer": {level: {} for level in LEVEL_NAMES},
        "temporal": {},
    }

    model_rows = {}
    inputs_device = inputs.to(resolved_device)
    targets_cpu = targets.long().cpu()
    empty_mask_cpu = empty_mask.bool().cpu()
    with open(log_path, "w") as log_file:
        def log(message=""):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(
            f"ARM 02 margin: {len(puzzles)} puzzles, splits={split_counts}, "
            f"snapshots={list(SNAPSHOTS)}, device={resolved_device}"
        )
        for model_config in model_configs:
            model_name = model_config["name"]
            model_started = time.time()
            log(f"COLLECT {model_name}: {model_config['path']}")
            model = _load_model(model_config, resolved_device)
            states, logits = collect_snapshots(model, inputs_device)
            rows = prepare_model_rows(
                states,
                logits,
                empty_mask_cpu,
                split_by_puzzle,
                bucket_names,
            )
            predictions = logits.argmax(dim=-1)
            solved_by_snapshot = []
            for time_index in range(len(SNAPSHOTS)):
                correct = (predictions[:, time_index] == targets_cpu) | ~empty_mask_cpu
                solved_by_snapshot.append(int(correct.all(dim=1).sum()))
            rows["solved_by_snapshot"] = solved_by_snapshot
            model_rows[model_name] = rows
            log(
                f"  states={tuple(states.shape)}, cell_rows={len(rows['cell']['features'])}, "
                f"elapsed={time.time() - model_started:.1f}s"
            )
            del model, states, logits, predictions
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

        candidate_fits = {}
        candidate_summary = {}
        for target_name in TARGET_NAMES:
            target_scores = []
            candidate_summary[target_name] = {}
            for model_name, rows in model_rows.items():
                fits, scores = _fit_target_candidate(rows, target_name)
                candidate_fits[(target_name, model_name)] = fits
                candidate_summary[target_name][model_name] = {
                    level_name: {
                        "selected_alpha": fits[level_name]["alpha"],
                        "validation_metrics": fits[level_name]["validation_metrics"],
                        "alpha_candidates": fits[level_name]["candidates"],
                    }
                    for level_name in LEVEL_NAMES
                }
                target_scores.extend(scores)
            candidate_summary[target_name]["mean_validation_partial_r2"] = float(
                np.mean(target_scores)
            )
            log(
                f"VALIDATE target {target_name}: mean partial R2 "
                f"{np.mean(target_scores):.4f}"
            )
        selected_target = max(
            TARGET_NAMES,
            key=lambda name: candidate_summary[name]["mean_validation_partial_r2"],
        )
        metrics["selection"] = {
            "rule": (
                "maximum mean validation partial R2 across four checkpoints and "
                "both predeclared levels"
            ),
            "target_candidates": candidate_summary,
            "selected_target": selected_target,
            "selected_target_label": selected_target.replace("_", " "),
            "final_split_used_for_selection": False,
        }
        log(f"SELECTED {selected_target}")

        probes = {}
        curves = {}
        for model_index, (model_name, rows) in enumerate(model_rows.items()):
            metrics["within_checkpoint"][model_name] = {
                "model_config": next(
                    config for config in model_configs if config["name"] == model_name
                ),
                "solved_by_snapshot": {
                    str(iteration): solved
                    for iteration, solved in zip(
                        SNAPSHOTS, rows["solved_by_snapshot"]
                    )
                },
            }
            metrics["controls"][model_name] = {}
            metrics["temporal"][model_name] = {}
            curves[model_name] = {}
            for level_index, level_name in enumerate(LEVEL_NAMES):
                discovery = _split_rows(rows[level_name], "discovery", selected_target)
                validation = _split_rows(rows[level_name], "validation", selected_target)
                final = _split_rows(rows[level_name], "final", selected_target)
                ordered_selection = candidate_fits[(selected_target, model_name)][level_name]
                probe = ordered_selection["probe"]
                probes[(model_name, level_name)] = probe
                thresholds = make_bin_thresholds(
                    discovery["target"], discovery["weights"], bin_count=5
                )
                categorical_selection = select_categorical_alpha(
                    discovery["features"],
                    discovery["target"],
                    discovery["iterations"],
                    discovery["weights"],
                    validation["features"],
                    validation["target"],
                    validation["iterations"],
                    validation["weights"],
                    thresholds=thresholds,
                    candidates=RIDGE_CANDIDATES,
                )
                categorical_probe = categorical_selection["probe"]

                ordered_final = evaluate_ordered_probe(
                    probe,
                    final["features"],
                    final["target"],
                    final["iterations"],
                    final["weights"],
                )
                ordered_bins = ordered_bin_metrics(
                    probe,
                    thresholds,
                    final["features"],
                    final["target"],
                    final["iterations"],
                    final["weights"],
                )
                categorical_final = evaluate_categorical_probe(
                    categorical_probe,
                    final["features"],
                    final["target"],
                    final["iterations"],
                    final["weights"],
                )
                final_labels = assign_bins(final["target"], thresholds)
                final_ordered_bin_predictions = assign_bins(
                    ordered_final["prediction"], thresholds
                )
                final_categorical_predictions = categorical_probe.predict(
                    final["features"], final["iterations"]
                )
                bootstrap = _cluster_bootstrap_final(
                    final["target"],
                    ordered_final["prediction"],
                    ordered_final["baseline"],
                    final["puzzle_ids"],
                    final["buckets"],
                    labels=final_labels,
                    ordered_bins=final_ordered_bin_predictions,
                    categorical_bins=final_categorical_predictions,
                    repetitions=bootstrap_repetitions,
                    seed=seed + 1000 * model_index + 100 * level_index,
                )
                metrics["within_checkpoint"][model_name][level_name] = {
                    "observation_counts": {
                        "discovery": len(discovery["target"]),
                        "validation": len(validation["target"]),
                        "final": len(final["target"]),
                        "puzzles_per_split": examples_per_split_bucket * 5,
                    },
                    "ordered_probe": {
                        "rank": 1,
                        "selected_alpha": probe.alpha,
                        "axis_norm_raw_features": float(
                            np.linalg.norm(probe.raw_axis)
                        ),
                        "validation_metrics": ordered_selection[
                            "validation_metrics"
                        ],
                    },
                    "categorical_probe": {
                        "unconstrained_class_count": 5,
                        "coefficient_rank": int(
                            np.linalg.matrix_rank(categorical_probe.coefficients)
                        ),
                        "selected_alpha": categorical_probe.alpha,
                        "validation_metrics": categorical_selection[
                            "validation_metrics"
                        ],
                    },
                    "discovery_bin_thresholds": thresholds,
                    "final": {
                        "ordered_continuous": _scalar_metrics(ordered_final),
                        "ordered_bins": _scalar_metrics(ordered_bins),
                        "categorical_bins": _scalar_metrics(categorical_final),
                        "ordered_fraction_of_categorical_accuracy_gain": (
                            ordered_fraction_of_categorical_gain(
                                ordered_bins, categorical_final
                            )
                        ),
                        "cluster_bootstrap_95_percent": bootstrap,
                    },
                }
                controls = _run_controls(
                    discovery,
                    final,
                    probe,
                    ordered_final["partial_r2_over_iteration"],
                    repetitions=label_and_projection_repetitions,
                    seed=seed + 10000 + 1000 * model_index + 100 * level_index,
                )
                metrics["controls"][model_name][level_name] = controls

                raw_coordinate = final["features"] @ probe.raw_axis
                _, iteration_values, true_matrix = aggregate_rows_by_puzzle_iteration(
                    final["target"], final["puzzle_ids"], final["iterations"]
                )
                final_puzzles, coordinate_iterations, coordinate_matrix = (
                    aggregate_rows_by_puzzle_iteration(
                        raw_coordinate,
                        final["puzzle_ids"],
                        final["iterations"],
                    )
                )
                if not np.array_equal(iteration_values, coordinate_iterations):
                    raise RuntimeError("true margin and coordinate iterations differ")
                puzzle_bucket_lookup = {
                    puzzle: final["buckets"][
                        np.flatnonzero(final["puzzle_ids"] == puzzle)[0]
                    ]
                    for puzzle in final_puzzles
                }
                final_puzzle_buckets = [
                    puzzle_bucket_lookup[puzzle] for puzzle in final_puzzles
                ]
                temporal = deep_temporal_metrics(
                    true_matrix,
                    coordinate_matrix,
                    iteration_values,
                    final_puzzle_buckets,
                    minimum_iteration=128,
                    permutation_repetitions=iteration_permutations,
                    bootstrap_repetitions=bootstrap_repetitions,
                    seed=seed + 20000 + 1000 * model_index + 100 * level_index,
                )
                metrics["temporal"][model_name][level_name] = temporal
                curves[model_name][level_name] = {
                    "iterations": iteration_values,
                    "true_mean": true_matrix.mean(axis=0),
                    "true_sem": true_matrix.std(axis=0, ddof=1)
                    / math.sqrt(len(true_matrix)),
                    "coordinate_mean": coordinate_matrix.mean(axis=0),
                    "coordinate_sem": coordinate_matrix.std(axis=0, ddof=1)
                    / math.sqrt(len(coordinate_matrix)),
                }
                log(
                    f"FINAL {model_name}/{level_name}: partial R2 "
                    f"{ordered_final['partial_r2_over_iteration']:.4f}, "
                    f"ordered bins {ordered_bins['accuracy']:.3f}, "
                    f"categorical bins {categorical_final['accuracy']:.3f}"
                )

        for level_name in LEVEL_NAMES:
            for source_model in model_rows:
                metrics["cross_checkpoint_transfer"][level_name][source_model] = {}
                source_axis = probes[(source_model, level_name)].raw_axis
                for target_model, target_rows in model_rows.items():
                    validation = _split_rows(
                        target_rows[level_name], "validation", selected_target
                    )
                    final = _split_rows(target_rows[level_name], "final", selected_target)
                    transferred = fit_transferred_axis(
                        source_axis,
                        validation["features"],
                        validation["target"],
                        validation["iterations"],
                        validation["weights"],
                        preserve_orientation=True,
                    )
                    transfer_final = evaluate_transferred_axis(
                        transferred,
                        final["features"],
                        final["target"],
                        final["iterations"],
                        final["weights"],
                    )
                    metrics["cross_checkpoint_transfer"][level_name][source_model][
                        target_model
                    ] = _scalar_metrics(transfer_final)

        metrics["elapsed_seconds"] = time.time() - started_at
        _atomic_json(os.path.join(output_dir, "metrics.json"), metrics)
        _plot_probe_performance(metrics, output_dir)
        _plot_controls(metrics, output_dir)
        _plot_transfer(metrics, output_dir)
        _plot_dynamics(metrics, curves, output_dir)
        with open(os.path.join(output_dir, "REPORT.md"), "w") as report_file:
            report_file.write(_generated_report(metrics))
        log(f"Wrote aggregate artifacts to {output_dir}")
        log(f"Total time: {metrics['elapsed_seconds']:.1f}s")
    return metrics


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--examples-per-split-bucket", type=int, default=4)
    parser.add_argument("--controls", type=int, default=99)
    parser.add_argument("--iteration-permutations", type=int, default=199)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = _parse_args()
    run(
        arguments.output_dir,
        examples_per_split_bucket=arguments.examples_per_split_bucket,
        label_and_projection_repetitions=arguments.controls,
        iteration_permutations=arguments.iteration_permutations,
        bootstrap_repetitions=arguments.bootstrap,
        seed=arguments.seed,
        device=arguments.device,
    )
