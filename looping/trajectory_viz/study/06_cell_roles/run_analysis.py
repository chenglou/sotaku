"""Run the protocol-defined cell-role trajectory analysis."""

from __future__ import annotations

import html
import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from analysis_core import (  # noqa: E402
    ITERATIONS,
    REPRESENTATIONS,
    TASKS,
    build_representations,
    compute_cell_roles,
    exact_iteration_order_test,
    fit_selected_readout,
    grouped_regression_predictions,
    label_shuffle_control,
    pairwise_centroid_geometry,
    random_subspace_control,
    stratified_three_way_split,
    task_examples,
)

import stabilize.exp_testbed_20k as model_module  # noqa: E402
from looping.eval_loop_diagnostics import (  # noqa: E402
    _load_balanced_sample,
    _load_model,
)
from looping.eval_trajectory_geometry import DEFAULT_MODELS  # noqa: E402


MODEL_COLORS = {
    "stable_plain": "#147D64",
    "collapsed_plain": "#C63D3D",
    "late_state_ce": "#2E67B1",
    "combined_margin": "#8B5AA5",
}
MODEL_LABELS = {
    "stable_plain": "Stable plain",
    "collapsed_plain": "Collapsed plain",
    "late_state_ce": "Late-state CE",
    "combined_margin": "Combined",
}
TASK_LABELS = {
    "clue": "Clue vs blank",
    "row": "Row",
    "column": "Column",
    "box": "Box",
    "candidate_size_categorical": "Candidate size (categorical)",
    "candidate_size_ordinal": "Candidate size (ordered axis)",
}


def _atomic_json_dump(data, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def collect_states(model, inputs, targets, empty_mask):
    """Collect only predefined states and their puzzle-level accuracy."""

    selected = set(ITERATIONS)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    states = {}
    accuracy = {}
    hidden_state = model.initial_encoder(inputs)
    feedback_predictions = torch.zeros(
        len(inputs),
        81,
        9,
        device=inputs.device,
    )
    with torch.no_grad():
        for iteration in range(ITERATIONS[-1] + 1):
            if iteration in selected:
                logits = model.output_head(hidden_state)
                predictions = logits.argmax(-1)
                correct_cells = predictions == targets
                solved = (correct_cells | ~empty_mask).all(1)
                accuracy[str(iteration)] = {
                    "puzzle_accuracy": float(solved.float().mean().item()),
                    "blank_cell_accuracy": float(
                        correct_cells[empty_mask].float().mean().item()
                    ),
                    "solved": int(solved.sum().item()),
                    "total": len(inputs),
                }
                states[iteration] = hidden_state.detach().float().cpu()
            if iteration == ITERATIONS[-1]:
                break
            hidden_state = model.recurrent_step(
                hidden_state,
                feedback_predictions,
                rope_cos,
                rope_sin,
            )
            feedback_predictions = F.softmax(
                model.output_head(hidden_state),
                dim=-1,
            )
    return states, accuracy


def _split_examples(features, roles, task, split_indices):
    return {
        split_name: task_examples(features, roles, task, puzzle_indices)
        for split_name, puzzle_indices in split_indices.items()
    }


def _task_seed(base_seed, model_index, iteration_index, representation_index, task_index):
    return (
        base_seed
        + 100_000 * model_index
        + 10_000 * iteration_index
        + 1_000 * representation_index
        + 100 * task_index
    )


def analyze_model_states(
    states,
    roles,
    split_indices,
    *,
    model_index,
    seed,
    label_shuffle_repeats,
    random_subspace_repeats,
):
    model_result = {"iterations": {}, "temporal_order_controls": {}}
    primary_series = {
        representation: {task.name: [] for task in TASKS}
        for representation in REPRESENTATIONS
    }

    for iteration_index, iteration in enumerate(ITERATIONS):
        representations, input_group_counts = build_representations(
            states[iteration],
            roles["input_symbol"],
            split_indices["discovery"],
        )
        iteration_result = {
            "input_symbol_discovery_counts": input_group_counts.tolist(),
            "representations": {},
        }
        for representation_index, representation_name in enumerate(REPRESENTATIONS):
            features = representations[representation_name]
            representation_result = {}
            for task_index, task in enumerate(TASKS):
                examples = _split_examples(features, roles, task, split_indices)
                probe_result, readout = fit_selected_readout(
                    examples["discovery"],
                    examples["validation"],
                    examples["final"],
                    task,
                )
                final_primary = probe_result["final"][task.primary_metric]
                primary_series[representation_name][task.name].append(final_primary)

                discovery_features, discovery_labels, _ = examples["discovery"]
                final_features, final_labels, final_puzzles = examples["final"]
                if task.kind == "classification" and task.class_count >= 3:
                    probe_result["centroid_geometry"] = pairwise_centroid_geometry(
                        readout.standardized(discovery_features),
                        discovery_labels,
                        readout.standardized(final_features),
                        final_labels,
                        task.class_count,
                        ordered=task.name == "candidate_size_categorical",
                        seed=_task_seed(
                            seed,
                            model_index,
                            iteration_index,
                            representation_index,
                            task_index,
                        ),
                    )

                if task.name == "candidate_size_ordinal":
                    final_predictions = readout.predict(final_features)[:, 0]
                    probe_result["final_grouped_predictions"] = (
                        grouped_regression_predictions(
                            final_labels,
                            final_predictions,
                        )
                    )

                if (
                    iteration == ITERATIONS[-1]
                    and representation_name in ("state", "input_residual")
                ):
                    control_seed = _task_seed(
                        seed,
                        model_index,
                        iteration_index,
                        representation_index,
                        task_index,
                    )
                    probe_result["label_shuffle_control"] = label_shuffle_control(
                        examples["discovery"],
                        examples["final"],
                        task,
                        alpha=probe_result["selected_alpha"],
                        observed=final_primary,
                        repeats=label_shuffle_repeats,
                        seed=control_seed,
                    )
                    probe_result["random_subspace_control"] = (
                        random_subspace_control(
                            readout,
                            examples["discovery"],
                            examples["final"],
                            task,
                            alpha=probe_result["selected_alpha"],
                            observed=final_primary,
                            repeats=random_subspace_repeats,
                            seed=control_seed + 50,
                        )
                    )
                representation_result[task.name] = probe_result
            iteration_result["representations"][representation_name] = (
                representation_result
            )
        model_result["iterations"][str(iteration)] = iteration_result

    for representation_name, task_series in primary_series.items():
        model_result["temporal_order_controls"][representation_name] = {}
        for task_name, values in task_series.items():
            model_result["temporal_order_controls"][representation_name][task_name] = (
                exact_iteration_order_test(ITERATIONS, values)
            )
    return model_result


def _primary_metric(result, model_name, iteration, representation, task_name):
    task = next(task for task in TASKS if task.name == task_name)
    return result["models"][model_name]["analysis"]["iterations"][str(iteration)][
        "representations"
    ][representation][task_name]["final"][task.primary_metric]


def render_probe_trajectories(result, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    for axis, task in zip(axes.flat, TASKS):
        for model_name in result["models"]:
            for representation, linestyle, alpha in (
                ("state", "-", 1.0),
                ("input_residual", "--", 0.8),
            ):
                values = [
                    _primary_metric(
                        result,
                        model_name,
                        iteration,
                        representation,
                        task.name,
                    )
                    for iteration in ITERATIONS
                ]
                axis.plot(
                    np.log1p(ITERATIONS),
                    values,
                    color=MODEL_COLORS[model_name],
                    linestyle=linestyle,
                    marker="o" if representation == "state" else None,
                    markersize=3,
                    linewidth=1.8,
                    alpha=alpha,
                    label=(
                        MODEL_LABELS[model_name]
                        if representation == "state"
                        else None
                    ),
                )
        chance = 0.5 if task.name == "clue" else (1 / 9 if task.kind == "classification" else 0)
        axis.axhline(chance, color="#777777", linewidth=0.8, linestyle=":")
        axis.set_title(TASK_LABELS[task.name])
        axis.set_xticks(np.log1p(ITERATIONS), [str(value) for value in ITERATIONS])
        axis.set_xlabel("Recurrent iteration")
        axis.set_ylabel(
            "Balanced accuracy"
            if task.kind == "classification"
            else "Within-puzzle Spearman"
        )
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Cell-role information on final held-out puzzles\n"
        "solid: full state; dashed: after removing discovery-fitted input-symbol means",
        fontsize=14,
        y=1.12,
    )
    path = Path(output_dir) / "probe_trajectories.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path.name


def render_candidate_geometry(result, output_dir):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    shown_iterations = (0, 16, 128, 1024)
    iteration_colors = {
        0: "#999999",
        16: "#E69F00",
        128: "#2E67B1",
        1024: "#147D64",
    }
    for axis, model_name in zip(axes.flat, result["models"]):
        observed_sizes = set()
        for iteration in shown_iterations:
            rows = result["models"][model_name]["analysis"]["iterations"][
                str(iteration)
            ]["representations"]["input_residual"]["candidate_size_ordinal"][
                "final_grouped_predictions"
            ]
            sizes = np.asarray([row["candidate_size"] for row in rows])
            means = np.asarray([row["prediction_mean"] for row in rows])
            sems = np.asarray([row["prediction_sem"] for row in rows])
            observed_sizes.update(sizes.tolist())
            axis.errorbar(
                sizes,
                means,
                yerr=1.96 * sems,
                color=iteration_colors[iteration],
                marker="o",
                linewidth=1.5,
                markersize=4,
                capsize=2,
                label=f"iter {iteration}",
            )
        minimum = min(observed_sizes)
        maximum = max(observed_sizes)
        axis.plot([minimum, maximum], [minimum, maximum], color="#333333", linestyle=":")
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xlabel("Legal candidates from the input puzzle")
        axis.set_ylabel("Discovery-fitted linear prediction")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Candidate-set size readout after removing direct input-symbol persistence\n"
        "points and 95% error bars use final held-out puzzles",
        fontsize=14,
        y=1.12,
    )
    path = Path(output_dir) / "candidate_size_geometry.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path.name


def render_direction_control(result, output_dir):
    task_names = [task.name for task in TASKS]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, model_name in zip(axes.flat, result["models"]):
        x = np.arange(len(task_names))
        width = 0.18
        for offset, representation in enumerate(REPRESENTATIONS):
            values = [
                _primary_metric(
                    result,
                    model_name,
                    1024,
                    representation,
                    task_name,
                )
                for task_name in task_names
            ]
            axis.bar(
                x + (offset - 1.5) * width,
                values,
                width,
                label=representation.replace("_", " "),
            )
        axis.set_title(MODEL_LABELS[model_name])
        axis.set_xticks(x, [TASK_LABELS[name] for name in task_names], rotation=25, ha="right")
        axis.set_ylabel("Primary held-out metric")
        axis.set_ylim(-0.15, 1.05)
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
    )
    figure.suptitle(
        "Iteration 1024: magnitude and direct-input controls\n"
        "classification uses balanced accuracy; ordered candidate size uses within-puzzle Spearman",
        fontsize=14,
        y=1.12,
    )
    path = Path(output_dir) / "direction_and_input_controls.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path.name


def render_shuffle_controls(result, output_dir):
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    model_names = list(result["models"])
    x = np.arange(len(model_names))
    for axis, task in zip(axes.flat, TASKS):
        observed = []
        shuffled_p95 = []
        random_p95 = []
        for model_name in model_names:
            probe = result["models"][model_name]["analysis"]["iterations"]["1024"][
                "representations"
            ]["input_residual"][task.name]
            observed.append(probe["final"][task.primary_metric])
            shuffled_p95.append(probe["label_shuffle_control"]["p95"])
            random_p95.append(probe["random_subspace_control"]["p95"])
        axis.bar(x, observed, width=0.55, color=[MODEL_COLORS[name] for name in model_names])
        axis.scatter(x - 0.09, shuffled_p95, marker="x", color="#111111", label="shuffled-label p95")
        axis.scatter(x + 0.09, random_p95, marker="_", s=120, color="#111111", label="random-subspace p95")
        axis.set_title(TASK_LABELS[task.name])
        axis.set_xticks(x, [MODEL_LABELS[name] for name in model_names], rotation=25, ha="right")
        axis.set_ylabel("Primary held-out metric")
        axis.grid(axis="y", alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Iteration 1024 input-residual readouts and falsification controls\n"
        "bars are final holdout; markers are 95th percentiles of discovery-only null fits",
        fontsize=14,
        y=1.12,
    )
    path = Path(output_dir) / "shuffle_and_subspace_controls.png"
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path.name


def render_index(result, artifact_names, output_dir):
    rows = []
    for model_name in result["models"]:
        candidate = _primary_metric(
            result,
            model_name,
            1024,
            "input_residual",
            "candidate_size_ordinal",
        )
        row = _primary_metric(result, model_name, 1024, "input_residual", "row")
        column = _primary_metric(result, model_name, 1024, "input_residual", "column")
        solved = result["models"][model_name]["accuracy"]["1024"]["puzzle_accuracy"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS[model_name])}</td>"
            f"<td>{solved:.3f}</td><td>{row:.3f}</td><td>{column:.3f}</td>"
            f"<td>{candidate:.3f}</td></tr>"
        )
    images = "\n".join(
        f'<section><h2>{html.escape(name.replace("_", " ").removesuffix(".png").title())}</h2>'
        f'<img src="{html.escape(name)}" alt="{html.escape(name)}"></section>'
        for name in artifact_names
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sotaku cell-role geometry</title><style>
body{{font-family:system-ui,sans-serif;max-width:1280px;margin:32px auto;padding:0 20px;color:#202124}} h1{{font-size:28px}} h2{{font-size:20px;margin-top:36px}} img{{width:100%;height:auto;border:1px solid #ddd}} table{{border-collapse:collapse;width:100%}} th,td{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:right}} th:first-child,td:first-child{{text-align:left}} code{{background:#f2f2f2;padding:2px 4px}}</style></head><body>
<h1>Sotaku cell-role geometry</h1><p>All readouts were fit on 20 discovery puzzles, selected on 20 validation puzzles, and evaluated once on 20 final puzzles. The table uses iteration 1024 and input-symbol residuals.</p>
<table><thead><tr><th>Model</th><th>Puzzle accuracy</th><th>Row balanced accuracy</th><th>Column balanced accuracy</th><th>Candidate within-puzzle Spearman</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{images}<p>Full machine-readable results: <a href="metrics.json"><code>metrics.json</code></a>.</p></body></html>"""
    path = Path(output_dir) / "index.html"
    path.write_text(page)
    return path.name


def _role_distribution(roles, split_indices):
    distribution = {}
    for split_name, puzzle_indices in split_indices.items():
        blank = roles["blank"][puzzle_indices]
        candidates = roles["candidate_size"][puzzle_indices][blank]
        distribution[split_name] = {
            "puzzles": len(puzzle_indices),
            "blank_cells": int(blank.sum().item()),
            "clue_cells": int((~blank).sum().item()),
            "candidate_size_counts": {
                str(size): int((candidates == size).sum().item())
                for size in sorted(candidates.unique().tolist())
            },
        }
    return distribution


def run(
    output_dir,
    *,
    examples_per_bucket=12,
    seed=20260811,
    split_seed=20260812,
    device="cuda",
    label_shuffle_repeats=8,
    random_subspace_repeats=16,
):
    if examples_per_bucket % 3:
        raise ValueError("examples_per_bucket must be divisible by three")
    if examples_per_bucket < 12:
        raise ValueError("the protocol requires at least 20 puzzles per split")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_handle = (output_dir / "run.log").open("w")

    def log(message=""):
        print(message, flush=True)
        log_handle.write(message + "\n")
        log_handle.flush()

    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    split_indices = stratified_three_way_split(bucket_names, split_seed)
    roles = compute_cell_roles(puzzles)
    result = {
        "protocol": {
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(puzzles),
            "puzzles_per_split": len(split_indices["discovery"]),
            "rating_buckets": sorted(set(bucket_names)),
            "seed": seed,
            "split_seed": split_seed,
            "iterations": list(ITERATIONS),
            "representations": list(REPRESENTATIONS),
            "label_shuffle_repeats": label_shuffle_repeats,
            "random_subspace_repeats": random_subspace_repeats,
            "device": str(resolved_device),
            "selection": (
                "Ridge strength selected on validation; every projection and readout "
                "fit on discovery only; final puzzles evaluated once without refitting."
            ),
            "input_residual": (
                "Subtract the discovery-set mean state for the cell's exact input "
                "symbol (empty or digit 1-9), separately at each iteration."
            ),
            "candidate_size": (
                "Number of digits absent from the blank cell's input row, column, "
                "and box; clue cells are excluded."
            ),
        },
        "sample": {
            "puzzles": puzzles,
            "solutions": solutions,
            "buckets": bucket_names,
            "splits": split_indices,
            "role_distribution": _role_distribution(roles, split_indices),
        },
        "models": {},
    }
    inputs = inputs.to(resolved_device)
    targets = targets.to(resolved_device)
    empty_mask = empty_mask.to(resolved_device)
    started_at = time.time()
    log(
        f"Cell-role study: {len(puzzles)} puzzles, "
        f"{len(split_indices['discovery'])} per split, device={resolved_device}"
    )

    for model_index, model_config in enumerate(DEFAULT_MODELS):
        model_name = model_config["name"]
        model_started_at = time.time()
        log(f"\nMODEL {model_name}: {model_config['path']}")
        model = _load_model(model_config, resolved_device)
        states, accuracy = collect_states(model, inputs, targets, empty_mask)
        log(
            "  inference: "
            + ", ".join(
                f"{iteration}={accuracy[str(iteration)]['solved']}/{len(puzzles)}"
                for iteration in ITERATIONS
            )
        )
        analysis = analyze_model_states(
            states,
            roles,
            split_indices,
            model_index=model_index,
            seed=seed,
            label_shuffle_repeats=label_shuffle_repeats,
            random_subspace_repeats=random_subspace_repeats,
        )
        result["models"][model_name] = {
            "model_config": model_config,
            "accuracy": accuracy,
            "analysis": analysis,
            "elapsed_seconds": time.time() - model_started_at,
        }
        candidate_metric = analysis["iterations"]["1024"]["representations"][
            "input_residual"
        ]["candidate_size_ordinal"]["final"]["within_puzzle_spearman"]
        log(f"  final candidate-size within-puzzle rho: {candidate_metric:.3f}")
        _atomic_json_dump(result, output_dir / "metrics.partial.json")
        del model, states
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    result["elapsed_seconds"] = time.time() - started_at
    artifact_names = [
        render_probe_trajectories(result, output_dir),
        render_candidate_geometry(result, output_dir),
        render_direction_control(result, output_dir),
        render_shuffle_controls(result, output_dir),
    ]
    artifact_names.append(render_index(result, artifact_names, output_dir))
    result["artifacts"] = artifact_names
    _atomic_json_dump(result, output_dir / "metrics.json")
    partial_path = output_dir / "metrics.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    log(f"\nArtifacts: {', '.join(artifact_names)}")
    log(f"Structured metrics: {output_dir / 'metrics.json'}")
    log(f"Total time: {result['elapsed_seconds']:.1f}s")
    log_handle.close()
    return result


if __name__ == "__main__":
    run(MODULE_DIR / "artifacts" / "cell_roles_local", device="cpu")
