"""Run the preregistered held-out uncertainty-coordinate analysis."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from entropy_core import (
    TARGET_SPECS,
    evaluate_probe,
    fit_shuffled_target_controls,
    monotonicity_metrics,
    predictive_targets,
    puzzle_equal_weights,
    random_axis_controls,
    rating_balanced_three_way_split,
    select_ridge_alpha,
    shuffled_time_controls,
    summarize_null,
    transform_features,
    weighted_mean,
)
from looping.eval_loop_diagnostics import _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS
from looping.trajectory_viz.helix_tests.statistics.trajectory_data import (
    SNAPSHOT_ITERATIONS,
    collect_trajectory,
    load_balanced_trajectory_sample,
    output_head_contrast_basis,
    project_outside_rowspace,
)


DEFAULT_ALPHAS = (0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0)
DISPLAY_NAMES = {
    "stable_plain": "stable plain",
    "collapsed_plain": "collapsed plain",
    "late_state_ce": "late-state CE",
    "combined_margin": "combined",
}


def _safe_run_name(run_name):
    if not run_name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
        for character in run_name
    ):
        raise ValueError(f"unsafe run name: {run_name!r}")
    return run_name


def _puzzle_hash(puzzle):
    return hashlib.sha256(puzzle.encode("ascii")).hexdigest()


def _split_rows(features, target, blank_mask, puzzle_indices):
    puzzle_indices = puzzle_indices.long()
    selected_features = features[puzzle_indices]
    selected_targets = target[puzzle_indices]
    selected_mask = blank_mask[puzzle_indices]
    puzzle_count, time_count, cell_count = selected_targets.shape
    expanded_mask = selected_mask[:, None, :].expand(
        puzzle_count, time_count, cell_count
    )
    iteration_grid = torch.arange(time_count).view(1, -1, 1).expand_as(
        selected_targets
    )
    puzzle_grid = puzzle_indices.view(-1, 1, 1).expand_as(selected_targets)
    return (
        selected_features[expanded_mask],
        selected_targets[expanded_mask],
        iteration_grid[expanded_mask],
        puzzle_grid[expanded_mask],
    )


def _paths(values, blank_mask, puzzle_indices):
    puzzle_indices = puzzle_indices.long()
    selected_values = values[puzzle_indices].permute(0, 2, 1)
    selected_mask = blank_mask[puzzle_indices]
    puzzle_grid = puzzle_indices[:, None].expand_as(selected_mask)
    return selected_values[selected_mask], puzzle_grid[selected_mask]


def _axis_values(features, probe):
    standardized = (
        features.double() - probe.preprocessor.feature_mean
    ) / probe.preprocessor.feature_scale
    return standardized @ probe.coefficient


def _prediction_values(features, probe):
    puzzle_count, time_count, cell_count, feature_count = features.shape
    flat = features.reshape(-1, feature_count)
    iteration_indices = torch.arange(time_count).view(1, -1, 1).expand(
        puzzle_count, time_count, cell_count
    ).reshape(-1)
    _, residual = transform_features(flat, iteration_indices, probe.preprocessor)
    baseline = probe.iteration_target_means[iteration_indices]
    return (baseline + residual @ probe.coefficient).reshape(
        puzzle_count, time_count, cell_count
    )


def _weighted_snapshot_means(values, blank_mask, puzzle_indices):
    paths, path_puzzles = _paths(values, blank_mask, puzzle_indices)
    weights = puzzle_equal_weights(path_puzzles)
    return [
        float(weighted_mean(paths[:, time_index], weights))
        for time_index in range(paths.size(1))
    ]


def _weighted_snapshot_accuracy(logits, targets, blank_mask, puzzle_indices):
    predictions = logits.argmax(dim=-1)
    correctness = predictions == targets[:, None, :]
    return _weighted_snapshot_means(
        correctness.float(), blank_mask, puzzle_indices
    )


def _analyze_representation(
    features,
    targets_by_name,
    blank_mask,
    split,
    *,
    alphas,
):
    results = {}
    probes = {}
    for target_name, target_values in targets_by_name.items():
        discovery = _split_rows(
            features, target_values, blank_mask, split.discovery
        )
        validation = _split_rows(
            features, target_values, blank_mask, split.validation
        )
        final = _split_rows(features, target_values, blank_mask, split.final)
        probe, candidates = select_ridge_alpha(
            discovery,
            validation,
            time_count=features.size(1),
            alphas=alphas,
        )
        probes[target_name] = probe
        results[target_name] = {
            "selected_alpha": probe.alpha,
            "validation_candidates": candidates,
            "validation": evaluate_probe(probe, *validation),
            "final": evaluate_probe(probe, *final),
        }
    return results, probes


def _analyze_primary_controls(
    features,
    targets_by_name,
    blank_mask,
    split,
    probes,
    *,
    control_count,
    seed,
):
    controls = {}
    for target_offset, (target_name, target_values) in enumerate(
        targets_by_name.items()
    ):
        probe = probes[target_name]
        discovery = _split_rows(
            features, target_values, blank_mask, split.discovery
        )
        final = _split_rows(features, target_values, blank_mask, split.final)
        shuffled_r2 = fit_shuffled_target_controls(
            probe,
            discovery,
            final,
            count=control_count,
            seed=seed + 1000 + target_offset,
        )
        random_r2 = random_axis_controls(
            probe,
            discovery,
            final,
            count=control_count,
            seed=seed + 2000 + target_offset,
        )

        axis_values = _axis_values(features, probe)
        axis_paths, axis_puzzles = _paths(
            axis_values, blank_mask, split.final
        )
        target_paths, target_puzzles = _paths(
            target_values, blank_mask, split.final
        )
        if not torch.equal(axis_puzzles, target_puzzles):
            raise RuntimeError("axis and target trajectories are misaligned")
        direction = TARGET_SPECS[target_name]["expected_direction"]
        axis_monotonicity = monotonicity_metrics(
            axis_paths, axis_puzzles, direction
        )
        target_monotonicity = monotonicity_metrics(
            target_paths, target_puzzles, direction
        )
        time_null = shuffled_time_controls(
            axis_paths,
            axis_puzzles,
            direction,
            count=control_count,
            seed=seed + 3000 + target_offset,
        )

        controls[target_name] = {
            "shuffled_discovery_targets": summarize_null(
                shuffled_r2,
                observed=evaluate_probe(probe, *final)[
                    "partial_r2_over_iteration"
                ],
            ),
            "random_one_dimensional_axes": summarize_null(
                random_r2,
                observed=evaluate_probe(probe, *final)[
                    "partial_r2_over_iteration"
                ],
            ),
            "axis_monotonicity": axis_monotonicity,
            "target_monotonicity": target_monotonicity,
            "shuffled_iteration_order": {
                metric_name: summarize_null(
                    metric_values,
                    observed=axis_monotonicity[metric_name],
                )
                for metric_name, metric_values in time_null.items()
            },
            "final_target_snapshot_means": _weighted_snapshot_means(
                target_values, blank_mask, split.final
            ),
            "final_probe_prediction_snapshot_means": (
                _weighted_snapshot_means(
                    _prediction_values(features, probe),
                    blank_mask,
                    split.final,
                )
            ),
        }
    return controls


def analyze_model(
    model,
    sample,
    split,
    *,
    iterations,
    alphas,
    control_count,
    seed,
    device,
):
    device_sample = sample.to(device)
    trajectory = collect_trajectory(
        model,
        device_sample.inputs,
        device_sample.targets,
        device_sample.originally_blank,
        iterations=iterations,
        output_device="cpu",
    )
    targets_by_name = predictive_targets(trajectory.logits)
    cell_centered = trajectory.hidden_states.float() - (
        trajectory.hidden_states.float().mean(dim=2, keepdim=True)
    )
    output_basis = output_head_contrast_basis(
        model.output_head, device="cpu"
    )
    output_null = project_outside_rowspace(cell_centered, output_basis)

    primary_results, primary_probes = _analyze_representation(
        cell_centered,
        targets_by_name,
        trajectory.originally_blank,
        split,
        alphas=alphas,
    )
    output_null_results, _ = _analyze_representation(
        output_null,
        targets_by_name,
        trajectory.originally_blank,
        split,
        alphas=alphas,
    )
    controls = _analyze_primary_controls(
        cell_centered,
        targets_by_name,
        trajectory.originally_blank,
        split,
        primary_probes,
        control_count=control_count,
        seed=seed,
    )
    return {
        "output_contrast_rank_removed": int(output_basis.size(0)),
        "primary_cell_centered_hidden": primary_results,
        "output_head_null_sensitivity": output_null_results,
        "controls": controls,
        "final_blank_cell_accuracy_by_iteration": _weighted_snapshot_accuracy(
            trajectory.logits,
            trajectory.targets,
            trajectory.originally_blank,
            split.final,
        ),
    }


def _plot_encoding(summary, output_path):
    model_names = list(summary["models"])
    target_names = list(TARGET_SPECS)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=False)
    x = torch.arange(len(model_names), dtype=torch.float64).numpy()
    width = 0.34
    for axis, target_name in zip(axes, target_names):
        primary = [
            summary["models"][model_name]["primary_cell_centered_hidden"]
            [target_name]["final"]["partial_r2_over_iteration"]
            for model_name in model_names
        ]
        output_null = [
            summary["models"][model_name]["output_head_null_sensitivity"]
            [target_name]["final"]["partial_r2_over_iteration"]
            for model_name in model_names
        ]
        axis.bar(x - width / 2, primary, width, label="cell-centered state")
        axis.bar(x + width / 2, output_null, width, label="output-head removed")
        for model_index, model_name in enumerate(model_names):
            control = summary["models"][model_name]["controls"][target_name]
            for null_name, color, marker in (
                ("shuffled_discovery_targets", "#ba3b46", "x"),
                ("random_one_dimensional_axes", "#555555", "o"),
            ):
                null = control[null_name]
                axis.vlines(
                    model_index,
                    null["p05"],
                    null["p95"],
                    color=color,
                    linewidth=1.3,
                    alpha=0.85,
                )
                axis.scatter(
                    model_index,
                    null["median"],
                    color=color,
                    marker=marker,
                    s=24,
                    zorder=4,
                )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(TARGET_SPECS[target_name]["label"])
        axis.set_xticks(x, [DISPLAY_NAMES.get(name, name) for name in model_names])
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("final partial R2 beyond iteration")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Unseen-puzzle uncertainty encoding", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_monotonicity(summary, output_path):
    model_names = list(summary["models"])
    target_names = list(TARGET_SPECS)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharey=True)
    x = torch.arange(len(model_names), dtype=torch.float64).numpy()
    width = 0.34
    for axis, target_name in zip(axes, target_names):
        fitted = []
        actual = []
        for model_name in model_names:
            controls = summary["models"][model_name]["controls"][target_name]
            fitted.append(
                controls["axis_monotonicity"]["all_pairs_direction_fraction"]
            )
            actual.append(
                controls["target_monotonicity"]["all_pairs_direction_fraction"]
            )
        axis.bar(x - width / 2, fitted, width, label="fitted hidden axis")
        axis.bar(x + width / 2, actual, width, label="actual model uncertainty")
        for model_index, model_name in enumerate(model_names):
            null = summary["models"][model_name]["controls"][target_name][
                "shuffled_iteration_order"
            ]["all_pairs_direction_fraction"]
            axis.vlines(
                model_index,
                null["p05"],
                null["p95"],
                color="#ba3b46",
                linewidth=1.5,
            )
            axis.scatter(
                model_index,
                null["median"],
                color="#ba3b46",
                marker="x",
                s=28,
                zorder=4,
            )
        axis.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        axis.set_ylim(0.35, 1.01)
        axis.set_title(TARGET_SPECS[target_name]["label"])
        axis.set_xticks(x, [DISPLAY_NAMES.get(name, name) for name in model_names])
        axis.tick_params(axis="x", rotation=20)
        axis.set_ylabel("ordered snapshot-pair fraction")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Does the fitted coordinate move monotonically?", fontsize=14)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _plot_iteration_profiles(summary, output_path):
    model_names = list(summary["models"])
    target_names = list(TARGET_SPECS)
    iterations = summary["config"]["iterations"]
    x = torch.log2(torch.tensor(iterations, dtype=torch.float64) + 1).numpy()
    display_floor = -0.30
    figure, axes = plt.subplots(
        2, 2, figsize=(13, 8), sharex=True, sharey=True
    )
    target_labels = {
        "entropy": "entropy",
        "top_two_gap": "top-two gap",
        "residual_uncertainty": "1 − max probability",
    }
    for axis, model_name in zip(axes.flat, model_names):
        clipped_by_iteration = {}
        for target_name in target_names:
            raw_values = [
                item["partial_r2_over_discovery_iteration_mean"]
                for item in summary["models"][model_name][
                    "primary_cell_centered_hidden"
                ][target_name]["final"]["per_iteration"]
            ]
            values = [
                float("nan") if value is None else max(value, display_floor)
                for value in raw_values
            ]
            axis.plot(
                x,
                values,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=target_labels[target_name],
            )
            for time_index, value in enumerate(raw_values):
                if value is not None and value < display_floor:
                    clipped_by_iteration.setdefault(
                        iterations[time_index], []
                    ).append(value)
                    axis.scatter(
                        x[time_index],
                        display_floor,
                        marker="v",
                        s=32,
                        color=axis.lines[-1].get_color(),
                        zorder=4,
                    )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(DISPLAY_NAMES.get(model_name, model_name))
        axis.set_xticks(x, [str(value) for value in iterations], rotation=45)
        axis.set_xlabel("recurrent iteration")
        axis.set_ylabel("within-iteration partial R2")
        axis.set_ylim(display_floor - 0.04, 1.02)
        axis.grid(alpha=0.2)
        if clipped_by_iteration:
            lines = []
            for iteration, clipped in clipped_by_iteration.items():
                lines.append(
                    f"iteration {iteration}: "
                    f"{min(clipped):.2e} to {max(clipped):.2e}"
                )
            axis.text(
                0.02,
                0.04,
                "clipped below −0.30\n" + "\n".join(lines),
                transform=axis.transAxes,
                fontsize=8,
                va="bottom",
                bbox={
                    "facecolor": "white",
                    "edgecolor": "0.8",
                    "alpha": 0.9,
                    "pad": 3,
                },
            )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
    )
    figure.suptitle("Held-out encoding at each snapshot", fontsize=14)
    figure.text(
        0.5,
        0.01,
        "Downward triangles mark values below the display floor; exact ranges "
        "are printed in the affected panels.",
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.93))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_json(path, value):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w") as output_file:
        json.dump(value, output_file, indent=2, allow_nan=False)
        output_file.write("\n")
    os.replace(temporary_path, path)


def run(
    *,
    output_root,
    run_name="entropy_v1_20260811",
    examples_per_bucket=12,
    control_count=64,
    seed=20260811,
    iterations=SNAPSHOT_ITERATIONS,
    alphas=DEFAULT_ALPHAS,
    device="cuda",
):
    """Collect trajectories and execute the locked three-way analysis."""

    run_name = _safe_run_name(run_name)
    if examples_per_bucket < 12 or examples_per_bucket % 3:
        raise ValueError(
            "examples_per_bucket must be a multiple of three and at least 12"
        )
    if control_count <= 0 or control_count > 128:
        raise ValueError("control_count must be between one and 128")
    output_dir = Path(output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "run.log"
    started_at = time.time()

    def log(message=""):
        print(message, flush=True)
        with log_path.open("a") as log_file:
            log_file.write(message + "\n")

    resolved_device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    log(
        f"Entropy-coordinate study: seed={seed}, "
        f"examples_per_bucket={examples_per_bucket}, controls={control_count}"
    )
    sample = load_balanced_trajectory_sample(examples_per_bucket, seed)
    split = rating_balanced_three_way_split(sample.rating_buckets, seed + 1)
    for split_name in ("discovery", "validation", "final"):
        indices = getattr(split, split_name)
        log(f"{split_name}: {len(indices)} whole puzzles")

    summary = {
        "study": "linear uncertainty coordinates and trajectory monotonicity",
        "status": "final holdout evaluated after discovery-only fit and validation selection",
        "config": {
            "seed": seed,
            "examples_per_bucket": examples_per_bucket,
            "sample_size": sample.puzzle_count,
            "iterations": list(iterations),
            "alphas": list(alphas),
            "control_count": control_count,
            "device": str(resolved_device),
            "primary_representation": (
                "hidden state minus the 81-cell board mean at each snapshot"
            ),
            "fit": (
                "ridge regression on within-iteration residuals; fit on discovery "
                "only; alpha selected on validation"
            ),
            "targets": TARGET_SPECS,
            "model_configs": list(DEFAULT_MODELS),
        },
        "sample": {
            "puzzle_hashes": [_puzzle_hash(value) for value in sample.puzzles],
            "rating_buckets": list(sample.rating_buckets),
            "splits": {
                split_name: getattr(split, split_name).tolist()
                for split_name in ("discovery", "validation", "final")
            },
        },
        "models": {},
    }

    for model_index, model_config in enumerate(DEFAULT_MODELS):
        model_name = model_config["name"]
        model_started_at = time.time()
        log(f"\nMODEL {model_name}: {model_config['path']}")
        model = _load_model(model_config, resolved_device)
        model_result = analyze_model(
            model,
            sample,
            split,
            iterations=iterations,
            alphas=alphas,
            control_count=control_count,
            seed=seed + 10000 * model_index,
            device=resolved_device,
        )
        model_result["elapsed_seconds"] = time.time() - model_started_at
        summary["models"][model_name] = model_result
        for target_name in TARGET_SPECS:
            result = model_result["primary_cell_centered_hidden"][target_name][
                "final"
            ]
            monotonic = model_result["controls"][target_name][
                "axis_monotonicity"
            ]["all_pairs_direction_fraction"]
            log(
                f"  {target_name}: final partial R2 "
                f"{result['partial_r2_over_iteration']:.4f}, "
                f"axis order {monotonic:.3f}"
            )
        log(f"  elapsed {model_result['elapsed_seconds']:.1f}s")
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    summary["elapsed_seconds"] = time.time() - started_at
    _write_json(output_dir / "metrics.json", summary)
    _plot_encoding(summary, output_dir / "heldout_encoding.png")
    _plot_monotonicity(summary, output_dir / "trajectory_monotonicity.png")
    _plot_iteration_profiles(summary, output_dir / "iteration_profiles.png")
    log(f"\nArtifacts written to {output_dir}")
    log(f"Total time: {summary['elapsed_seconds']:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--run-name", default="entropy_v1_20260811")
    parser.add_argument("--examples-per-bucket", type=int, default=12)
    parser.add_argument("--control-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    run(
        output_root=arguments.output_root,
        run_name=arguments.run_name,
        examples_per_bucket=arguments.examples_per_bucket,
        control_count=arguments.control_count,
        seed=arguments.seed,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
