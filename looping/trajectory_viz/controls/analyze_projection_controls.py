"""Test whether recurrent trajectory curves survive projection controls."""

import json
import os
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


DEFAULT_STRIDE = 4
DEFAULT_FINAL_ITERATION = 1024
DEFAULT_EXAMPLES_PER_BUCKET = 2


def _fit_pca(matrix, components=3):
    matrix = matrix.float()
    mean = matrix.mean(0, keepdim=True)
    centered = matrix - mean
    rank = min(components, centered.size(0), centered.size(1))
    _, singular_values, basis = torch.pca_lowrank(
        centered,
        q=rank,
        center=False,
        niter=4,
    )
    explained = singular_values.square().sum() / centered.square().sum().clamp_min(1e-30)
    return mean, basis, float(explained)


def _project(matrix, mean, basis):
    return (matrix.float() - mean) @ basis


def _normalize_rows(matrix):
    return F.normalize(matrix.float(), dim=-1, eps=1e-12)


def _path_metrics(points, order=None):
    if order is not None:
        points = points[order]
    steps = points[1:] - points[:-1]
    path_length = steps.norm(dim=-1).sum()
    displacement = (points[-1] - points[0]).norm()
    cosines = F.cosine_similarity(steps[:-1], steps[1:], dim=-1, eps=1e-12)
    return {
        "path_efficiency": float(displacement / path_length.clamp_min(1e-12)),
        "turn_cosine": float(cosines.mean()) if len(cosines) else 0.0,
    }


def _random_basis(dimension, components, seed):
    generator = torch.Generator().manual_seed(seed)
    matrix = torch.randn(dimension, components, generator=generator)
    return torch.linalg.qr(matrix, mode="reduced").Q


def collect_trajectories(model, inputs, final_iteration, stride):
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    hidden = model.initial_encoder(inputs)
    predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
    states = []
    updates = []
    iterations = []
    with torch.no_grad():
        for iteration in range(final_iteration + 1):
            if iteration % stride == 0:
                states.append(hidden.detach().float().cpu().flatten(1))
                iterations.append(iteration)
            if iteration == final_iteration:
                break
            next_hidden = model.recurrent_step(
                hidden, predictions, rope_cos, rope_sin
            )
            if iteration % stride == 0:
                updates.append(
                    (next_hidden - hidden).detach().float().cpu().flatten(1)
                )
            hidden = next_hidden
            predictions = F.softmax(model.output_head(hidden), dim=-1)
    return {
        "iterations": torch.tensor(iterations),
        "states": torch.stack(states, dim=1),
        "updates": torch.stack(updates, dim=1),
    }


def _analyze_representation(values, seed):
    puzzle_count, time_count, dimension = values.shape
    split = max(1, puzzle_count // 2)
    global_mean, global_basis, global_ev = _fit_pca(
        values[:split].flatten(0, 1)
    )
    random_bases = [_random_basis(dimension, 3, seed + i) for i in range(20)]
    generator = torch.Generator().manual_seed(seed + 1000)
    shuffled_order = torch.randperm(time_count, generator=generator)
    puzzle_results = []
    plot_data = []
    for puzzle_index in range(puzzle_count):
        trajectory = values[puzzle_index]
        local_mean, local_basis, local_ev = _fit_pca(trajectory)
        local_points = _project(trajectory, local_mean, local_basis)
        global_points = _project(trajectory, global_mean, global_basis)
        random_metrics = []
        random_points = []
        for basis in random_bases:
            points = trajectory.float() @ basis
            random_metrics.append(_path_metrics(points))
            random_points.append(points)
        metrics = {
            "puzzle": puzzle_index,
            "local_pca_explained_3d": local_ev,
            "global_train_pca_explained_3d": global_ev,
            "local": _path_metrics(local_points),
            "global": _path_metrics(global_points),
            "global_shuffled_time": _path_metrics(global_points, shuffled_order),
            "random_path_efficiency_median": float(np.median([
                item["path_efficiency"] for item in random_metrics
            ])),
            "random_turn_cosine_median": float(np.median([
                item["turn_cosine"] for item in random_metrics
            ])),
        }
        puzzle_results.append(metrics)
        plot_data.append({
            "local": local_points,
            "global": global_points,
            "random": random_points[0],
            "shuffled_order": shuffled_order,
        })
    return puzzle_results, plot_data


def _draw_panel(axis, points, title, order=None):
    points = points.numpy()
    if order is not None:
        points = points[order.numpy()]
    colors = np.linspace(0, 1, len(points))
    axis.plot(points[:, 0], points[:, 1], color="#b9c0c8", linewidth=0.7)
    scatter = axis.scatter(
        points[:, 0], points[:, 1], c=colors, cmap="viridis", s=8
    )
    axis.scatter(points[0, 0], points[0, 1], marker="s", color="#d62728", s=24)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    return scatter


def _plot_model(model_name, plot_groups, output_dir):
    representations = list(plot_groups)
    puzzle_count = len(plot_groups[representations[0]])
    shown = min(4, puzzle_count)
    for puzzle_index in range(shown):
        figure, axes = plt.subplots(
            len(representations), 4,
            figsize=(12, 2.7 * len(representations)),
            constrained_layout=True,
        )
        for row, representation in enumerate(representations):
            data = plot_groups[representation][puzzle_index]
            _draw_panel(axes[row, 0], data["local"], f"{representation}: local PCA")
            _draw_panel(axes[row, 1], data["global"], "cross-puzzle global PCA")
            _draw_panel(axes[row, 2], data["random"], "fixed random projection")
            _draw_panel(
                axes[row, 3], data["global"], "global PCA, shuffled time",
                data["shuffled_order"],
            )
        figure.suptitle(
            f"{model_name}, puzzle {puzzle_index}; color is displayed order",
            fontsize=12,
        )
        path = os.path.join(output_dir, f"{model_name}_puzzle_{puzzle_index}.png")
        figure.savefig(path, dpi=160)
        plt.close(figure)

    figure = plt.figure(figsize=(12, 3.2 * shown), constrained_layout=True)
    columns = (("local", "local PCA"), ("global", "cross-puzzle global PCA"),
               ("random", "fixed random projection"))
    for puzzle_index in range(shown):
        data = plot_groups["normalized states"][puzzle_index]
        for column, (key, title) in enumerate(columns):
            axis = figure.add_subplot(shown, 3, puzzle_index * 3 + column + 1,
                                      projection="3d")
            points = data[key].numpy()
            colors = np.linspace(0, 1, len(points))
            axis.plot(points[:, 0], points[:, 1], points[:, 2],
                      color="#b9c0c8", linewidth=0.7)
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors,
                         cmap="viridis", s=6)
            axis.set_title(f"puzzle {puzzle_index}: {title}", fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_zticks([])
    figure.suptitle(f"{model_name}: normalized states in 3D", fontsize=12)
    figure.savefig(os.path.join(output_dir, f"{model_name}_normalized_states_3d.png"),
                   dpi=160)
    plt.close(figure)


def run(
    output_dir,
    examples_per_bucket=DEFAULT_EXAMPLES_PER_BUCKET,
    final_iteration=DEFAULT_FINAL_ITERATION,
    stride=DEFAULT_STRIDE,
    seed=42,
    device="cuda",
    model_configs=DEFAULT_MODELS,
):
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, _, _, puzzles, _, buckets = _load_balanced_sample(
        examples_per_bucket, seed
    )
    inputs = inputs.to(resolved_device)
    summary = {
        "config": {
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(inputs),
            "final_iteration": final_iteration,
            "stride": stride,
            "seed": seed,
            "models": list(model_configs),
        },
        "sample": {"puzzles": puzzles, "buckets": buckets},
        "models": {},
    }
    started = time.time()
    for model_index, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        model = _load_model(model_config, resolved_device)
        collected = collect_trajectories(model, inputs, final_iteration, stride)
        representations = {
            "raw states": collected["states"],
            "normalized states": _normalize_rows(collected["states"]),
            "raw updates": collected["updates"],
            "normalized updates": _normalize_rows(collected["updates"]),
        }
        model_results = {}
        plots = {}
        for representation_index, (name, values) in enumerate(representations.items()):
            results, plot_data = _analyze_representation(
                values, seed + model_index * 100 + representation_index * 10
            )
            model_results[name] = results
            plots[name] = plot_data
        summary["models"][model_name] = model_results
        _plot_model(model_name, plots, output_dir)
        del model, collected
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    summary["elapsed_seconds"] = time.time() - started
    result_path = os.path.join(output_dir, "projection_controls.json")
    with open(result_path + ".tmp", "w") as result_file:
        json.dump(summary, result_file, indent=2)
        result_file.write("\n")
    os.replace(result_path + ".tmp", result_path)
    return summary


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu", final_iteration=64)
