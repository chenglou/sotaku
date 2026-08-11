"""Visualize recurrent hidden-state trajectories in a shared PCA per model."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


DEFAULT_ITERATIONS = tuple(range(0, 65, 2)) + tuple(range(72, 1025, 8))


def fit_pca(values, component_count=3):
    values = values.float()
    mean = values.mean(dim=0, keepdim=True)
    centered = values - mean
    _, singular_values, basis = torch.pca_lowrank(
        centered,
        q=min(component_count, *centered.shape),
        center=False,
        niter=6,
    )
    explained = singular_values.square() / centered.square().sum().clamp_min(1e-30)
    return mean, basis, explained


def collect_states(model, inputs, iterations):
    requested = set(iterations)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    hidden = model.initial_encoder(inputs)
    predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
    states = []
    with torch.no_grad():
        for iteration in range(max(iterations) + 1):
            if iteration in requested:
                states.append(hidden.detach().float().cpu().flatten(1))
            if iteration == max(iterations):
                break
            hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
            predictions = F.softmax(model.output_head(hidden), dim=-1)
    return torch.stack(states, dim=1)


def project_global(states, fit_puzzle_count):
    train = states[:fit_puzzle_count].flatten(0, 1)
    mean, basis, explained = fit_pca(train)
    projected = (states.float() - mean) @ basis
    reconstruction = (states.float() - mean) @ basis @ basis.T
    heldout = states[fit_puzzle_count:].float() - mean
    heldout_explained = reconstruction[fit_puzzle_count:].square().sum() / heldout.square().sum().clamp_min(1e-30)
    return projected, explained, float(heldout_explained)


def _draw_path(axis, points, iterations, color, label=None, late_start=0):
    mask = iterations >= late_start
    points = points[mask]
    shown_iterations = iterations[mask]
    axis.plot(points[:, 0], points[:, 1], color=color, alpha=0.8, linewidth=1.0, label=label)
    axis.scatter(points[:, 0], points[:, 1], c=shown_iterations, cmap="viridis", s=7, alpha=0.75)
    axis.scatter(points[0, 0], points[0, 1], marker="s", color=color, s=24)


def plot_model(model_name, projected, iterations, output_dir):
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")
    shown = min(5, projected.size(0))
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    views = (
        (0, 1, 0, "PC1 / PC2, full trajectory"),
        (0, 2, 0, "PC1 / PC3, full trajectory"),
        (1, 2, 0, "PC2 / PC3, full trajectory"),
        (0, 1, 128, "PC1 / PC2, iterations 128-1024"),
        (0, 2, 128, "PC1 / PC3, iterations 128-1024"),
        (1, 2, 128, "PC2 / PC3, iterations 128-1024"),
    )
    for axis, (x, y, late_start, title) in zip(axes.flat, views):
        for puzzle_index in range(shown):
            points = projected[puzzle_index][:, [x, y]].numpy()
            _draw_path(axis, points, np.asarray(iterations), colors[puzzle_index], f"puzzle {puzzle_index}", late_start)
        axis.set_title(title)
        axis.set_xlabel(f"PC{x + 1}")
        axis.set_ylabel(f"PC{y + 1}")
        axis.set_aspect("equal", adjustable="datalim")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(f"{model_name}: one PCA basis shared across puzzles and iterations")
    path = os.path.join(output_dir, f"{model_name}_global_pca.png")
    figure.savefig(path, dpi=180)
    plt.close(figure)

    centered = projected - projected[:, :1]
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, late_start, title in zip(axes, (0, 128), ("start-centered, full", "start-centered, iterations 128-1024")):
        for puzzle_index in range(shown):
            points = centered[puzzle_index, :, :2].numpy()
            _draw_path(axis, points, np.asarray(iterations), colors[puzzle_index], f"puzzle {puzzle_index}", late_start)
        axis.set_title(title)
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_aspect("equal", adjustable="datalim")
    path = os.path.join(output_dir, f"{model_name}_start_centered.png")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(output_dir, examples_per_bucket=2, seed=42, device="cuda", model_configs=DEFAULT_MODELS):
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, _, _, _, _, _ = _load_balanced_sample(examples_per_bucket, seed)
    inputs = inputs.to(resolved_device)
    fit_puzzle_count = max(1, inputs.size(0) // 2)
    report = {"iterations": list(DEFAULT_ITERATIONS), "models": {}}
    for config in model_configs:
        model = _load_model(config, resolved_device)
        states = collect_states(model, inputs, DEFAULT_ITERATIONS)
        projected, explained, heldout_explained = project_global(states, fit_puzzle_count)
        plot_model(config["name"], projected, DEFAULT_ITERATIONS, output_dir)
        report["models"][config["name"]] = {
            "pca_explained_variance": explained.tolist(),
            "heldout_three_component_explained": heldout_explained,
            "state_shape": list(states.shape),
        }
        del model, states
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    with open(os.path.join(output_dir, "global_pca_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    return report


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu")
