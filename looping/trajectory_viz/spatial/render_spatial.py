"""Render per-cell recurrent geometry for healthy and collapsed Sudoku models."""

import base64
import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model


MODELS = {
    "healthy": "/outputs/model_baseline_lr2e3.pt",
    "collapsed": "/outputs/model_baseline_lr2e3_clean_a.pt",
}
SNAPSHOTS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _collect(model, inputs):
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    states = []
    updates = []
    predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
    with torch.no_grad():
        hidden = model.initial_encoder(inputs)
        for iteration in range(max(SNAPSHOTS) + 1):
            if iteration in SNAPSHOTS:
                states.append(hidden.float().cpu())
            next_hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
            if iteration in SNAPSHOTS:
                updates.append((next_hidden - hidden).float().cpu())
            hidden = next_hidden
            predictions = F.softmax(model.output_head(hidden), dim=-1)
    return torch.stack(states, 1), torch.stack(updates, 1)


def _pca_basis(updates):
    samples = updates.flatten(0, 2).double()
    mean = samples.mean(0)
    _, _, basis = torch.pca_lowrank(samples - mean, q=3, center=False, niter=6)
    return mean.float(), basis.float()


def _project(updates, mean, basis):
    return (updates - mean) @ basis


def _grid(ax, values, title, cmap="viridis", symmetric=False):
    values = np.asarray(values).reshape(9, 9)
    limit = np.nanmax(np.abs(values)) if symmetric else None
    image = ax.imshow(
        values,
        cmap=cmap,
        vmin=-limit if symmetric else None,
        vmax=limit if symmetric else None,
    )
    for edge in (2.5, 5.5):
        ax.axhline(edge, color="white", lw=1.2, alpha=.8)
        ax.axvline(edge, color="white", lw=1.2, alpha=.8)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def _save_figure(figure, path):
    figure.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render(output_dir, examples_per_bucket=2, device="cuda", seed=20260807):
    os.makedirs(output_dir, exist_ok=True)
    inputs, targets, empty_mask, _, _, _ = _load_balanced_sample(
        examples_per_bucket,
        seed,
    )
    inputs = inputs.to(device)
    targets = targets.to(device)
    empty_mask = empty_mask.to(device)
    collected = {}
    for name, path in MODELS.items():
        model = _load_model({"path": path, "model_kwargs": {}}, device)
        collected[name] = _collect(model, inputs)

    all_updates = torch.cat([value[1] for value in collected.values()], dim=0)
    mean, basis = _pca_basis(all_updates)
    projected = {
        name: _project(updates, mean, basis)
        for name, (_, updates) in collected.items()
    }

    artifacts = []
    for puzzle in range(min(4, inputs.size(0))):
        figure, axes = plt.subplots(4, len(SNAPSHOTS), figsize=(22, 7.2))
        for column, iteration in enumerate(SNAPSHOTS):
            for row, name in enumerate(("healthy", "collapsed")):
                update = collected[name][1][puzzle, column]
                magnitude = update.norm(dim=-1).numpy()
                angle = torch.atan2(
                    projected[name][puzzle, column, :, 1],
                    projected[name][puzzle, column, :, 0],
                ).numpy()
                _grid(axes[row * 2, column], magnitude, f"{name} |t={iteration}")
                _grid(
                    axes[row * 2 + 1, column],
                    angle,
                    "PC phase",
                    cmap="twilight",
                    symmetric=True,
                )
        figure.suptitle(f"Puzzle {puzzle}: update magnitude and shared-PC phase", fontsize=14)
        path = os.path.join(output_dir, f"puzzle_{puzzle}_spatial.png")
        _save_figure(figure, path)
        artifacts.append(os.path.basename(path))

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    summary = {}
    for row, name in enumerate(("healthy", "collapsed")):
        coordinates = projected[name]
        mean_norm = collected[name][1].norm(dim=-1).mean(dim=(0, 2)).numpy()
        coherence = (
            coordinates[..., :2].mean(dim=2).norm(dim=-1)
            / coordinates[..., :2].norm(dim=-1).mean(dim=2).clamp_min(1e-8)
        ).mean(dim=0).numpy()
        angle = torch.atan2(coordinates[..., 1], coordinates[..., 0])
        phase_step = torch.angle(
            torch.exp(1j * (angle[:, 1:] - angle[:, :-1])).mean(dim=(0, 2))
        ).numpy()
        axes[row, 0].plot(SNAPSHOTS, mean_norm, marker="o")
        axes[row, 1].plot(SNAPSHOTS, coherence, marker="o")
        axes[row, 2].plot(SNAPSHOTS[1:], phase_step, marker="o")
        for column, title in enumerate(("Mean update norm", "Spatial phase coherence", "Mean phase change")):
            axes[row, column].set_xscale("symlog", linthresh=1)
            axes[row, column].set_title(f"{name}: {title}")
            axes[row, column].grid(alpha=.25)
        summary[name] = {
            "mean_update_norm": mean_norm.tolist(),
            "spatial_phase_coherence": coherence.tolist(),
            "mean_phase_step": phase_step.tolist(),
        }
    path = os.path.join(output_dir, "aggregate_geometry.png")
    _save_figure(figure, path)
    artifacts.append(os.path.basename(path))

    report = {
        "snapshots": list(SNAPSHOTS),
        "models": MODELS,
        "examples": int(inputs.size(0)),
        "summary": summary,
        "artifacts": artifacts,
    }
    with open(os.path.join(output_dir, "metrics.json"), "w") as handle:
        json.dump(report, handle, indent=2)

    cards = []
    for filename in artifacts:
        with open(os.path.join(output_dir, filename), "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        cards.append(f'<section><h2>{filename}</h2><img src="data:image/png;base64,{encoded}"></section>')
    html = """<!doctype html><meta charset=utf-8><title>Sotaku spatial trajectories</title>
<style>body{font:14px system-ui;margin:24px;background:#f4f4f1;color:#171717}section{margin:0 0 32px}img{max-width:100%;background:white;border:1px solid #ccc}h1,h2{letter-spacing:0}</style>
<h1>Sotaku recurrent spatial trajectories</h1><p>Healthy and collapsed checkpoints use one shared PCA basis. Rows alternate update magnitude and phase in the first two shared feature directions.</p>""" + "".join(cards)
    with open(os.path.join(output_dir, "index.html"), "w") as handle:
        handle.write(html)
    return report
