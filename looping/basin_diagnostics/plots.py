"""Static scientific figures; no projection search or answer selection."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def pca_paths(states):
    shape = states.shape[:2]
    values = states.reshape(-1, int(np.prod(states.shape[2:]))).astype(np.float64)
    centered = values - values.mean(0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    coordinates = vectors[:, order] * np.sqrt(np.maximum(eigenvalues[order], 0))[None]
    return coordinates.reshape(*shape, 2), float(np.maximum(eigenvalues[order], 0).sum() / max(np.trace(gram), 1e-30))


def render_map(input_path, output_path, spec, info):
    with np.load(input_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    resolution, horizon = info["resolution"], info["horizon"]
    count = resolution ** 2
    figure, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    fields = [
        ("First correct iteration", arrays["first_correct"][:count].astype(float), "viridis", info["anchor"], horizon),
        ("Last answer change (finite run)", arrays["last_change"][:count].astype(float), "viridis", info["anchor"], horizon),
        ("Outcome through final iteration", arrays["outcome"][:count], ListedColormap(["#808080", "#c43e45", "#d39b1a", "#23866b"]), 0, 3),
        ("Correct-to-incorrect transitions", arrays["regressions"][:count], "magma", 0, max(1, int(arrays["regressions"][:count].max()))),
    ]
    fields[0][1][fields[0][1] > horizon] = np.nan
    for axis, (title, field, cmap, low, high) in zip(axes[0], fields):
        colormap = plt.get_cmap(cmap).copy() if isinstance(cmap, str) else cmap.copy()
        colormap.set_bad("#c9c9c9")
        image = axis.imshow(field.reshape(resolution, resolution), origin="lower", interpolation="nearest",
                            extent=(-1, 1, -1, 1), cmap=colormap, vmin=low, vmax=high)
        axis.set(title=title, xlabel="Perturbation direction 1", ylabel="Perturbation direction 2")
        if title.startswith("Outcome"):
            bar = figure.colorbar(image, ax=axis, ticks=[0, 1, 2, 3], shrink=0.75)
            bar.ax.set_yticklabels(["Never correct", "Lost at end", "Recovered", "Stayed correct"], fontsize=8)
        else:
            figure.colorbar(image, ax=axis, shrink=0.75)
    gain = np.log10(np.maximum(arrays["max_neighbor"], 1e-12) / np.maximum(arrays["initial_neighbor"], 1e-12))
    image = axes[1, 0].imshow(gain, origin="lower", interpolation="nearest", extent=(-1, 1, -1, 1), cmap="magma", vmin=0)
    axes[1, 0].set(title="Peak neighbor separation / initial (log10)", xlabel="Direction 1", ylabel="Direction 2")
    figure.colorbar(image, ax=axes[1, 0], shrink=0.75)
    paths, variance = pca_paths(arrays["hidden_snapshots"])
    names, colors = ("Unperturbed", "Adjacent x", "Adjacent y"), ("#202020", "#a33d7c", "#2781b2")
    for index, (name, color) in enumerate(zip(names, colors)):
        axes[1, 1].plot(paths[:, index, 0], paths[:, index, 1], color=color, label=name, linewidth=1)
        axes[1, 1].scatter(*paths[0, index], color=color, marker="o", s=20)
        axes[1, 1].scatter(*paths[-1, index], color=color, marker="x", s=30)
        axes[1, 2].plot(arrays["steps"], arrays["minimum_margin"][:, index], color=color, label=name)
    axes[1, 1].set(title=f"Three neighboring paths, PCA ({variance:.1%})", xlabel="PC 1", ylabel="PC 2")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].axhline(0, color="#777777", linewidth=0.8)
    axes[1, 2].set(title="Weakest correct-digit margin", xlabel="Absolute iteration", ylabel="Correct score minus best wrong score")
    steps = arrays["steps"]
    axes[1, 3].plot(steps, arrays["correct_curve"][:, :count].mean(-1) * 100, color="#23866b", label="Grid")
    axes[1, 3].plot(steps, arrays["correct_curve"][:, count // 2] * 100, color="#202020", alpha=0.65, label="Unperturbed")
    axes[1, 3].set(title="Solved fraction", xlabel="Absolute iteration", ylabel="Percent", ylim=(-3, 103))
    axes[1, 3].legend(fontsize=8)
    figure.suptitle(f"{spec['key']} | {spec['expected_status']} | puzzle {info['gallery']['index']} "
                   f"({info['gallery']['difficulty']})\nNudge at {info['anchor']}; per-axis RMS {info['rms_fraction']:.1%} "
                   f"of state; plane {info['plane_seed']}. Gray first-hit pixels never solve. No infinite-time convergence claim.", fontsize=12)
    figure.savefig(output_path, dpi=130)
    plt.close(figure)


def render_probe(directory, settings, spec, names):
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    with np.load(directory / "baseline.npz", allow_pickle=False) as arrays:
        axes[0].plot(arrays["steps"], 100 * arrays["correct_curve"].mean(-1), color="#222222")
    records = []
    for name in names:
        with np.load(directory / f"{name}.npz", allow_pickle=False) as data:
            count = len(data["final_correct"]) // 5
            final = data["final_correct"].reshape(count, 5)
            regressions = data["regressions"].reshape(count, 5)
            zero = final[:, 0]
            records.append({"name": name, "n_puzzles": count, "zero_final": float(zero.mean()),
                            "perturbed_final": float(final[:, 1:].mean()),
                            "rescued_fraction": float((~zero[:, None] & final[:, 1:]).mean()),
                            "harmed_fraction": float((zero[:, None] & ~final[:, 1:]).mean()),
                            "perturbed_regressed": float((regressions[:, 1:] > 0).mean())})
    positions = np.arange(len(records))
    labels = [item["name"].replace("probe_", "").replace("_", "\n") for item in records]
    axes[1].bar(positions - .18, [100 * item["zero_final"] for item in records], .36, color="#777777", label="Zero nudge")
    axes[1].bar(positions + .18, [100 * item["perturbed_final"] for item in records], .36, color="#2781b2", label="Nudged")
    axes[2].bar(positions - .18, [100 * item["rescued_fraction"] for item in records], .36, color="#23866b", label="Rescued")
    axes[2].bar(positions + .18, [100 * item["harmed_fraction"] for item in records], .36, color="#c43e45", label="Harmed")
    axes[0].set(title="Unperturbed fixed puzzle sample", xlabel="Iteration", ylabel="Percent solved", ylim=(-3, 103))
    axes[1].set(title="Final accuracy: paired perturbations", ylabel="Percent solved", ylim=(-3, 103), xticks=positions, xticklabels=labels)
    axes[2].set(title="Changed final outcome vs zero nudge", ylabel="Percent of perturbations", xticks=positions, xticklabels=labels)
    for axis in axes[1:]:
        axis.legend(fontsize=8)
    figure.suptitle(f"{spec['key']} | {spec['expected_status']} | {len(records) and records[0]['n_puzzles']} fixed puzzles; exploratory sample")
    figure.savefig(directory / "probe_summary.png", dpi=150)
    plt.close(figure)
    from checkpoint_utils import atomic_json_save
    atomic_json_save(records, directory / "probe_summary.json")


def comparison(output_dir):
    root = Path(output_dir)
    directories = sorted((root / "models").glob("*"))
    directories = [directory for directory in directories if (directory / "completed.json").exists()]
    if not directories:
        raise ValueError("No completed models")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    colors = {"20260907": "#23866b", "20260908": "#c43e45", "20260909": "#2781b2",
              "20260910": "#bd7020", "20260911": "#23866b", "20260912": "#2781b2"}
    for directory in directories:
        cohort, seed = directory.name.split("_")
        with np.load(directory / "baseline.npz", allow_pickle=False) as arrays:
            axis = axes[0 if cohort == "20k" else 1]
            axis.plot(arrays["steps"], 100 * arrays["correct_curve"].mean(-1), label=seed, color=colors[seed])
    for axis, cohort in zip(axes, ("20K training", "50K training")):
        axis.set(title=cohort, xlabel="Inference iteration", ylabel="Percent solved", ylim=(-3, 103))
        axis.legend()
    figure.suptitle("Same fixed 50-puzzle sample; all seeds shown within their training budget")
    figure.savefig(root / "comparison.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    comparison(args.output_dir)
