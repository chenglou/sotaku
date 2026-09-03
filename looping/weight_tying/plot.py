"""Plot individual training seeds and the locked final-checkpoint evaluations."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from looping.weight_tying.common import protocol, run_name

NAMES = {"tied": "Shared weights", "untied_compute": "Untied: matched compute",
         "untied_parameters": "Untied: matched parameters"}
COLORS = {"tied": "#009E73", "untied_compute": "#CC5877", "untied_parameters": "#0072B2"}
SEED_COLORS = ("#0072B2", "#D55E00", "#009E73")


def style_axis(axis):
    axis.set_ylim(0, 102)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure, path):
    if path.exists():
        plt.close(figure)
        raise FileExistsError(f"Refusing to overwrite figure: {path}")
    figure.savefig(path, dpi=160, facecolor="white")
    plt.close(figure)
    return path


def plot_training(report, histories, directory):
    settings = protocol()
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True, sharey=True)
    for row, regime in enumerate(settings["regimes"]):
        iteration = "16" if regime == "early" else "1024"
        for column, architecture in enumerate(settings["architectures"]):
            axis = axes[row, column]
            count = 0
            for seed, color in zip(settings["seeds"], SEED_COLORS):
                history = histories.get(run_name(architecture, regime, seed), [])
                if not history:
                    continue
                count += 1
                axis.plot([item["updates"] for item in history],
                          [100 * item["scores"][iteration]["accuracy"] for item in history],
                          color=color, linewidth=1.6, label=f"Seed {seed}")
            if not count:
                axis.text(0.5, 0.5, "No completed run", ha="center", transform=axis.transAxes, color="#666666")
            for step in (4000, 8000, 12000):
                axis.axvline(step, color="#BBBBBB", linestyle=":", linewidth=0.7)
            axis.set_title(f"{NAMES[architecture]}\n{'Early' if regime == 'early' else 'Late-state'} training, evaluated at {iteration}", fontsize=10)
            axis.set_xlim(0, 20000)
            axis.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5K", "10K", "15K", "20K"])
            style_axis(axis)
            if column == 0:
                axis.set_ylabel("Validation puzzles solved (%)")
            if row == 1:
                axis.set_xlabel("Optimizer updates")
    handles = [plt.Line2D([], [], color=color, label=f"Seed {seed}")
               for seed, color in zip(settings["seeds"], SEED_COLORS)]
    figure.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.005))
    title = "Synthetic layout check (not study results)" if report.get("synthetic") else "Validation during training" + (" (partial report)" if report["partial"] else "")
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    return save_figure(figure, directory / "training_curves.png")


def plot_horizons(report, directory, dataset):
    settings = protocol()
    horizons = settings["evaluation"]["iterations"]
    x = np.log2(horizons)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.6), sharey=True)
    for axis, regime in zip(axes, settings["regimes"]):
        for architecture in settings["architectures"]:
            traces = []
            style = "--" if regime == "early" and architecture != "tied" else "-"
            for seed in settings["seeds"]:
                summary = report["runs"].get(run_name(architecture, regime, seed), {})
                scores = summary.get("evaluations", {}).get("final", {}).get(dataset)
                if scores is None:
                    continue
                values = np.asarray([100 * scores[str(horizon)]["accuracy"] for horizon in horizons])
                traces.append(values)
                axis.plot(x, values, color=COLORS[architecture], alpha=0.25, linewidth=1, linestyle=style)
            if traces:
                axis.plot(x, np.mean(traces, axis=0), color=COLORS[architecture], linewidth=2.2,
                          marker="o", markersize=4, linestyle=style,
                          label=f"{NAMES[architecture]} ({len(traces)}/3)")
        axis.set_title("Early training: primary at 16" if regime == "early" else "Late-state training: primary at 1024", fontsize=11)
        axis.set_xticks(x, [str(horizon) for horizon in horizons])
        axis.set_xlabel("Inference iterations (log scale)")
        style_axis(axis)
        if axis.lines:
            axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=8, frameon=False)
    axes[0].set_ylabel("Puzzles solved (%)")
    label = "New 10K test set" if dataset == "holdout" else "Reused 25K development benchmark"
    title = "Synthetic layout check (not study results)" if report.get("synthetic") else f"{label}: final checkpoints" + (" (partial report)" if report["partial"] else "")
    figure.suptitle(title, fontsize=14)
    figure.text(0.5, 0.02, "Thin lines: individual seeds. Thick lines: mean of evaluated seeds; failures remain in the report.\n"
                "Dashed lines beyond 16 repeat an early-trained untied stack, not additional independently trained layers.",
                ha="center", fontsize=8)
    figure.tight_layout(rect=(0, 0.075, 1, 0.94))
    return save_figure(figure, directory / f"{dataset}_final_horizons.png")


def render_report(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    histories = json.loads((directory / "learning_curves.json").read_text())
    paths = [plot_training(report, histories, directory)]
    for dataset in ("development", "holdout"):
        if any(dataset in result.get("evaluations", {}).get("final", {}) for result in report["runs"].values()):
            paths.append(plot_horizons(report, directory, dataset))
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    for path in render_report(arguments.directory):
        print(path)
