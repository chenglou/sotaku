"""Render curvature and Fourier summaries from the persisted metrics."""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(directory):
    with open(os.path.join(directory, "metrics.json")) as handle:
        results = json.load(handle)["results"]
    model_names = list(results["variants"])
    metric_specs = (
        ("turn_cosine_mean", "Consecutive velocity cosine"),
        ("relative_acceleration_mean", "Relative acceleration"),
        ("high_frequency_power_fraction", "Power at period <= 32"),
        ("dominant_period", "Dominant Fourier period"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    positions = np.arange(len(model_names))
    for axis, (metric_name, title) in zip(axes.flat, metric_specs):
        offset = -0.3
        for window_name in ("early", "late"):
            for basis_name in ("a", "b"):
                values = [
                    results["variants"][model_name][basis_name][window_name][
                        "normalized_update"
                    ]["final_summary"][metric_name]
                    for model_name in model_names
                ]
                axis.bar(
                    positions + offset,
                    values,
                    0.2,
                    label=f"{window_name}, projection {basis_name}",
                )
                offset += 0.2
        axis.set_title(title)
        axis.set_xticks(positions, model_names, rotation=20, ha="right")
        axis.axhline(0, color="black", linewidth=0.7)
        if metric_name == "dominant_period":
            axis.set_yscale("log")
        axis.legend(fontsize=8)
    figure.suptitle("Final-holdout normalized-update temporal statistics")
    figure.tight_layout()
    figure.savefig(
        os.path.join(directory, "curvature_fourier_summary.png"),
        dpi=170,
    )
    plt.close(figure)


if __name__ == "__main__":
    render(os.path.dirname(__file__))
