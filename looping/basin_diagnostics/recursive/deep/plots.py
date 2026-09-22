"""Deeper zooms and explicitly labelled precision comparisons."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from looping.basin_diagnostics.recursive.plots import panel


def render(folder, prepared, levels, audits):
    folder = Path(folder)
    count = len(levels)
    figure, axes = plt.subplots(2, 4, figsize=(19.5, 10), layout="constrained")
    for col in range(4):
        if col >= count:
            for axis in axes[:, col]:
                axis.set_axis_off()
            continue
        info = levels[col]
        with np.load(folder / f"level_{info['level']}.npz", allow_pickle=False) as data:
            field = {name: data[name] for name in ("last_change", "confirmed", "separation")}
        for row in range(2):
            image = panel(axes[row, col], field, info, separation=bool(row), outline=col < count - 1)
            if row == 0:
                times = field["last_change"][field["confirmed"]]
                if len(times) and times.min() > 0 and times.max() > times.min():
                    image.set_norm(LogNorm(vmin=float(times.min()), vmax=float(times.max())))
            label = "Last answer change" if row == 0 else "Neighboring-output separation"
            axes[row, col].set_title(f"{info['magnification']}x | {label}", fontsize=12)
            axes[row, col].set_xlabel("Initial-state direction 1", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Initial-state direction 2", fontsize=9)
    key = prepared["checkpoint"]["key"]
    label = "Healthy at 4096" if key == "20k_20260907" else "Collapses by 4096"
    figure.suptitle(f"Sotaku: {label} | Puzzle {prepared['puzzle']['index']}\n401x401 new starts per panel | First 1024 iterations | FP32", fontsize=17)
    figure.supxlabel("Boxes mark the next view. Local colour scales: logarithmic for answer-change time, linear for separation. Gray = changed in the final 128 iterations.", fontsize=10)
    figure.savefig(folder / "deep_zoom.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 4, figsize=(19.5, 10), layout="constrained")
    for col in range(4):
        if col >= count:
            for axis in axes[:, col]:
                axis.set_axis_off()
            continue
        with np.load(folder / f"audit_{col}.npz", allow_pickle=False) as data:
            low = min(data["fp32_last_change"].min(), data["fp64_last_change"].min())
            high = max(data["fp32_last_change"].max(), data["fp64_last_change"].max())
            for row, precision in enumerate(("fp32", "fp64")):
                field = {name: data[f"{precision}_{name}"] for name in ("last_change", "confirmed", "separation")}
                image = panel(axes[row, col], field, levels[col])
                image.set_clim(float(low), float(high) if high > low else float(low) + 1)
                axes[row, col].set_title(f"{levels[col]['magnification']}x | {precision.upper()}", fontsize=12)
            stats = audits[col]["fp32_vs_fp64"]
            axes[1, col].set_xlabel(f"Same settling time: {stats['same_last_change_fraction']:.1%}\nMedian difference: {stats['median_absolute_time_difference']:g} iterations", fontsize=10)
    figure.suptitle(f"Numerical precision check: {label}\n21x21 identical coordinates per panel | Shared colour range within each column", fontsize=17)
    figure.supxlabel("FP64 retains the original encoded puzzle and plane, then constructs perturbations and runs the recurrence in double precision. It is an analysis check, not a new checkpoint.", fontsize=10)
    figure.savefig(folder / "precision_check.png", dpi=150)
    plt.close(figure)
