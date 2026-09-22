"""Settling-time and decoded-output separation panels with nested zoom boxes."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np


def panel(axis, field, info, *, separation=False, outline=False):
    center, width = info["center"], info["width"]
    extent = (center[0] - width, center[0] + width, center[1] - width, center[1] + width)
    values = field["separation"] if separation else np.ma.masked_where(~field["confirmed"], field["last_change"])
    colormap = plt.get_cmap("magma" if separation else "viridis").copy()
    colormap.set_bad("#c6c6c6")
    empty = values.count() == 0 if np.ma.isMaskedArray(values) else values.size == 0
    if empty:
        low, high = 0., 1.
    else:
        low, high = float(values.min()), float(values.max())
        if low == high:
            low, high = low - .5, high + .5
    image = axis.imshow(values, origin="lower", interpolation="nearest", extent=extent,
                        cmap=colormap, vmin=low, vmax=high, aspect="equal")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.ticklabel_format(style="sci", scilimits=(-2, 3), useOffset=False)
    axis.tick_params(labelsize=8)
    bar = axis.figure.colorbar(image, ax=axis, fraction=.043, pad=.035)
    bar.locator = MaxNLocator(4, integer=not separation)
    bar.update_ticks()
    bar.ax.tick_params(labelsize=8)
    if outline:
        child = info["next_zoom"]
        origin = np.asarray(child["center"]) - child["width"]
        axis.add_patch(Rectangle(origin, 2 * child["width"], 2 * child["width"], fill=False,
                                 edgecolor="white", linewidth=2))
        axis.add_patch(Rectangle(origin, 2 * child["width"], 2 * child["width"], fill=False,
                                 edgecolor="#1a1a1a", linewidth=.7))
    return image


def plot_sequence(folder, prepared, spec, levels):
    count = len(levels)
    figure, axes = plt.subplots(2, count, figsize=(4.8 * count, 8.4), squeeze=False, layout="constrained")
    for col, info in enumerate(levels):
        with np.load(Path(folder) / f"level_{info['level']}.npz", allow_pickle=False) as data:
            field = {key: data[key] for key in ("last_change", "confirmed", "separation")}
        for row in range(2):
            panel(axes[row, col], field, info, separation=bool(row), outline=col < count - 1)
            axes[row, col].set_xlabel("Initial-state direction 1", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Initial-state direction 2", fontsize=9)
            label = "Last answer change" if row == 0 else "Peak neighboring-output separation"
            axes[row, col].set_title(f"{info['magnification']}x zoom | {label}", fontsize=10)
        axes[0, col].text(.5, -.28, f"{info['final_correct']:.1%} correct at end; {info['confirmed']:.1%} unchanged for last 128 loops",
                          ha="center", transform=axes[0, col].transAxes, fontsize=8)
    label = "Healthy" if spec["key"] == "20k_20260907" else "Late-collapsing"
    figure.suptitle(f"Sotaku: {label.lower()} width-128 checkpoint\nPuzzle {prepared['puzzle']['index']}, difficulty 51+ | Initial state to iteration {prepared['settings']['horizon']}", fontsize=14)
    figure.supxlabel("Each zoom is newly evaluated. Colour ranges are local to each panel. Gray: answer still changed within the final 128 loops.", fontsize=9)
    figure.savefig(Path(folder) / "recursive.png", dpi=180)
    plt.close(figure)


def plot_previews(root, prepared):
    widths = prepared["settings"]["preview_rms_fractions"]
    figure, axes = plt.subplots(2, len(widths), figsize=(4 * len(widths), 7.2), layout="constrained")
    for row, key in enumerate(prepared["settings"]["models"]):
        for col, width in enumerate(widths):
            path = root / "previews" / key / f"width_{width:g}" / "field.npz"
            with np.load(path) as data:
                field = {name: data[name] for name in ("last_change", "confirmed", "separation")}
            info = {"center": [0., 0.], "width": width}
            panel(axes[row, col], field, info)
            axes[row, col].set_title(f"{key}\nPer-axis RMS {width:.0%}", fontsize=10)
    figure.suptitle(f"Framing previews only: 33x33 grids through 512 iterations | Selected per-axis RMS {prepared['width']:.0%}")
    figure.savefig(root / "framing_previews.png", dpi=150)
    plt.close(figure)


def comparison(root):
    root = Path(root)
    prepared = json.loads((root / "prepared.json").read_text())
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9), layout="constrained")
    for row, key in enumerate(prepared["settings"]["models"]):
        result = json.loads((root / key / "completed.json").read_text())
        for col, info in enumerate(result["levels"]):
            with np.load(root / key / f"level_{col}.npz") as data:
                field = {name: data[name] for name in ("last_change", "confirmed", "separation")}
            panel(axes[row, col], field, info, outline=col < 2)
            label = "Healthy" if row == 0 else "Late-collapsing"
            axes[row, col].set_title(f"{label} | {info['magnification']}x zoom", fontsize=12)
            axes[row, col].set_xlabel("Initial-state direction 1", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Initial-state direction 2", fontsize=9)
            axes[row, col].text(.5, -.25, f"{info['final_correct']:.1%} correct at end; {info['distinct_confirmed_times']} settling-time values",
                               ha="center", transform=axes[row, col].transAxes, fontsize=9)
    figure.suptitle(f"Sotaku: recursive zooms of answer-settling time\nSame hard puzzle {prepared['puzzle']['index']}; 201x201 newly evaluated starts per panel; through iteration {prepared['settings']['horizon']}", fontsize=15)
    figure.supxlabel("Boxes show the next zoom. Zoom locations are chosen separately for each model. Each colour bar shows its own iteration range. Gray: not settled.", fontsize=9)
    figure.savefig(root / "comparison.png", dpi=180)
    plt.close(figure)
