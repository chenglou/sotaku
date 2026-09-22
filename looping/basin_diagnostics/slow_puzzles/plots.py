"""Recursive settling-time figures with explicit unsuccessful outcomes."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np


def panel(axis, field, info, *, separation=False):
    center, width = info["center"], info["width"]
    extent = [center[0] - width, center[0] + width, center[1] - width, center[1] + width]
    if separation:
        values = np.asarray(field["separation"])
        image = axis.imshow(values, origin="lower", extent=extent, cmap="magma", interpolation="nearest",
                            vmin=0, vmax=max(1., float(values.max())))
    else:
        valid = field["successful"]
        values = np.ma.masked_where(~valid, field["last_change"])
        times = field["last_change"][valid]
        low, high = (max(1, int(times.min())), max(2, int(times.max()))) if len(times) else (1, 2)
        high = max(high, low + 1)
        palette = plt.get_cmap("viridis").copy()
        palette.set_bad("#d4d4d4")
        image = axis.imshow(values, origin="lower", extent=extent, cmap=palette,
                            norm=LogNorm(vmin=low, vmax=high), interpolation="nearest")
        wrong = field["confirmed_answer"] & ~field["final_correct"] & ~field["ever_nonfinite"]
        overlay = np.zeros((*wrong.shape, 4))
        overlay[wrong] = [.55, .10, .18, 1.]
        axis.imshow(overlay, origin="lower", extent=extent, interpolation="nearest")
    axis.xaxis.set_major_locator(MaxNLocator(3))
    axis.yaxis.set_major_locator(MaxNLocator(3))
    axis.ticklabel_format(style="sci", scilimits=(-2, 2), useOffset=False)
    axis.tick_params(labelsize=8)
    return image


def plot_case(folder, puzzle, settings, levels):
    folder = Path(folder)
    figure, axes = plt.subplots(2, settings["levels"], figsize=(19, 7.8), squeeze=False)
    for column, info in enumerate(levels):
        with np.load(folder / f"level_{info['level']}.npz", allow_pickle=False) as data:
            field = {key: data[key].copy() for key in data.files}
        for row in range(2):
            axis = axes[row, column]
            image = panel(axis, field, info, separation=row == 1)
            figure.colorbar(image, ax=axis, fraction=.046, pad=.025).ax.tick_params(labelsize=8)
            if column < settings["levels"] - 1:
                child = info["next_zoom"]
                axis.add_patch(Rectangle((child["center"][0] - child["width"], child["center"][1] - child["width"]),
                                         2 * child["width"], 2 * child["width"], fill=False, edgecolor="white", linewidth=1.1))
        axes[0, column].set_title(f"{info['magnification']}x | {100 * info['successful_fraction']:.1f}% settled correctly", fontsize=10, pad=12)
    for column in range(len(levels), settings["levels"]):
        for axis in axes[:, column]:
            axis.axis("off")
    axes[0, 0].set_ylabel("Last answer-change iteration", fontsize=11)
    axes[1, 0].set_ylabel("Peak neighboring-board separation", fontsize=11)
    figure.suptitle(f"Sotaku: {puzzle['model_key']} | {puzzle['case']} puzzle {puzzle['index']}\n"
                   f"FP64 throughout | {settings['resolution']}x{settings['resolution']} fresh starts per view | "
                   f"{settings['horizon']} iterations\n"
                   f"Nine-start screening mean: {puzzle['mean_last_change']:.1f} iterations", fontsize=16, y=.97)
    figure.legend(handles=[Patch(color="#d4d4d4", label="Not settled / nonfinite"),
                           Patch(color="#8c1a2e", label="Settled wrong answer")], loc="lower center", ncol=2,
                  frameon=False, bbox_to_anchor=(.5, .025), fontsize=10)
    figure.text(.5, .085, "Coordinates are fractions of initial-state RMS. Local color ranges; boxes mark the next evaluated region.",
                ha="center", fontsize=10)
    figure.subplots_adjust(left=.045, right=.985, top=.78, bottom=.16, wspace=.45, hspace=.35)
    figure.savefig(folder / "recursive_fp64.png", dpi=170)
    plt.close(figure)


def plot_orientation_control(folder, puzzle, settings, field):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.8))
    info = {"center": [0., 0.], "width": settings["initial_half_width_rms"]}
    for index, axis in enumerate(axes):
        image = panel(axis, field, info, separation=index == 1)
        figure.colorbar(image, ax=axis, fraction=.046, pad=.04)
        axis.set_title(("Last answer-change iteration", "Peak neighboring-board separation")[index], fontsize=11)
    figure.suptitle(f"Second-plane control: {puzzle['model_key']} | {puzzle['case']} puzzle {puzzle['index']}\n"
                   f"FP64 | {settings['orientation_control_resolution']}x{settings['orientation_control_resolution']} starts | "
                   f"{100 * field['successful'].mean():.1f}% settled correctly", fontsize=12)
    figure.subplots_adjust(top=.78, bottom=.15, wspace=.35)
    figure.savefig(Path(folder) / "orientation_control.png", dpi=160)
    plt.close(figure)
