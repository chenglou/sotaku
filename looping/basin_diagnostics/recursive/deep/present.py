"""Present both verified completed sequences in one compact comparison."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

from checkpoint_utils import validate_config
from looping.basin_diagnostics.recursive.plots import panel
from runtime_utils import file_sha256


def present_precision(folder, result):
    figure, axes = plt.subplots(2, 4, figsize=(19.5, 11.5))
    figure.subplots_adjust(left=.055, right=.97, bottom=.13, top=.85, wspace=.42, hspace=.36)
    for col, info in enumerate(result["levels"]):
        with np.load(folder / f"audit_{col}.npz", allow_pickle=False) as data:
            low = min(data["fp32_last_change"].min(), data["fp64_last_change"].min())
            high = max(data["fp32_last_change"].max(), data["fp64_last_change"].max())
            for row, precision in enumerate(("fp32", "fp64")):
                field = {name: data[f"{precision}_{name}"] for name in ("last_change", "confirmed", "separation")}
                image = panel(axes[row, col], field, info)
                image.set_clim(float(low), float(high) if high > low else float(low) + 1)
                axes[row, col].set_title(f"{info['magnification']}x | {precision.upper()}", fontsize=12, pad=12)
        stats = result["audits"][col]["fp32_vs_fp64"]
        axes[1, col].set_xlabel(f"Same settling time: {stats['same_last_change_fraction']:.1%}\nMedian difference: {stats['median_absolute_time_difference']:g} iterations", fontsize=10, labelpad=12)
    label = "Healthy at 4096" if result["identity"]["checkpoint"]["key"] == "20k_20260907" else "Collapses by 4096"
    figure.suptitle(f"Numerical precision check: {label}\n21x21 identical starting coordinates | First 1024 iterations | Shared colour range within each column", fontsize=17, y=.97)
    figure.text(.5, .025, "FP64 uses the same FP32-encoded puzzle and plane, with double-precision perturbation construction and recurrence. This is a precision check, not a new checkpoint.", ha="center", fontsize=10)
    figure.savefig(folder / "precision_comparison.png", dpi=160)
    plt.close(figure)


def present(root):
    root = Path(root)
    settings = json.loads(Path(__file__).with_name("protocol.json").read_text())
    figure, axes = plt.subplots(2, 4, figsize=(19.5, 10.5), layout="constrained")
    for row, key in enumerate(settings["models"]):
        folder = root / key
        result = json.loads((folder / "completed.json").read_text())
        if result["status"] != "complete" or len(result["levels"]) != 4:
            raise ValueError("All four levels must be complete")
        validate_config(result["identity"]["settings"], settings)
        for name, checksum in result["sha256"].items():
            if file_sha256(folder / name) != checksum:
                raise ValueError(f"Artifact changed: {key}/{name}")
        environment = json.loads((folder / "environment.json").read_text())
        if environment["matmul_precision"] != "highest":
            raise ValueError("The completed worker did not retain full FP32 matrix multiplication")
        present_precision(folder, result)
        for col, info in enumerate(result["levels"]):
            with np.load(folder / f"level_{col}.npz", allow_pickle=False) as data:
                field = {name: data[name] for name in ("last_change", "confirmed", "separation")}
            image = panel(axes[row, col], field, info, outline=col < 3)
            times = field["last_change"][field["confirmed"]]
            if len(times) and times.min() > 0 and times.max() > times.min():
                image.set_norm(LogNorm(vmin=float(times.min()), vmax=float(times.max())))
                image.colorbar.locator = FixedLocator(np.unique(np.round(np.geomspace(times.min(), times.max(), 4))))
                image.colorbar.formatter = ScalarFormatter()
                image.colorbar.update_ticks()
                image.colorbar.ax.minorticks_off()
            label = "Healthy at 4096" if row == 0 else "Collapses by 4096"
            axes[row, col].set_title(f"{label} | {info['magnification']}x", fontsize=13)
            axes[row, col].set_xlabel("Initial-state direction 1", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Initial-state direction 2", fontsize=9)
    figure.suptitle("Sotaku: deeper recursive zooms\nSame puzzle and checkpoints | 401x401 new starts per panel | First 1024 iterations", fontsize=18)
    figure.supxlabel("Colour = last answer-change iteration (local logarithmic scales). Boxes mark the next view. Each model follows its own centres. FP32; precision checks are separate.", fontsize=10)
    target = root / "deep_comparison.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    print(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).with_name("results"))
    present(parser.parse_args().directory)
