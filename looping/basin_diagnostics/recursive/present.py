"""Present completed zoom arrays without changing the frozen GPU renderer."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, ScalarFormatter
import numpy as np

from looping.basin_diagnostics.recursive.plots import comparison, panel
from runtime_utils import file_sha256


def present(root):
    root = Path(root)
    prepared = json.loads((root / "prepared.json").read_text())
    results = []
    for key in prepared["settings"]["models"]:
        folder = root / key
        result = json.loads((folder / "completed.json").read_text())
        if result["status"] != "complete" or len(result["levels"]) != 3:
            raise ValueError("All three zoom levels must be complete")
        if result["identity"]["prepared_sha256"] != file_sha256(root / "prepared.json"):
            raise ValueError("Maps used a different puzzle or framing")
        for filename, checksum in result["sha256"].items():
            if file_sha256(folder / filename) != checksum:
                raise ValueError(f"Downloaded artifact changed: {key}/{filename}")
        results.append((key, result))
    comparison(root)
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), layout="constrained")
    for row, (key, result) in enumerate(results):
        for col, info in enumerate(result["levels"]):
            with np.load(root / key / f"level_{col}.npz", allow_pickle=False) as data:
                field = {name: data[name] for name in ("last_change", "confirmed", "separation")}
            image = panel(axes[row, col], field, info, outline=col < 2)
            times = field["last_change"][field["confirmed"]]
            if len(times) and times.min() > 0 and times.max() > times.min():
                image.set_norm(LogNorm(vmin=float(times.min()), vmax=float(times.max())))
                ticks = np.unique(np.round(np.geomspace(times.min(), times.max(), 4)))
                image.colorbar.locator = FixedLocator(ticks)
                image.colorbar.formatter = ScalarFormatter()
                image.colorbar.update_ticks()
                image.colorbar.ax.minorticks_off()
            label = "Healthy at 4096" if row == 0 else "Collapses by 4096"
            axes[row, col].set_title(f"{label} | {info['magnification']}x zoom", fontsize=12)
            axes[row, col].set_xlabel("Initial-state direction 1", fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel("Initial-state direction 2", fontsize=9)
    figure.suptitle(f"Sotaku: recursive zooms into the solving process\nPuzzle {prepared['puzzle']['index']} | First 1024 iterations | 201x201 new evaluations per panel", fontsize=15)
    figure.supxlabel("Colour = last answer-change iteration, logarithmic scale. Boxes mark the next view. Each model follows its own zoom centres. Gray = not settled.", fontsize=9)
    figure.savefig(root / "recursive_comparison.png", dpi=180)
    plt.close(figure)
    print(root / "recursive_comparison.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).with_name("results_1024"))
    present(parser.parse_args().directory)
