"""Refresh derived null fields and comparison plots without rerunning models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from looping.trajectory_viz.helix_tests.visual.helix_geometry import (
    percentile_at,
    upper_tail_fraction,
)
from looping.trajectory_viz.helix_tests.visual.render_helix import (
    render_model_comparison,
)


def refresh(metrics_path: str) -> None:
    with open(metrics_path) as result_file:
        summary = json.load(result_file)
    for model in summary["models"].values():
        for representation in model["representations"].values():
            null = representation["periodic_readout"]["cycle_null"]
            alignments = torch.tensor(null["alignments"])
            median = float(alignments.median().item())
            for prefix in ("natural", "train_shortest"):
                observed = float(null[f"{prefix}_centroid_alignment"])
                null[f"{prefix}_alignment_percentile"] = percentile_at(
                    alignments, observed
                )
                null[f"{prefix}_alignment_upper_tail_fraction"] = (
                    upper_tail_fraction(alignments, observed)
                )
                null[f"{prefix}_alignment_advantage_over_median"] = (
                    observed - median
                )
            null["alignment_null"]["median"] = median
    temporary_path = metrics_path + ".tmp"
    with open(temporary_path, "w") as result_file:
        json.dump(summary, result_file, indent=2)
        result_file.write("\n")
    os.replace(temporary_path, metrics_path)
    render_model_comparison(
        summary["models"],
        os.path.join(os.path.dirname(metrics_path), "model_comparison.png"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "metrics_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "helix_metrics.json"),
    )
    refresh(parser.parse_args().metrics_path)
