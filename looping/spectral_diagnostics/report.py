"""Summarize only completed, checksum-verified diagnostic runs."""

import json
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save
from runtime_utils import file_sha256

NAMES = {"old_stable": "Original successful", "old_collapsing": "Original collapsing", "v2": "Sotaku 2"}


def summarize(records):
    result = {}
    for iteration in sorted({row["iteration"] for row in records}):
        rows = [row for row in records if row["iteration"] == iteration]
        radii = [row["spectral_radius"] for row in rows if row["spectral_radius"] is not None]
        result[str(iteration)] = {
            "points": len(rows), "validated": len(radii), "solved": sum(row["solved"] for row in rows),
            "radius_median": float(np.median(radii)) if radii else None,
            "radius_min": min(radii) if radii else None,
            "radius_max": max(radii) if radii else None,
            "historical_power_median": {
                mode: float(np.median([row["precision_audits"][mode]["historical_power"]["last_gain"]
                                       for row in rows]))
                for mode in ("fp64", "fp32_highest", "fp32_high")},
        }
    return result


def run():
    directory = Path(__file__).parent
    settings = json.loads((directory / "protocol.json").read_text())
    expected_sample = None
    summaries = {}
    runs = {}
    for key in NAMES:
        root = directory / "results" / key
        data = json.loads((root / "completed.json").read_text())
        if data["status"] != "complete" or data["identity"]["settings"] != settings:
            raise ValueError(f"Incomplete or mismatched run: {key}")
        sample = data["identity"]["sample"]
        if expected_sample is not None and sample != expected_sample:
            raise ValueError("Models used different samples")
        expected_sample = sample
        for relative, checksum in data["sha256"].items():
            if file_sha256(root / relative) != checksum:
                raise ValueError(f"Result checksum mismatch: {key}/{relative}")
        records = data["records"]
        expected = {(iteration, index) for iteration in settings["iterations"] for index in sample["indices"]}
        if len(records) != len(expected) or {(row["iteration"], row["puzzle_index"]) for row in records} != expected:
            raise ValueError("Missing or duplicate measurements")
        for row in records:
            saved = json.loads((root / f"iter{row['iteration']}_puzzle{row['puzzle_index']}.json").read_text())
            if saved != row:
                raise ValueError("Completion record disagrees with independently saved point")
        summaries[key] = summarize(records)
        runs[key] = data
    atomic_json_save({"sample": expected_sample, "models": summaries}, directory / "summary.json")
    print("Model | Iteration | Validated | Solved | Radius median [min,max] | Old-style FP32-high median")
    for key, summary in summaries.items():
        for iteration, row in summary.items():
            radius = (f"{row['radius_median']:.6f} [{row['radius_min']:.6f},{row['radius_max']:.6f}]"
                      if row["radius_median"] is not None else "unresolved")
            print(f"{key} | {iteration} | {row['validated']}/{row['points']} | "
                  f"{row['solved']}/{row['points']} | {radius} | "
                  f"{row['historical_power_median']['fp32_high']:.3f}")
    return summaries, runs


if __name__ == "__main__":
    run()
