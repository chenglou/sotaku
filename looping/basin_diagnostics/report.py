"""Verify downloaded results and generate an index of every diagnostic plot."""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save, validate_config
from looping.basin_diagnostics.common import checkpoint_spec, protocol
from looping.basin_diagnostics.plots import comparison
from runtime_utils import file_sha256


def generate(root):
    root = Path(root)
    settings = protocol()
    models = {}
    gallery = ["# Diagnostic Plots", "", "All six fixed checkpoints and all 96 maps are included. Each map starts from the same two selected puzzles; no plot was selected for its appearance.", "", "Each pixel is a different nudge to the internal state. The center has no nudge. Green means the answer became correct and stayed correct for the rest of the run; yellow means it recovered after a regression; red means it was once correct but ended wrong; gray means it never became correct after the nudge. Times start at the intervention, so a last-change time equal to the starting iteration means no later change was observed. None of these finite-run plots proves permanent convergence or fractality.", "", "[All-model accuracy curves](comparison.png)", ""]
    for key in settings["models"]:
        directory = root / "models" / key
        completed = json.loads((directory / "completed.json").read_text())
        if completed["status"] != "complete":
            raise ValueError(f"Incomplete model: {key}")
        validate_config(completed["identity"]["checkpoint"], checkpoint_spec(key))
        validate_config(completed["identity"]["settings"], settings)
        if len(completed["maps"]) != 16:
            raise ValueError(f"Missing maps: {key}")
        for name, checksum in completed["artifact_sha256"].items():
            if name.startswith("map_") and name.endswith(".npz") and not (directory / name).exists():
                continue
            if file_sha256(directory / name) != checksum:
                raise ValueError(f"Downloaded artifact changed: {key}/{name}")
        with np.load(directory / "baseline.npz", allow_pickle=False) as arrays:
            baseline = {"n_puzzles": len(arrays["final_correct"]),
                        "ever_correct": int((arrays["first_correct"] <= settings["horizon"]).sum()),
                        "final_correct": int(arrays["final_correct"].sum()),
                        "regressed": int((arrays["regressions"] > 0).sum()),
                        "confirmed_wrong": int((arrays["confirmed_answer"] & ~arrays["final_correct"]).sum()),
                        "accuracy": {str(step): float(arrays["correct_curve"][arrays["steps"] == step].mean())
                                     for step in (128, 1024, 2048, 4096)}}
        probes = []
        for record in json.loads((directory / "probe_summary.json").read_text()):
            with np.load(directory / f"{record['name']}.npz", allow_pickle=False) as arrays:
                final = arrays["final_correct"].reshape(-1, 5)
                initial = arrays["initial_correct"].reshape(-1, 5)
                harmed_immediately = initial[:, :1] & ~initial[:, 1:]
                item = {**record, "n_perturbations": int(final[:, 1:].size),
                        "rescued": int((~final[:, :1] & final[:, 1:]).sum()),
                        "harmed": int((final[:, :1] & ~final[:, 1:]).sum()),
                        "harmed_immediately": int(harmed_immediately.sum()),
                        "recovered_after_immediate_harm": int((harmed_immediately & final[:, 1:]).sum())}
                probes.append(item)
        for record in completed["maps"]:
            checks = record["checks"]
            if checks["zero_control_board_mismatches"] or checks["zero_control_max_hidden_error"] or checks["nonfinite"]:
                raise ValueError(f"Diagnostic control failed: {key}/{record['name']}")
        models[key] = {"baseline": baseline, "probes": probes, "seconds": completed["elapsed_seconds"],
                       "maps": [{name: value for name, value in record.items() if name not in ("arrays_sha256", "settling_uncertainty")}
                                for record in completed["maps"]]}
        gallery.extend([f"## {key}", "", f"[Perturbation summary](models/{key}/probe_summary.png)", "",
                        "| Puzzle | Starting Iteration | Per-Axis Nudge | Plane | Plot |",
                        "|---|---:|---:|---:|---|"])
        for record in completed["maps"]:
            gallery.append(f"| {record['gallery']['index']} | {record['anchor']} | {record['rms_fraction']:.1%} | {record['plane_seed']} | [Open](models/{key}/{record['name']}.png) |")
        gallery.append("")
    comparison(root)
    atomic_json_save({"study_id": settings["study_id"], "models": models}, root / "summary.json")
    (root / "GALLERY.md").write_text("\n".join(gallery))
    for key, result in models.items():
        print(key, json.dumps({"baseline": result["baseline"], "seconds": round(result["seconds"], 1), "probes": result["probes"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path, default=Path(__file__).with_name("results"))
    generate(parser.parse_args().directory)
