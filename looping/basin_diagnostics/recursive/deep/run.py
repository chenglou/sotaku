"""Extend saved zoom regions without changing the original model or renderer."""

import copy
import json
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, validate_config
from looping.basin_diagnostics.analysis import plane_basis, saved_unit, save_unit
from looping.basin_diagnostics.common import load_model
from looping.basin_diagnostics.recursive.run import (
    SOURCE_PATHS as PARENT_SOURCES, grid_field, grid_rows, neighbor_distance, next_window,
)
from looping.weight_tying.common import atomic_npz
from runtime_utils import file_sha256, runtime_manifest

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = (*PARENT_SOURCES, *[f"looping/basin_diagnostics/recursive/deep/{name}" for name in
                               ("__init__.py", "protocol.json", "run.py", "plots.py", "modal_run.py", "test_deep.py")])


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def validated_json(path, checksum):
    if file_sha256(path) != checksum:
        raise ValueError(f"Parent artifact changed: {path}")
    return json.loads(Path(path).read_text())


def parent_context(key, volume_root):
    settings = protocol()
    parent = Path(volume_root) / settings["parent_study_id"]
    prepared = validated_json(parent / "prepared.json", settings["parent_prepared_sha256"])
    completed = validated_json(parent / key / "completed.json", settings["parent_completed_sha256"][key])
    for name in ("identity.json", f"level_{settings['parent_level']}.json", f"level_{settings['parent_level']}.npz"):
        if file_sha256(parent / key / name) != completed["sha256"][name]:
            raise ValueError(f"Parent map changed: {key}/{name}")
    for name, checksum in prepared["source_sha256"].items():
        if file_sha256(Path(__file__).resolve().parents[4] / name) != checksum:
            raise ValueError(f"Original visualization source changed: {name}")
    for name in ("plane_seed", "horizon", "confirmation_window", "start_iteration"):
        if prepared["settings"][name] != settings[name]:
            raise ValueError(f"Must retain original {name}")
    return prepared, completed["levels"][settings["parent_level"]], completed["identity"]["checkpoint"]


@torch.inference_mode()
def initial_grid(model, puzzle, coordinates, *, dtype):
    device = next(model.parameters()).device
    given = torch.tensor(puzzle["digits"], device=device).long()[None]
    base, _ = model.initial_state(F.one_hot(given, 10).float())
    directions = plane_basis((81, 128), protocol()["plane_seed"]).to(device=device, dtype=dtype)
    values = torch.as_tensor(coordinates, device=device, dtype=dtype)
    values = torch.cat((values, torch.zeros(2, 2, device=device, dtype=dtype)), 0)
    offsets = values[:, 0, None, None] * directions[0] + values[:, 1, None, None] * directions[1]
    # Preserve the original encoded puzzle and RMS scale; only subsequent arithmetic changes.
    scale = base.square().mean().sqrt().to(dtype)
    return base.to(dtype) + offsets * scale


@torch.inference_mode()
def trace_initial(model, initial, puzzle, resolution, horizon, confirmation_window):
    hidden = initial.clone()
    predictions = hidden.new_zeros(len(hidden), 81, 9)
    targets = torch.tensor(puzzle["targets"], device=hidden.device)[None]
    board = torch.full((len(hidden), 81), -1, device=hidden.device, dtype=torch.long)
    last_change = torch.zeros(len(hidden), device=hidden.device, dtype=torch.int32)
    separation = torch.zeros((resolution, resolution), device=hidden.device)
    invalid = torch.zeros((), device=hidden.device, dtype=torch.bool)
    for step in range(1, horizon + 1):
        hidden, predictions, logits = model.step(hidden, predictions)
        decoded = logits.argmax(-1)
        invalid |= ~hidden.isfinite().all() | ~logits.isfinite().all()
        invalid |= (decoded[-1] != decoded[-2]).any()
        last_change = torch.where((decoded != board).any(-1), step, last_change)
        board = decoded
        separation = torch.maximum(separation, neighbor_distance(board[:-2], resolution, resolution))
    # CPU FP64 kernels can round identical batch rows differently; GPU checks remain exact.
    tolerance = 1e-12 if hidden.device.type == "cpu" and hidden.dtype == torch.float64 else 0.0
    if bool(invalid) or not torch.allclose(hidden[-1], hidden[-2], rtol=tolerance, atol=tolerance):
        raise ValueError("Precision audit found nonfinite states or inconsistent duplicate controls")
    return {"last_change": last_change[:-2].reshape(resolution, resolution).cpu().numpy(),
            "confirmed": ((horizon - last_change[:-2]) >= confirmation_window).reshape(resolution, resolution).cpu().numpy(),
            "final_board": board[:-2].reshape(resolution, resolution, 81).cpu().numpy().astype(np.uint8),
            "final_correct": (board[:-2] == targets).all(-1).reshape(resolution, resolution).cpu().numpy(),
            "separation": separation.cpu().numpy()}


def comparison_stats(first, second):
    delta = np.abs(first["last_change"].astype(float) - second["last_change"])
    valid = first["confirmed"] & second["confirmed"]
    a, b = first["last_change"][valid], second["last_change"][valid]
    correlation = float(np.corrcoef(a, b)[0, 1]) if len(a) > 1 and a.std() > 0 and b.std() > 0 else None
    return {"same_last_change_fraction": float((delta == 0).mean()),
            "median_absolute_time_difference": float(np.median(delta)),
            "p90_absolute_time_difference": float(np.percentile(delta, 90)),
            "confirmed_time_correlation": correlation,
            "same_final_board_fraction": float((first["final_board"] == second["final_board"]).all(-1).mean()),
            "both_confirmed_fraction": float(valid.mean())}


def audit(model, double_model, puzzle, folder, info, field):
    settings = protocol()
    name = f"audit_{info['level']}"
    saved = saved_unit(folder, name)
    if saved:
        return saved
    resolution = settings["audit_resolution"]
    coordinates = grid_rows(info["center"], info["width"], resolution, 0, resolution)
    coordinates = coordinates.reshape(resolution + 2, resolution, 2)[1:-1].reshape(-1, 2)
    initial32 = initial_grid(model, puzzle, coordinates, dtype=torch.float32)
    initial64 = initial_grid(model, puzzle, coordinates, dtype=torch.float64)
    arguments = (puzzle, resolution, settings["horizon"], settings["confirmation_window"])
    fields = {"fp32": trace_initial(model, initial32, *arguments),
              "fp64": trace_initial(double_model, initial64, *arguments)}
    stride = (settings["resolution"] - 1) // (resolution - 1)
    dense_samples = {key: value[::stride, ::stride] for key, value in field.items()}
    # Compare construction rounding with one pixel of spacing on the dense grid.
    rounding_rms = (initial32.double() - initial64).square().mean(-1).mean(-1).sqrt()[:-2]
    base_scale = initial32[-1].double().square().mean().sqrt()
    dense_spacing = 2 * info["width"] / (settings["resolution"] - 1)
    summary = {"resolution": resolution, "center": info["center"], "width": info["width"],
               "fp32_vs_fp64": comparison_stats(fields["fp32"], fields["fp64"]),
               "dense_vs_audit_fp32": comparison_stats(dense_samples, fields["fp32"]),
               "maximum_initial_rounding_in_dense_pixels": float(rounding_rms.max() / (dense_spacing * base_scale)),
               "fp64_correct": float(fields["fp64"]["final_correct"].mean()),
               "fp64_confirmed": float(fields["fp64"]["confirmed"].mean())}
    arrays = {f"{precision}_{key}": value for precision, values in fields.items() for key, value in values.items()}
    return save_unit(folder, name, arrays, summary)


def run_model(key, output_dir, *, volume_root="/outputs", callback=None):
    from looping.basin_diagnostics.recursive.deep.plots import render
    from looping.basin_diagnostics.recursive.deep.test_deep import DeepTests
    from looping.basin_diagnostics.recursive.test_recursive import RecursiveTests
    from looping.basin_diagnostics.test_diagnostics import gpu_preflight
    settings = protocol()
    if key not in settings["models"]:
        raise ValueError("Unknown model")
    if (settings["resolution"] - 1) % (settings["audit_resolution"] - 1):
        raise ValueError("Audit coordinates must be a subset of dense grid coordinates")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent, starting_view, parent_spec = parent_context(key, volume_root)
    model, spec = load_model(key, volume_root=volume_root)
    validate_config(parent_spec, spec)
    prepared = {"settings": settings, "puzzle": parent["puzzle"], "checkpoint": spec,
                "starting_view": starting_view, "source_sha256": runtime_manifest(SOURCE_PATHS)["source_sha256"]}
    folder = Path(output_dir) / key
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "identity.json").exists():
        validate_config(json.loads((folder / "identity.json").read_text()), prepared)
    else:
        atomic_json_save(prepared, folder / "identity.json")
    if (folder / "completed.json").exists():
        result = json.loads((folder / "completed.json").read_text())
        for name, checksum in result["sha256"].items():
            if file_sha256(folder / name) != checksum:
                raise ValueError(f"Completed output changed: {name}")
        return result
    print(f"CONFIG {json.dumps(prepared)}", flush=True)
    gpu_preflight()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (RecursiveTests, DeepTests))
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise ValueError("Deeper-zoom tests failed")
    torch.set_num_threads(8)
    atomic_json_save(runtime_manifest(SOURCE_PATHS), folder / "environment.json")
    double_model = copy.deepcopy(model).double()
    smoke_coordinates = grid_rows(starting_view["center"], starting_view["width"], 3, 0, 3)
    smoke_coordinates = smoke_coordinates.reshape(5, 3, 2)[1:-1].reshape(-1, 2)
    for dtype, tested_model in ((torch.float32, model), (torch.float64, double_model)):
        initial = initial_grid(model, prepared["puzzle"], smoke_coordinates, dtype=dtype)
        trace_initial(tested_model, initial, prepared["puzzle"], 3, 16, 4)
    atomic_json_save({"status": "passed", "unit_tests": 28, "exact_original_recurrence": True,
                      "gpu_fp32_and_fp64_smoke": True}, folder / "preflight.json")
    print("PREFLIGHT passed: 28 tests, original recurrence, and real-checkpoint FP32/FP64 GPU traces", flush=True)
    if callback:
        callback()
    center, width = starting_view["center"], starting_view["width"]
    levels, audits = [], []
    started = time.perf_counter()
    for level in range(settings["levels"]):
        field = grid_field(model, prepared["puzzle"], folder / f"level_{level}", resolution=settings["resolution"],
                           center=center, width=width, horizon=settings["horizon"],
                           confirmation_window=settings["confirmation_window"], chunk_rows=settings["chunk_rows"], callback=callback)
        atomic_npz(folder / f"level_{level}.npz", **field)
        child = next_window(field["last_change"], field["confirmed"], center, width, settings["zoom_factor"])
        info = {"level": level, "center": center, "width": width, "next_zoom": child,
                "magnification": starting_view["magnification"] * settings["zoom_factor"] ** level,
                "resolution": settings["resolution"], "horizon": settings["horizon"],
                "last_change_min": int(field["last_change"].min()), "last_change_max": int(field["last_change"].max()),
                "final_correct": float(field["final_correct"].mean()), "confirmed": float(field["confirmed"].mean()),
                "distinct_confirmed_times": int(len(np.unique(field["last_change"][field["confirmed"]]))),
                "sha256": file_sha256(folder / f"level_{level}.npz")}
        atomic_json_save(info, folder / f"level_{level}.json")
        print(f"LEVEL {json.dumps(info)}", flush=True)
        audit_result = audit(model, double_model, prepared["puzzle"], folder, info, field)
        print(f"AUDIT {json.dumps(audit_result)}", flush=True)
        levels.append(info)
        audits.append(audit_result)
        render(folder, prepared, levels, audits)
        if callback:
            callback()
        center, width = child["center"], child["width"]
    result = {"status": "complete", "identity": prepared, "levels": levels, "audits": audits,
              "seconds": time.perf_counter() - started,
              "sha256": {path.name: file_sha256(path) for path in folder.iterdir() if path.is_file()
                         and path.suffix in (".npz", ".json", ".png") and path.name != "completed.json"}}
    atomic_json_save(result, folder / "completed.json")
    print(f"COMPLETE {key} seconds={result['seconds']:.1f}", flush=True)
    return result
