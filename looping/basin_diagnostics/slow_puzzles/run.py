"""Select slow and easy solves before rendering double-precision state slices."""

import json
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, validate_config
from looping.basin_diagnostics.analysis import OutcomeTracker, saved_unit, save_unit
from looping.basin_diagnostics.common import load_model
from looping.basin_diagnostics.recursive.run import (
    SOURCE_PATHS as PARENT_SOURCES, grid_rows, neighbor_distance, next_window,
)
from looping.weight_tying.common import atomic_npz
from runtime_utils import file_sha256, runtime_manifest

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = (*PARENT_SOURCES, *[
    f"looping/basin_diagnostics/slow_puzzles/{name}" for name in
    ("__init__.py", "protocol.json", "run.py", "plots.py", "modal_run.py", "test_slow_puzzles.py")
])
BUCKETS = ("0", "1-2", "3-10", "11-50", "51+")


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def select_rows(labels, indices, settings):
    rng = np.random.default_rng(settings["sample_seed"])
    selected = []
    for bucket in BUCKETS:
        eligible = np.flatnonzero(labels == bucket)[settings["exclude_first_per_bucket"]:]
        eligible = eligible[~np.isin(indices[eligible], settings["exclude_indices"])]
        chosen = rng.choice(eligible, settings["examples_per_bucket"], replace=False)
        selected.extend(sorted(chosen.tolist()))
    return np.asarray(selected, dtype=np.int64)


def plane64(shape, seed):
    dimension = int(np.prod(shape))
    generator = torch.Generator().manual_seed(seed)
    basis, _ = torch.linalg.qr(torch.randn(dimension, 2, generator=generator, dtype=torch.float64))
    return basis.T.reshape(2, *shape) * dimension ** 0.5


def screen_offsets(shape, settings):
    offsets = [torch.zeros(shape, dtype=torch.float64)]
    for seed in settings["plane_seeds"]:
        for direction in plane64(shape, seed):
            offsets.extend((direction * settings["initial_half_width_rms"],
                            -direction * settings["initial_half_width_rms"]))
    return torch.stack(offsets)


def require_double(model):
    for name, tensor in (*model.named_parameters(), *model.named_buffers()):
        if tensor.is_floating_point() and tensor.dtype != torch.float64:
            raise ValueError(f"Expected FP64 model tensor: {name} is {tensor.dtype}")
    if model.training:
        raise ValueError("Diagnostics require evaluation mode")


def configure_precision():
    # Model imports change the global setting, so call this after loading and testing.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.inference_mode()
def encode(model, digits):
    parameter = next(model.parameters())
    inputs = F.one_hot(torch.as_tensor(digits, device=parameter.device).long(), 10).to(parameter.dtype)
    hidden, _ = model.initial_state(inputs)
    return hidden


@torch.inference_mode()
def initial_states(model, digits, offsets):
    base = encode(model, digits)
    offsets = offsets.to(device=base.device, dtype=base.dtype)
    scale = base.square().mean((-1, -2), keepdim=True).sqrt()
    hidden = base[:, None] + scale[:, None] * offsets[None]
    return hidden.flatten(0, 1)


@torch.inference_mode()
def trace_states(model, hidden, targets, horizon, confirmation_window, *, pairs=(), grid_shape=None):
    if horizon < 1 or not 0 <= confirmation_window < horizon:
        raise ValueError("Invalid trajectory horizon or confirmation window")
    dtype = hidden.dtype
    predictions = hidden.new_zeros(len(hidden), 81, 9)
    targets = torch.as_tensor(targets, device=hidden.device).long().expand(len(hidden), -1)
    board = torch.full_like(targets, -1)
    finite = hidden.flatten(1).isfinite().all(-1)
    tracker = OutcomeTracker(board, targets, finite, 0, horizon)
    pairs = torch.as_tensor(pairs, device=hidden.device, dtype=torch.long).reshape(-1, 2)
    mismatch = torch.zeros((), device=hidden.device, dtype=torch.bool)
    # CPU FP64 kernels can round identical batch rows differently; GPU checks remain exact.
    tolerance = 1e-12 if hidden.device.type == "cpu" and dtype == torch.float64 else 0.0
    separation = torch.zeros(grid_shape, device=hidden.device) if grid_shape else None
    for step in range(1, horizon + 1):
        hidden, predictions, logits = model.step(hidden, predictions)
        if any(value.dtype != dtype for value in (hidden, predictions, logits)):
            raise ValueError("Recurrence changed the requested numerical precision")
        finite = (hidden.flatten(1).isfinite().all(-1) & logits.flatten(1).isfinite().all(-1)
                  & predictions.flatten(1).isfinite().all(-1))
        board = logits.argmax(-1)
        tracker.update(board, finite, step)
        if len(pairs):
            mismatch |= (board[pairs[:, 0]] != board[pairs[:, 1]]).any()
            if step % 64 == 0 or step == horizon:
                mismatch |= ~torch.isclose(hidden[pairs[:, 0]], hidden[pairs[:, 1]],
                                          rtol=tolerance, atol=tolerance).all()
                mismatch |= ~torch.isclose(predictions[pairs[:, 0]], predictions[pairs[:, 1]],
                                          rtol=tolerance, atol=tolerance).all()
        if grid_shape:
            separation = torch.maximum(separation, neighbor_distance(board[:int(np.prod(grid_shape))], *grid_shape))
    if bool(mismatch):
        raise ValueError("Identical starting-state controls diverged")
    arrays = tracker.arrays(confirmation_window)
    # A historical nonfinite state is always excluded, even if a later output looks correct.
    arrays["successful"] = arrays["final_correct"] & arrays["confirmed_answer"] & ~arrays["ever_nonfinite"]
    if grid_shape:
        arrays["separation"] = separation.cpu().numpy()
    return arrays


def load_arrays(directory, name):
    directory = Path(directory)
    if saved_unit(directory, name) is None:
        raise ValueError(f"Missing saved unit: {name}")
    with np.load(Path(directory) / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def identity_file(directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "identity.json"
    if path.exists():
        validate_config(json.loads(path.read_text()), identity)
    else:
        atomic_json_save(identity, path)


def select_cases(rows, indices, fields):
    eligible = np.flatnonzero(fields["successful"].all(axis=1))
    if len(eligible) < 2:
        raise ValueError("Fewer than two puzzles settle correctly at all nine starts")
    means = fields["last_change"].mean(axis=1)
    ordered = sorted(eligible.tolist(), key=lambda position: (means[position], int(indices[rows[position]])))
    easy = ordered[0]
    slow = min((position for position in ordered if position != easy),
               key=lambda position: (-means[position], int(indices[rows[position]])))
    return {name: {"position": position, "row": int(rows[position]), "index": int(indices[rows[position]]),
                   "mean_last_change": float(means[position]),
                   "last_change": fields["last_change"][position].tolist()}
            for name, position in (("slow", slow), ("easy", easy))}


def screen(model, data, rows, folder, settings, callback=None):
    offsets = screen_offsets((81, 128), settings)
    count = len(offsets)
    with_controls = torch.cat((offsets, torch.zeros(2, 81, 128, dtype=torch.float64)))
    pieces = []
    started = time.perf_counter()
    for start in range(0, len(rows), settings["screen_batch_puzzles"]):
        batch_rows = rows[start:start + settings["screen_batch_puzzles"]]
        name = f"batch_{start:04d}"
        if saved_unit(folder, name) is None:
            hidden = initial_states(model, data["digits"][batch_rows], with_controls)
            targets = np.repeat(data["targets"][batch_rows], count + 2, axis=0)
            pairs = [(base + a, base + b) for base in range(0, len(hidden), count + 2)
                     for a, b in ((0, count), (count, count + 1))]
            fields = trace_states(model, hidden, targets, settings["horizon"], settings["confirmation_window"], pairs=pairs)
            fields = {key: value.reshape(len(batch_rows), count + 2, *value.shape[1:])[:, :count]
                      for key, value in fields.items()}
            save_unit(folder, name, fields, {"rows": batch_rows.tolist(), "precision": "fp64"})
            if callback:
                callback()
        saved = saved_unit(folder, name)
        if saved["rows"] != batch_rows.tolist():
            raise ValueError("Screening resume uses different puzzle rows")
        pieces.append(load_arrays(folder, name))
        print(f"SCREEN {start + len(batch_rows)}/{len(rows)} seconds={time.perf_counter() - started:.1f}", flush=True)
    fields = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    save_unit(folder, "screen", {**fields, "rows": rows, "indices": data["indices"][rows]},
              {"puzzles": len(rows), "starts_per_puzzle": count, "horizon": settings["horizon"],
               "all_starts_successful": int(fields["successful"].all(axis=1).sum()),
               "nonfinite_starts": int(fields["ever_nonfinite"].sum()),
               "wrong_final_starts": int((~fields["final_correct"]).sum()),
               "unsettled_starts": int((~fields["confirmed_answer"]).sum()),
               "seconds": time.perf_counter() - started})
    return fields


def grid_field64(model, puzzle, folder, *, resolution, center, width, horizon, confirmation_window,
                 chunk_rows, plane_seed, callback=None):
    settings = dict(resolution=resolution, center=center, width=width, horizon=horizon,
                    confirmation_window=confirmation_window, chunk_rows=chunk_rows, plane_seed=plane_seed,
                    puzzle=puzzle, precision="fp64")
    identity_file(folder, settings)
    require_double(model)
    parameter = next(model.parameters())
    plane = plane64((81, 128), plane_seed).to(parameter.device)
    previous, pieces = None, []
    started = time.perf_counter()
    for start in range(0, resolution, chunk_rows):
        name = f"rows_{start:04d}"
        if saved_unit(folder, name) is None:
            coordinates = torch.as_tensor(grid_rows(center, width, resolution, start, chunk_rows),
                                          device=parameter.device, dtype=torch.float64)
            offsets = coordinates[:, 0, None, None] * plane[0] + coordinates[:, 1, None, None] * plane[1]
            offsets = torch.cat((offsets, offsets.new_zeros(2, 81, 128)))
            hidden = initial_states(model, [puzzle["digits"]], offsets)
            result = trace_states(model, hidden, [puzzle["targets"]], horizon, confirmation_window,
                                  pairs=[(len(hidden) - 2, len(hidden) - 1)], grid_shape=(chunk_rows + 2, resolution))
            fields = {key: value[:-2].reshape(chunk_rows + 2, resolution, *value.shape[1:])
                      for key, value in result.items() if key != "separation"}
            fields.update(separation=result["separation"], zero_last_change=result["last_change"][-2:],
                          zero_final_board=result["final_board"][-2:])
            save_unit(folder, name, fields, {"start": start})
            if callback:
                callback()
        fields = load_arrays(folder, name)
        if previous is not None:
            for key in ("last_change", "final_board", "ever_nonfinite"):
                np.testing.assert_array_equal(previous[key][-2:], fields[key][:2], err_msg=f"Row-overlap mismatch: {key}")
            for key in ("zero_last_change", "zero_final_board"):
                np.testing.assert_array_equal(previous[key], fields[key], err_msg=f"Zero-control mismatch: {key}")
        previous = fields
        count = min(chunk_rows, resolution - start)
        pieces.append({key: value[1:count + 1] for key, value in fields.items() if not key.startswith("zero_")})
        print(f"ROWS {Path(folder).name} {start + count}/{resolution} fp64 seconds={time.perf_counter() - started:.1f}", flush=True)
    return {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}


def summarize_grid(field):
    successful = field["successful"]
    times = field["last_change"][successful]
    return {"successful_fraction": float(successful.mean()),
            "final_correct_fraction": float(field["final_correct"].mean()),
            "unsettled_fraction": float((~field["confirmed_answer"]).mean()),
            "nonfinite_count": int(field["ever_nonfinite"].sum()),
            "minimum_successful_time": int(times.min()) if len(times) else None,
            "maximum_successful_time": int(times.max()) if len(times) else None,
            "mean_successful_time": float(times.mean()) if len(times) else None}


def render_case(model, puzzle, folder, settings, callback=None):
    from looping.basin_diagnostics.slow_puzzles.plots import plot_case, plot_orientation_control
    levels, center, width = [], [0., 0.], settings["initial_half_width_rms"]
    arguments = dict(horizon=settings["horizon"], confirmation_window=settings["confirmation_window"],
                     chunk_rows=settings["chunk_rows"], callback=callback)
    for level in range(settings["levels"]):
        field = grid_field64(model, puzzle, folder / f"level_{level}", resolution=settings["resolution"],
                             center=center, width=width, plane_seed=settings["plane_seeds"][0], **arguments)
        child = next_window(field["last_change"], field["successful"], center, width, settings["zoom_factor"])
        info = {"level": level, "center": center, "width": width, "next_zoom": child,
                "magnification": settings["zoom_factor"] ** level, **summarize_grid(field)}
        save_unit(folder, f"level_{level}", field, info)
        levels.append(info)
        plot_case(folder, puzzle, settings, levels)
        print(f"LEVEL {Path(folder).name} {json.dumps(info)}", flush=True)
        if callback:
            callback()
        center, width = child["center"], child["width"]
    control = grid_field64(model, puzzle, folder / "orientation_control", center=[0., 0.],
                           width=settings["initial_half_width_rms"], resolution=settings["orientation_control_resolution"],
                           plane_seed=settings["plane_seeds"][1], **arguments)
    save_unit(folder, "orientation_control", control, summarize_grid(control))
    plot_orientation_control(folder, puzzle, settings, control)
    if callback:
        callback()
    return {"puzzle": puzzle, "levels": levels, "orientation_control": summarize_grid(control)}


def load_data(settings):
    path = Path(settings["data_directory"]) / "development.npz"
    if file_sha256(path) != settings["data_sha256"]:
        raise ValueError("Benchmark data checksum mismatch")
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name].copy() for name in ("digits", "targets", "labels", "indices")}


def preflight(model, puzzle, folder):
    from looping.basin_diagnostics.slow_puzzles.test_slow_puzzles import SlowPuzzleTests
    require_double(model)
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(SlowPuzzleTests)
    if not unittest.TextTestRunner(verbosity=2).run(tests).wasSuccessful():
        raise ValueError("Unit tests failed in worker environment")
    configure_precision()
    torch.set_num_threads(8)
    observed_dtypes = set()

    def inspect(module, inputs, output):
        for value in (*inputs, *(output if isinstance(output, tuple) else (output,))):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                observed_dtypes.add(str(value.dtype))

    handles = [module.register_forward_hook(inspect) for module in model.modules() if not list(module.children())]
    try:
        initial = initial_states(model, [puzzle["digits"]], torch.zeros(3, 81, 128, dtype=torch.float64))
        result = trace_states(model, initial, [puzzle["targets"]], 16, 4, pairs=[(0, 1), (1, 2)])
    finally:
        for handle in handles:
            handle.remove()
    if observed_dtypes != {"torch.float64"} or result["ever_nonfinite"].any():
        raise ValueError(f"Real model is not finite FP64 throughout: {observed_dtypes}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    grid_field64(model, puzzle, folder / "grid", resolution=17, center=[0., 0.], width=.3,
                 horizon=64, confirmation_window=16, chunk_rows=8, plane_seed=2701)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    settings = protocol()
    coordinates = torch.as_tensor(grid_rows([0., 0.], .3, settings["resolution"], 0, settings["chunk_rows"]),
                                  device=next(model.parameters()).device, dtype=torch.float64)
    plane = plane64((81, 128), settings["plane_seeds"][0]).to(coordinates.device)
    offsets = coordinates[:, 0, None, None] * plane[0] + coordinates[:, 1, None, None] * plane[1]
    offsets = torch.cat((offsets, offsets.new_zeros(2, 81, 128)))
    initial = initial_states(model, [puzzle["digits"]], offsets)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    benchmark_started = time.perf_counter()
    row_result = trace_states(model, initial, [puzzle["targets"]], 64, 16,
                              pairs=[(len(initial) - 2, len(initial) - 1)],
                              grid_shape=(settings["chunk_rows"] + 2, settings["resolution"]))
    if row_result["ever_nonfinite"].any():
        raise ValueError("Nonfinite real-model full-size row preflight")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    row_seconds = time.perf_counter() - benchmark_started
    report = {"status": "passed", "unit_tests": tests.countTestCases(), "activation_dtypes": sorted(observed_dtypes),
              "model_source": __import__(model.__class__.__module__, fromlist=["__file__"]).__file__,
              "grid_seconds": seconds, "grid_shape": [17, 17], "grid_horizon": 64,
              "full_row_batch": len(initial), "full_row_64_iterations_seconds": row_seconds,
              "estimated_grid_hours_per_case": row_seconds / 64 * settings["horizon"]
                  * int(np.ceil(settings["resolution"] / settings["chunk_rows"])) * settings["levels"] / 3600,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
              "environment": runtime_manifest(SOURCE_PATHS)}
    atomic_json_save(report, folder / "preflight.json")
    print(f"PREFLIGHT passed: {report['unit_tests']} tests, FP64 activations, duplicate and overlap checks; "
          f"full_row={row_seconds:.1f}s/64 iterations, estimated_hours_per_case={report['estimated_grid_hours_per_case']:.2f}", flush=True)
    return report


def run_model(key, output_dir, *, volume_root="/outputs", callback=None, smoke=False):
    settings = protocol()
    if key not in settings["models"]:
        raise ValueError("Unknown model")
    folder = Path(output_dir) / (f"smoke_{key}" if smoke else key)
    data = load_data(settings)
    rows = select_rows(data["labels"], data["indices"], settings)
    model, spec = load_model(key, volume_root=volume_root)
    model = model.double().eval()
    configure_precision()
    require_double(model)
    environment = runtime_manifest(SOURCE_PATHS)
    identity = {"settings": settings, "checkpoint": spec, "rows": rows.tolist(), "indices": data["indices"][rows].tolist(),
                "source_sha256": environment["source_sha256"], "smoke": smoke}
    identity_file(folder, identity)
    if (folder / "completed.json").exists():
        result = json.loads((folder / "completed.json").read_text())
        for name, checksum in result["sha256"].items():
            if file_sha256(folder / name) != checksum:
                raise ValueError(f"Completed artifact changed: {name}")
        return result
    print(f"CONFIG {json.dumps(identity)}", flush=True)
    row = int(rows[0])
    puzzle = {"digits": data["digits"][row].tolist(), "targets": data["targets"][row].tolist(), "index": int(data["indices"][row])}
    preflight(model, puzzle, folder / "preflight")
    atomic_json_save(runtime_manifest(SOURCE_PATHS), folder / "environment.json")
    if callback:
        callback()
    if smoke:
        return {"status": "smoke_passed", "output": str(folder)}
    started = time.perf_counter()
    fields = screen(model, data, rows, folder / "screen", settings, callback)
    cases = select_cases(rows, data["indices"], fields)
    for name, selected in cases.items():
        row = selected["row"]
        selected.update(digits=data["digits"][row].tolist(), targets=data["targets"][row].tolist(),
                        difficulty=str(data["labels"][row]), case=name, model_key=key)
    selection = {"cases": cases, "screen_sha256": file_sha256(folder / "screen" / "screen.npz"),
                 "criterion": settings["selection"]}
    if (folder / "selection.json").exists():
        validate_config(json.loads((folder / "selection.json").read_text()), selection)
    else:
        atomic_json_save(selection, folder / "selection.json")
    print(f"SELECTED {json.dumps(selection)}", flush=True)
    if callback:
        callback()
    results = {name: render_case(model, puzzle, folder / name, settings, callback) for name, puzzle in cases.items()}
    result = {"status": "complete", "identity": identity, "selection": selection, "cases": results,
              "seconds": time.perf_counter() - started,
              "sha256": {str(path.relative_to(folder)): file_sha256(path) for path in folder.rglob("*")
                         if path.is_file() and path.name != "completed.json"}}
    atomic_json_save(result, folder / "completed.json")
    print(f"COMPLETE {key} seconds={result['seconds']:.1f}", flush=True)
    return result
