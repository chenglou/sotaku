"""Evaluate fresh initial-state grids, then zoom into settling-time variation."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, validate_config
from looping.basin_diagnostics.analysis import plane_basis, saved_unit, save_unit
from looping.basin_diagnostics.common import SOURCE_PATHS as BASE_SOURCES, checkpoint_spec, load_model
from looping.weight_tying.common import atomic_npz
from runtime_utils import file_sha256, runtime_manifest

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = (*BASE_SOURCES, *[f"looping/basin_diagnostics/recursive/{name}" for name in
                               ("__init__.py", "protocol.json", "run.py", "plots.py", "modal_run.py", "test_recursive.py")])


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def select_puzzle():
    settings = protocol()
    path = Path(settings["data_directory"]) / "development.npz"
    if file_sha256(path) != settings["data_sha256"]:
        raise ValueError("Benchmark data changed")
    with np.load(path, allow_pickle=False) as data:
        digits, targets, labels, indices = [data[name].copy() for name in ("digits", "targets", "labels", "indices")]
    records = []
    for key in settings["models"]:
        spec = checkpoint_spec(key)
        folder = Path(spec["path"]).parent / "evaluations" / "final"
        result = json.loads((folder / "result.json").read_text())
        if result["identity"]["weights_sha256"] != spec["weights_sha256"]:
            raise ValueError("Archived predictions use another checkpoint")
        if file_sha256(folder / "predictions.npz") != result["predictions_sha256"]:
            raise ValueError("Archived predictions changed")
        with np.load(folder / "predictions.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(indices, data["indices"])
            records.append({name: data[name].copy() for name in ("solved_16", "solved_512", "solved_4096")})
    healthy, failing = records
    eligible = np.flatnonzero(labels == "51+")[200:]
    matched = (healthy["solved_512"] & failing["solved_512"] & ~healthy["solved_16"] & ~failing["solved_16"]
               & healthy["solved_4096"] & ~failing["solved_4096"])
    eligible = eligible[matched[eligible]]
    if not len(eligible):
        raise ValueError("No puzzle meets the fixed visualization selection rule")
    row = int(eligible[0])
    return {"row": row, "index": int(indices[row]), "difficulty": str(labels[row]),
            "digits": digits[row].tolist(), "targets": targets[row].tolist(),
            "data_sha256": settings["data_sha256"], "selection": settings["selection"]}


def grid_rows(center, width, resolution, start, count):
    x = np.linspace(center[0] - width, center[0] + width, resolution)
    y = np.linspace(center[1] - width, center[1] + width, resolution)
    rows = np.clip(np.arange(start - 1, start + count + 1), 0, resolution - 1)
    xx, yy = np.meshgrid(x, y[rows])
    return np.stack((xx.flatten(), yy.flatten()), -1)


def neighbor_distance(board, rows, resolution):
    grid = board.float().reshape(rows, resolution, -1)
    horizontal = (grid[:, 1:] - grid[:, :-1]).square().sum(-1).sqrt()
    vertical = (grid[1:] - grid[:-1]).square().sum(-1).sqrt()
    result = torch.zeros((rows, resolution), device=board.device)
    result[:, :-1] = horizontal
    result[:, 1:] = torch.maximum(result[:, 1:], horizontal)
    result[:-1] = torch.maximum(result[:-1], vertical)
    result[1:] = torch.maximum(result[1:], vertical)
    return result


@torch.inference_mode()
def trace_grid_chunk(model, puzzle, coords, directions, rows, resolution, horizon, confirmation_window):
    device = next(model.parameters()).device
    given = torch.tensor(puzzle["digits"], device=device).long()[None]
    answers = torch.tensor(puzzle["targets"], device=device).long()[None]
    base, _ = model.initial_state(F.one_hot(given, 10).float())
    coordinates = torch.as_tensor(coords, device=device, dtype=torch.float32)
    coordinates = torch.cat((coordinates, torch.zeros(2, 2, device=device)), 0)
    offsets = coordinates[:, 0, None, None] * directions[0] + coordinates[:, 1, None, None] * directions[1]
    hidden = base + offsets * base.square().mean().sqrt()
    probabilities = torch.zeros(len(hidden), 81, 9, device=device)
    board = torch.full((len(hidden), 81), -1, device=device, dtype=torch.long)
    last_change = torch.zeros(len(hidden), device=device, dtype=torch.int32)
    first_correct = torch.full_like(last_change, horizon + 1)
    nonfinite = torch.zeros(len(hidden), device=device, dtype=torch.bool)
    separation = torch.zeros((rows, resolution), device=device)
    zero_mismatches = torch.zeros((), device=device, dtype=torch.int64)
    zero_error = torch.zeros((), device=device)
    for step in range(1, horizon + 1):
        hidden, probabilities, logits = model.step(hidden, probabilities)
        decoded = logits.argmax(-1)
        finite = hidden.flatten(1).isfinite().all(-1) & logits.flatten(1).isfinite().all(-1)
        nonfinite |= ~finite
        last_change = torch.where((decoded != board).any(-1) | ~finite, step, last_change)
        correct = (decoded == answers).all(-1) & finite
        first_correct = torch.where(correct & (first_correct > horizon), step, first_correct)
        board = decoded
        separation = torch.maximum(separation, neighbor_distance(board[:-2], rows, resolution))
        zero_mismatches += (board[-1] != board[-2]).any().long()
        if step % 64 == 0 or step == horizon:
            zero_error = torch.maximum(zero_error, (hidden[-1] - hidden[-2]).abs().max())
    if int(zero_mismatches) or float(zero_error) or bool(nonfinite.any()):
        raise ValueError("Zero-control mismatch or nonfinite trajectory")
    arrays = {"last_change": last_change[:-2].reshape(rows, resolution).cpu().numpy(),
              "first_correct": first_correct[:-2].reshape(rows, resolution).cpu().numpy(),
              "confirmed": ((horizon - last_change[:-2]) >= confirmation_window).reshape(rows, resolution).cpu().numpy(),
              "final_correct": correct[:-2].reshape(rows, resolution).cpu().numpy(),
              "final_board": board[:-2].reshape(rows, resolution, 81).cpu().numpy().astype(np.uint8),
              "separation": separation.cpu().numpy(),
              "zero_last_change": last_change[-2:].cpu().numpy(), "zero_final_board": board[-2:].cpu().numpy(),
              "zero_final_correct": correct[-2:].cpu().numpy()}
    return arrays


def rectangle_sums(values, height, width):
    cumulative = np.pad(np.asarray(values, dtype=float), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return cumulative[height:, width:] - cumulative[:-height, width:] - cumulative[height:, :-width] + cumulative[:-height, :-width]


def next_window(field, confirmed, center, width, factor):
    resolution = len(field)
    if (resolution - 1) % factor:
        raise ValueError("Grid intervals must be divisible by zoom factor")
    size = (resolution - 1) // factor + 1
    horizontal = confirmed[:, 1:] & confirmed[:, :-1]
    vertical = confirmed[1:] & confirmed[:-1]
    total = (rectangle_sums(np.abs(np.diff(field.astype(float), axis=1)) * horizontal, size, size - 1)
             + rectangle_sums(np.abs(np.diff(field.astype(float), axis=0)) * vertical, size - 1, size))
    count = rectangle_sums(horizontal, size, size - 1) + rectangle_sums(vertical, size - 1, size)
    enough = count >= .9 * (2 * size * (size - 1))
    score = np.where(enough, total / np.maximum(count, 1), -1)
    maximum = float(score.max())
    if maximum <= 0:
        return {"center": list(center), "width": width / factor, "score": max(maximum, 0),
                "selection": "center fallback: no varying window with sufficient confirmed pixels"}
    row, col = np.unravel_index(np.argmax(score), score.shape)
    x = np.linspace(center[0] - width, center[0] + width, resolution)
    y = np.linspace(center[1] - width, center[1] + width, resolution)
    return {"center": [float((x[col] + x[col + size - 1]) / 2), float((y[row] + y[row + size - 1]) / 2)],
            "width": width / factor, "score": maximum, "selection": "largest mean neighboring settling-time difference"}


def grid_field(model, puzzle, directory, *, resolution, center, width, horizon, confirmation_window, chunk_rows, callback=None):
    started = time.perf_counter()
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    directions = plane_basis((81, 128), protocol()["plane_seed"]).to(device)
    pieces, previous = [], None
    for start in range(0, resolution, chunk_rows):
        name = f"rows_{start:03d}"
        saved = saved_unit(directory, name)
        if saved is None:
            coords = grid_rows(center, width, resolution, start, chunk_rows)
            arrays = trace_grid_chunk(model, puzzle, coords, directions, chunk_rows + 2, resolution, horizon, confirmation_window)
            save_unit(directory, name, arrays, {"start": start, "horizon": horizon, "center": center, "width": width})
            if callback:
                callback()
        with np.load(directory / f"{name}.npz", allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        if previous is not None:
            for key in ("last_change", "final_board"):
                np.testing.assert_array_equal(previous[key][-2:], arrays[key][:2], err_msg=f"Row-halo repeatability: {key}")
            np.testing.assert_array_equal(previous["zero_final_board"], arrays["zero_final_board"])
            np.testing.assert_array_equal(previous["zero_last_change"], arrays["zero_last_change"])
        previous = arrays
        count = min(chunk_rows, resolution - start)
        pieces.append({key: arrays[key][1:count + 1] for key in ("last_change", "first_correct", "confirmed", "final_correct", "final_board", "separation")})
        print(f"ROWS {directory.name} {start + count}/{resolution} horizon={horizon} seconds={time.perf_counter() - started:.1f}", flush=True)
    return {key: np.concatenate([piece[key] for piece in pieces], axis=0) for key in pieces[0]}


def prepare(root, callback=None):
    from looping.basin_diagnostics.recursive.plots import plot_previews
    from looping.basin_diagnostics.test_diagnostics import gpu_preflight
    import unittest
    from looping.basin_diagnostics.recursive.test_recursive import RecursiveTests
    settings = protocol()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    gpu_preflight()
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecursiveTests)).wasSuccessful():
        raise ValueError("Recursive-map tests failed")
    puzzle = select_puzzle()
    identity = {"settings": settings, "puzzle": puzzle, "source_sha256": runtime_manifest(SOURCE_PATHS)["source_sha256"]}
    path = root / "identity.json"
    if path.exists():
        validate_config(json.loads(path.read_text()), identity)
    else:
        atomic_json_save(identity, path)
    previews = []
    for key in settings["models"]:
        model, spec = load_model(key)
        grid_field(model, puzzle, root / "smoke" / key, resolution=17, center=[0., 0.], width=.1,
                   horizon=128, confirmation_window=32, chunk_rows=8, callback=callback)
        for width in settings["preview_rms_fractions"]:
            folder = root / "previews" / key / f"width_{width:g}"
            field = grid_field(model, puzzle, folder, resolution=settings["preview_resolution"], center=[0., 0.], width=width,
                               horizon=settings["preview_horizon"], confirmation_window=64,
                               chunk_rows=settings["preview_resolution"], callback=callback)
            atomic_npz(folder / "field.npz", **field)
            zoom = next_window(field["last_change"], field["confirmed"], [0., 0.], width, settings["zoom_factor"])
            previews.append({"key": key, "width": width, "correct": float(field["final_correct"].mean()),
                             "confirmed": float(field["confirmed"].mean()), "roughness": zoom["score"]})
        del model
        torch.cuda.empty_cache()
    candidates = []
    for width in settings["preview_rms_fractions"]:
        pair = [row for row in previews if row["width"] == width]
        minimum_correct = min(row["correct"] for row in pair)
        candidates.append((minimum_correct >= .9, sum(row["roughness"] for row in pair) if minimum_correct >= .9 else minimum_correct, width))
    chosen = max(candidates)[2]
    selection = {"status": "passed", **identity, "width": chosen, "previews": previews,
                 "criterion": "Most neighboring-time variation among common widths with >=90% correct in both 512-iteration previews; otherwise highest minimum correctness"}
    plot_previews(root, selection)
    atomic_json_save(selection, root / "prepared.json")
    print(f"PREPARED {json.dumps(selection)}", flush=True)
    return selection


def render_model(key, root, callback=None):
    from looping.basin_diagnostics.recursive.plots import plot_sequence
    settings = protocol()
    root = Path(root)
    prepared = json.loads((root / "prepared.json").read_text())
    validate_config(prepared["settings"], settings)
    validate_config(prepared["source_sha256"], runtime_manifest(SOURCE_PATHS)["source_sha256"])
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    model, spec = load_model(key)
    folder = root / key
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"prepared_sha256": file_sha256(root / "prepared.json"), "checkpoint": spec, "settings": settings,
                "source_sha256": prepared["source_sha256"]}
    if (folder / "identity.json").exists():
        validate_config(json.loads((folder / "identity.json").read_text()), identity)
    else:
        atomic_json_save(identity, folder / "identity.json")
    if (folder / "completed.json").exists():
        result = json.loads((folder / "completed.json").read_text())
        for name, checksum in result["sha256"].items():
            if file_sha256(folder / name) != checksum:
                raise ValueError(f"Completed artifact changed: {name}")
        return result
    atomic_json_save(runtime_manifest(SOURCE_PATHS), folder / "environment.json")
    print(f"CONFIG {json.dumps(identity)}", flush=True)
    started = time.perf_counter()
    center, width, levels = [0., 0.], prepared["width"], []
    for level in range(settings["levels"]):
        path = folder / f"level_{level}"
        field = grid_field(model, prepared["puzzle"], path, resolution=settings["resolution"], center=center, width=width,
                           horizon=settings["horizon"], confirmation_window=settings["confirmation_window"],
                           chunk_rows=settings["chunk_rows"], callback=callback)
        atomic_npz(folder / f"level_{level}.npz", **field)
        next_zoom = next_window(field["last_change"], field["confirmed"], center, width, settings["zoom_factor"])
        info = {"level": level, "center": center, "width": width, "magnification": settings["zoom_factor"] ** level,
                "horizon": settings["horizon"], "resolution": settings["resolution"], "next_zoom": next_zoom,
                "final_correct": float(field["final_correct"].mean()), "confirmed": float(field["confirmed"].mean()),
                "distinct_confirmed_times": int(len(np.unique(field["last_change"][field["confirmed"]]))),
                "sha256": file_sha256(folder / f"level_{level}.npz")}
        atomic_json_save(info, folder / f"level_{level}.json")
        levels.append(info)
        plot_sequence(folder, prepared, spec, levels)
        if callback:
            callback()
        print(f"LEVEL {json.dumps(info)}", flush=True)
        center, width = next_zoom["center"], next_zoom["width"]
    result = {"status": "complete", "identity": identity, "levels": levels, "seconds": time.perf_counter() - started,
              "sha256": {path.name: file_sha256(path) for path in folder.iterdir() if path.is_file() and path.suffix in (".npz", ".json", ".png") and path.name != "completed.json"}}
    atomic_json_save(result, folder / "completed.json")
    print(f"COMPLETE {key} seconds={result['seconds']:.1f}", flush=True)
    return result
