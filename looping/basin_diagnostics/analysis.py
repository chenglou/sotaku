"""Measure answer retention and neighboring trajectories around reached states."""

import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from checkpoint_utils import atomic_json_save, validate_config
from looping.basin_diagnostics.common import ROOT, SOURCE_PATHS, checkpoint_spec, load_model, protocol
from looping.weight_tying.common import atomic_npz
from runtime_utils import file_sha256, runtime_manifest


def select_rows(labels, settings):
    rng = np.random.default_rng(settings["sample_seed"])
    discovery, evaluation = [], []
    for label in ("0", "1-2", "3-10", "11-50", "51+"):
        eligible = np.flatnonzero(labels == label)[settings["exclude_first_per_bucket"]:]
        chosen = rng.choice(eligible, settings["examples_per_bucket"], replace=False)
        split = settings["discovery_per_bucket"]
        discovery.extend(chosen[:split].tolist())
        evaluation.extend(chosen[split:].tolist())
    return np.asarray(discovery), np.asarray(evaluation)


def prepare_selection(output_dir, data_dir=None, volume_root="/outputs"):
    settings = protocol()
    path = Path(data_dir or settings["data_directory"]) / "development.npz"
    if file_sha256(path) != settings["data_sha256"]:
        raise ValueError("Prepared benchmark data changed")
    with np.load(path, allow_pickle=False) as data:
        discovery, evaluation = select_rows(data["labels"], settings)
        indices = data["indices"].copy()
        labels = data["labels"].copy()
    # Gallery choices use only the small discovery set and previously saved outputs.
    archives = []
    for key in ("20k_20260907", "20k_20260908"):
        spec = checkpoint_spec(key, volume_root)
        folder = Path(spec["path"]).parent / "evaluations" / "final"
        saved = json.loads((folder / "result.json").read_text())
        if saved["identity"]["weights_sha256"] != spec["weights_sha256"]:
            raise ValueError("Gallery selection used another checkpoint")
        if file_sha256(folder / "predictions.npz") != saved["predictions_sha256"]:
            raise ValueError("Archived predictions changed")
        with np.load(folder / "predictions.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["indices"], indices):
                raise ValueError("Archived predictions use different puzzles")
            archives.append({name: arrays[name].copy() for name in ("solved_16", "solved_1024", "solved_4096")})
    healthy, failing = archives
    contrast = discovery[healthy["solved_4096"][discovery] & failing["solved_1024"][discovery]
                         & ~failing["solved_4096"][discovery]]
    slow = discovery[~healthy["solved_16"][discovery] & healthy["solved_4096"][discovery]]
    first = int(contrast[0] if len(contrast) else discovery[0])
    other = [int(row) for row in slow if row != first]
    second = other[0] if other else next(int(row) for row in discovery if row != first)
    result = {"settings": settings, "data_sha256": file_sha256(path),
              "discovery_rows": discovery.tolist(), "evaluation_rows": evaluation.tolist(),
              "gallery": [{"row": first, "reason": "late-collapse contrast" if len(contrast) else "fallback first discovery puzzle"},
                          {"row": second, "reason": "slow-solving example" if other else "fallback next discovery puzzle"}]}
    for entry in result["gallery"]:
        entry.update(index=int(indices[entry["row"]]), difficulty=str(labels[entry["row"]]))
    target = Path(output_dir) / "selection.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        validate_config(json.loads(target.read_text()), result)
    else:
        atomic_json_save(result, target)
    return result


def plane_basis(shape, seed):
    generator = torch.Generator().manual_seed(seed)
    dimension = int(np.prod(shape))
    directions, _ = torch.linalg.qr(torch.randn(dimension, 2, generator=generator, dtype=torch.float64))
    return directions.T.reshape(2, *shape).float() * dimension ** 0.5


def coordinates(resolution):
    if resolution < 3 or resolution % 2 != 1:
        raise ValueError("An odd grid resolution >=3 is required for an exact zero control")
    axis = torch.linspace(-1, 1, resolution)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((x.flatten(), y.flatten()), -1)


def inject_state(model, hidden, probabilities, offsets, *, anchor):
    perturbed = hidden + offsets
    if anchor == 0:
        updated_probabilities = probabilities.expand(len(perturbed), -1, -1).clone()
    else:
        # Feedback probabilities must remain consistent with the changed hidden state.
        updated_probabilities = model.output_head(perturbed).softmax(-1)
        unchanged = offsets.flatten(1).eq(0).all(-1)
        updated_probabilities[unchanged] = probabilities.expand(len(perturbed), -1, -1)[unchanged]
    return perturbed, updated_probabilities


def completed_board(logits, givens):
    predicted = logits.argmax(-1)
    return torch.where(givens != 0, givens - 1, predicted)


class OutcomeTracker:
    def __init__(self, board, targets, finite, anchor, horizon):
        self.horizon = horizon
        self.anchor = anchor
        self.targets = targets
        self.board = board
        self.correct = (board == targets).all(-1) & finite
        self.first_correct = torch.where(self.correct, anchor, horizon + 1)
        self.last_incorrect = torch.where(self.correct, anchor - 1, anchor)
        self.last_change = torch.full_like(self.first_correct, anchor)
        self.regressions = torch.zeros_like(self.first_correct)
        self.switches = torch.zeros_like(self.first_correct)
        self.ever_nonfinite = ~finite
        self.initial_correct = self.correct.clone()

    def update(self, board, finite, step):
        correct = (board == self.targets).all(-1) & finite
        changed = (board != self.board).any(-1) | ~finite
        self.first_correct = torch.where(correct & (self.first_correct > self.horizon), step, self.first_correct)
        self.last_incorrect = torch.where(~correct, step, self.last_incorrect)
        self.last_change = torch.where(changed, step, self.last_change)
        self.regressions += (self.correct & ~correct).long()
        self.switches += changed.long()
        self.ever_nonfinite |= ~finite
        self.board, self.correct = board, correct

    def arrays(self, confirmation_window):
        suffix_start = self.last_incorrect + 1
        retained = (self.first_correct <= self.horizon) & (self.regressions == 0)
        category = torch.where(self.first_correct > self.horizon, 0,
                               torch.where(~self.correct, 1, torch.where(retained, 3, 2)))
        values = {"first_correct": self.first_correct, "correct_suffix_start": suffix_start,
                  "last_change": self.last_change, "regressions": self.regressions,
                  "switches": self.switches, "final_correct": self.correct,
                  "initial_correct": self.initial_correct, "ever_nonfinite": self.ever_nonfinite,
                  "outcome": category, "final_board": self.board,
                  "confirmed_answer": (self.horizon - self.last_change >= confirmation_window) & ~self.ever_nonfinite,
                  "confirmed_correct": self.correct & (self.horizon - suffix_start >= confirmation_window)}
        return {name: tensor.detach().cpu().numpy() for name, tensor in values.items()}


def sensitivity_map(hidden, resolution):
    grid = F.normalize(hidden[:resolution * resolution].flatten(1), dim=-1).reshape(resolution, resolution, -1)
    horizontal = (grid[:, 1:] - grid[:, :-1]).norm(dim=-1)
    vertical = (grid[1:] - grid[:-1]).norm(dim=-1)
    out = torch.zeros((resolution, resolution), device=hidden.device)
    out[:, :-1] = horizontal
    out[:, 1:] = torch.maximum(out[:, 1:], horizontal)
    out[:-1] = torch.maximum(out[:-1], vertical)
    out[1:] = torch.maximum(out[1:], vertical)
    return out


@torch.inference_mode()
def reached_states(model, digits, anchors):
    given = torch.as_tensor(digits, device=next(model.parameters()).device).long()
    hidden, probabilities = model.initial_state(F.one_hot(given, 10).float())
    result = {0: (hidden.clone(), probabilities.clone())}
    for step in range(1, max(anchors, default=0) + 1):
        hidden, probabilities, _ = model.step(hidden, probabilities)
        if step in anchors:
            result[step] = (hidden.clone(), probabilities.clone())
    return result


@torch.inference_mode()
def trace(model, hidden, probabilities, digits, targets, *, anchor, settings, resolution=None):
    device = hidden.device
    given = torch.as_tensor(digits, device=device).long().expand(len(hidden), -1)
    answers = torch.as_tensor(targets, device=device).long().expand(len(hidden), -1)
    horizon = settings["horizon"]
    logits = model.output_head(hidden)
    board = completed_board(logits, given) if anchor else given - 1
    finite = hidden.flatten(1).isfinite().all(-1) & logits.flatten(1).isfinite().all(-1)
    tracker = OutcomeTracker(board, answers, finite, anchor, horizon)
    if resolution:
        center = resolution * resolution // 2
        selected = [center, center + 1, center + resolution]
        initial_neighbor = sensitivity_map(hidden, resolution)
        max_neighbor = initial_neighbor.clone()
        zero_controls = [center, resolution * resolution, resolution * resolution + 1]
    else:
        selected, zero_controls = [], []
    steps, correct_curve, state_rms, margins, hidden_snapshots, boards = [], [], [], [], [], []
    zero_mismatch = 0
    zero_max_error = 0.0

    def record(step):
        nonlocal max_neighbor, zero_mismatch, zero_max_error
        steps.append(step)
        correct_curve.append(tracker.correct.cpu().numpy())
        if resolution:
            max_neighbor = torch.maximum(max_neighbor, sensitivity_map(hidden, resolution))
            differences = (hidden[zero_controls] - hidden[zero_controls[0]]).abs().amax()
            zero_max_error = max(zero_max_error, float(differences))
            zero_mismatch += int((tracker.board[zero_controls] != tracker.board[zero_controls[0]]).any(-1).sum())
            values = hidden[selected]
            state_rms.append(values.square().mean((1, 2)).sqrt().cpu().numpy())
            competitors = logits[selected].clone()
            competitors.scatter_(-1, answers[selected].unsqueeze(-1), -torch.inf)
            margin = logits[selected].gather(-1, answers[selected].unsqueeze(-1)).squeeze(-1) - competitors.amax(-1)
            margin = margin.masked_fill(given[selected] != 0, torch.inf).amin(-1)
            margins.append(margin.cpu().numpy())
        if selected and (step == anchor or step == horizon or step % settings["snapshot_interval"] == 0):
            hidden_snapshots.append((step, hidden[selected].cpu().numpy()))
            boards.append((step, tracker.board[selected].cpu().numpy()))

    record(anchor)
    for step in range(anchor + 1, horizon + 1):
        hidden, probabilities, logits = model.step(hidden, probabilities)
        finite = (hidden.flatten(1).isfinite().all(-1) & logits.flatten(1).isfinite().all(-1)
                  & probabilities.flatten(1).isfinite().all(-1))
        tracker.update(completed_board(logits, given), finite, step)
        if step == horizon or step % settings["measurement_interval"] == 0:
            record(step)
    arrays = tracker.arrays(settings["confirmation_window"])
    arrays.update(steps=np.asarray(steps), correct_curve=np.asarray(correct_curve),
                  digits=given.cpu().numpy(), targets=answers.cpu().numpy())
    if resolution:
        arrays.update(initial_neighbor=initial_neighbor.cpu().numpy(), max_neighbor=max_neighbor.cpu().numpy(),
                      state_rms=np.asarray(state_rms), minimum_margin=np.asarray(margins),
                      snapshot_steps=np.asarray([item[0] for item in hidden_snapshots]),
                      hidden_snapshots=np.asarray([item[1] for item in hidden_snapshots]),
                      board_snapshots=np.asarray([item[1] for item in boards]))
    summary = {"initial_correct": float(arrays["initial_correct"].mean()),
               "final_correct": float(arrays["final_correct"].mean()),
               "ever_correct": float((arrays["first_correct"] <= horizon).mean()),
               "regressed": float((arrays["regressions"] > 0).mean()),
               "confirmed_answer": float(arrays["confirmed_answer"].mean()),
               "nonfinite": int(arrays["ever_nonfinite"].sum()),
               "zero_control_board_mismatches": zero_mismatch, "zero_control_max_hidden_error": zero_max_error}
    return arrays, summary


def uncertainty_curve(field, valid=None, seed=10):
    field = np.asarray(field)
    valid = np.ones_like(field, dtype=bool) if valid is None else np.asarray(valid)
    rng = np.random.default_rng(seed)
    shuffled = field.copy()
    shuffled[valid] = rng.permutation(field[valid])
    records = []
    for distance in (1, 2, 4):
        if distance >= min(field.shape):
            continue
        a = np.concatenate((field[:, distance:].flatten(), field[distance:, :].flatten()))
        b = np.concatenate((field[:, :-distance].flatten(), field[:-distance, :].flatten()))
        eligible = np.concatenate(((valid[:, distance:] & valid[:, :-distance]).flatten(),
                                   (valid[distance:, :] & valid[:-distance, :]).flatten()))
        sa = np.concatenate((shuffled[:, distance:].flatten(), shuffled[distance:, :].flatten()))
        sb = np.concatenate((shuffled[:, :-distance].flatten(), shuffled[:-distance, :].flatten()))
        records.append({"pixel_distance": distance, "pairs": int(eligible.sum()),
                        "different_fraction": float((a[eligible] != b[eligible]).mean()) if eligible.any() else None,
                        "shuffled_fraction": float((sa[eligible] != sb[eligible]).mean()) if eligible.any() else None})
    return records


def summarize_map(arrays, resolution, horizon):
    count = resolution ** 2
    correct = arrays["final_correct"][:count]
    ever = arrays["first_correct"][:count] <= horizon
    return {"final_correct": float(correct.mean()), "ever_correct": float(ever.mean()),
            "regressed": float((arrays["regressions"][:count] > 0).mean()),
            "confirmed_answer": float(arrays["confirmed_answer"][:count].mean()),
            "outcomes": {name: int((arrays["outcome"][:count] == index).sum()) for index, name in
                         enumerate(("never_correct", "lost_at_end", "recovered", "stayed_correct"))},
            "settling_uncertainty": uncertainty_curve(arrays["last_change"][:count].reshape(resolution, resolution),
                                                       arrays["confirmed_answer"][:count].reshape(resolution, resolution))}


def saved_unit(directory, name):
    summary_path = directory / f"{name}.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    if file_sha256(directory / f"{name}.npz") != summary["arrays_sha256"]:
        raise ValueError(f"Saved analysis arrays changed: {name}")
    return summary


def save_unit(directory, name, arrays, summary):
    path = directory / f"{name}.npz"
    atomic_npz(path, **arrays)
    summary = {**summary, "arrays_sha256": file_sha256(path)}
    atomic_json_save(summary, directory / f"{name}.json")
    return summary


@torch.inference_mode()
def run_model(key, output_dir, *, device="cuda", smoke=False, checkpoint_callback=None):
    from looping.basin_diagnostics.plots import render_map, render_probe

    started = time.perf_counter()
    settings = protocol()
    root = Path(output_dir)
    selection = json.loads((root / "selection.json").read_text())
    validate_config(selection["settings"], settings)
    if smoke:
        settings.update(horizon=192, anchors=[128], resolution=5, plane_seeds=[2701],
                        perturbation_rms_fractions=[0.001], confirmation_window=16, snapshot_interval=16)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    model, spec = load_model(key, device)
    environment = runtime_manifest(SOURCE_PATHS)
    identity = {"settings": settings, "checkpoint": spec, "source_sha256": environment["source_sha256"],
                "selection_sha256": file_sha256(root / "selection.json"), "smoke": smoke}
    directory = root / ("smoke" if smoke else "models") / key
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / "identity.json"
    if identity_path.exists():
        validate_config(json.loads(identity_path.read_text()), identity)
    else:
        atomic_json_save(identity, identity_path)
    if (directory / "completed.json").exists():
        completed = json.loads((directory / "completed.json").read_text())
        for filename, checksum in completed["artifact_sha256"].items():
            if file_sha256(directory / filename) != checksum:
                raise ValueError(f"Completed artifact changed: {filename}")
        return completed
    atomic_json_save(environment, directory / "environment.json")
    data_path = Path(settings["data_directory"]) / "development.npz"
    if file_sha256(data_path) != settings["data_sha256"]:
        raise ValueError("Prepared arrays changed")
    with np.load(data_path, allow_pickle=False) as data:
        digits, targets, indices = (data[name].copy() for name in ("digits", "targets", "indices"))
    print(f"CONFIG {json.dumps(identity, sort_keys=True)}", flush=True)
    rows = selection["evaluation_rows"][:2] if smoke else selection["evaluation_rows"]
    baseline_path = directory / "baseline.npz"
    if saved_unit(directory, "baseline") is None:
        states = reached_states(model, digits[rows], [0])
        arrays, summary = trace(model, *states[0], digits[rows], targets[rows], anchor=0, settings=settings)
        arrays["indices"] = indices[rows]
        save_unit(directory, "baseline", arrays, summary)
        print(f"BASELINE {json.dumps(summary)}", flush=True)
    probe_records = []
    for anchor, fraction in itertools.product(settings["anchors"], settings["perturbation_rms_fractions"]):
        name = f"probe_t{anchor}_r{fraction:g}"
        path = directory / f"{name}.npz"
        if saved_unit(directory, name) is None:
            states = reached_states(model, digits[rows], [anchor])
            hidden, probabilities = states[anchor]
            directions = plane_basis(hidden.shape[1:], 2701).to(device)
            offsets = torch.stack((torch.zeros_like(directions[0]), directions[0], -directions[0], directions[1], -directions[1]))
            offsets = offsets[None] * fraction * hidden.square().mean((1, 2), keepdim=True).sqrt()[:, None]
            hidden = hidden[:, None].expand(-1, 5, -1, -1).flatten(0, 1)
            probabilities = probabilities[:, None].expand(-1, 5, -1, -1).flatten(0, 1)
            perturbed = inject_state(model, hidden, probabilities, offsets.flatten(0, 1), anchor=anchor)
            given = np.repeat(digits[rows], 5, axis=0)
            answers = np.repeat(targets[rows], 5, axis=0)
            arrays, summary = trace(model, *perturbed, given, answers, anchor=anchor, settings=settings)
            arrays["indices"] = np.repeat(indices[rows], 5)
            arrays["sample_order"] = np.tile(np.arange(5), len(rows))
            save_unit(directory, name, arrays, summary)
            print(f"PROBE {name} {json.dumps(summary)}", flush=True)
            if checkpoint_callback:
                checkpoint_callback()
        probe_records.append(name)
    render_probe(directory, settings, spec, probe_records)
    records = []
    for gallery, anchor, fraction, plane_seed in itertools.product(selection["gallery"][:1] if smoke else selection["gallery"],
            settings["anchors"], settings["perturbation_rms_fractions"], settings["plane_seeds"]):
        name = f"map_row{gallery['row']}_t{anchor}_r{fraction:g}_p{plane_seed}"
        path = directory / f"{name}.npz"
        info = {"gallery": gallery, "anchor": anchor, "rms_fraction": fraction,
                "plane_seed": plane_seed, "resolution": settings["resolution"], "horizon": settings["horizon"]}
        summary = saved_unit(directory, name)
        if summary is None:
            hidden, probabilities = reached_states(model, digits[[gallery["row"]]], [anchor])[anchor]
            directions = plane_basis(hidden.shape[1:], plane_seed).to(device)
            coords = coordinates(settings["resolution"]).to(device)
            offsets = (coords[:, 0, None, None] * directions[0] + coords[:, 1, None, None] * directions[1])
            offsets *= fraction * hidden.square().mean().sqrt()
            offsets = torch.cat((offsets, torch.zeros_like(offsets[:2])), dim=0)
            perturbed = inject_state(model, hidden, probabilities, offsets, anchor=anchor)
            arrays, checks = trace(model, *perturbed, digits[[gallery["row"]]], targets[[gallery["row"]]],
                                   anchor=anchor, settings=settings, resolution=settings["resolution"])
            arrays["coordinates"] = coords.cpu().numpy()
            summary = {**info, **summarize_map(arrays, settings["resolution"], settings["horizon"]), "checks": checks}
            summary = save_unit(directory, name, arrays, summary)
            print(f"MAP {name} {json.dumps(summary)}", flush=True)
        render_map(path, directory / f"{name}.png", spec, summary)
        records.append({"name": name, **summary})
        if checkpoint_callback:
            checkpoint_callback()
    result = {"status": "complete", "identity": identity, "baseline": json.loads((directory / "baseline.json").read_text()),
              "maps": records, "elapsed_seconds": time.perf_counter() - started,
              "artifact_sha256": {path.name: file_sha256(path) for path in directory.iterdir()
                                  if path.suffix in (".npz", ".png", ".json") and path.name != "completed.json"}}
    atomic_json_save(result, directory / "completed.json")
    print(f"COMPLETE {key} seconds={result['elapsed_seconds']:.1f}", flush=True)
    return result
