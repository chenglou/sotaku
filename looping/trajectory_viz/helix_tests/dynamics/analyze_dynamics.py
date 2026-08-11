"""Held-out per-cell tests for helix-like recurrent geometry.

The analysis deliberately separates accumulated translation from rotation. All
feature-space bases are fitted on a rating-stratified set of complete puzzles
and evaluated on different puzzles. The result is a collection of numerical
artifacts and plots rather than a single favorable projection.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import time
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


PHASES = OrderedDict(
    (
        ("initial 0-16", (0, 16)),
        ("solving 16-64", (16, 64)),
        ("settling 64-128", (64, 128)),
        ("late 128-512", (128, 512)),
        ("deep 512-1024", (512, 1024)),
    )
)
TRANSITION_NAMES = ("wrong→wrong", "wrong→correct", "correct→wrong", "correct→correct")
REPRESENTATION_NAMES = (
    "raw state residual",
    "unit state residual",
    "raw update",
    "unit update",
)
DEFAULT_SEED = 20260807
DEFAULT_EXAMPLES_PER_BUCKET = 10
DEFAULT_FINAL_ITERATION = 1024
DEFAULT_PHASE_SAMPLES = 65
BOOTSTRAP_REPETITIONS = 500
TIME_CONTROL_REPETITIONS = 24
RANDOM_PLANE_REPETITIONS = 12
MODEL_CYCLE_CANDIDATES = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path, value):
    temporary_path = path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(_json_ready(value), handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def stratified_puzzle_split(bucket_names):
    """Alternate complete puzzles within every rating bucket."""
    fit_indices = []
    heldout_indices = []
    for bucket in dict.fromkeys(bucket_names):
        indices = [index for index, name in enumerate(bucket_names) if name == bucket]
        if len(indices) < 2:
            raise ValueError(f"bucket {bucket!r} needs at least two puzzles")
        fit_indices.extend(indices[::2])
        heldout_indices.extend(indices[1::2])
    return fit_indices, heldout_indices


def _phase_times(start, end, maximum_count, *, updates=False, device=None):
    inclusive_end = end - 1 if updates else end
    length = inclusive_end - start + 1
    if length <= maximum_count:
        return torch.arange(start, inclusive_end + 1, device=device)
    return torch.linspace(start, inclusive_end, maximum_count, device=device).round().long().unique()


def _cell_time_values(values, puzzle_indices, times, cell_mask):
    """Return [selected cells, time, features] without splitting cells across sets."""
    selected = values[puzzle_indices][:, times].permute(0, 2, 1, 3)
    selected_mask = cell_mask[puzzle_indices]
    return selected[selected_mask]


def _chord_residual(values, start, end, times):
    selected = values[:, times]
    alpha = (times.float() - start) / max(end - start, 1)
    shape = (1, len(times), 1, 1)
    chord = values[:, start : start + 1] + alpha.view(shape) * (
        values[:, end : end + 1] - values[:, start : start + 1]
    )
    return selected - chord


def _normalize_tokens(values):
    return F.normalize(values.float(), dim=-1, eps=1e-12)


def fit_feature_basis(rows, component_count=3):
    rows = rows.float()
    mean = rows.mean(dim=0, keepdim=True)
    centered = rows - mean
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order].clamp_min(0)
    basis = eigenvectors[:, order[:component_count]]
    total = eigenvalues.sum().clamp_min(1e-30)
    return mean, basis, eigenvalues[:component_count] / total


def fit_representation_bases(states, updates, fit_indices, empty_mask, phase_samples):
    rows = {name: [] for name in REPRESENTATION_NAMES}
    for start, end in PHASES.values():
        state_times = _phase_times(start, end, phase_samples, device=states.device)
        update_times = _phase_times(
            start,
            end,
            phase_samples,
            updates=True,
            device=states.device,
        )
        raw_state_residual = _chord_residual(states[fit_indices], start, end, state_times)
        unit_states = _normalize_tokens(states[fit_indices])
        unit_state_residual = _chord_residual(unit_states, start, end, state_times)
        selected_mask = empty_mask[fit_indices]
        for name, values in (
            ("raw state residual", raw_state_residual),
            ("unit state residual", unit_state_residual),
        ):
            cell_time = values.permute(0, 2, 1, 3)[selected_mask]
            rows[name].append(cell_time.reshape(-1, cell_time.size(-1)))
        raw_updates = _cell_time_values(updates, fit_indices, update_times, empty_mask)
        unit_updates = _normalize_tokens(raw_updates)
        rows["raw update"].append(raw_updates.reshape(-1, raw_updates.size(-1)))
        rows["unit update"].append(unit_updates.reshape(-1, unit_updates.size(-1)))

    bases = {}
    for name, chunks in rows.items():
        mean, basis, explained = fit_feature_basis(torch.cat(chunks), 3)
        bases[name] = {
            "mean": mean,
            "basis": basis,
            "train_explained": explained,
        }
    return bases


def collect_trajectory(model, inputs, final_iteration):
    """Collect h_t, h_{t+1}-h_t, and logits for every integer iteration."""
    device = inputs.device
    batch_size = inputs.size(0)
    hidden_size = model_module.d_model
    states = torch.empty(
        batch_size,
        final_iteration + 1,
        81,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    updates = torch.empty(
        batch_size,
        final_iteration,
        81,
        hidden_size,
        device=device,
        dtype=torch.float32,
    )
    logits = torch.empty(
        batch_size,
        final_iteration + 1,
        81,
        9,
        device=device,
        dtype=torch.float32,
    )
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    with torch.no_grad():
        hidden = model.initial_encoder(inputs)
        predictions = torch.zeros(batch_size, 81, 9, device=device)
        states[:, 0] = hidden
        logits[:, 0] = model.output_head(hidden)
        for iteration in range(final_iteration):
            next_hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
            next_logits = model.output_head(next_hidden)
            updates[:, iteration] = next_hidden - hidden
            states[:, iteration + 1] = next_hidden
            logits[:, iteration + 1] = next_logits
            hidden = next_hidden
            predictions = F.softmax(next_logits, dim=-1)
    return states, updates, logits


def prediction_statistics(logits, targets):
    target_index = targets[:, None, :, None].expand(
        logits.size(0), logits.size(1), logits.size(2), 1
    )
    target_logits = logits.gather(-1, target_index).squeeze(-1)
    wrong_logits = logits.clone()
    wrong_logits.scatter_(-1, target_index, -torch.inf)
    margin = target_logits - wrong_logits.max(dim=-1).values
    probabilities = F.softmax(logits, dim=-1)
    target_probability = probabilities.gather(-1, target_index).squeeze(-1)
    confidence, predictions = probabilities.max(dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    correct = predictions.eq(targets[:, None, :])
    return {
        "margin": margin,
        "target_probability": target_probability,
        "confidence": confidence,
        "entropy": entropy,
        "predictions": predictions,
        "correct": correct,
    }


def _linear_r2(x, y):
    """Vectorized R² for y[..., time] against one shared x[time]."""
    x = x.float()
    y = y.float()
    if x.ndim == 1:
        centered_x = x - x.mean()
        x_energy = centered_x.square().sum().clamp_min(1e-12)
    else:
        centered_x = x - x.mean(dim=-1, keepdim=True)
        x_energy = centered_x.square().sum(dim=-1).clamp_min(1e-12)
    centered_y = y - y.mean(dim=-1, keepdim=True)
    slope = (centered_y * centered_x).sum(dim=-1) / x_energy
    fitted = y.mean(dim=-1, keepdim=True) + slope[..., None] * centered_x
    residual = (y - fitted).square().sum(dim=-1)
    total = centered_y.square().sum(dim=-1)
    return (1 - residual / total.clamp_min(1e-12)).clamp(-10, 1)


def phase_geometry_metrics(
    points,
    *,
    state_residual,
    prepared_state_residual=False,
    time=None,
):
    """Return per-cell rotation, spiral, and helix metrics for [P,C,T,3]."""
    if time is None:
        time = torch.linspace(0, 1, points.size(2), device=points.device)
    else:
        time = time.to(points.device).float()
    if state_residual and not prepared_state_residual and points.size(2) > 4:
        points = points[:, :, 1:-1]
        time = time[1:-1]
    time = (time - time[0]) / (time[-1] - time[0]).clamp_min(1e-12)
    centered = points if state_residual else points - points.mean(dim=2, keepdim=True)
    planar = centered[..., :2]
    radius = planar.norm(dim=-1)
    angle = torch.atan2(planar[..., 1], planar[..., 0])
    angle_step = torch.atan2(
        torch.sin(angle[..., 1:] - angle[..., :-1]),
        torch.cos(angle[..., 1:] - angle[..., :-1]),
    )
    unwrapped = torch.cat(
        [angle[..., :1], angle[..., :1] + angle_step.cumsum(dim=-1)],
        dim=-1,
    )
    total_angle = angle_step.abs().sum(dim=-1)
    net_angle = angle_step.sum(dim=-1).abs()
    phase_r2 = _linear_r2(time, unwrapped)
    radial_r2 = _linear_r2(unwrapped, radius)
    phase_resultant = torch.sqrt(
        torch.cos(angle_step).mean(dim=-1).square()
        + torch.sin(angle_step).mean(dim=-1).square()
    )
    radius_mean = radius.mean(dim=-1)
    radius_reference = radius_mean[radius_mean > 0].median().clamp_min(1e-12)
    planar_fraction = planar.square().sum(dim=(-1, -2)) / centered.square().sum(
        dim=(-1, -2)
    ).clamp_min(1e-12)
    valid = (radius_mean >= 0.1 * radius_reference) & (planar_fraction >= 0.05)
    return {
        "turns_total": total_angle / (2 * math.pi),
        "turns_net": net_angle / (2 * math.pi),
        "rotation_monotonicity": net_angle / total_angle.clamp_min(1e-12),
        "phase_linearity_r2": phase_r2,
        "phase_step_resultant": phase_resultant,
        "radius_cv": radius.std(dim=-1) / radius_mean.clamp_min(1e-12),
        "radial_spiral_r2": radial_r2,
        "valid": valid,
        "angle": angle,
        "angle_step": angle_step,
    }


def translation_metrics(values, start, end, times):
    """High-dimensional per-cell translation measures for [P,T,C,D]."""
    selected = values[:, times].permute(0, 2, 1, 3)
    steps = selected[:, :, 1:] - selected[:, :, :-1]
    iteration_gaps = (times[1:] - times[:-1]).float().clamp_min(1)
    velocity = steps / iteration_gaps.view(1, 1, -1, 1)
    displacement = (selected[:, :, -1] - selected[:, :, 0]).norm(dim=-1)
    path_length = steps.norm(dim=-1).sum(dim=-1)
    alpha = (times.float() - start) / max(end - start, 1)
    chord = selected[:, :, :1] + alpha.view(1, 1, -1, 1) * (
        selected[:, :, -1:] - selected[:, :, :1]
    )
    residual_energy = (selected - chord).square().sum(dim=(-1, -2))
    total_energy = (selected - selected.mean(dim=2, keepdim=True)).square().sum(dim=(-1, -2))
    mean_step = velocity.mean(dim=2).norm(dim=-1)
    mean_norm = velocity.norm(dim=-1).mean(dim=-1)
    return {
        "path_efficiency": displacement / path_length.clamp_min(1e-12),
        "translation_r2": (1 - residual_energy / total_energy.clamp_min(1e-12)).clamp(-10, 1),
        "velocity_coherence": mean_step / mean_norm.clamp_min(1e-12),
    }


def _bootstrap_summary(values, buckets, seed, repetitions=BOOTSTRAP_REPETITIONS):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    buckets = np.asarray(buckets)[finite]
    if len(values) == 0:
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "puzzles": 0}
    estimate = float(values.mean())
    generator = np.random.default_rng(seed)
    samples = []
    unique_buckets = list(dict.fromkeys(buckets.tolist()))
    for _ in range(repetitions):
        pieces = []
        for bucket in unique_buckets:
            group = values[buckets == bucket]
            pieces.append(generator.choice(group, size=len(group), replace=True))
        samples.append(np.concatenate(pieces).mean())
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "estimate": estimate,
        "ci_low": float(low),
        "ci_high": float(high),
        "puzzles": int(len(values)),
    }


def summarize_cell_metric(metric, cell_mask, buckets, seed):
    metric = metric.detach().float().cpu()
    cell_mask = cell_mask.detach().cpu()
    puzzle_values = []
    for puzzle_index in range(metric.size(0)):
        selected = metric[puzzle_index][cell_mask[puzzle_index]]
        selected = selected[torch.isfinite(selected)]
        puzzle_values.append(float(selected.median()) if len(selected) else float("nan"))
    summary = _bootstrap_summary(puzzle_values, buckets, seed)
    summary["per_puzzle"] = puzzle_values
    return summary


def _block_permutation(length, generator):
    block_size = max(2, length // 8)
    blocks = [torch.arange(start, min(start + block_size, length)) for start in range(0, length, block_size)]
    order = torch.randperm(len(blocks), generator=generator).tolist()
    return torch.cat([blocks[index] for index in order])


def _project_phase(
    values,
    puzzle_indices,
    cell_mask,
    start,
    end,
    representation,
    basis_info,
    maximum_count,
    detrend_state=True,
):
    is_state = "state" in representation
    source = values[puzzle_indices].float()
    if representation.startswith("unit"):
        source = _normalize_tokens(source)
    times = _phase_times(
        start,
        end,
        maximum_count,
        updates=not is_state,
        device=values.device,
    )
    if is_state and detrend_state:
        selected = _chord_residual(source, start, end, times)
    else:
        selected = source[:, times]
    projected = (selected - basis_info["mean"]) @ basis_info["basis"]
    return projected.permute(0, 2, 1, 3), times


def analyze_phase_geometry(
    states,
    updates,
    bases,
    heldout_indices,
    empty_mask,
    bucket_names,
    seed,
    phase_samples,
):
    records = []
    details = {}
    heldout_buckets = [bucket_names[index] for index in heldout_indices]
    masks = {
        "blank": empty_mask[heldout_indices],
        "given": ~empty_mask[heldout_indices],
    }
    for phase_index, (phase_name, (start, end)) in enumerate(PHASES.items()):
        state_times = _phase_times(start, end, phase_samples, device=states.device)
        for normalized, representation_name in ((False, "raw state"), (True, "unit state")):
            source = _normalize_tokens(states[heldout_indices]) if normalized else states[heldout_indices]
            translation = translation_metrics(source, start, end, state_times)
            for cell_type, mask in masks.items():
                for metric_name, metric in translation.items():
                    summary = summarize_cell_metric(
                        metric,
                        mask,
                        heldout_buckets,
                        seed + phase_index * 100 + len(records),
                    )
                    records.append(
                        {
                            "phase": phase_name,
                            "representation": representation_name,
                            "cell_type": cell_type,
                            "control": "ordered",
                            "metric": metric_name,
                            **{key: value for key, value in summary.items() if key != "per_puzzle"},
                        }
                    )

        for representation_index, representation in enumerate(REPRESENTATION_NAMES):
            source = states if "state" in representation else updates
            points, times = _project_phase(
                source,
                heldout_indices,
                empty_mask,
                start,
                end,
                representation,
                bases[representation],
                phase_samples,
            )
            normalized_times = (times.float() - times[0]) / (times[-1] - times[0]).clamp_min(1)
            ordered = phase_geometry_metrics(
                points,
                state_residual="state" in representation,
                time=normalized_times,
            )
            detail_key = f"{phase_name}|{representation}"
            details[detail_key] = {
                "times": times.detach().cpu().tolist(),
                "ordered_angle": ordered["angle"].detach().cpu(),
                "points": points.detach().cpu(),
            }
            for cell_type, mask in masks.items():
                valid_mask = mask & ordered["valid"]
                for metric_name in (
                    "turns_total",
                    "turns_net",
                    "rotation_monotonicity",
                    "phase_linearity_r2",
                    "phase_step_resultant",
                    "radius_cv",
                    "radial_spiral_r2",
                ):
                    summary = summarize_cell_metric(
                        ordered[metric_name],
                        valid_mask,
                        heldout_buckets,
                        seed + phase_index * 1000 + representation_index * 100 + len(records),
                    )
                    records.append(
                        {
                            "phase": phase_name,
                            "representation": representation,
                            "cell_type": cell_type,
                            "control": "ordered",
                            "metric": metric_name,
                            **{
                                key: value
                                for key, value in summary.items()
                                if key != "per_puzzle"
                            },
                        }
                    )
                valid_fraction = []
                for puzzle_index in range(ordered["valid"].size(0)):
                    denominator = mask[puzzle_index].sum().clamp_min(1)
                    valid_fraction.append(
                        float((ordered["valid"][puzzle_index] & mask[puzzle_index]).sum() / denominator)
                    )
                valid_summary = _bootstrap_summary(
                    valid_fraction,
                    heldout_buckets,
                    seed + len(records),
                )
                records.append(
                    {
                        "phase": phase_name,
                        "representation": representation,
                        "cell_type": cell_type,
                        "control": "ordered",
                        "metric": "valid_cell_fraction",
                        **valid_summary,
                    }
                )

            generator = torch.Generator().manual_seed(seed + phase_index * 100 + representation_index)
            control_points = (
                points[:, :, 1:-1]
                if "state" in representation and points.size(2) > 4
                else points
            )
            controls = {"full time shuffle": [], "block time shuffle": []}
            for _ in range(TIME_CONTROL_REPETITIONS):
                full_order = torch.randperm(control_points.size(2), generator=generator).to(points.device)
                block_order = _block_permutation(control_points.size(2), generator).to(points.device)
                controls["full time shuffle"].append(
                    phase_geometry_metrics(
                        control_points[:, :, full_order],
                        state_residual="state" in representation,
                        prepared_state_residual="state" in representation,
                        time=normalized_times[1:-1]
                        if "state" in representation
                        else normalized_times,
                    )
                )
                controls["block time shuffle"].append(
                    phase_geometry_metrics(
                        control_points[:, :, block_order],
                        state_residual="state" in representation,
                        prepared_state_residual="state" in representation,
                        time=normalized_times[1:-1]
                        if "state" in representation
                        else normalized_times,
                    )
                )
            reverse = phase_geometry_metrics(
                points.flip(2),
                state_residual="state" in representation,
                time=normalized_times,
            )
            control_metrics = {
                "time reversal": reverse,
                **{
                    control_name: {
                        metric_name: torch.stack([item[metric_name] for item in repetitions]).median(dim=0).values
                        for metric_name in (
                            "turns_total",
                            "turns_net",
                            "rotation_monotonicity",
                            "phase_linearity_r2",
                        )
                    }
                    for control_name, repetitions in controls.items()
                },
            }
            for control_name, metrics in control_metrics.items():
                for metric_name in (
                    "turns_total",
                    "turns_net",
                    "rotation_monotonicity",
                    "phase_linearity_r2",
                ):
                    summary = summarize_cell_metric(
                        metrics[metric_name],
                        masks["blank"] & ordered["valid"],
                        heldout_buckets,
                        seed + len(records),
                    )
                    records.append(
                        {
                            "phase": phase_name,
                            "representation": representation,
                            "cell_type": "blank",
                            "control": control_name,
                            "metric": metric_name,
                            **{key: value for key, value in summary.items() if key != "per_puzzle"},
                        }
                    )
    return records, details


def _design_matrix(time, model_name, cycles=None):
    time = time.float()
    columns = [torch.ones_like(time), time]
    if model_name == "translation":
        pass
    elif model_name == "cubic":
        columns.extend([time.square(), time.pow(3)])
    elif model_name == "quintic":
        columns.extend([time.pow(power) for power in range(2, 6)])
    elif model_name in ("rotation", "spiral"):
        if cycles is None:
            raise ValueError("cycles is required for periodic models")
        angle = 2 * math.pi * cycles * time
        cosine = torch.cos(angle)
        sine = torch.sin(angle)
        columns.extend([cosine, sine])
        if model_name == "spiral":
            columns.extend([time * cosine, time * sine])
    else:
        raise ValueError(f"unknown model {model_name!r}")
    return torch.stack(columns, dim=1)


def _sequence_model_sse(
    points,
    cell_mask,
    model_name,
    cycles=None,
    calibration_fraction=0.6,
    time=None,
):
    """Fit cell nuisance coefficients on a prefix and score a later suffix."""
    point_count = points.size(2)
    parameter_count = {
        "translation": 2,
        "cubic": 4,
        "rotation": 4,
        "quintic": 6,
        "spiral": 6,
    }[model_name]
    calibration_count = max(parameter_count + 2, int(round(point_count * calibration_fraction)))
    calibration_count = min(calibration_count, point_count - 2)
    if calibration_count <= parameter_count or calibration_count >= point_count:
        raise ValueError(f"phase with {point_count} points is too short for {model_name}")
    if time is None:
        time = torch.linspace(0, 1, point_count, device=points.device)
    else:
        time = time.to(points.device).float()
        time = (time - time[0]) / (time[-1] - time[0]).clamp_min(1e-12)
    design = _design_matrix(time, model_name, cycles)
    fit_design = design[:calibration_count]
    test_design = design[calibration_count:]
    selected = points[cell_mask]
    coefficients = torch.linalg.pinv(fit_design) @ selected[:, :calibration_count]
    predicted = torch.einsum("tp,npd->ntd", test_design, coefficients)
    residual = selected[:, calibration_count:] - predicted
    sse_by_cell = residual.square().sum(dim=(1, 2))

    counts = cell_mask.sum(dim=1)
    puzzle_ids = torch.repeat_interleave(
        torch.arange(points.size(0), device=points.device), counts
    )
    sse_by_puzzle = torch.zeros(points.size(0), device=points.device)
    sse_by_puzzle.scatter_add_(0, puzzle_ids, sse_by_cell)
    calibration_mean = selected[:, :calibration_count].mean(dim=1, keepdim=True)
    constant_sse = (selected[:, calibration_count:] - calibration_mean).square().sum(dim=(1, 2))
    constant_by_puzzle = torch.zeros(points.size(0), device=points.device)
    constant_by_puzzle.scatter_add_(0, puzzle_ids, constant_sse)
    return sse_by_puzzle, constant_by_puzzle


def _select_periodic_frequency(points, cell_mask, model_name, time=None):
    translation_sse, _ = _sequence_model_sse(
        points, cell_mask, "translation", time=time
    )
    denominator = translation_sse.sum().clamp_min(1e-30)
    candidates = []
    for cycles in MODEL_CYCLE_CANDIDATES:
        sse, _ = _sequence_model_sse(
            points, cell_mask, model_name, cycles, time=time
        )
        candidates.append((float(sse.sum() / denominator), cycles))
    return min(candidates)[1], candidates


def _score_predictive_models(
    train_points,
    heldout_points,
    train_mask,
    heldout_mask,
    *,
    rotation_cycles=None,
    spiral_cycles=None,
    time=None,
):
    if rotation_cycles is None:
        rotation_cycles, rotation_search = _select_periodic_frequency(
            train_points, train_mask, "rotation", time=time
        )
    else:
        rotation_search = []
    if spiral_cycles is None:
        spiral_cycles, spiral_search = _select_periodic_frequency(
            train_points, train_mask, "spiral", time=time
        )
    else:
        spiral_search = []
    model_specs = (
        ("translation", None),
        ("cubic", None),
        ("quintic", None),
        ("rotation", rotation_cycles),
        ("spiral", spiral_cycles),
    )
    scores = {}
    constant = None
    for model_name, cycles in model_specs:
        sse, constant_sse = _sequence_model_sse(
            heldout_points, heldout_mask, model_name, cycles, time=time
        )
        scores[model_name] = sse
        constant = constant_sse
    per_puzzle = {
        "translation_predictive_r2": 1 - scores["translation"] / constant.clamp_min(1e-30),
        "rotation_skill_vs_translation": 1 - scores["rotation"] / scores["translation"].clamp_min(1e-30),
        "rotation_skill_vs_cubic": 1 - scores["rotation"] / scores["cubic"].clamp_min(1e-30),
        "spiral_skill_vs_translation": 1 - scores["spiral"] / scores["translation"].clamp_min(1e-30),
        "spiral_skill_vs_quintic": 1 - scores["spiral"] / scores["quintic"].clamp_min(1e-30),
    }
    return {
        "rotation_cycles": rotation_cycles,
        "spiral_cycles": spiral_cycles,
        "rotation_search": rotation_search,
        "spiral_search": spiral_search,
        "per_puzzle": per_puzzle,
    }


def analyze_predictive_geometry(
    states,
    updates,
    bases,
    fit_indices,
    heldout_indices,
    empty_mask,
    bucket_names,
    seed,
    phase_samples,
):
    records = []
    details = {}
    heldout_buckets = [bucket_names[index] for index in heldout_indices]
    train_mask = empty_mask[fit_indices]
    heldout_mask = empty_mask[heldout_indices]
    for phase_index, (phase_name, (start, end)) in enumerate(PHASES.items()):
        for representation_index, representation in enumerate(REPRESENTATION_NAMES):
            source = states if "state" in representation else updates
            train_points, times = _project_phase(
                source,
                fit_indices,
                empty_mask,
                start,
                end,
                representation,
                bases[representation],
                phase_samples,
                detrend_state=False,
            )
            heldout_points, _ = _project_phase(
                source,
                heldout_indices,
                empty_mask,
                start,
                end,
                representation,
                bases[representation],
                phase_samples,
                detrend_state=False,
            )
            scored = _score_predictive_models(
                train_points,
                heldout_points,
                train_mask,
                heldout_mask,
                time=times,
            )
            reported_representation = representation.replace(
                "state residual",
                "state in residual-fitted basis",
            )
            details[f"{phase_name}|{representation}"] = {
                "times": times.detach().cpu().tolist(),
                "rotation_cycles": scored["rotation_cycles"],
                "spiral_cycles": scored["spiral_cycles"],
                "rotation_search": scored["rotation_search"],
                "spiral_search": scored["spiral_search"],
            }
            for metric_index, (metric_name, values) in enumerate(scored["per_puzzle"].items()):
                summary = _bootstrap_summary(
                    values.detach().cpu().numpy(),
                    heldout_buckets,
                    seed + phase_index * 1000 + representation_index * 100 + metric_index,
                )
                records.append(
                    {
                        "phase": phase_name,
                        "representation": reported_representation,
                        "control": "ordered",
                        "metric": metric_name,
                        "selected_rotation_cycles": scored["rotation_cycles"],
                        "selected_spiral_cycles": scored["spiral_cycles"],
                        **summary,
                    }
                )

            generator = torch.Generator().manual_seed(seed + phase_index * 100 + representation_index)
            for control_name in ("full time shuffle", "block time shuffle", "time reversal"):
                repetitions = 1 if control_name == "time reversal" else TIME_CONTROL_REPETITIONS
                control_values = {name: [] for name in scored["per_puzzle"]}
                for _ in range(repetitions):
                    if control_name == "time reversal":
                        order = torch.arange(heldout_points.size(2) - 1, -1, -1, device=states.device)
                    elif control_name == "full time shuffle":
                        order = torch.randperm(heldout_points.size(2), generator=generator).to(states.device)
                    else:
                        order = _block_permutation(heldout_points.size(2), generator).to(states.device)
                    controlled = heldout_points[:, :, order]
                    control_score = _score_predictive_models(
                        train_points,
                        controlled,
                        train_mask,
                        heldout_mask,
                        rotation_cycles=scored["rotation_cycles"],
                        spiral_cycles=scored["spiral_cycles"],
                        time=times,
                    )
                    for metric_name, values in control_score["per_puzzle"].items():
                        control_values[metric_name].append(values)
                for metric_index, (metric_name, repetitions_values) in enumerate(control_values.items()):
                    values = torch.stack(repetitions_values).median(dim=0).values
                    summary = _bootstrap_summary(
                        values.detach().cpu().numpy(),
                        heldout_buckets,
                        seed + len(records),
                    )
                    records.append(
                        {
                            "phase": phase_name,
                            "representation": reported_representation,
                            "control": control_name,
                            "metric": metric_name,
                            "selected_rotation_cycles": scored["rotation_cycles"],
                            "selected_spiral_cycles": scored["spiral_cycles"],
                            **summary,
                        }
                    )
    return records, details


def _cycle_catalog():
    """Return the 20,160 distinct unoriented cycles on nine labeled digits."""
    cycles = []
    for tail in itertools.permutations(range(1, 9)):
        if tail[0] < tail[-1]:
            cycles.append((0,) + tail)
    pairs = list(itertools.combinations(range(9), 2))
    patterns = np.empty((len(cycles), len(pairs)), dtype=np.float64)
    adjacency = np.empty_like(patterns)
    for cycle_index, cycle in enumerate(cycles):
        positions = np.empty(9, dtype=np.int8)
        for position, digit in enumerate(cycle):
            positions[digit] = position
        distances = []
        for first, second in pairs:
            distance = abs(int(positions[first]) - int(positions[second]))
            distances.append(min(distance, 9 - distance))
        patterns[cycle_index] = distances
        adjacency[cycle_index] = np.asarray(distances) == 1
    centered = patterns - patterns.mean(axis=1, keepdims=True)
    normalized = centered / np.linalg.norm(centered, axis=1, keepdims=True).clip(1e-12)
    natural_index = cycles.index(tuple(range(9)))
    return {
        "cycles": cycles,
        "pairs": pairs,
        "patterns": patterns,
        "normalized_patterns": normalized,
        "adjacency": adjacency,
        "natural_index": natural_index,
    }


def _rdm(centroids):
    pairs = list(itertools.combinations(range(9), 2))
    return np.asarray(
        [np.linalg.norm(centroids[first] - centroids[second]) for first, second in pairs],
        dtype=np.float64,
    )


def _cycle_correlations(rdm, catalog):
    centered = rdm - rdm.mean()
    normalized = centered / max(np.linalg.norm(centered), 1e-12)
    return np.einsum(
        "ij,j->i",
        catalog["normalized_patterns"],
        normalized.astype(np.float64, copy=False),
        optimize=False,
    )


def _digit_centroids(values, targets, mask):
    values = _normalize_tokens(values).detach()
    centroids = []
    for digit in range(9):
        selected = values[(targets == digit) & mask]
        centroids.append(selected.mean(dim=0))
    return torch.stack(centroids)


def _head_null_basis(model):
    centered_head = model.output_head.weight.float() - model.output_head.weight.float().mean(dim=0, keepdim=True)
    _, singular_values, right = torch.linalg.svd(centered_head, full_matrices=False)
    rank = int((singular_values > singular_values.max() * 1e-6).sum())
    return right[:rank].T


def _remove_subspace(values, basis):
    return values - (values @ basis) @ basis.T


def _digit_geometry_record(train_centroids, heldout_centroids, catalog):
    train_rdm = _rdm(train_centroids)
    heldout_rdm = _rdm(heldout_centroids)
    train_correlations = _cycle_correlations(train_rdm, catalog)
    heldout_correlations = _cycle_correlations(heldout_rdm, catalog)
    natural_index = catalog["natural_index"]
    learned_index = int(np.argmax(train_correlations))
    natural_score = float(heldout_correlations[natural_index])
    learned_score = float(heldout_correlations[learned_index])
    centered_train = train_centroids - train_centroids.mean(axis=0, keepdims=True)
    _, singular_values, basis = np.linalg.svd(centered_train, full_matrices=False)
    heldout_centered = heldout_centroids - train_centroids.mean(axis=0, keepdims=True)
    projected = heldout_centered @ basis[:2].T
    plane_explained = float((projected**2).sum() / max((heldout_centered**2).sum(), 1e-30))
    eigenvalue_balance = float(
        singular_values[1] ** 2 / max(singular_values[0] ** 2, 1e-30)
    )
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    target_angles = np.arange(9) * 2 * np.pi / 9
    angular_forward = abs(np.exp(1j * (angles - target_angles)).mean())
    angular_reverse = abs(np.exp(1j * (angles + target_angles)).mean())
    angular_score = float(max(angular_forward, angular_reverse))
    return {
        "natural_cycle_correlation": natural_score,
        "natural_cycle_exact_p": float((heldout_correlations >= natural_score - 1e-12).mean()),
        "natural_cycle_rank": int((heldout_correlations > natural_score + 1e-12).sum() + 1),
        "learned_cycle_correlation": learned_score,
        "learned_cycle_exact_p": float((heldout_correlations >= learned_score - 1e-12).mean()),
        "learned_cycle_rank": int((heldout_correlations > learned_score + 1e-12).sum() + 1),
        "learned_cycle_digits": [digit + 1 for digit in catalog["cycles"][learned_index]],
        "heldout_plane_explained": plane_explained,
        "train_second_to_first_eigenvalue": eigenvalue_balance,
        "heldout_angular_natural_score": angular_score,
        "train_centroids": train_centroids,
        "heldout_centroids": heldout_centroids,
        "heldout_plane_coordinates": projected,
        "heldout_null_q025": float(np.quantile(heldout_correlations, 0.025)),
        "heldout_null_q975": float(np.quantile(heldout_correlations, 0.975)),
    }


def _prediction_transition_cycle(predictions, mask, puzzle_indices, start, end, catalog):
    selected = predictions[puzzle_indices, start : end + 1]
    selected_mask = mask[puzzle_indices]
    before = selected[:, :-1][selected_mask[:, None, :].expand(-1, selected.size(1) - 1, -1)]
    after = selected[:, 1:][selected_mask[:, None, :].expand(-1, selected.size(1) - 1, -1)]
    changed = before != after
    before = before[changed].detach().cpu().numpy()
    after = after[changed].detach().cpu().numpy()
    unordered_counts = np.zeros(len(catalog["pairs"]), dtype=np.float64)
    pair_to_index = {pair: index for index, pair in enumerate(catalog["pairs"])}
    for first, second in zip(before, after):
        pair = tuple(sorted((int(first), int(second))))
        unordered_counts[pair_to_index[pair]] += 1
    total = unordered_counts.sum()
    if total == 0:
        return None
    scores = np.einsum(
        "ij,j->i",
        catalog["adjacency"],
        unordered_counts,
        optimize=False,
    ) / total
    return scores, int(total)


def analyze_digit_geometry(
    model,
    states,
    statistics,
    targets,
    empty_mask,
    fit_indices,
    heldout_indices,
):
    catalog = _cycle_catalog()
    head_basis = _head_null_basis(model)
    records = []
    plot_data = {}
    endpoints = [end for _, end in PHASES.values()]
    for endpoint in endpoints:
        fit_values = states[fit_indices, endpoint]
        heldout_values = states[heldout_indices, endpoint]
        for representation, fit_source, heldout_source in (
            ("full hidden state", fit_values, heldout_values),
            (
                "output-head nullspace",
                _remove_subspace(fit_values, head_basis),
                _remove_subspace(heldout_values, head_basis),
            ),
        ):
            train_centroids = _digit_centroids(
                fit_source,
                targets[fit_indices],
                empty_mask[fit_indices],
            ).cpu().numpy()
            heldout_centroids = _digit_centroids(
                heldout_source,
                targets[heldout_indices],
                empty_mask[heldout_indices],
            ).cpu().numpy()
            result = _digit_geometry_record(train_centroids, heldout_centroids, catalog)
            plot_data[f"{endpoint}|{representation}"] = result
            records.append(
                {
                    "iteration": endpoint,
                    "representation": representation,
                    **{
                        key: value
                        for key, value in result.items()
                        if key not in (
                            "train_centroids",
                            "heldout_centroids",
                            "heldout_plane_coordinates",
                        )
                    },
                }
            )

        train_cycle = _prediction_transition_cycle(
            statistics["predictions"], empty_mask, fit_indices, max(0, endpoint - 64), endpoint, catalog
        )
        heldout_cycle = _prediction_transition_cycle(
            statistics["predictions"], empty_mask, heldout_indices, max(0, endpoint - 64), endpoint, catalog
        )
        if train_cycle and heldout_cycle:
            train_scores, train_count = train_cycle
            heldout_scores, heldout_count = heldout_cycle
            learned_index = int(np.argmax(train_scores))
            natural_index = catalog["natural_index"]
            natural_score = float(heldout_scores[natural_index])
            learned_score = float(heldout_scores[learned_index])
            records.append(
                {
                    "iteration": endpoint,
                    "representation": "predicted-digit changes (preceding 64 iters)",
                    "natural_cycle_adjacency_fraction": natural_score,
                    "natural_cycle_exact_p": float((heldout_scores >= natural_score - 1e-12).mean()),
                    "natural_cycle_rank": int((heldout_scores > natural_score + 1e-12).sum() + 1),
                    "learned_cycle_adjacency_fraction": learned_score,
                    "learned_cycle_exact_p": float((heldout_scores >= learned_score - 1e-12).mean()),
                    "learned_cycle_rank": int((heldout_scores > learned_score + 1e-12).sum() + 1),
                    "learned_cycle_digits": [digit + 1 for digit in catalog["cycles"][learned_index]],
                    "fit_prediction_changes": train_count,
                    "heldout_prediction_changes": heldout_count,
                }
            )

    head = _normalize_tokens(model.output_head.weight.float()).detach().cpu().numpy()
    head_result = _digit_geometry_record(head, head, catalog)
    pairwise = _rdm(head)
    head_public = {
        key: value
        for key, value in head_result.items()
        if key
        not in (
            "train_centroids",
            "heldout_centroids",
            "heldout_plane_coordinates",
            "learned_cycle_correlation",
            "learned_cycle_exact_p",
            "learned_cycle_rank",
            "learned_cycle_digits",
        )
    }
    records.append(
        {
            "iteration": "output head",
            "representation": "output-head rows",
            "pairwise_distance_cv": float(pairwise.std() / pairwise.mean()),
            "learned_cycle_note": "omitted because fitting and evaluation would use the same nine rows",
            **head_public,
        }
    )
    return records, plot_data


def _transition_code(correct):
    return correct[:, :-1].long() * 2 + correct[:, 1:].long()


def _turn_and_radial_geometry(states, updates):
    update_norm = updates.norm(dim=-1)
    radial_cosine = F.cosine_similarity(states[:, :-1], updates, dim=-1, eps=1e-12)
    turn_cosine = torch.full_like(update_norm, torch.nan)
    turn_cosine[:, 1:] = F.cosine_similarity(
        updates[:, :-1], updates[:, 1:], dim=-1, eps=1e-12
    )
    return update_norm, radial_cosine, turn_cosine


def analyze_correctness_transitions(
    states,
    updates,
    statistics,
    empty_mask,
    heldout_indices,
    bucket_names,
    update_basis,
    seed,
):
    records = []
    event_curves = []
    event_phase = []
    heldout_buckets = [bucket_names[index] for index in heldout_indices]
    heldout_mask = empty_mask[heldout_indices]
    transition = _transition_code(statistics["correct"])[heldout_indices]
    update_norm, radial_cosine, turn_cosine = _turn_and_radial_geometry(
        states[heldout_indices], updates[heldout_indices]
    )
    margin = statistics["margin"][heldout_indices]
    target_probability = statistics["target_probability"][heldout_indices]
    confidence = statistics["confidence"][heldout_indices]
    entropy = statistics["entropy"][heldout_indices]
    update_coordinates = (
        updates[heldout_indices] - update_basis["mean"]
    ) @ update_basis["basis"][:, :2]

    for phase_index, (phase_name, (start, end)) in enumerate(PHASES.items()):
        phase_transition = transition[:, start:end]
        phase_mask = heldout_mask[:, None, :].expand_as(phase_transition)
        phase_update_coordinates = update_coordinates[:, start:end]
        centered_coordinates = phase_update_coordinates - phase_update_coordinates.mean(dim=1, keepdim=True)
        update_angle = torch.atan2(centered_coordinates[..., 1], centered_coordinates[..., 0])
        for transition_code, transition_name in enumerate(TRANSITION_NAMES):
            event_mask = phase_mask & phase_transition.eq(transition_code)
            counts_by_puzzle = event_mask.sum(dim=(1, 2)).detach().cpu().numpy()
            metrics = {
                "update_norm": update_norm[:, start:end],
                "state_update_radial_cosine": radial_cosine[:, start:end],
                "consecutive_update_cosine": turn_cosine[:, start:end],
                "margin_before": margin[:, start:end],
                "margin_after": margin[:, start + 1 : end + 1],
                "margin_change": margin[:, start + 1 : end + 1] - margin[:, start:end],
                "target_probability_before": target_probability[:, start:end],
                "target_probability_change": (
                    target_probability[:, start + 1 : end + 1] - target_probability[:, start:end]
                ),
                "confidence_before": confidence[:, start:end],
                "entropy_before": entropy[:, start:end],
            }
            for metric_index, (metric_name, values) in enumerate(metrics.items()):
                puzzle_values = []
                for puzzle_index in range(values.size(0)):
                    selected = values[puzzle_index][event_mask[puzzle_index]]
                    selected = selected[torch.isfinite(selected)]
                    puzzle_values.append(float(selected.median()) if len(selected) else float("nan"))
                summary = _bootstrap_summary(
                    puzzle_values,
                    heldout_buckets,
                    seed + phase_index * 1000 + transition_code * 100 + metric_index,
                )
                records.append(
                    {
                        "phase": phase_name,
                        "transition": transition_name,
                        "metric": metric_name,
                        "event_count": int(counts_by_puzzle.sum()),
                        "puzzles_with_event": int((counts_by_puzzle > 0).sum()),
                        **summary,
                    }
                )

            if transition_code in (1, 2) and int(event_mask.sum()) > 0:
                observed_angles = update_angle[event_mask]
                observed_resultant = float(
                    torch.sqrt(
                        torch.cos(observed_angles).mean().square()
                        + torch.sin(observed_angles).mean().square()
                    )
                )
                generator = torch.Generator(device=states.device).manual_seed(
                    seed + phase_index * 10 + transition_code
                )
                null_values = []
                time_count = update_angle.size(1)
                base_time = torch.arange(time_count, device=states.device).view(1, -1, 1)
                for _ in range(128):
                    shifts = torch.randint(
                        time_count,
                        (update_angle.size(0), 1, update_angle.size(2)),
                        device=states.device,
                        generator=generator,
                    )
                    indices = (base_time + shifts) % time_count
                    shifted_angles = update_angle.gather(1, indices.expand_as(update_angle))
                    selected_angles = shifted_angles[event_mask]
                    null_values.append(
                        float(
                            torch.sqrt(
                                torch.cos(selected_angles).mean().square()
                                + torch.sin(selected_angles).mean().square()
                            )
                        )
                    )
                event_phase.append(
                    {
                        "phase": phase_name,
                        "transition": transition_name,
                        "event_count": int(event_mask.sum()),
                        "observed_resultant": observed_resultant,
                        "circular_shift_null_median": float(np.median(null_values)),
                        "circular_shift_null_q95": float(np.quantile(null_values, 0.95)),
                        "circular_shift_p": float(
                            (1 + np.sum(np.asarray(null_values) >= observed_resultant))
                            / (len(null_values) + 1)
                        ),
                    }
                )

    for transition_code, transition_name in ((1, "wrong→correct"), (2, "correct→wrong")):
        event_locations = torch.nonzero(
            transition.eq(transition_code) & heldout_mask[:, None, :],
            as_tuple=False,
        ).cpu()
        curve_margin = margin.detach().cpu()
        curve_target_probability = target_probability.detach().cpu()
        curve_confidence = confidence.detach().cpu()
        curve_update_norm = update_norm.detach().cpu()
        for offset in range(-8, 9):
            state_iterations = event_locations[:, 1] + 1 + offset
            valid = (state_iterations >= 0) & (state_iterations < margin.size(1))
            puzzle_indices = event_locations[valid, 0]
            cell_indices = event_locations[valid, 2]
            state_iterations = state_iterations[valid]
            values = {
                "margin": curve_margin[
                    puzzle_indices, state_iterations, cell_indices
                ].numpy(),
                "target_probability": curve_target_probability[
                    puzzle_indices, state_iterations, cell_indices
                ].numpy(),
                "confidence": curve_confidence[
                    puzzle_indices, state_iterations, cell_indices
                ].numpy(),
            }
            has_incoming_update = state_iterations > 0
            values["update_norm"] = curve_update_norm[
                puzzle_indices[has_incoming_update],
                state_iterations[has_incoming_update] - 1,
                cell_indices[has_incoming_update],
            ].numpy()
            for metric_name, metric_values in values.items():
                event_curves.append(
                    {
                        "transition": transition_name,
                        "offset": offset,
                        "metric": metric_name,
                        "event_count": int(len(metric_values)),
                        "median": float(np.median(metric_values))
                        if len(metric_values)
                        else float("nan"),
                        "q10": float(np.quantile(metric_values, 0.1))
                        if len(metric_values)
                        else float("nan"),
                        "q90": float(np.quantile(metric_values, 0.9))
                        if len(metric_values)
                        else float("nan"),
                    }
                )
    return records, event_curves, event_phase


def _binary_auc(labels, scores):
    labels = labels.detach().cpu().numpy().astype(np.int64)
    scores = scores.detach().cpu().numpy()
    positive_count = labels.sum()
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive_rank_sum = ranks[labels == 1].sum()
    return float(
        (positive_rank_sum - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def _sample_binary_rows(features, labels, maximum_negatives_per_positive, maximum_rows, seed):
    positive = torch.nonzero(labels, as_tuple=False).flatten()
    negative = torch.nonzero(~labels, as_tuple=False).flatten()
    generator = torch.Generator(device=labels.device).manual_seed(seed)
    if len(positive) == 0:
        return features[:0], labels[:0]
    maximum_negative = min(
        len(negative),
        maximum_negatives_per_positive * len(positive),
        max(0, maximum_rows - len(positive)),
    )
    if maximum_negative < len(negative):
        negative = negative[torch.randperm(len(negative), generator=generator, device=labels.device)[:maximum_negative]]
    indices = torch.cat([positive, negative])
    if len(indices) > maximum_rows:
        indices = indices[torch.randperm(len(indices), generator=generator, device=labels.device)[:maximum_rows]]
    return features[indices], labels[indices]


def _fit_logistic(train_features, train_labels, test_features, steps=300):
    mean = train_features.mean(dim=0)
    scale = train_features.std(dim=0).clamp_min(1e-5)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    weights = torch.zeros(train.size(1), device=train.device, requires_grad=True)
    bias = torch.zeros((), device=train.device, requires_grad=True)
    optimizer = torch.optim.Adam([weights, bias], lr=0.05)
    positive_count = train_labels.sum().clamp_min(1)
    negative_count = (~train_labels).sum().clamp_min(1)
    positive_weight = negative_count.float() / positive_count.float()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = train @ weights + bias
        loss = F.binary_cross_entropy_with_logits(
            logits,
            train_labels.float(),
            pos_weight=positive_weight,
        ) + 1e-3 * weights.square().mean()
        loss.backward()
        optimizer.step()
    return test @ weights.detach() + bias.detach()


def analyze_transition_probes(
    states,
    updates,
    statistics,
    targets,
    empty_mask,
    bucket_names,
    fit_indices,
    heldout_indices,
    update_basis,
    seed,
):
    margin = statistics["margin"]
    margin_velocity = torch.zeros_like(margin)
    margin_velocity[:, 1:] = margin[:, 1:] - margin[:, :-1]
    entropy = statistics["entropy"]
    correct = statistics["correct"]
    update_norm, radial_cosine, turn_cosine = _turn_and_radial_geometry(states, updates)
    update_coordinates = (updates - update_basis["mean"]) @ update_basis["basis"][:, :2]
    update_angle = torch.atan2(update_coordinates[..., 1], update_coordinates[..., 0])
    iteration = torch.arange(updates.size(1), device=updates.device).float()
    bucket_map = {name: index for index, name in enumerate(dict.fromkeys(bucket_names))}
    bucket_value = torch.tensor(
        [bucket_map[name] for name in bucket_names],
        device=updates.device,
        dtype=torch.float32,
    )
    records = []
    for outcome_name, current_correct, next_correct in (
        ("gain: wrong→correct", False, True),
        ("loss: correct→wrong", True, False),
    ):
        baseline_features = torch.stack(
            (
                margin[:, :-1],
                margin_velocity[:, :-1],
                entropy[:, :-1],
                torch.log1p(iteration)[None, :, None].expand_as(margin[:, :-1]),
                bucket_value[:, None, None].expand_as(margin[:, :-1]),
            ),
            dim=-1,
        )
        current_update_geometry = torch.stack(
            (
                torch.sin(update_angle),
                torch.cos(update_angle),
                torch.log(update_norm.clamp_min(1e-12)),
                torch.nan_to_num(turn_cosine, nan=1.0),
                radial_cosine,
            ),
            dim=-1,
        )
        geometry_features = torch.zeros_like(current_update_geometry)
        geometry_features[:, 1:] = current_update_geometry[:, :-1]
        full_features = torch.cat([baseline_features, geometry_features], dim=-1)
        eligible = (
            empty_mask[:, None, :]
            & correct[:, :-1].eq(current_correct)
        )
        eligible[:, 0] = False
        labels = correct[:, 1:].eq(next_correct)

        def extract(indices, feature_tensor):
            index_mask = torch.zeros(states.size(0), dtype=torch.bool, device=states.device)
            index_mask[indices] = True
            mask = eligible & index_mask[:, None, None]
            return feature_tensor[mask], labels[mask]

        train_baseline, train_labels = extract(fit_indices, baseline_features)
        train_full, _ = extract(fit_indices, full_features)
        test_baseline, test_labels = extract(heldout_indices, baseline_features)
        test_full, _ = extract(heldout_indices, full_features)
        # Repeat the exact row choice for the full feature tensor by sampling a
        # joint tensor once; the leading baseline columns remain identical.
        joint_train, joint_train_labels = _sample_binary_rows(
            train_full,
            train_labels,
            maximum_negatives_per_positive=30,
            maximum_rows=250_000,
            seed=seed + len(records),
        )
        joint_test, sampled_test_labels = _sample_binary_rows(
            test_full,
            test_labels,
            maximum_negatives_per_positive=50,
            maximum_rows=300_000,
            seed=seed + 100 + len(records),
        )
        if joint_train_labels.sum() == 0 or sampled_test_labels.sum() == 0:
            records.append(
                {
                    "outcome": outcome_name,
                    "fit_positive_events": int(train_labels.sum()),
                    "heldout_positive_events": int(test_labels.sum()),
                    "baseline_auc": float("nan"),
                    "baseline_plus_geometry_auc": float("nan"),
                    "geometry_delta_auc": float("nan"),
                }
            )
            continue
        # Use joint samples for both probes so the comparison is paired.
        train_baseline = joint_train[:, : baseline_features.size(-1)]
        test_baseline = joint_test[:, : baseline_features.size(-1)]
        baseline_scores = _fit_logistic(
            train_baseline,
            joint_train_labels,
            test_baseline,
        )
        full_scores = _fit_logistic(
            joint_train,
            joint_train_labels,
            joint_test,
        )
        baseline_auc = _binary_auc(sampled_test_labels, baseline_scores)
        full_auc = _binary_auc(sampled_test_labels, full_scores)
        records.append(
            {
                "outcome": outcome_name,
                "fit_eligible_rows": int(train_labels.numel()),
                "fit_positive_events": int(train_labels.sum()),
                "heldout_eligible_rows": int(test_labels.numel()),
                "heldout_positive_events": int(test_labels.sum()),
                "sampled_fit_rows": int(joint_train_labels.numel()),
                "sampled_heldout_rows": int(sampled_test_labels.numel()),
                "baseline_auc": baseline_auc,
                "baseline_plus_geometry_auc": full_auc,
                "geometry_delta_auc": full_auc - baseline_auc,
            }
        )
    return records


def _representation_rows(values, puzzle_indices, cell_mask, representation, phase_samples):
    chunks = []
    is_state = "state" in representation
    source = values[puzzle_indices].float()
    if representation.startswith("unit"):
        source = _normalize_tokens(source)
    for start, end in PHASES.values():
        times = _phase_times(
            start,
            end,
            phase_samples,
            updates=not is_state,
            device=values.device,
        )
        if is_state:
            selected = _chord_residual(source, start, end, times)
            selected = selected.permute(0, 2, 1, 3)[cell_mask[puzzle_indices]]
        else:
            selected = source[:, times].permute(0, 2, 1, 3)[cell_mask[puzzle_indices]]
        chunks.append(selected.reshape(-1, selected.size(-1)))
    return torch.cat(chunks)


def analyze_basis_controls(
    states,
    updates,
    bases,
    heldout_indices,
    empty_mask,
    bucket_names,
    seed,
    phase_samples,
):
    basis_records = []
    phase_records = []
    generator = torch.Generator(device=states.device).manual_seed(seed)
    for representation_index, representation in enumerate(REPRESENTATION_NAMES):
        source = states if "state" in representation else updates
        rows = _representation_rows(
            source,
            heldout_indices,
            empty_mask,
            representation,
            phase_samples,
        )
        centered = rows - bases[representation]["mean"]
        total = centered.square().sum().clamp_min(1e-30)
        coordinates = centered @ bases[representation]["basis"]
        random_explained = []
        for _ in range(RANDOM_PLANE_REPETITIONS):
            random_matrix = torch.randn(
                centered.size(1), 3, device=states.device, generator=generator
            )
            random_basis = torch.linalg.qr(random_matrix, mode="reduced").Q
            random_explained.append(float((centered @ random_basis).square().sum() / total))
        basis_records.append(
            {
                "representation": representation,
                "train_top2_explained": float(bases[representation]["train_explained"][:2].sum()),
                "train_top3_explained": float(bases[representation]["train_explained"][:3].sum()),
                "heldout_top2_explained": float(coordinates[:, :2].square().sum() / total),
                "heldout_top3_explained": float(coordinates.square().sum() / total),
                "random_top3_explained_median": float(np.median(random_explained)),
                "random_top3_explained_q95": float(np.quantile(random_explained, 0.95)),
            }
        )

    for state_name, update_name, normalization in (
        ("raw state residual", "raw update", "raw"),
        ("unit state residual", "unit update", "unit"),
    ):
        state_basis = bases[state_name]["basis"]
        update_basis = bases[update_name]["basis"]
        singular_values = torch.linalg.svdvals(state_basis.T @ update_basis)
        basis_records.append(
            {
                "representation": f"{normalization} state/update subspace",
                "mean_squared_principal_cosine": float(singular_values.square().mean()),
                "minimum_principal_cosine": float(singular_values.min()),
            }
        )

    heldout_buckets = [bucket_names[index] for index in heldout_indices]
    heldout_mask = empty_mask[heldout_indices]
    for phase_index, (phase_name, (start, end)) in enumerate(PHASES.items()):
        for representation_index, representation in enumerate(REPRESENTATION_NAMES):
            source = states if "state" in representation else updates
            normalized_source = source[heldout_indices].float()
            if representation.startswith("unit"):
                normalized_source = _normalize_tokens(normalized_source)
            times = _phase_times(
                start,
                end,
                phase_samples,
                updates="state" not in representation,
                device=states.device,
            )
            if "state" in representation:
                high_dimensional = _chord_residual(
                    normalized_source, start, end, times
                ).permute(0, 2, 1, 3)
            else:
                high_dimensional = normalized_source[:, times].permute(0, 2, 1, 3)
            repeated_metrics = {name: [] for name in ("turns_net", "phase_linearity_r2")}
            for _ in range(RANDOM_PLANE_REPETITIONS):
                random_matrix = torch.randn(
                    high_dimensional.size(-1),
                    3,
                    device=states.device,
                    generator=generator,
                )
                random_basis = torch.linalg.qr(random_matrix, mode="reduced").Q
                points = (high_dimensional - bases[representation]["mean"]) @ random_basis
                normalized_times = (times.float() - times[0]) / (
                    times[-1] - times[0]
                ).clamp_min(1)
                metrics = phase_geometry_metrics(
                    points,
                    state_residual="state" in representation,
                    time=normalized_times,
                )
                for metric_name in repeated_metrics:
                    values = metrics[metric_name].clone()
                    values[~metrics["valid"]] = torch.nan
                    repeated_metrics[metric_name].append(values)
            for metric_name, repetitions in repeated_metrics.items():
                values = torch.nanmedian(torch.stack(repetitions), dim=0).values
                summary = summarize_cell_metric(
                    values,
                    heldout_mask,
                    heldout_buckets,
                    seed + phase_index * 100 + representation_index * 10,
                )
                phase_records.append(
                    {
                        "phase": phase_name,
                        "representation": representation,
                        "cell_type": "blank",
                        "control": "random orthonormal plane",
                        "metric": metric_name,
                        **{key: value for key, value in summary.items() if key != "per_puzzle"},
                    }
                )
    return basis_records, phase_records


def _phase_midpoints():
    return np.arange(len(PHASES), dtype=np.float64)


def _model_color_map(model_names):
    palette = plt.get_cmap("tab10")
    return {name: palette(index) for index, name in enumerate(model_names)}


def _records_where(records, **conditions):
    return [
        row
        for row in records
        if all(row.get(key) == value for key, value in conditions.items())
    ]


def plot_phase_summary(records, output_path, model_names):
    colors = _model_color_map(model_names)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        ("raw state", "path_efficiency", "Raw-state path efficiency", ("ordered",)),
        ("unit state", "translation_r2", "Unit-state linear-translation R²", ("ordered",)),
        (
            "raw state residual",
            "phase_linearity_r2",
            "Detrended-state phase linearity",
            ("ordered", "block time shuffle", "full time shuffle", "random orthonormal plane"),
        ),
        (
            "raw update",
            "phase_linearity_r2",
            "Update-direction phase linearity",
            ("ordered", "block time shuffle", "full time shuffle", "random orthonormal plane"),
        ),
    )
    line_styles = {
        "ordered": "-",
        "block time shuffle": "--",
        "full time shuffle": ":",
        "random orthonormal plane": "-.",
    }
    for axis, (representation, metric, title, controls) in zip(axes.flat, panels):
        for model_name in model_names:
            for control in controls:
                rows = _records_where(
                    records,
                    model=model_name,
                    representation=representation,
                    metric=metric,
                    control=control,
                    cell_type="blank",
                )
                rows.sort(key=lambda row: list(PHASES).index(row["phase"]))
                if len(rows) != len(PHASES):
                    continue
                label = model_name if len(controls) == 1 else f"{model_name}, {control}"
                axis.plot(
                    _phase_midpoints(),
                    [row["estimate"] for row in rows],
                    marker="o" if control == "ordered" else None,
                    linewidth=1.8 if control == "ordered" else 1.0,
                    linestyle=line_styles[control],
                    color=colors[model_name],
                    alpha=1.0 if control == "ordered" else 0.55,
                    label=label,
                )
        axis.set_title(title)
        axis.set_xticks(_phase_midpoints(), list(PHASES), rotation=25, ha="right")
        axis.grid(alpha=0.25)
        axis.set_ylabel(metric.replace("_", " "))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, fontsize=8)
    axes[1, 0].legend(fontsize=6, ncols=2)
    figure.suptitle("Held-out per-cell translation and rotation controls")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_predictive_summary(records, output_path, model_names):
    colors = _model_color_map(model_names)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        (
            "raw state in residual-fitted basis",
            "rotation_skill_vs_translation",
            "State: rotation vs translation",
        ),
        (
            "raw state in residual-fitted basis",
            "rotation_skill_vs_cubic",
            "State: rotation vs same-DF cubic",
        ),
        ("raw update", "rotation_skill_vs_translation", "Update: rotation vs translation"),
        ("raw update", "rotation_skill_vs_cubic", "Update: rotation vs same-DF cubic"),
    )
    controls = ("ordered", "block time shuffle", "full time shuffle")
    styles = {"ordered": "-", "block time shuffle": "--", "full time shuffle": ":"}
    for axis, (representation, metric, title) in zip(axes.flat, panels):
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        for model_name in model_names:
            for control in controls:
                rows = _records_where(
                    records,
                    model=model_name,
                    representation=representation,
                    metric=metric,
                    control=control,
                )
                rows.sort(key=lambda row: list(PHASES).index(row["phase"]))
                if len(rows) != len(PHASES):
                    continue
                axis.plot(
                    _phase_midpoints(),
                    [row["estimate"] for row in rows],
                    color=colors[model_name],
                    linestyle=styles[control],
                    marker="o" if control == "ordered" else None,
                    linewidth=1.8 if control == "ordered" else 1.0,
                    alpha=1.0 if control == "ordered" else 0.55,
                    label=f"{model_name}, {control}",
                )
        axis.set_title(title)
        axis.set_xticks(_phase_midpoints(), list(PHASES), rotation=25, ha="right")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_ylabel("held-out predictive skill (symlog)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, ncols=2)
    figure.suptitle("Periodic model must beat translation and an equal-parameter smooth curve")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_spiral_summary(records, output_path, model_names):
    colors = _model_color_map(model_names)
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    panels = (
        (
            "unit state in residual-fitted basis",
            "spiral_skill_vs_translation",
            "Unit state: spiral vs translation",
        ),
        (
            "unit state in residual-fitted basis",
            "spiral_skill_vs_quintic",
            "Unit state: spiral vs same-DF quintic",
        ),
        ("unit update", "spiral_skill_vs_translation", "Unit update: spiral vs translation"),
        ("unit update", "spiral_skill_vs_quintic", "Unit update: spiral vs same-DF quintic"),
    )
    controls = ("ordered", "block time shuffle", "full time shuffle")
    styles = {"ordered": "-", "block time shuffle": "--", "full time shuffle": ":"}
    for axis, (representation, metric, title) in zip(axes.flat, panels):
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        for model_name in model_names:
            for control in controls:
                rows = _records_where(
                    records,
                    model=model_name,
                    representation=representation,
                    metric=metric,
                    control=control,
                )
                rows.sort(key=lambda row: list(PHASES).index(row["phase"]))
                if len(rows) != len(PHASES):
                    continue
                axis.plot(
                    _phase_midpoints(),
                    [row["estimate"] for row in rows],
                    color=colors[model_name],
                    linestyle=styles[control],
                    marker="o" if control == "ordered" else None,
                    linewidth=1.8 if control == "ordered" else 1.0,
                    alpha=1.0 if control == "ordered" else 0.55,
                    label=f"{model_name}, {control}",
                )
        axis.set_title(title)
        axis.set_xticks(_phase_midpoints(), list(PHASES), rotation=25, ha="right")
        axis.set_yscale("symlog", linthresh=0.1)
        axis.set_ylabel("held-out predictive skill (symlog)")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=6, ncols=2)
    figure.suptitle("Spiral model must beat translation and an equal-parameter smooth curve")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_event_curves(records, output_path, model_names):
    colors = _model_color_map(model_names)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    for row_index, transition in enumerate(("wrong→correct", "correct→wrong")):
        for column_index, metric in enumerate(("margin", "update_norm")):
            axis = axes[row_index, column_index]
            for model_name in model_names:
                rows = _records_where(
                    records,
                    model=model_name,
                    transition=transition,
                    metric=metric,
                )
                rows.sort(key=lambda row: row["offset"])
                if not rows:
                    continue
                x = np.asarray([row["offset"] for row in rows])
                median = np.asarray([row["median"] for row in rows])
                low = np.asarray([row["q10"] for row in rows])
                high = np.asarray([row["q90"] for row in rows])
                axis.plot(x, median, color=colors[model_name], label=model_name)
                axis.fill_between(x, low, high, color=colors[model_name], alpha=0.08)
            axis.axvline(0, color="black", linewidth=0.8, alpha=0.6)
            if metric == "margin":
                axis.axhline(0, color="black", linewidth=0.8, alpha=0.4)
            axis.set_title(f"{transition}: {metric.replace('_', ' ')}")
            axis.set_xlabel("iterations from transition (0 is new state)")
            axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out cell events: output boundary and motion")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_transition_summary(transition_records, probe_records, output_path, model_names):
    colors = _model_color_map(model_names)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for model_name in model_names:
        for transition, style in (("wrong→correct", "-"), ("correct→wrong", "--")):
            count_rows = _records_where(
                transition_records,
                model=model_name,
                transition=transition,
                metric="margin_change",
            )
            count_rows.sort(key=lambda row: list(PHASES).index(row["phase"]))
            axes[0].plot(
                _phase_midpoints(),
                [row["event_count"] for row in count_rows],
                color=colors[model_name],
                linestyle=style,
                marker="o",
                label=f"{model_name}, {transition}",
            )
            axes[1].plot(
                _phase_midpoints(),
                [row["estimate"] for row in count_rows],
                color=colors[model_name],
                linestyle=style,
                marker="o",
            )
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_title("Transition event count")
    axes[0].legend(fontsize=6)
    axes[1].set_title("Median margin change at transition")
    for axis in axes[:2]:
        axis.set_xticks(_phase_midpoints(), list(PHASES), rotation=25, ha="right")
        axis.grid(alpha=0.25)

    width = 0.18
    positions = np.arange(2)
    for model_index, model_name in enumerate(model_names):
        rows = _records_where(probe_records, model=model_name)
        rows.sort(key=lambda row: row["outcome"])
        values = [row.get("geometry_delta_auc", np.nan) for row in rows]
        if len(values) == 2:
            axes[2].bar(
                positions + (model_index - 1.5) * width,
                values,
                width,
                color=colors[model_name],
                label=model_name,
            )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_xticks(positions, ("gain", "loss"))
    axes[2].set_ylabel("Δ held-out AUROC")
    axes[2].set_title("Geometry beyond margin/history")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=7)
    figure.suptitle("Correctness transitions: output-boundary crossings and update timing")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_digit_summary(records, output_path, model_names):
    colors = _model_color_map(model_names)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis, representation in zip(axes, ("full hidden state", "output-head nullspace")):
        for model_name in model_names:
            rows = [
                row
                for row in records
                if row.get("model") == model_name
                and row.get("representation") == representation
                and isinstance(row.get("iteration"), int)
            ]
            rows.sort(key=lambda row: row["iteration"])
            x = [row["iteration"] for row in rows]
            axis.plot(
                x,
                [row["natural_cycle_correlation"] for row in rows],
                color=colors[model_name],
                marker="o",
                label=f"{model_name}, natural 1→9",
            )
            axis.plot(
                x,
                [row["learned_cycle_correlation"] for row in rows],
                color=colors[model_name],
                linestyle="--",
                marker="x",
                label=f"{model_name}, train-learned",
            )
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_xscale("log", base=2)
        axis.set_xticks([16, 64, 128, 512, 1024], [16, 64, 128, 512, 1024])
        axis.set_xlabel("iteration")
        axis.set_ylabel("held-out RDM correlation with cycle distance")
        axis.set_title(representation)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=6, ncols=2)
    figure.suptitle("Exact 20,160-cycle test of digit order")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_digit_centroids(plot_data_by_model, output_path, model_names):
    endpoints = (16, 1024)
    figure, axes = plt.subplots(
        len(model_names), len(endpoints), figsize=(10, 3.5 * len(model_names)), constrained_layout=True
    )
    if len(model_names) == 1:
        axes = np.asarray([axes])
    for row, model_name in enumerate(model_names):
        for column, endpoint in enumerate(endpoints):
            axis = axes[row, column]
            result = plot_data_by_model[model_name][f"{endpoint}|full hidden state"]
            points = result["heldout_plane_coordinates"]
            closed = np.vstack([points, points[:1]])
            axis.plot(closed[:, 0], closed[:, 1], color="#999999", linewidth=1.0)
            axis.scatter(points[:, 0], points[:, 1], c=np.arange(9), cmap="tab10", s=36)
            for digit, (x, y) in enumerate(points, start=1):
                axis.annotate(str(digit), (x, y), xytext=(4, 3), textcoords="offset points", fontsize=9)
            axis.axhline(0, color="#cccccc", linewidth=0.7)
            axis.axvline(0, color="#cccccc", linewidth=0.7)
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_title(
                f"{model_name}, t={endpoint}\n"
                f"natural p={result['natural_cycle_exact_p']:.3f}, plane EV={result['heldout_plane_explained']:.2f}"
            )
            axis.set_xlabel("train digit-centroid PC1")
            axis.set_ylabel("train digit-centroid PC2")
    figure.suptitle("Held-out digit centroids; gray line forces the natural numeric order")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _representative_cell(points, mask, *, state_residual, time=None):
    metrics = phase_geometry_metrics(
        points,
        state_residual=state_residual,
        time=time,
    )
    values = metrics["phase_linearity_r2"].clone()
    values[~(mask & metrics["valid"])] = torch.nan
    finite = values[torch.isfinite(values)]
    target = finite.median()
    distance = (values - target).abs()
    distance[~torch.isfinite(distance)] = torch.inf
    flat_index = int(distance.argmin())
    puzzle_index = flat_index // values.size(1)
    cell_index = flat_index % values.size(1)
    return puzzle_index, cell_index, float(values[puzzle_index, cell_index])


def plot_representative_paths(details_by_model, heldout_masks, output_path, model_names):
    columns = (
        ("solving 16-64", "raw state residual", "state, solving"),
        ("solving 16-64", "raw update", "update, solving"),
        ("deep 512-1024", "raw state residual", "state, deep"),
        ("deep 512-1024", "raw update", "update, deep"),
    )
    figure, axes = plt.subplots(
        len(model_names), len(columns), figsize=(14, 3.2 * len(model_names)), constrained_layout=True
    )
    if len(model_names) == 1:
        axes = np.asarray([axes])
    for row, model_name in enumerate(model_names):
        for column, (phase, representation, title) in enumerate(columns):
            axis = axes[row, column]
            detail = details_by_model[model_name][f"{phase}|{representation}"]
            points = detail["points"].float()
            times = torch.tensor(detail["times"], dtype=torch.float32)
            if "update" in representation:
                points = points - points.mean(dim=2, keepdim=True)
            puzzle_index, cell_index, phase_r2 = _representative_cell(
                points,
                heldout_masks[model_name],
                state_residual="state" in representation,
                time=times,
            )
            path = points[puzzle_index, cell_index].numpy()
            axis.plot(path[:, 0], path[:, 1], color="#aaaaaa", linewidth=0.8)
            axis.scatter(
                path[:, 0], path[:, 1], c=np.linspace(0, 1, len(path)), cmap="viridis", s=12
            )
            axis.scatter(path[0, 0], path[0, 1], marker="s", color="#d62728", s=30)
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(
                f"{model_name}: {title}\nmedian cell p{puzzle_index}/c{cell_index}, phase R²={phase_r2:.2f}",
                fontsize=9,
            )
    figure.suptitle("Representative held-out cells selected by the median phase-linearity score")
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _model_accuracy(statistics, empty_mask, puzzle_indices):
    result = []
    selected_mask = empty_mask[puzzle_indices]
    for endpoint in [end for _, end in PHASES.values()]:
        correct = statistics["correct"][puzzle_indices, endpoint]
        cell_accuracy = float(correct[selected_mask].float().mean())
        solved = ((correct | ~selected_mask).all(dim=1)).float().mean()
        margin = statistics["margin"][puzzle_indices, endpoint][selected_mask]
        result.append(
            {
                "iteration": endpoint,
                "blank_cell_accuracy": cell_accuracy,
                "puzzle_accuracy": float(solved),
                "blank_margin_median": float(margin.median()),
                "blank_margin_p10": float(torch.quantile(margin, 0.1)),
            }
        )
    return result


def run(
    output_dir,
    *,
    examples_per_bucket=DEFAULT_EXAMPLES_PER_BUCKET,
    final_iteration=DEFAULT_FINAL_ITERATION,
    phase_samples=DEFAULT_PHASE_SAMPLES,
    seed=DEFAULT_SEED,
    device="cuda",
    model_configs=DEFAULT_MODELS,
):
    if final_iteration < max(end for _, end in PHASES.values()):
        raise ValueError("final_iteration must cover all configured phases")
    if examples_per_bucket < 2:
        raise ValueError("examples_per_bucket must be at least two")
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = _load_balanced_sample(
        examples_per_bucket,
        seed,
    )
    fit_indices, heldout_indices = stratified_puzzle_split(bucket_names)
    inputs = inputs.to(resolved_device)
    targets = targets.to(resolved_device)
    empty_mask = empty_mask.to(resolved_device)
    model_names = [config["name"] for config in model_configs]
    started_at = time.time()
    progress_path = os.path.join(output_dir, "progress.log")
    progress_handle = open(progress_path, "w")

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        progress_handle.write(line + "\n")
        progress_handle.flush()

    summary = {
        "config": {
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(inputs),
            "final_iteration": final_iteration,
            "phase_samples": phase_samples,
            "seed": seed,
            "device": str(resolved_device),
            "phases": PHASES,
            "model_cycle_candidates": MODEL_CYCLE_CANDIDATES,
            "time_control_repetitions": TIME_CONTROL_REPETITIONS,
            "random_plane_repetitions": RANDOM_PLANE_REPETITIONS,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "models": list(model_configs),
        },
        "sample": {
            "puzzles": puzzles,
            "solutions": solutions,
            "buckets": bucket_names,
            "fit_indices": fit_indices,
            "heldout_indices": heldout_indices,
            "fit_buckets": [bucket_names[index] for index in fit_indices],
            "heldout_buckets": [bucket_names[index] for index in heldout_indices],
        },
        "definitions": {
            "state_iteration_zero": "initial encoder output before any recurrent step",
            "update_t": "h[t+1] - h[t]",
            "confidence": "maximum softmax probability",
            "target_probability": "softmax probability of the answer digit",
            "margin": "answer logit minus the largest competing logit",
            "blank_cells_primary": True,
            "basis_fit": "alternating complete puzzles within each rating bucket",
            "state_rotation_centering": "subtract the endpoint chord separately within each phase",
            "update_rotation_centering": "subtract each cell's temporal mean after projection",
            "predictive_state_input": "raw or token-unit state projected into a basis fitted on train-puzzle chord residuals; the intercept and slope are fitted only on the calibration prefix",
            "transition_probe_geometry": "u[t-1] and its geometry, available at h[t], for predicting correctness at t+1; t=0 is excluded",
        },
        "models": {},
    }
    phase_records = []
    predictive_records = []
    digit_records = []
    transition_records = []
    event_curve_records = []
    event_phase_records = []
    probe_records = []
    basis_records = []
    phase_details_by_model = {}
    digit_plot_data_by_model = {}
    heldout_masks = {}

    log(
        f"Starting {len(model_configs)} models on {len(inputs)} puzzles; "
        f"fit={len(fit_indices)}, heldout={len(heldout_indices)}"
    )
    for model_index, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        model_started = time.time()
        log(f"MODEL {model_name}: loading {model_config['path']}")
        model = _load_model(model_config, resolved_device)
        states, updates, logits = collect_trajectory(model, inputs, final_iteration)
        statistics = prediction_statistics(logits, targets)
        log(f"MODEL {model_name}: fitting puzzle-held-out feature bases")
        bases = fit_representation_bases(
            states,
            updates,
            fit_indices,
            empty_mask,
            phase_samples,
        )
        model_phase_records, phase_details = analyze_phase_geometry(
            states,
            updates,
            bases,
            heldout_indices,
            empty_mask,
            bucket_names,
            seed + model_index * 10_000,
            phase_samples,
        )
        model_basis_records, random_phase_records = analyze_basis_controls(
            states,
            updates,
            bases,
            heldout_indices,
            empty_mask,
            bucket_names,
            seed + model_index * 10_000 + 1000,
            phase_samples,
        )
        model_phase_records.extend(random_phase_records)
        model_predictive_records, predictive_details = analyze_predictive_geometry(
            states,
            updates,
            bases,
            fit_indices,
            heldout_indices,
            empty_mask,
            bucket_names,
            seed + model_index * 10_000 + 2000,
            phase_samples,
        )
        log(f"MODEL {model_name}: exact digit-cycle controls")
        model_digit_records, digit_plot_data = analyze_digit_geometry(
            model,
            states,
            statistics,
            targets,
            empty_mask,
            fit_indices,
            heldout_indices,
        )
        log(f"MODEL {model_name}: correctness transitions and held-out probes")
        model_transition_records, model_event_curves, model_event_phase = (
            analyze_correctness_transitions(
                states,
                updates,
                statistics,
                empty_mask,
                heldout_indices,
                bucket_names,
                bases["raw update"],
                seed + model_index * 10_000 + 3000,
            )
        )
        model_probe_records = analyze_transition_probes(
            states,
            updates,
            statistics,
            targets,
            empty_mask,
            bucket_names,
            fit_indices,
            heldout_indices,
            bases["raw update"],
            seed + model_index * 10_000 + 4000,
        )

        for record_group in (
            model_phase_records,
            model_predictive_records,
            model_digit_records,
            model_transition_records,
            model_event_curves,
            model_event_phase,
            model_probe_records,
            model_basis_records,
        ):
            for record in record_group:
                record["model"] = model_name
        phase_records.extend(model_phase_records)
        predictive_records.extend(model_predictive_records)
        digit_records.extend(model_digit_records)
        transition_records.extend(model_transition_records)
        event_curve_records.extend(model_event_curves)
        event_phase_records.extend(model_event_phase)
        probe_records.extend(model_probe_records)
        basis_records.extend(model_basis_records)
        phase_details_by_model[model_name] = phase_details
        digit_plot_data_by_model[model_name] = digit_plot_data
        heldout_masks[model_name] = empty_mask[heldout_indices].detach().cpu()
        summary["models"][model_name] = {
            "model_config": model_config,
            "heldout_accuracy": _model_accuracy(statistics, empty_mask, heldout_indices),
            "bases": model_basis_records,
            "predictive_frequency_selection": predictive_details,
            "event_phase_alignment": model_event_phase,
            "transition_probes": model_probe_records,
            "elapsed_seconds": time.time() - model_started,
        }
        _atomic_json(os.path.join(output_dir, "metrics.partial.json"), summary)
        log(f"MODEL {model_name}: complete in {time.time() - model_started:.1f}s")
        del model, states, updates, logits, statistics, bases
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    log("Writing numerical artifacts and figures")
    _write_csv(os.path.join(output_dir, "phase_geometry.csv"), phase_records)
    _write_csv(os.path.join(output_dir, "predictive_models.csv"), predictive_records)
    _write_csv(os.path.join(output_dir, "digit_cycles.csv"), digit_records)
    _write_csv(os.path.join(output_dir, "correctness_transitions.csv"), transition_records)
    _write_csv(os.path.join(output_dir, "event_aligned_curves.csv"), event_curve_records)
    _write_csv(os.path.join(output_dir, "event_phase_alignment.csv"), event_phase_records)
    _write_csv(os.path.join(output_dir, "transition_probes.csv"), probe_records)
    _write_csv(os.path.join(output_dir, "basis_controls.csv"), basis_records)
    plot_phase_summary(
        phase_records,
        os.path.join(output_dir, "phase_geometry.png"),
        model_names,
    )
    plot_predictive_summary(
        predictive_records,
        os.path.join(output_dir, "predictive_helix_models.png"),
        model_names,
    )
    plot_spiral_summary(
        predictive_records,
        os.path.join(output_dir, "predictive_spiral_models.png"),
        model_names,
    )
    plot_event_curves(
        event_curve_records,
        os.path.join(output_dir, "correctness_event_dynamics.png"),
        model_names,
    )
    plot_transition_summary(
        transition_records,
        probe_records,
        os.path.join(output_dir, "correctness_transition_summary.png"),
        model_names,
    )
    plot_digit_summary(
        digit_records,
        os.path.join(output_dir, "digit_cycle_controls.png"),
        model_names,
    )
    plot_digit_centroids(
        digit_plot_data_by_model,
        os.path.join(output_dir, "digit_centroid_projections.png"),
        model_names,
    )
    plot_representative_paths(
        phase_details_by_model,
        heldout_masks,
        os.path.join(output_dir, "representative_cell_paths.png"),
        model_names,
    )
    summary["elapsed_seconds"] = time.time() - started_at
    summary["artifacts"] = [
        "metrics.json",
        "phase_geometry.csv",
        "predictive_models.csv",
        "digit_cycles.csv",
        "correctness_transitions.csv",
        "event_aligned_curves.csv",
        "event_phase_alignment.csv",
        "transition_probes.csv",
        "basis_controls.csv",
        "phase_geometry.png",
        "predictive_helix_models.png",
        "predictive_spiral_models.png",
        "correctness_event_dynamics.png",
        "correctness_transition_summary.png",
        "digit_cycle_controls.png",
        "digit_centroid_projections.png",
        "representative_cell_paths.png",
        "progress.log",
    ]
    _atomic_json(os.path.join(output_dir, "metrics.json"), summary)
    log(f"Complete in {summary['elapsed_seconds']:.1f}s")
    progress_handle.close()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=os.path.dirname(__file__))
    parser.add_argument("--examples-per-bucket", type=int, default=DEFAULT_EXAMPLES_PER_BUCKET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(
        args.output_dir,
        examples_per_bucket=args.examples_per_bucket,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
