"""Held-out causal interventions on semantic recurrent-state axes.

The script deliberately runs discovery/validation before materializing the
final split.  It fits one supervised board-progress axis per checkpoint and
uses the output-head answer-evidence directions as a predefined positive
control.  One-shot pulses are norm matched to each puzzle's natural recurrent
update and compared with signed doses and random directions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/sotaku-causal-axes-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_model


ARM_DIR = Path(__file__).resolve().parent
RATING_BUCKETS = (
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
)
SPLIT_NAMES = ("discovery", "validation", "final")
SNAPSHOTS = (8, 16, 32, 64, 128, 256, 512)
RIDGE_LAMBDAS = (0.01, 0.1, 1.0, 10.0, 100.0)
DOSES = (-0.5, -0.25, -0.125, 0.125, 0.25, 0.5)
INTERVENTION_WINDOWS = {
    "early": (16, 128),
    "late": (512, 1024),
}
SEED = 20260811
EXAMPLES_PER_SPLIT_BUCKET = 4
N_LABEL_SHUFFLES = 64
N_RANDOM_DIRECTIONS = 8
BOOTSTRAP_DRAWS = 2000

DEFAULT_MODELS = (
    {
        "name": "stable_plain",
        "path": "model_baseline_lr2e3.pt",
        "model_kwargs": {},
    },
    {
        "name": "collapsed_plain",
        "path": (
            "/private/tmp/sotaku_causal_axes_checkpoints/"
            "model_baseline_lr2e3_clean_a.pt"
        ),
        "model_kwargs": {},
    },
    {
        "name": "late_state_ce",
        "path": (
            "/private/tmp/sotaku_causal_axes_checkpoints/"
            "model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt"
        ),
        "model_kwargs": {},
    },
    {
        "name": "combined_margin",
        "path": (
            "/private/tmp/sotaku_causal_axes_checkpoints/"
            "model_loop_stay_late_switch_margin_floor5_from39k.pt"
        ),
        "model_kwargs": {},
    },
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json_dump(payload, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with open(temporary_path, "w") as output_file:
        json.dump(_jsonable(payload), output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    os.replace(temporary_path, path)


def assign_balanced_splits(bucket_names, examples_per_split_bucket):
    expected = examples_per_split_bucket * len(SPLIT_NAMES)
    seen = {}
    assignments = []
    for bucket_name in bucket_names:
        offset = seen.get(bucket_name, 0)
        if offset >= expected:
            raise ValueError(f"too many examples in bucket {bucket_name!r}")
        assignments.append(SPLIT_NAMES[offset // examples_per_split_bucket])
        seen[bucket_name] = offset + 1
    if set(seen) != {bucket[2] for bucket in RATING_BUCKETS}:
        raise ValueError("all five rating buckets are required")
    if any(count != expected for count in seen.values()):
        raise ValueError("each rating bucket must have equal split counts")
    return assignments


def _cached_test_dataset():
    cache_root = Path.home() / ".cache/huggingface/datasets"
    candidates = sorted(cache_root.glob(
        "sapientinc___sudoku-extreme/default/*/*/sudoku-extreme-test.arrow"
    ))
    if not candidates:
        raise FileNotFoundError("no cached sudoku-extreme test Arrow file")
    return Dataset.from_file(str(candidates[-1]))


def load_balanced_sample(examples_per_split_bucket, seed):
    try:
        dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    except (PermissionError, OSError):
        dataset = _cached_test_dataset()

    per_bucket = examples_per_split_bucket * len(SPLIT_NAMES)
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, example in enumerate(dataset):
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= example["rating"] <= maximum:
                bucket_indices[name].append(index)
                break

    generator = random.Random(seed)
    selected_indices = []
    bucket_names = []
    for _, _, name in RATING_BUCKETS:
        choices = generator.sample(bucket_indices[name], per_bucket)
        selected_indices.extend(choices)
        bucket_names.extend([name] * per_bucket)

    puzzles = [dataset[index]["question"] for index in selected_indices]
    solutions = [dataset[index]["answer"] for index in selected_indices]
    inputs = model_module.encode_puzzles(puzzles)
    targets = model_module.encode_solutions(solutions).long()
    empty_mask = inputs[:, :, 0].bool()
    split_names = assign_balanced_splits(
        bucket_names,
        examples_per_split_bucket,
    )
    return {
        "inputs": inputs,
        "targets": targets,
        "empty_mask": empty_mask,
        "bucket_names": bucket_names,
        "split_names": split_names,
        "dataset_indices": selected_indices,
        "puzzle_hashes": [
            hashlib.sha256(puzzle.encode("ascii")).hexdigest()[:16]
            for puzzle in puzzles
        ],
    }


def split_indices(split_names, split_name):
    return torch.tensor([
        index for index, name in enumerate(split_names) if name == split_name
    ], dtype=torch.long)


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()[:16]


def collect_snapshots(model, inputs, snapshots=SNAPSHOTS):
    requested = set(snapshots)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    hidden_state = model.initial_encoder(inputs)
    predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
    states = []
    logits = []
    with torch.no_grad():
        for iteration in range(max(snapshots) + 1):
            if iteration in requested:
                states.append(hidden_state.detach().float().cpu())
                logits.append(
                    model.output_head(hidden_state).detach().float().cpu()
                )
            if iteration == max(snapshots):
                break
            hidden_state = model.recurrent_step(
                hidden_state,
                predictions,
                rope_cos,
                rope_sin,
            )
            predictions = F.softmax(model.output_head(hidden_state), dim=-1)
    return torch.stack(states, dim=1), torch.stack(logits, dim=1)


def board_progress_dataset(states, logits, targets, empty_mask):
    mask = empty_mask[:, None, :, None].float()
    counts = empty_mask.sum(dim=1).clamp_min(1).float()
    pooled_states = (states * mask).sum(dim=2) / counts[:, None, None]
    predictions = logits.argmax(dim=-1)
    correctness = (
        ((predictions == targets[:, None, :]) & empty_mask[:, None, :])
        .sum(dim=2).float()
        / counts[:, None]
    )
    puzzle_count, snapshot_count, features = pooled_states.shape
    groups = torch.arange(snapshot_count).repeat(puzzle_count)
    return (
        pooled_states.reshape(-1, features),
        correctness.reshape(-1),
        groups,
    )


def center_within_groups(values, groups):
    centered = values.clone().float()
    for group in torch.unique(groups):
        mask = groups == group
        centered[mask] -= centered[mask].mean(dim=0, keepdim=True)
    return centered


def fit_ridge_axis(features, labels, groups, ridge_lambda):
    feature_residual = center_within_groups(features, groups)
    label_residual = center_within_groups(labels[:, None], groups).squeeze(1)
    scales = feature_residual.square().mean(dim=0).sqrt().clamp_min(1e-5)
    standardized = feature_residual / scales
    covariance = standardized.T @ standardized / len(standardized)
    cross_covariance = standardized.T @ label_residual / len(standardized)
    coefficient = torch.linalg.solve(
        covariance + ridge_lambda * torch.eye(covariance.size(0)),
        cross_covariance,
    ) / scales
    return F.normalize(coefficient, dim=0, eps=1e-12)


def fit_shuffled_axes(
    features,
    labels,
    groups,
    ridge_lambda,
    count,
    seed,
):
    feature_residual = center_within_groups(features, groups)
    label_residual = center_within_groups(labels[:, None], groups).squeeze(1)
    scales = feature_residual.square().mean(dim=0).sqrt().clamp_min(1e-5)
    standardized = feature_residual / scales
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.empty(len(labels), count)
    for group in torch.unique(groups):
        mask = torch.where(groups == group)[0]
        for shuffle_index in range(count):
            order = mask[torch.randperm(len(mask), generator=generator)]
            shuffled[mask, shuffle_index] = label_residual[order]
    covariance = standardized.T @ standardized / len(standardized)
    cross_covariance = standardized.T @ shuffled / len(standardized)
    coefficients = torch.linalg.solve(
        covariance + ridge_lambda * torch.eye(covariance.size(0)),
        cross_covariance,
    ) / scales[:, None]
    return F.normalize(coefficients.T, dim=1, eps=1e-12)


def partial_correlation(scores, labels, groups):
    score_residual = center_within_groups(scores[:, None], groups).squeeze(1)
    label_residual = center_within_groups(labels[:, None], groups).squeeze(1)
    denominator = score_residual.norm() * label_residual.norm()
    if denominator <= 1e-12:
        return 0.0
    return float((score_residual @ label_residual / denominator).item())


def answer_evidence_axes(output_weight):
    directions = []
    for digit in range(9):
        other_rows = torch.cat((output_weight[:digit], output_weight[digit + 1:]))
        directions.append(output_weight[digit] - other_rows.mean(dim=0))
    return F.normalize(torch.stack(directions).float(), dim=1, eps=1e-12)


def answer_evidence_probe(states, logits, targets, empty_mask, axes):
    puzzle_count, snapshot_count, _, features = states.shape
    target_axes = axes[targets]
    scores = (states * target_axes[:, None, :, :]).sum(dim=-1)
    correct_logits = logits.gather(
        -1,
        targets[:, None, :, None].expand(-1, snapshot_count, -1, 1),
    ).squeeze(-1)
    wrong_mask = F.one_hot(targets, num_classes=9).bool()[:, None, :, :]
    wrong_max = logits.masked_fill(wrong_mask, -torch.inf).max(dim=-1).values
    margins = correct_logits - wrong_max
    selected = empty_mask[:, None, :].expand(-1, snapshot_count, -1)
    snapshot_groups = (
        torch.arange(snapshot_count)[None, :, None]
        .expand(puzzle_count, -1, 81)
    )
    digit_groups = targets[:, None, :].expand(-1, snapshot_count, -1)
    groups = snapshot_groups * 9 + digit_groups
    return partial_correlation(
        scores[selected],
        margins[selected],
        groups[selected],
    )


def random_global_axes(real_axis, count, seed):
    generator = torch.Generator().manual_seed(seed)
    random_axes = torch.randn(count, real_axis.numel(), generator=generator)
    random_axes -= (random_axes @ real_axis)[:, None] * real_axis[None, :]
    return F.normalize(random_axes, dim=1, eps=1e-12)


def random_evidence_axes(real_axes, count, seed):
    generator = torch.Generator().manual_seed(seed)
    random_axes = torch.randn(
        count,
        real_axes.size(0),
        real_axes.size(1),
        generator=generator,
    )
    alignments = (random_axes * real_axes[None, :, :]).sum(dim=-1)
    random_axes -= alignments[:, :, None] * real_axes[None, :, :]
    return F.normalize(random_axes, dim=2, eps=1e-12)


def make_perturbation(
    natural_update,
    empty_mask,
    direction,
    dose,
    targets=None,
):
    if direction.ndim == 1:
        cell_directions = direction[None, None, :].expand(
            natural_update.size(0), natural_update.size(1), -1
        )
    elif direction.ndim == 2 and direction.size(0) == 9:
        if targets is None:
            raise ValueError("digit-conditioned directions require targets")
        cell_directions = direction[targets]
    else:
        raise ValueError("direction must have shape [features] or [9, features]")
    mask = empty_mask[:, :, None].float()
    cell_directions = cell_directions * mask
    direction_norm = cell_directions.flatten(1).norm(dim=1).clamp_min(1e-12)
    update_norm = (natural_update * mask).flatten(1).norm(dim=1)
    scales = dose * update_norm / direction_norm
    return cell_directions * scales[:, None, None]


def semantic_metrics(logits, targets, empty_mask):
    probabilities = F.softmax(logits.float(), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    confidence = 1.0 - entropy / math.log(9.0)
    correct_logits = logits.gather(-1, targets[:, :, None]).squeeze(-1)
    wrong_mask = F.one_hot(targets, num_classes=9).bool()
    wrong_logits = logits.masked_fill(wrong_mask, -torch.inf)
    margin = correct_logits - wrong_logits.max(dim=-1).values
    correct = logits.argmax(dim=-1) == targets
    counts = empty_mask.sum(dim=1).clamp_min(1)

    def masked_mean(values):
        return (values * empty_mask).sum(dim=1) / counts

    minimum_margin = margin.masked_fill(~empty_mask, torch.inf).min(dim=1).values
    cell_accuracy = ((correct & empty_mask).sum(dim=1) / counts).float()
    solved = (correct | ~empty_mask).all(dim=1).float()
    return {
        "confidence": masked_mean(confidence),
        "mean_margin": masked_mean(margin),
        "minimum_margin": minimum_margin,
        "cell_accuracy": cell_accuracy,
        "solved": solved,
    }


def collect_baseline_contexts(model, inputs, targets, empty_mask):
    needed = {16, 17, 128, 512, 513, 1024}
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    hidden_state = model.initial_encoder(inputs)
    predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
    states = {}
    metrics = {}
    with torch.no_grad():
        for iteration in range(1025):
            if iteration in needed:
                states[iteration] = hidden_state.detach().clone()
                metrics[iteration] = {
                    name: values.detach().float().cpu()
                    for name, values in semantic_metrics(
                        model.output_head(hidden_state), targets, empty_mask
                    ).items()
                }
            if iteration == 1024:
                break
            hidden_state = model.recurrent_step(
                hidden_state,
                predictions,
                rope_cos,
                rope_sin,
            )
            predictions = F.softmax(model.output_head(hidden_state), dim=-1)
    return states, metrics


def condition_specs(axis_bundle):
    specs = []
    for axis_name in ("answer_evidence", "solvedness_progress"):
        for dose in DOSES:
            specs.append({
                "axis": axis_name,
                "dose": dose,
                "control": "real",
                "control_index": -1,
                "direction": axis_bundle[axis_name],
            })
            for control_index, direction in enumerate(
                axis_bundle[f"{axis_name}_random"]
            ):
                specs.append({
                    "axis": axis_name,
                    "dose": dose,
                    "control": "random",
                    "control_index": control_index,
                    "direction": direction,
                })
    return specs


def evaluate_interventions(
    model,
    targets,
    empty_mask,
    baseline_states,
    axis_bundle,
    condition_batch_size=4,
):
    rope_cos = model_module.ROPE_COS.to(targets.device)
    rope_sin = model_module.ROPE_SIN.to(targets.device)
    specs = condition_specs(axis_bundle)
    raw = {}
    with torch.no_grad():
        for window_name, (pulse, endpoint) in INTERVENTION_WINDOWS.items():
            base_state = baseline_states[pulse]
            natural_update = baseline_states[pulse + 1] - base_state
            for start in range(0, len(specs), condition_batch_size):
                chunk = specs[start:start + condition_batch_size]
                perturbed = []
                for spec in chunk:
                    perturbation = make_perturbation(
                        natural_update,
                        empty_mask,
                        spec["direction"].to(base_state.device),
                        spec["dose"],
                        targets,
                    )
                    perturbed.append(base_state + perturbation)
                hidden_state = torch.stack(perturbed).flatten(0, 1)
                chunk_count = len(chunk)
                repeated_targets = targets.repeat(chunk_count, 1)
                repeated_mask = empty_mask.repeat(chunk_count, 1)

                stage_values = {}
                logits = model.output_head(hidden_state)
                stage_values["immediate"] = semantic_metrics(
                    logits, repeated_targets, repeated_mask
                )
                predictions = F.softmax(logits, dim=-1)
                for iteration in range(pulse, endpoint):
                    hidden_state = model.recurrent_step(
                        hidden_state,
                        predictions,
                        rope_cos,
                        rope_sin,
                    )
                    logits = model.output_head(hidden_state)
                    predictions = F.softmax(logits, dim=-1)
                    if iteration + 1 == pulse + 1:
                        stage_values["next_step"] = semantic_metrics(
                            logits, repeated_targets, repeated_mask
                        )
                stage_values["endpoint"] = semantic_metrics(
                    logits, repeated_targets, repeated_mask
                )

                for chunk_index, spec in enumerate(chunk):
                    key = (
                        window_name,
                        spec["axis"],
                        spec["dose"],
                        spec["control"],
                        spec["control_index"],
                    )
                    raw[key] = {
                        stage: {
                            metric: values.view(chunk_count, -1)[chunk_index]
                            .detach().float().cpu().numpy()
                            for metric, values in metrics.items()
                        }
                        for stage, metrics in stage_values.items()
                    }
    return raw


def bootstrap_mean_ci(values, seed, draws=BOOTSTRAP_DRAWS):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return [float("nan"), float("nan")]
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def summarize_interventions(raw, baseline_metrics, seed):
    summary = {}
    metrics = (
        "confidence", "mean_margin", "minimum_margin", "cell_accuracy", "solved"
    )
    for window_index, (window_name, (pulse, endpoint)) in enumerate(
        INTERVENTION_WINDOWS.items()
    ):
        summary[window_name] = {}
        baseline_by_stage = {
            "immediate": baseline_metrics[pulse],
            "next_step": baseline_metrics[pulse + 1],
            "endpoint": baseline_metrics[endpoint],
        }
        for axis_index, axis_name in enumerate(
            ("answer_evidence", "solvedness_progress")
        ):
            axis_summary = {}
            for dose_index, dose in enumerate(DOSES):
                real_key = (window_name, axis_name, dose, "real", -1)
                real = raw[real_key]
                record = {
                    "perturbation_to_natural_update_norm": abs(dose),
                    "real": {},
                    "random_controls": {},
                }
                for stage in ("immediate", "next_step", "endpoint"):
                    record["real"][stage] = {}
                    record["random_controls"][stage] = {}
                    for metric_index, metric in enumerate(metrics):
                        baseline = baseline_by_stage[stage][metric].numpy()
                        effect = real[stage][metric] - baseline
                        bootstrap_seed = (
                            seed + 100000 * window_index + 10000 * axis_index
                            + 1000 * dose_index + 10 * metric_index
                            + (0 if stage == "immediate" else 1 if stage == "next_step" else 2)
                        )
                        record["real"][stage][metric] = {
                            "mean_effect": float(effect.mean()),
                            "bootstrap_95_ci": bootstrap_mean_ci(
                                effect, bootstrap_seed
                            ),
                            "puzzle_effects": effect.tolist(),
                        }
                        random_means = []
                        for control_index in range(N_RANDOM_DIRECTIONS):
                            control_key = (
                                window_name, axis_name, dose, "random", control_index
                            )
                            control_effect = (
                                raw[control_key][stage][metric] - baseline
                            )
                            random_means.append(float(control_effect.mean()))
                        random_means_array = np.asarray(random_means)
                        record["random_controls"][stage][metric] = {
                            "direction_mean_effects": random_means,
                            "p10": float(np.quantile(random_means_array, 0.10)),
                            "median": float(np.median(random_means_array)),
                            "p90": float(np.quantile(random_means_array, 0.90)),
                            "real_percentile": float(
                                np.mean(random_means_array <= effect.mean())
                            ),
                        }

                immediate_solved = baseline_metrics[pulse]["solved"].numpy() > 0.5
                endpoint_solved = real["endpoint"]["solved"] > 0.5
                baseline_endpoint = (
                    baseline_metrics[endpoint]["solved"].numpy() > 0.5
                )
                if immediate_solved.any():
                    collapse_rate = float((~endpoint_solved[immediate_solved]).mean())
                    baseline_collapse = float(
                        (~baseline_endpoint[immediate_solved]).mean()
                    )
                else:
                    collapse_rate = None
                    baseline_collapse = None
                initially_unsolved = ~immediate_solved
                if initially_unsolved.any():
                    recovery_rate = float(endpoint_solved[initially_unsolved].mean())
                    baseline_recovery = float(
                        baseline_endpoint[initially_unsolved].mean()
                    )
                else:
                    recovery_rate = None
                    baseline_recovery = None
                record["outcomes"] = {
                    "collapse_rate": collapse_rate,
                    "baseline_collapse_rate": baseline_collapse,
                    "recovery_rate": recovery_rate,
                    "baseline_recovery_rate": baseline_recovery,
                }
                axis_summary[str(dose)] = record
            summary[window_name][axis_name] = axis_summary
    return summary


def summarize_null(real_value, null_values):
    null_values = np.asarray(null_values, dtype=np.float64)
    return {
        "real": float(real_value),
        "null_values": null_values.tolist(),
        "null_p95": float(np.quantile(null_values, 0.95)),
        "one_sided_empirical_p": float(
            (1 + np.sum(null_values >= real_value)) / (1 + len(null_values))
        ),
    }


def plot_results(results, output_dir):
    model_names = list(results["models"])
    colors = dict(zip(model_names, plt.cm.tab10(np.linspace(0, 1, len(model_names)))))
    dose_values = np.asarray(DOSES)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for row, axis_name in enumerate(("answer_evidence", "solvedness_progress")):
        for column, window_name in enumerate(("early", "late")):
            axis = axes[row, column]
            for model_name in model_names:
                records = results["models"][model_name]["interventions"][window_name][axis_name]
                real = np.asarray([
                    records[str(dose)]["real"]["endpoint"]["solved"]["mean_effect"] * 100
                    for dose in DOSES
                ])
                low = np.asarray([
                    records[str(dose)]["random_controls"]["endpoint"]["solved"]["p10"] * 100
                    for dose in DOSES
                ])
                high = np.asarray([
                    records[str(dose)]["random_controls"]["endpoint"]["solved"]["p90"] * 100
                    for dose in DOSES
                ])
                axis.plot(dose_values, real, marker="o", color=colors[model_name], label=model_name)
                axis.fill_between(dose_values, low, high, color=colors[model_name], alpha=0.10)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(0, color="black", linewidth=0.6, alpha=0.5)
            axis.set_title(f"{axis_name.replace('_', ' ')} — {window_name}")
            axis.set_ylabel("endpoint solved change (percentage points)")
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("signed pulse norm / natural update norm")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Held-out causal-axis dose response; shading is random-direction p10–p90")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "dose_response.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    semantic_metric = {
        "answer_evidence": "mean_margin",
        "solvedness_progress": "cell_accuracy",
    }
    for row, axis_name in enumerate(("answer_evidence", "solvedness_progress")):
        metric = semantic_metric[axis_name]
        scale = 100 if metric == "cell_accuracy" else 1
        for column, window_name in enumerate(("early", "late")):
            axis = axes[row, column]
            for model_name in model_names:
                records = results["models"][model_name]["interventions"][window_name][axis_name]
                real = np.asarray([
                    records[str(dose)]["real"]["immediate"][metric]["mean_effect"] * scale
                    for dose in DOSES
                ])
                low = np.asarray([
                    records[str(dose)]["random_controls"]["immediate"][metric]["p10"] * scale
                    for dose in DOSES
                ])
                high = np.asarray([
                    records[str(dose)]["random_controls"]["immediate"][metric]["p90"] * scale
                    for dose in DOSES
                ])
                axis.plot(dose_values, real, marker="o", color=colors[model_name], label=model_name)
                axis.fill_between(dose_values, low, high, color=colors[model_name], alpha=0.10)
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axvline(0, color="black", linewidth=0.6, alpha=0.5)
            unit = "percentage points" if scale == 100 else "logit units"
            axis.set_ylabel(f"immediate {metric.replace('_', ' ')} change ({unit})")
            axis.set_title(f"{axis_name.replace('_', ' ')} — {window_name}")
            axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("signed pulse norm / natural update norm")
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.suptitle("Immediate semantic response; shading is random-direction p10–p90")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "immediate_semantics.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(model_names))
    for axis, probe_name, title in (
        (axes[0], "answer_evidence", "Answer-evidence probe"),
        (axes[1], "solvedness_progress", "Solvedness-progress probe"),
    ):
        real = [results["models"][name]["final_probes"][probe_name]["real"] for name in model_names]
        null = [results["models"][name]["final_probes"][probe_name]["null_p95"] for name in model_names]
        axis.bar(x - 0.18, real, width=0.36, label="real axis")
        axis.bar(x + 0.18, null, width=0.36, label="95th percentile null")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, model_names, rotation=25, ha="right")
        axis.set_ylabel("held-out partial correlation")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Final-holdout semantic decoding versus matched null axes")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "heldout_probes.png", dpi=180)
    plt.close(fig)


def run_analysis(
    output_dir=ARM_DIR,
    model_configs=DEFAULT_MODELS,
    condition_batch_size=4,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    sample = load_balanced_sample(EXAMPLES_PER_SPLIT_BUCKET, SEED)
    discovery_indices = split_indices(sample["split_names"], "discovery")
    validation_indices = split_indices(sample["split_names"], "validation")
    final_indices = split_indices(sample["split_names"], "final")
    assert len(discovery_indices) == len(validation_indices) == len(final_indices) == 20

    discovery_validation = {
        "format_version": 1,
        "protocol": {
            "seed": SEED,
            "snapshots": SNAPSHOTS,
            "ridge_lambdas": RIDGE_LAMBDAS,
            "examples_per_split": 20,
            "examples_per_rating_bucket_per_split": EXAMPLES_PER_SPLIT_BUCKET,
            "final_holdout_accessed": False,
        },
        "sample": {
            "dataset_indices": sample["dataset_indices"],
            "puzzle_hashes": sample["puzzle_hashes"],
            "bucket_names": sample["bucket_names"],
            "split_names": sample["split_names"],
        },
        "models": {},
    }
    candidate_axes = {}
    discovery_datasets = {}
    fingerprints = {}
    validation_scores = {ridge_lambda: [] for ridge_lambda in RIDGE_LAMBDAS}

    discovery_validation_indices = torch.cat((discovery_indices, validation_indices))
    development_inputs = sample["inputs"][discovery_validation_indices]
    development_targets = sample["targets"][discovery_validation_indices]
    development_mask = sample["empty_mask"][discovery_validation_indices]
    for model_config in model_configs:
        model_name = model_config["name"]
        print(f"Development pass: {model_name}", flush=True)
        model = _load_model(model_config, torch.device("cpu"))
        fingerprints[model_name] = model_fingerprint(model)
        states, logits = collect_snapshots(model, development_inputs)
        discovery_states, validation_states = states[:20], states[20:]
        discovery_logits, validation_logits = logits[:20], logits[20:]
        discovery_data = board_progress_dataset(
            discovery_states,
            discovery_logits,
            development_targets[:20],
            development_mask[:20],
        )
        validation_data = board_progress_dataset(
            validation_states,
            validation_logits,
            development_targets[20:],
            development_mask[20:],
        )
        discovery_datasets[model_name] = discovery_data
        candidate_axes[model_name] = {}
        per_lambda = {}
        for ridge_lambda in RIDGE_LAMBDAS:
            direction = fit_ridge_axis(*discovery_data, ridge_lambda)
            candidate_axes[model_name][ridge_lambda] = direction
            correlation = partial_correlation(
                validation_data[0] @ direction,
                validation_data[1],
                validation_data[2],
            )
            validation_scores[ridge_lambda].append(correlation)
            per_lambda[str(ridge_lambda)] = correlation
        evidence_axes = answer_evidence_axes(model.output_head.weight.detach().cpu())
        evidence_validation = answer_evidence_probe(
            validation_states,
            validation_logits,
            development_targets[20:],
            development_mask[20:],
            evidence_axes,
        )
        discovery_validation["models"][model_name] = {
            "model_config": model_config,
            "fingerprint": fingerprints[model_name],
            "solvedness_progress_validation_by_lambda": per_lambda,
            "answer_evidence_validation_correlation": evidence_validation,
        }
        del model, states, logits

    selected_lambda = max(
        RIDGE_LAMBDAS,
        key=lambda value: (float(np.mean(validation_scores[value])), -value),
    )
    discovery_validation["selection"] = {
        "criterion": "maximum mean validation partial correlation across four checkpoints",
        "selected_ridge_lambda": selected_lambda,
        "mean_validation_correlation_by_lambda": {
            str(value): float(np.mean(validation_scores[value]))
            for value in RIDGE_LAMBDAS
        },
    }

    frozen_axes = {}
    for model_index, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        model = _load_model(model_config, torch.device("cpu"))
        progress_axis = candidate_axes[model_name][selected_lambda]
        shuffled_axes = fit_shuffled_axes(
            *discovery_datasets[model_name],
            selected_lambda,
            N_LABEL_SHUFFLES,
            SEED + 1000 + model_index,
        )
        evidence_axes = answer_evidence_axes(model.output_head.weight.detach().cpu())
        frozen_axes[model_name] = {
            "solvedness_progress": progress_axis,
            "solvedness_progress_shuffled": shuffled_axes,
            "solvedness_progress_random": random_global_axes(
                progress_axis,
                N_RANDOM_DIRECTIONS,
                SEED + 2000 + model_index,
            ),
            "answer_evidence": evidence_axes,
            "answer_evidence_random": random_evidence_axes(
                evidence_axes,
                N_RANDOM_DIRECTIONS,
                SEED + 3000 + model_index,
            ),
        }
        del model

    discovery_validation["protocol"]["final_holdout_accessed"] = False
    atomic_json_dump(discovery_validation, output_dir / "discovery_validation.json")
    torch.save(
        {
            "selected_ridge_lambda": selected_lambda,
            "fingerprints": fingerprints,
            "axes": frozen_axes,
            "sample_hashes": sample["puzzle_hashes"],
        },
        output_dir / "frozen_axes.pt",
    )
    print(
        f"Frozen lambda={selected_lambda}; beginning one final-holdout pass",
        flush=True,
    )

    results = {
        "format_version": 1,
        "config": {
            "seed": SEED,
            "models": list(model_configs),
            "snapshots": SNAPSHOTS,
            "ridge_lambdas": RIDGE_LAMBDAS,
            "selected_ridge_lambda": selected_lambda,
            "doses": DOSES,
            "intervention_windows": INTERVENTION_WINDOWS,
            "random_directions": N_RANDOM_DIRECTIONS,
            "label_shuffles": N_LABEL_SHUFFLES,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "condition_batch_size": condition_batch_size,
            "device": "cpu",
        },
        "split_audit": {
            "discovery_count": len(discovery_indices),
            "validation_count": len(validation_indices),
            "final_count": len(final_indices),
            "final_puzzle_hashes": [sample["puzzle_hashes"][i] for i in final_indices],
            "selection_frozen_before_final": True,
        },
        "models": {},
    }
    final_inputs = sample["inputs"][final_indices]
    final_targets = sample["targets"][final_indices]
    final_mask = sample["empty_mask"][final_indices]

    for model_index, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        model_started = time.time()
        print(f"Final interventions: {model_name}", flush=True)
        model = _load_model(model_config, torch.device("cpu"))
        if model_fingerprint(model) != fingerprints[model_name]:
            raise RuntimeError(f"checkpoint changed after axis freezing: {model_name}")
        final_states, final_logits = collect_snapshots(model, final_inputs)
        final_progress_data = board_progress_dataset(
            final_states, final_logits, final_targets, final_mask
        )
        model_axes = frozen_axes[model_name]
        progress_real = partial_correlation(
            final_progress_data[0] @ model_axes["solvedness_progress"],
            final_progress_data[1],
            final_progress_data[2],
        )
        progress_null = [
            partial_correlation(
                final_progress_data[0] @ direction,
                final_progress_data[1],
                final_progress_data[2],
            )
            for direction in model_axes["solvedness_progress_shuffled"]
        ]
        evidence_real = answer_evidence_probe(
            final_states,
            final_logits,
            final_targets,
            final_mask,
            model_axes["answer_evidence"],
        )
        evidence_null_axes = random_evidence_axes(
            model_axes["answer_evidence"],
            N_LABEL_SHUFFLES,
            SEED + 4000 + model_index,
        )
        evidence_null = [
            answer_evidence_probe(
                final_states,
                final_logits,
                final_targets,
                final_mask,
                direction,
            )
            for direction in evidence_null_axes
        ]
        baseline_states, baseline_metrics = collect_baseline_contexts(
            model, final_inputs, final_targets, final_mask
        )
        raw = evaluate_interventions(
            model,
            final_targets,
            final_mask,
            baseline_states,
            model_axes,
            condition_batch_size=condition_batch_size,
        )
        intervention_summary = summarize_interventions(
            raw,
            baseline_metrics,
            SEED + 5000 + model_index,
        )
        results["models"][model_name] = {
            "fingerprint": fingerprints[model_name],
            "final_probes": {
                "answer_evidence": summarize_null(evidence_real, evidence_null),
                "solvedness_progress": summarize_null(progress_real, progress_null),
            },
            "baseline": {
                str(iteration): {
                    metric: {
                        "mean": float(values.mean()),
                        "puzzle_values": values.tolist(),
                    }
                    for metric, values in metrics.items()
                }
                for iteration, metrics in baseline_metrics.items()
            },
            "interventions": intervention_summary,
            "elapsed_seconds": time.time() - model_started,
        }
        atomic_json_dump(results, output_dir / "final_metrics.partial.json")
        del model, final_states, final_logits, baseline_states, raw

    results["elapsed_seconds"] = time.time() - started_at
    atomic_json_dump(results, output_dir / "final_metrics.json")
    plot_results(results, output_dir)
    partial_path = output_dir / "final_metrics.partial.json"
    if partial_path.exists():
        partial_path.unlink()
    print(f"Complete in {results['elapsed_seconds']:.1f}s", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ARM_DIR)
    parser.add_argument("--condition-batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.condition_batch_size <= 0:
        raise ValueError("condition batch size must be positive")
    run_analysis(
        output_dir=args.output_dir,
        condition_batch_size=args.condition_batch_size,
    )


if __name__ == "__main__":
    main()
