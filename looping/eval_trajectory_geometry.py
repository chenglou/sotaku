"""Measure the dimension and curvature of recurrent hidden-state trajectories."""

import json
import os
import re
import time

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model


DEFAULT_MODELS = (
    {
        "name": "stable_plain",
        "path": "/outputs/model_baseline_lr2e3.pt",
        "model_kwargs": {},
    },
    {
        "name": "collapsed_plain",
        "path": "/outputs/model_baseline_lr2e3_clean_a.pt",
        "model_kwargs": {},
    },
    {
        "name": "late_state_ce",
        "path": (
            "/outputs/looping/"
            "model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt"
        ),
        "model_kwargs": {},
    },
    {
        "name": "combined_margin",
        "path": (
            "/outputs/looping/"
            "model_loop_stay_late_switch_margin_floor5_from39k.pt"
        ),
        "model_kwargs": {},
    },
)

DEFAULT_WINDOW_STARTS = (16, 128, 512, 1024)
DEFAULT_WINDOW_LENGTH = 16
DEFAULT_COMPONENT_COUNTS = (1, 2, 4, 8, 16, 32, 64)


def validate_windows(window_starts, window_length):
    if not window_starts:
        raise ValueError("window_starts must not be empty")
    if tuple(sorted(set(window_starts))) != tuple(window_starts):
        raise ValueError("window_starts must be strictly increasing")
    if window_starts[0] < 0:
        raise ValueError("window starts must be non-negative")
    if window_length < 2:
        raise ValueError("window_length must be at least 2")


def participation_ratio(matrix, *, center):
    matrix = matrix.float()
    if center:
        matrix = matrix - matrix.mean(dim=0, keepdim=True)
    gram = matrix @ matrix.T
    trace = matrix.square().sum()
    denominator = gram.square().sum()
    if denominator <= 1e-24:
        return 0.0
    return (trace.square() / denominator).item()


def feature_spectrum(matrix, component_counts):
    matrix = matrix.float()
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    covariance = matrix.T @ matrix
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    total = eigenvalues.sum().clamp_min(1e-24)
    cumulative = eigenvalues.cumsum(0) / total
    explained = {
        str(count): cumulative[min(count, len(cumulative)) - 1].item()
        for count in component_counts
    }
    effective_rank = (
        eigenvalues.sum().square()
        / eigenvalues.square().sum().clamp_min(1e-24)
    ).item()
    return {
        "effective_rank": effective_rank,
        "explained_variance": explained,
    }


def fit_principal_basis(matrix, maximum_components, *, center):
    matrix = matrix.float()
    mean = (
        matrix.mean(dim=0, keepdim=True)
        if center
        else torch.zeros(1, matrix.size(1), device=matrix.device)
    )
    centered = matrix - mean
    rank = min(maximum_components, centered.size(0), centered.size(1))
    if rank <= 0:
        raise ValueError("matrix must have at least one row and column")
    _, _, basis = torch.pca_lowrank(
        centered,
        q=rank,
        center=False,
        niter=4,
    )
    return mean, basis


def explained_by_basis(matrix, mean, basis, component_counts):
    centered = matrix.float() - mean
    total_energy = centered.square().sum().clamp_min(1e-24)
    coordinates = centered @ basis
    cumulative = coordinates.square().sum(dim=0).cumsum(0) / total_energy
    return {
        str(count): cumulative[min(count, basis.size(1)) - 1].item()
        for count in component_counts
    }


def trajectory_shape_metrics(updates):
    if updates.ndim != 4 or updates.size(1) < 2:
        raise ValueError("updates must have shape [puzzles, steps, cells, features]")
    board_updates = updates.flatten(2).float()
    current = board_updates[:, :-1]
    following = board_updates[:, 1:]
    cosine = F.cosine_similarity(current, following, dim=-1, eps=1e-12)
    acceleration = (following - current).norm(dim=-1) / current.norm(
        dim=-1
    ).clamp_min(1e-12)
    return {
        "consecutive_update_cosine_mean": cosine.mean().item(),
        "consecutive_update_cosine_p10": torch.quantile(
            cosine.flatten(), 0.1
        ).item(),
        "relative_acceleration_mean": acceleration.mean().item(),
        "relative_acceleration_p90": torch.quantile(
            acceleration.flatten(), 0.9
        ).item(),
    }


def _solved_count(model, hidden_state, targets, empty_mask):
    predictions = model.output_head(hidden_state).argmax(dim=-1)
    solved = ((predictions == targets) | ~empty_mask).all(dim=1)
    return int(solved.sum().item())


def collect_update_windows(
    model,
    inputs,
    targets,
    empty_mask,
    window_starts,
    window_length,
):
    starts = set(window_starts)
    final_iteration = window_starts[-1] + window_length
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    windows = {start: [] for start in window_starts}
    solved = {}

    with torch.no_grad():
        hidden_state = model.initial_encoder(inputs)
        predictions = torch.zeros(
            inputs.size(0), 81, 9, device=inputs.device
        )
        for iteration in range(final_iteration):
            if iteration in starts:
                solved[iteration] = {
                    "start": _solved_count(
                        model,
                        hidden_state,
                        targets,
                        empty_mask,
                    )
                }

            next_state = model.recurrent_step(
                hidden_state,
                predictions,
                rope_cos,
                rope_sin,
            )
            update = next_state - hidden_state
            for start in window_starts:
                if start <= iteration < start + window_length:
                    windows[start].append(update.float())

            hidden_state = next_state
            predictions = F.softmax(model.output_head(hidden_state), dim=-1)
            finished_window = iteration + 1 - window_length
            if finished_window in starts:
                solved[finished_window]["end"] = _solved_count(
                    model,
                    hidden_state,
                    targets,
                    empty_mask,
                )

    return {
        start: torch.stack(window_updates, dim=1)
        for start, window_updates in windows.items()
    }, solved


def summarize_update_windows(updates_by_start, solved, component_counts):
    first_start = min(updates_by_start)
    puzzle_count = updates_by_start[first_start].size(0)
    split = puzzle_count // 2
    if split == 0 or split == puzzle_count:
        raise ValueError("at least two puzzles are required")

    maximum_components = max(component_counts)
    first_board_updates = updates_by_start[first_start].flatten(2)
    early_mean, early_basis = fit_principal_basis(
        first_board_updates[:split].flatten(0, 1),
        maximum_components,
        center=True,
    )
    first_token_updates = updates_by_start[first_start]
    early_token_mean, early_token_basis = fit_principal_basis(
        first_token_updates[:split].flatten(0, 2),
        maximum_components,
        center=True,
    )

    results = {}
    for start, updates in updates_by_start.items():
        board_updates = updates.flatten(2)
        train_board = board_updates[:split].flatten(0, 1)
        heldout_board = board_updates[split:].flatten(0, 1)
        window_mean, window_basis = fit_principal_basis(
            train_board,
            maximum_components,
            center=True,
        )
        train_token = updates[:split].flatten(0, 2)
        heldout_token = updates[split:].flatten(0, 2)
        token_updates = updates.flatten(0, 2)
        token_mean, token_basis = fit_principal_basis(
            train_token,
            maximum_components,
            center=True,
        )
        unit_board_updates = F.normalize(
            board_updates.flatten(0, 1),
            dim=1,
            eps=1e-12,
        )
        results[str(start)] = {
            "solved_start": solved[start]["start"],
            "solved_end": solved[start]["end"],
            "total": puzzle_count,
            "board_update_effective_rank": participation_ratio(
                train_board,
                center=True,
            ),
            "board_direction_effective_rank": participation_ratio(
                unit_board_updates,
                center=False,
            ),
            "token_feature_spectrum": feature_spectrum(
                token_updates,
                component_counts,
            ),
            "heldout_token_same_window_explained": explained_by_basis(
                heldout_token,
                token_mean,
                token_basis,
                component_counts,
            ),
            "heldout_token_explained_by_first_window_basis": (
                explained_by_basis(
                    heldout_token,
                    early_token_mean,
                    early_token_basis,
                    component_counts,
                )
            ),
            "heldout_same_window_explained": explained_by_basis(
                heldout_board,
                window_mean,
                window_basis,
                component_counts,
            ),
            "heldout_explained_by_first_window_basis": explained_by_basis(
                heldout_board,
                early_mean,
                early_basis,
                component_counts,
            ),
            **trajectory_shape_metrics(updates),
        }
    return results


def evaluate(
    model_configs=DEFAULT_MODELS,
    window_starts=DEFAULT_WINDOW_STARTS,
    window_length=DEFAULT_WINDOW_LENGTH,
    component_counts=DEFAULT_COMPONENT_COUNTS,
    examples_per_bucket=10,
    seed=42,
    device="cuda",
    output_dir=None,
    output_prefix="trajectory_geometry_v1",
):
    validate_windows(window_starts, window_length)
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if tuple(sorted(set(component_counts))) != tuple(component_counts):
        raise ValueError("component_counts must be strictly increasing")
    if component_counts[0] <= 0:
        raise ValueError("component counts must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")

    resolved_device = torch.device(
        device if torch.cuda.is_available() else "cpu"
    )
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    inputs = inputs.to(resolved_device)
    targets = targets.to(resolved_device)
    empty_mask = empty_mask.to(resolved_device)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = open(os.path.join(output_dir, f"{output_prefix}.log"), "w")
    else:
        log_file = None

    def log(message=""):
        print(message, flush=True)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    summary = {
        "config": {
            "models": list(model_configs),
            "window_starts": list(window_starts),
            "window_length": window_length,
            "component_counts": list(component_counts),
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(inputs),
            "seed": seed,
            "device": str(resolved_device),
        },
        "sample": {
            "puzzles": puzzles,
            "solutions": solutions,
            "buckets": bucket_names,
        },
        "models": {},
    }
    started_at = time.time()
    log(
        f"Trajectory geometry: {len(inputs)} puzzles, "
        f"models={len(model_configs)}, windows={list(window_starts)}"
    )

    for model_config in model_configs:
        model_name = model_config["name"]
        model_started_at = time.time()
        log(f"\nMODEL {model_name}: {model_config['path']}")
        model = _load_model(model_config, resolved_device)
        updates, solved = collect_update_windows(
            model,
            inputs,
            targets,
            empty_mask,
            window_starts,
            window_length,
        )
        model_summary = summarize_update_windows(
            updates,
            solved,
            component_counts,
        )
        summary["models"][model_name] = {
            "model_config": model_config,
            "windows": model_summary,
            "elapsed_seconds": time.time() - model_started_at,
        }
        for start in window_starts:
            result = model_summary[str(start)]
            log(
                f"  {start:4d}-{start + window_length}: "
                f"solved {result['solved_start']}/{result['total']} -> "
                f"{result['solved_end']}/{result['total']}, "
                f"board rank {result['board_update_effective_rank']:.1f}, "
                f"turn cosine "
                f"{result['consecutive_update_cosine_mean']:.3f}"
            )
        del model, updates
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    summary["elapsed_seconds"] = time.time() - started_at
    if output_dir:
        result_path = os.path.join(output_dir, f"{output_prefix}.json")
        temporary_path = result_path + ".tmp"
        with open(temporary_path, "w") as result_file:
            json.dump(summary, result_file, indent=2)
            result_file.write("\n")
        os.replace(temporary_path, result_path)
        log(f"\nStructured results: {result_path}")
    log(f"Total time: {summary['elapsed_seconds']:.1f}s")
    if log_file:
        log_file.close()
    return summary


if __name__ == "__main__":
    evaluate()
