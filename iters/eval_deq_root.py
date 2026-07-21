"""Probe whether the recurrent Sudoku transformer has a latent fixed point.

The model is not modified. We first follow its normal recurrent trajectory, then
run Anderson acceleration from selected warm states and compare the resulting
state with the ordinary 1024-iteration prediction.
"""

import argparse
import importlib
import json
import os
import random
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from datasets import load_dataset


RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def _quantiles(values):
    values = torch.cat(values).float()
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, 0.9).item(),
        "max": values.max().item(),
    }


def _state_residual(state, next_state):
    difference = (next_state - state).float()
    flat_difference = difference.flatten(1)
    flat_state = state.float().flatten(1)
    absolute_rms = flat_difference.square().mean(dim=1).sqrt()
    relative = flat_difference.norm(dim=1) / flat_state.norm(dim=1).clamp_min(1e-12)
    state_rms = flat_state.square().mean(dim=1).sqrt()
    return absolute_rms, relative, state_rms


@torch.no_grad()
def anderson_solve(
    function,
    initial_state,
    max_steps=64,
    history_size=5,
    regularization=1e-4,
    mixing=1.0,
    absolute_tolerance=1e-3,
    relative_tolerance=1e-4,
):
    """Batched Anderson acceleration, returning each example's best iterate."""
    batch_size = initial_state.shape[0]
    state_shape = initial_state.shape[1:]
    flat_size = initial_state[0].numel()
    initial_flat = initial_state.float().reshape(batch_size, flat_size)

    states = torch.empty(
        batch_size, history_size, flat_size,
        device=initial_state.device, dtype=torch.float32,
    )
    function_values = torch.empty_like(states)

    def evaluate(flat_state):
        state = flat_state.reshape(batch_size, *state_shape)
        return function(state).float().reshape(batch_size, flat_size)

    states[:, 0] = initial_flat
    function_values[:, 0] = evaluate(states[:, 0])

    best_state = states[:, 0].clone()
    best_function_value = function_values[:, 0].clone()
    best_difference = best_function_value - best_state
    best_absolute = best_difference.square().mean(dim=1).sqrt()
    best_relative = (
        best_difference.norm(dim=1) /
        best_state.norm(dim=1).clamp_min(1e-12)
    )
    first_converged_step = torch.where(
        (best_absolute <= absolute_tolerance) & (best_relative <= relative_tolerance),
        torch.ones(batch_size, device=initial_state.device, dtype=torch.int64),
        torch.zeros(batch_size, device=initial_state.device, dtype=torch.int64),
    )

    if max_steps == 1:
        return {
            "state": best_state.reshape(batch_size, *state_shape),
            "absolute_residual": best_absolute,
            "relative_residual": best_relative,
            "first_converged_step": first_converged_step,
        }

    states[:, 1] = function_values[:, 0]
    function_values[:, 1] = evaluate(states[:, 1])

    second_difference = function_values[:, 1] - states[:, 1]
    second_absolute = second_difference.square().mean(dim=1).sqrt()
    second_relative = (
        second_difference.norm(dim=1) /
        states[:, 1].norm(dim=1).clamp_min(1e-12)
    )
    improved = second_absolute < best_absolute
    best_state[improved] = states[:, 1][improved]
    best_function_value[improved] = function_values[:, 1][improved]
    best_absolute[improved] = second_absolute[improved]
    best_relative[improved] = second_relative[improved]
    converged_now = (
        (first_converged_step == 0) &
        (second_absolute <= absolute_tolerance) &
        (second_relative <= relative_tolerance)
    )
    first_converged_step[converged_now] = 2

    identity = torch.eye(history_size, device=initial_state.device, dtype=torch.float32)

    for evaluation_number in range(3, max_steps + 1):
        history_count = min(evaluation_number - 1, history_size)
        residual_history = (
            function_values[:, :history_count] - states[:, :history_count]
        )

        system = torch.zeros(
            batch_size, history_count + 1, history_count + 1,
            device=initial_state.device, dtype=torch.float32,
        )
        system[:, 0, 1:] = 1
        system[:, 1:, 0] = 1
        system[:, 1:, 1:] = (
            torch.bmm(residual_history, residual_history.transpose(1, 2)) +
            regularization * identity[:history_count, :history_count]
        )
        target = torch.zeros(
            batch_size, history_count + 1, 1,
            device=initial_state.device, dtype=torch.float32,
        )
        target[:, 0] = 1

        try:
            coefficients = torch.linalg.solve(system, target)[:, 1:, 0]
        except torch.linalg.LinAlgError:
            coefficients = torch.linalg.lstsq(system, target).solution[:, 1:, 0]

        mixed_function = torch.bmm(
            coefficients.unsqueeze(1), function_values[:, :history_count]
        ).squeeze(1)
        mixed_state = torch.bmm(
            coefficients.unsqueeze(1), states[:, :history_count]
        ).squeeze(1)
        candidate = mixing * mixed_function + (1 - mixing) * mixed_state

        slot = (evaluation_number - 1) % history_size
        states[:, slot] = candidate
        function_values[:, slot] = evaluate(candidate)

        difference = function_values[:, slot] - states[:, slot]
        absolute = difference.square().mean(dim=1).sqrt()
        relative = (
            difference.norm(dim=1) /
            states[:, slot].norm(dim=1).clamp_min(1e-12)
        )
        finite = torch.isfinite(absolute) & torch.isfinite(relative)
        improved = finite & (absolute < best_absolute)
        best_state[improved] = states[:, slot][improved]
        best_function_value[improved] = function_values[:, slot][improved]
        best_absolute[improved] = absolute[improved]
        best_relative[improved] = relative[improved]

        converged_now = (
            (first_converged_step == 0) & finite &
            (absolute <= absolute_tolerance) &
            (relative <= relative_tolerance)
        )
        first_converged_step[converged_now] = evaluation_number

        if (first_converged_step > 0).all():
            break

    return {
        "state": best_state.reshape(batch_size, *state_shape),
        "absolute_residual": best_absolute,
        "relative_residual": best_relative,
        "first_converged_step": first_converged_step,
    }


@torch.no_grad()
def broyden_solve(
    function,
    initial_state,
    max_steps=64,
    absolute_tolerance=1e-3,
    relative_tolerance=1e-4,
):
    """Batched good Broyden solve with a low-rank inverse-Jacobian estimate."""
    batch_size = initial_state.shape[0]
    state_shape = initial_state.shape[1:]
    flat_size = initial_state[0].numel()
    current_state = initial_state.float().reshape(batch_size, flat_size).clone()

    def evaluate(flat_state):
        state = flat_state.reshape(batch_size, *state_shape)
        return function(state).float().reshape(batch_size, flat_size)

    def inverse_matvec(left_factors, right_factors, vector, rank):
        if rank == 0:
            return -vector
        projected = torch.bmm(
            right_factors[:, :rank], vector.unsqueeze(-1)
        )
        return -vector + torch.bmm(
            left_factors[:, :, :rank], projected
        ).squeeze(-1)

    def inverse_rmatvec(left_factors, right_factors, vector, rank):
        if rank == 0:
            return -vector
        projected = torch.bmm(
            vector.unsqueeze(1), left_factors[:, :, :rank]
        )
        return -vector + torch.bmm(
            projected, right_factors[:, :rank]
        ).squeeze(1)

    function_value = evaluate(current_state)
    residual = function_value - current_state
    best_state = current_state.clone()
    best_absolute = residual.square().mean(dim=1).sqrt()
    best_relative = (
        residual.norm(dim=1) /
        current_state.norm(dim=1).clamp_min(1e-12)
    )
    first_converged_step = torch.where(
        (best_absolute <= absolute_tolerance) & (best_relative <= relative_tolerance),
        torch.ones(batch_size, device=initial_state.device, dtype=torch.int64),
        torch.zeros(batch_size, device=initial_state.device, dtype=torch.int64),
    )

    max_rank = max(max_steps - 1, 1)
    left_factors = torch.zeros(
        batch_size, flat_size, max_rank,
        device=initial_state.device, dtype=torch.float32,
    )
    right_factors = torch.zeros(
        batch_size, max_rank, flat_size,
        device=initial_state.device, dtype=torch.float32,
    )
    rank = 0

    for evaluation_number in range(2, max_steps + 1):
        update = -inverse_matvec(left_factors, right_factors, residual, rank)
        candidate_state = current_state + update
        candidate_function_value = evaluate(candidate_state)
        candidate_residual = candidate_function_value - candidate_state

        finite = (
            torch.isfinite(candidate_state).all(dim=1) &
            torch.isfinite(candidate_residual).all(dim=1)
        )
        candidate_state = torch.where(
            finite[:, None], candidate_state, current_state
        )
        candidate_residual = torch.where(
            finite[:, None], candidate_residual, residual
        )

        absolute = candidate_residual.square().mean(dim=1).sqrt()
        relative = (
            candidate_residual.norm(dim=1) /
            candidate_state.norm(dim=1).clamp_min(1e-12)
        )
        improved = finite & (absolute < best_absolute)
        best_state[improved] = candidate_state[improved]
        best_absolute[improved] = absolute[improved]
        best_relative[improved] = relative[improved]

        converged_now = (
            (first_converged_step == 0) & finite &
            (absolute <= absolute_tolerance) &
            (relative <= relative_tolerance)
        )
        first_converged_step[converged_now] = evaluation_number

        state_difference = candidate_state - current_state
        residual_difference = candidate_residual - residual
        right_factor = inverse_rmatvec(
            left_factors, right_factors, state_difference, rank
        )
        numerator = state_difference - inverse_matvec(
            left_factors, right_factors, residual_difference, rank
        )
        denominator = (right_factor * residual_difference).sum(dim=1)
        safe_denominator = torch.isfinite(denominator) & (denominator.abs() > 1e-12)
        new_left_factor = torch.zeros_like(numerator)
        new_left_factor[safe_denominator] = (
            numerator[safe_denominator] /
            denominator[safe_denominator, None]
        )
        new_left_factor[~torch.isfinite(new_left_factor)] = 0
        right_factor[~torch.isfinite(right_factor)] = 0
        left_factors[:, :, rank] = new_left_factor
        right_factors[:, rank] = right_factor
        rank += 1

        current_state = candidate_state
        residual = candidate_residual
        if (first_converged_step > 0).all():
            break

    return {
        "state": best_state.reshape(batch_size, *state_shape),
        "absolute_residual": best_absolute,
        "relative_residual": best_relative,
        "first_converged_step": first_converged_step,
    }


def _load_test_sample(module, examples_per_bucket, seed):
    dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, example in enumerate(dataset):
        rating = example["rating"]
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= rating <= maximum:
                bucket_indices[name].append(index)
                break

    generator = random.Random(seed)
    selected_indices = []
    selected_buckets = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        count = min(examples_per_bucket, len(candidates))
        chosen = generator.sample(candidates, count)
        selected_indices.extend(chosen)
        selected_buckets.extend([name] * count)

    puzzles = [dataset[index]["question"] for index in selected_indices]
    solutions = [dataset[index]["answer"] for index in selected_indices]
    inputs = module.encode_puzzles(puzzles)
    targets = module.encode_solutions(solutions).long()
    empty_mask = torch.tensor(
        [[character == "." for character in puzzle] for puzzle in puzzles],
        dtype=torch.bool,
    )
    return inputs, targets, empty_mask, selected_buckets


def _puzzles_solved(predictions, targets, empty_mask):
    correct_cells = (predictions == targets) & empty_mask
    return correct_cells.sum(dim=1) == empty_mask.sum(dim=1)


def evaluate(
    model_path,
    experiment_module="iters.exp_baseline_lr2e3",
    examples_per_bucket=200,
    batch_size=100,
    warm_iterations=(16, 64, 128, 1024),
    mixing_values=(1.0, 0.5),
    max_solver_steps=64,
    history_size=5,
    regularization=1e-4,
    absolute_tolerance=1e-3,
    relative_tolerance=1e-4,
    precision="fp32",
    seed=42,
    device="cuda",
    output_dir=None,
):
    module = importlib.import_module(experiment_module)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    if precision not in {"fp32", "bf16"}:
        raise ValueError(f"Unsupported precision: {precision}")
    if min(warm_iterations) < 1:
        raise ValueError("Warm iterations must be positive")

    log_file = None
    if output_dir:
        model_name = os.path.basename(model_path).removesuffix(".pt")
        log_path = os.path.join(output_dir, f"{model_name}_deq_root_probe.log")
        log_file = open(log_path, "w")

    def log(message=""):
        print(message, flush=True)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    model = module.SudokuTransformer().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    if precision == "fp32":
        torch.set_float32_matmul_precision("highest")

    inputs, targets, empty_mask, bucket_names = _load_test_sample(
        module, examples_per_bucket, seed
    )
    sample_size = len(inputs)
    rope_cos = module.ROPE_COS.to(device)
    rope_sin = module.ROPE_SIN.to(device)

    def autocast_context():
        if precision == "bf16" and device.type == "cuda":
            return torch.autocast(device.type, dtype=torch.bfloat16)
        return nullcontext()

    def iteration_step(hidden_state):
        with autocast_context():
            predictions = F.softmax(model.output_head(hidden_state), dim=-1)
            next_hidden = hidden_state + model.pred_proj(predictions)
            for layer in model.layers:
                next_hidden = layer(next_hidden, rope_cos, rope_sin)
        return next_hidden.float()

    def decode(hidden_state):
        with autocast_context():
            return model.output_head(hidden_state).float()

    trajectory = {
        iteration: {
            "solved": 0,
            "absolute": [],
            "relative": [],
            "state_rms": [],
            "probability_rms": [],
            "stable_puzzles": 0,
            "changed_cells": 0,
            "empty_cells": 0,
        }
        for iteration in warm_iterations
    }
    solver_results = {
        (iteration, mixing): {
            "solved": 0,
            "converged": 0,
            "same_as_1024": 0,
            "absolute": [],
            "relative": [],
            "state_rms": [],
            "convergence_steps": [],
        }
        for iteration in warm_iterations
        for mixing in mixing_values
    }
    broyden_results = {
        iteration: {
            "solved": 0,
            "converged": 0,
            "same_as_1024": 0,
            "absolute": [],
            "relative": [],
            "state_rms": [],
            "convergence_steps": [],
        }
        for iteration in warm_iterations
    }

    max_warm_iterations = max(warm_iterations)
    if max_warm_iterations < 1024:
        raise ValueError("Include 1024 in warm_iterations for the reference prediction")

    log("DEQ root probe")
    log(f"Model: {model_path}")
    log(f"Examples: {sample_size} ({examples_per_bucket} per rating bucket)")
    log(f"Precision: {precision}; device: {device}")
    log(
        f"Anderson: max_steps={max_solver_steps}, history={history_size}, "
        f"regularization={regularization:g}, mixing={list(mixing_values)}"
    )
    log(
        f"Convergence requires absolute RMS <= {absolute_tolerance:g} and "
        f"relative residual <= {relative_tolerance:g}"
    )

    started_at = time.time()
    with torch.inference_mode():
        for batch_start in range(0, sample_size, batch_size):
            batch_end = min(batch_start + batch_size, sample_size)
            batch_inputs = inputs[batch_start:batch_end].to(device)
            batch_targets = targets[batch_start:batch_end]
            batch_empty_mask = empty_mask[batch_start:batch_end]
            current_batch_size = batch_end - batch_start

            with autocast_context():
                hidden_state = model.initial_encoder(batch_inputs)
                predictions = torch.zeros(
                    current_batch_size, 81, 9, device=device, dtype=torch.float32
                )

            warm_states = {}
            warm_predictions = {}
            for iteration in range(1, max_warm_iterations + 1):
                with autocast_context():
                    next_hidden = hidden_state + model.pred_proj(predictions)
                    for layer in model.layers:
                        next_hidden = layer(next_hidden, rope_cos, rope_sin)
                    logits = model.output_head(next_hidden)
                    predictions = F.softmax(logits, dim=-1)
                hidden_state = next_hidden.float()

                if iteration not in trajectory:
                    continue

                next_state = iteration_step(hidden_state)
                next_logits = decode(next_state)
                current_logits = logits.float()
                current_probabilities = F.softmax(current_logits, dim=-1)
                next_probabilities = F.softmax(next_logits, dim=-1)
                absolute, relative, state_rms = _state_residual(
                    hidden_state, next_state
                )
                current_predictions = current_logits.argmax(dim=-1).cpu()
                following_predictions = next_logits.argmax(dim=-1).cpu()
                changed = (
                    (current_predictions != following_predictions) & batch_empty_mask
                )

                metrics = trajectory[iteration]
                metrics["solved"] += _puzzles_solved(
                    current_predictions, batch_targets, batch_empty_mask
                ).sum().item()
                metrics["absolute"].append(absolute.cpu())
                metrics["relative"].append(relative.cpu())
                metrics["state_rms"].append(state_rms.cpu())
                metrics["probability_rms"].append(
                    (next_probabilities - current_probabilities)
                    .square().mean(dim=(1, 2)).sqrt().cpu()
                )
                metrics["stable_puzzles"] += (~changed.any(dim=1)).sum().item()
                metrics["changed_cells"] += changed.sum().item()
                metrics["empty_cells"] += batch_empty_mask.sum().item()
                warm_states[iteration] = hidden_state.clone()
                warm_predictions[iteration] = current_predictions

            reference_predictions = warm_predictions[1024]
            for iteration in warm_iterations:
                for mixing in mixing_values:
                    result = anderson_solve(
                        iteration_step,
                        warm_states[iteration],
                        max_steps=max_solver_steps,
                        history_size=history_size,
                        regularization=regularization,
                        mixing=mixing,
                        absolute_tolerance=absolute_tolerance,
                        relative_tolerance=relative_tolerance,
                    )
                    root_predictions = decode(result["state"]).argmax(dim=-1).cpu()
                    converged = (
                        (result["absolute_residual"] <= absolute_tolerance) &
                        (result["relative_residual"] <= relative_tolerance)
                    )
                    same_as_reference = (
                        ((root_predictions == reference_predictions) | ~batch_empty_mask)
                        .all(dim=1)
                    )

                    metrics = solver_results[(iteration, mixing)]
                    metrics["solved"] += _puzzles_solved(
                        root_predictions, batch_targets, batch_empty_mask
                    ).sum().item()
                    metrics["converged"] += converged.sum().item()
                    metrics["same_as_1024"] += same_as_reference.sum().item()
                    metrics["absolute"].append(result["absolute_residual"].cpu())
                    metrics["relative"].append(result["relative_residual"].cpu())
                    metrics["state_rms"].append(
                        result["state"].float().square().mean(dim=(1, 2)).sqrt().cpu()
                    )
                    positive_steps = result["first_converged_step"]
                    metrics["convergence_steps"].append(positive_steps.cpu())

                result = broyden_solve(
                    iteration_step,
                    warm_states[iteration],
                    max_steps=max_solver_steps,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                )
                root_predictions = decode(result["state"]).argmax(dim=-1).cpu()
                converged = (
                    (result["absolute_residual"] <= absolute_tolerance) &
                    (result["relative_residual"] <= relative_tolerance)
                )
                same_as_reference = (
                    ((root_predictions == reference_predictions) | ~batch_empty_mask)
                    .all(dim=1)
                )
                metrics = broyden_results[iteration]
                metrics["solved"] += _puzzles_solved(
                    root_predictions, batch_targets, batch_empty_mask
                ).sum().item()
                metrics["converged"] += converged.sum().item()
                metrics["same_as_1024"] += same_as_reference.sum().item()
                metrics["absolute"].append(result["absolute_residual"].cpu())
                metrics["relative"].append(result["relative_residual"].cpu())
                metrics["state_rms"].append(
                    result["state"].float().square().mean(dim=(1, 2)).sqrt().cpu()
                )
                metrics["convergence_steps"].append(
                    result["first_converged_step"].cpu()
                )

            elapsed = time.time() - started_at
            log(f"Processed {batch_end}/{sample_size} examples in {elapsed:.1f}s")

    summary = {
        "config": {
            "model_path": model_path,
            "experiment_module": experiment_module,
            "sample_size": sample_size,
            "examples_per_bucket": examples_per_bucket,
            "bucket_counts": {
                name: bucket_names.count(name) for _, _, name in RATING_BUCKETS
            },
            "batch_size": batch_size,
            "warm_iterations": list(warm_iterations),
            "mixing_values": list(mixing_values),
            "max_solver_steps": max_solver_steps,
            "history_size": history_size,
            "regularization": regularization,
            "absolute_tolerance": absolute_tolerance,
            "relative_tolerance": relative_tolerance,
            "precision": precision,
            "seed": seed,
        },
        "trajectory": {},
        "anderson": {},
        "broyden": {},
        "elapsed_seconds": time.time() - started_at,
    }

    log()
    log("Ordinary recurrent trajectory")
    log(
        " Iters | Solved | state RMS | residual RMS (median/p90) | "
        "relative (median/p90) | unchanged puzzles | changed empty cells"
    )
    for iteration in warm_iterations:
        metrics = trajectory[iteration]
        state_stats = _quantiles(metrics["state_rms"])
        absolute_stats = _quantiles(metrics["absolute"])
        relative_stats = _quantiles(metrics["relative"])
        probability_stats = _quantiles(metrics["probability_rms"])
        solved_rate = metrics["solved"] / sample_size
        stable_rate = metrics["stable_puzzles"] / sample_size
        changed_cell_rate = metrics["changed_cells"] / metrics["empty_cells"]
        summary["trajectory"][str(iteration)] = {
            "solved_rate": solved_rate,
            "state_rms": state_stats,
            "absolute_residual_rms": absolute_stats,
            "relative_residual": relative_stats,
            "probability_change_rms": probability_stats,
            "unchanged_puzzle_rate": stable_rate,
            "changed_empty_cell_rate": changed_cell_rate,
        }
        log(
            f"{iteration:6d} | {100 * solved_rate:5.1f}% | "
            f"{state_stats['median']:9.3g} | "
            f"{absolute_stats['median']:.3g}/{absolute_stats['p90']:.3g} | "
            f"{relative_stats['median']:.3g}/{relative_stats['p90']:.3g} | "
            f"{100 * stable_rate:6.2f}% | {100 * changed_cell_rate:7.4f}%"
        )

    log()
    log("Anderson root solve")
    log(
        " Warm | Mix | Solved | Converged | Same as iter 1024 | "
        "best residual RMS (median/p90) | root state RMS | relative (median/p90) | median NFE"
    )
    for iteration in warm_iterations:
        for mixing in mixing_values:
            metrics = solver_results[(iteration, mixing)]
            absolute_stats = _quantiles(metrics["absolute"])
            relative_stats = _quantiles(metrics["relative"])
            state_stats = _quantiles(metrics["state_rms"])
            convergence_steps = torch.cat(metrics["convergence_steps"])
            converged_steps = convergence_steps[convergence_steps > 0]
            median_nfe = (
                converged_steps.float().median().item()
                if len(converged_steps) else None
            )
            solved_rate = metrics["solved"] / sample_size
            converged_rate = metrics["converged"] / sample_size
            same_rate = metrics["same_as_1024"] / sample_size
            key = f"warm_{iteration}_mix_{mixing:g}"
            summary["anderson"][key] = {
                "solved_rate": solved_rate,
                "converged_rate": converged_rate,
                "same_as_1024_rate": same_rate,
                "absolute_residual_rms": absolute_stats,
                "relative_residual": relative_stats,
                "root_state_rms": state_stats,
                "median_nfe_among_converged": median_nfe,
            }
            nfe_text = f"{median_nfe:.0f}" if median_nfe is not None else "-"
            log(
                f"{iteration:5d} | {mixing:3.1f} | {100 * solved_rate:5.1f}% | "
                f"{100 * converged_rate:8.2f}% | {100 * same_rate:16.2f}% | "
                f"{absolute_stats['median']:.3g}/{absolute_stats['p90']:.3g} | "
                f"{state_stats['median']:.3g} | "
                f"{relative_stats['median']:.3g}/{relative_stats['p90']:.3g} | "
                f"{nfe_text}"
            )

    log()
    log("Broyden root solve")
    log(
        " Warm | Solved | Converged | Same as iter 1024 | "
        "best residual RMS (median/p90) | root state RMS | relative (median/p90) | median NFE"
    )
    for iteration in warm_iterations:
        metrics = broyden_results[iteration]
        absolute_stats = _quantiles(metrics["absolute"])
        relative_stats = _quantiles(metrics["relative"])
        state_stats = _quantiles(metrics["state_rms"])
        convergence_steps = torch.cat(metrics["convergence_steps"])
        converged_steps = convergence_steps[convergence_steps > 0]
        median_nfe = (
            converged_steps.float().median().item()
            if len(converged_steps) else None
        )
        solved_rate = metrics["solved"] / sample_size
        converged_rate = metrics["converged"] / sample_size
        same_rate = metrics["same_as_1024"] / sample_size
        summary["broyden"][f"warm_{iteration}"] = {
            "solved_rate": solved_rate,
            "converged_rate": converged_rate,
            "same_as_1024_rate": same_rate,
            "absolute_residual_rms": absolute_stats,
            "relative_residual": relative_stats,
            "root_state_rms": state_stats,
            "median_nfe_among_converged": median_nfe,
        }
        nfe_text = f"{median_nfe:.0f}" if median_nfe is not None else "-"
        log(
            f"{iteration:5d} | {100 * solved_rate:5.1f}% | "
            f"{100 * converged_rate:8.2f}% | {100 * same_rate:16.2f}% | "
            f"{absolute_stats['median']:.3g}/{absolute_stats['p90']:.3g} | "
            f"{state_stats['median']:.3g} | "
            f"{relative_stats['median']:.3g}/{relative_stats['p90']:.3g} | "
            f"{nfe_text}"
        )

    log(f"\nTotal time: {summary['elapsed_seconds']:.1f}s")
    if output_dir:
        model_name = os.path.basename(model_path).removesuffix(".pt")
        json_path = os.path.join(output_dir, f"{model_name}_deq_root_probe.json")
        with open(json_path, "w") as json_file:
            json.dump(summary, json_file, indent=2)
        log(f"Structured results: {json_path}")
        log_file.close()

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--exp", default="iters.exp_baseline_lr2e3")
    parser.add_argument("--examples-per-bucket", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--warm-iters", type=int, nargs="+", default=[16, 64, 128, 1024])
    parser.add_argument("--mixing", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--max-solver-steps", type=int, default=64)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-3)
    parser.add_argument("--relative-tolerance", type=float, default=1e-4)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    evaluate(
        arguments.model_path,
        experiment_module=arguments.exp,
        examples_per_bucket=arguments.examples_per_bucket,
        batch_size=arguments.batch_size,
        warm_iterations=tuple(arguments.warm_iters),
        mixing_values=tuple(arguments.mixing),
        max_solver_steps=arguments.max_solver_steps,
        history_size=arguments.history_size,
        regularization=arguments.regularization,
        absolute_tolerance=arguments.absolute_tolerance,
        relative_tolerance=arguments.relative_tolerance,
        precision=arguments.precision,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
    )
