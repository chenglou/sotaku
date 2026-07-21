"""Measure directional dynamics in stable, collapsed, and ES-rescued models."""

import argparse
import importlib
import json
import os
import random
import time

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

DEFAULT_CHECKPOINTS = (4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
DEFAULT_MODELS = (
    ("stable_bp", "/outputs/model_baseline_lr2e3.pt"),
    ("collapsed_pre_es", "/outputs/model_baseline_lr2e3_clean_a.pt"),
    ("rescued_post_es", "/outputs/model_es_ft_collapsed.pt"),
)


def _quantiles(chunks):
    values = torch.cat(chunks).float()
    return {
        "mean": values.mean().item(),
        "p10": torch.quantile(values, 0.1).item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, 0.9).item(),
        "max": values.max().item(),
    }


def _cosine(first, second):
    first_flat = first.float().flatten(1)
    second_flat = second.float().flatten(1)
    return F.cosine_similarity(first_flat, second_flat, dim=1).clamp(-1, 1)


def _rms(value):
    return value.float().flatten(1).square().mean(dim=1).sqrt()


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
    bucket_names = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        count = min(examples_per_bucket, len(candidates))
        selected_indices.extend(generator.sample(candidates, count))
        bucket_names.extend([name] * count)

    puzzles = [dataset[index]["question"] for index in selected_indices]
    solutions = [dataset[index]["answer"] for index in selected_indices]
    inputs = module.encode_puzzles(puzzles)
    targets = module.encode_solutions(solutions).long()
    empty_mask = torch.tensor(
        [[character == "." for character in puzzle] for puzzle in puzzles],
        dtype=torch.bool,
    )
    return inputs, targets, empty_mask, bucket_names


def _puzzles_solved(predictions, targets, empty_mask):
    correct = (predictions == targets) | ~empty_mask
    return correct.all(dim=1)


def _minimum_target_margin(logits, targets, empty_mask):
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    other_logits = logits.clone()
    other_logits.scatter_(-1, targets.unsqueeze(-1), -torch.inf)
    margins = target_logits - other_logits.max(dim=-1).values
    margins = margins.masked_fill(~empty_mask, torch.inf)
    return margins.min(dim=1).values


def _load_model(module, model_path, device):
    model = module.SudokuTransformer().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    state_dict = {
        key.removeprefix("_orig_mod."): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _parameter_difference(before_model, after_model):
    before_parameters = dict(before_model.named_parameters())
    after_parameters = dict(after_model.named_parameters())
    difference_squared = torch.zeros((), device=next(before_model.parameters()).device)
    before_squared = torch.zeros_like(difference_squared)
    after_squared = torch.zeros_like(difference_squared)
    dot_product = torch.zeros_like(difference_squared)
    maximum_absolute_difference = 0.0
    for name, before in before_parameters.items():
        after = after_parameters[name]
        difference = after.float() - before.float()
        difference_squared += difference.square().sum()
        before_squared += before.float().square().sum()
        after_squared += after.float().square().sum()
        dot_product += (before.float() * after.float()).sum()
        maximum_absolute_difference = max(
            maximum_absolute_difference, difference.abs().max().item()
        )
    return {
        "relative_l2": (difference_squared / before_squared).sqrt().item(),
        "cosine": (
            dot_product / (before_squared.sqrt() * after_squared.sqrt())
        ).item(),
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def _empty_model_metrics(checkpoints):
    distribution_names = (
        "state_rms",
        "update_rms",
        "state_update_cosine",
        "successive_update_cosine",
        "step_direction_change",
        "state_to_final_cosine",
        "update_to_final_cosine",
        "raw_minimum_target_margin",
        "directional_minimum_target_margin",
    )
    return {
        iteration: {
            "count": 0,
            "solved": 0,
            "directional_solved": 0,
            **{name: [] for name in distribution_names},
        }
        for iteration in checkpoints
    }


def _run_model(model, inputs, targets, empty_mask, checkpoints, rope_cos, rope_sin):
    batch_size = inputs.shape[0]
    hidden_state = model.initial_encoder(inputs)
    predictions = torch.zeros(batch_size, 81, 9, device=inputs.device)
    saved = {}

    for iteration in range(1, max(checkpoints) + 1):
        previous_hidden = hidden_state
        hidden_state = hidden_state + model.pred_proj(predictions)
        for layer in model.layers:
            hidden_state = layer(hidden_state, rope_cos, rope_sin)
        incoming_update = hidden_state - previous_hidden
        logits = model.output_head(hidden_state)
        predictions = F.softmax(logits, dim=-1)

        if iteration not in checkpoints:
            continue

        following_predictions = F.softmax(model.output_head(hidden_state), dim=-1)
        next_hidden = hidden_state + model.pred_proj(following_predictions)
        for layer in model.layers:
            next_hidden = layer(next_hidden, rope_cos, rope_sin)
        outgoing_update = next_hidden - hidden_state

        actual_predictions = logits.argmax(dim=-1)
        normalized_hidden = F.normalize(hidden_state.float(), dim=-1)
        directional_logits = F.linear(
            normalized_hidden,
            model.output_head.weight.float(),
            bias=None,
        )
        directional_predictions = directional_logits.argmax(dim=-1)

        saved[iteration] = {
            "state": hidden_state.clone(),
            "update": outgoing_update.clone(),
            "predictions": actual_predictions,
            "solved": _puzzles_solved(actual_predictions, targets, empty_mask),
            "directional_solved": _puzzles_solved(
                directional_predictions, targets, empty_mask
            ),
            "state_rms": _rms(hidden_state),
            "update_rms": _rms(outgoing_update),
            "state_update_cosine": _cosine(hidden_state, outgoing_update),
            "successive_update_cosine": _cosine(
                incoming_update, outgoing_update
            ),
            "step_direction_change": 1 - _cosine(hidden_state, next_hidden),
            "raw_minimum_target_margin": _minimum_target_margin(
                logits.float(), targets, empty_mask
            ),
            "directional_minimum_target_margin": _minimum_target_margin(
                directional_logits, targets, empty_mask
            ),
        }

    final_state = saved[max(checkpoints)]["state"]
    final_update = saved[max(checkpoints)]["update"]
    for iteration in checkpoints:
        saved[iteration]["state_to_final_cosine"] = _cosine(
            saved[iteration]["state"], final_state
        )
        saved[iteration]["update_to_final_cosine"] = _cosine(
            saved[iteration]["update"], final_update
        )
    return saved


def evaluate(
    model_configs=DEFAULT_MODELS,
    experiment_module="iters.exp_baseline_lr2e3",
    examples_per_bucket=200,
    batch_size=100,
    checkpoints=DEFAULT_CHECKPOINTS,
    seed=42,
    device="cuda",
    output_dir=None,
):
    module = importlib.import_module(experiment_module)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("highest")

    log_file = None
    if output_dir:
        log_file = open(os.path.join(output_dir, "ray_dynamics_probe.log"), "w")

    def log(message=""):
        print(message, flush=True)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    models = {
        name: _load_model(module, model_path, device)
        for name, model_path in model_configs
    }
    inputs, targets, empty_mask, bucket_names = _load_test_sample(
        module, examples_per_bucket, seed
    )
    sample_size = len(inputs)
    rope_cos = module.ROPE_COS.to(device)
    rope_sin = module.ROPE_SIN.to(device)

    model_metrics = {
        name: _empty_model_metrics(checkpoints) for name in models
    }
    cross_metrics = {
        iteration: {
            "state_cosine": [],
            "update_cosine": [],
            "puzzle_prediction_agreement": 0,
            "empty_cell_prediction_agreement": 0,
            "empty_cells": 0,
        }
        for iteration in checkpoints
    }
    parameter_difference = _parameter_difference(
        models["collapsed_pre_es"], models["rescued_post_es"]
    )

    log("Stable-ray dynamics probe")
    log(f"Examples: {sample_size} ({examples_per_bucket} per rating bucket)")
    log(f"Iterations: {list(checkpoints)}; precision: fp32; device: {device}")
    log(
        "Collapsed -> rescued parameter change: "
        f"relative L2={parameter_difference['relative_l2']:.6g}, "
        f"cosine={parameter_difference['cosine']:.9f}, "
        f"max abs={parameter_difference['maximum_absolute_difference']:.6g}"
    )

    started_at = time.time()
    distribution_names = tuple(
        name
        for name in model_metrics[next(iter(models))][checkpoints[0]]
        if name not in {"count", "solved", "directional_solved"}
    )

    with torch.inference_mode():
        for batch_start in range(0, sample_size, batch_size):
            batch_end = min(batch_start + batch_size, sample_size)
            batch_inputs = inputs[batch_start:batch_end].to(device)
            batch_targets = targets[batch_start:batch_end].to(device)
            batch_empty_mask = empty_mask[batch_start:batch_end].to(device)
            batch_saved = {}

            for name, model in models.items():
                saved = _run_model(
                    model,
                    batch_inputs,
                    batch_targets,
                    batch_empty_mask,
                    checkpoints,
                    rope_cos,
                    rope_sin,
                )
                batch_saved[name] = saved
                for iteration in checkpoints:
                    source = saved[iteration]
                    destination = model_metrics[name][iteration]
                    destination["count"] += batch_end - batch_start
                    destination["solved"] += source["solved"].sum().item()
                    destination["directional_solved"] += (
                        source["directional_solved"].sum().item()
                    )
                    for metric_name in distribution_names:
                        destination[metric_name].append(
                            source[metric_name].detach().cpu()
                        )

            collapsed = batch_saved["collapsed_pre_es"]
            rescued = batch_saved["rescued_post_es"]
            for iteration in checkpoints:
                destination = cross_metrics[iteration]
                destination["state_cosine"].append(
                    _cosine(
                        collapsed[iteration]["state"],
                        rescued[iteration]["state"],
                    ).cpu()
                )
                destination["update_cosine"].append(
                    _cosine(
                        collapsed[iteration]["update"],
                        rescued[iteration]["update"],
                    ).cpu()
                )
                collapsed_predictions = collapsed[iteration]["predictions"]
                rescued_predictions = rescued[iteration]["predictions"]
                agreement = collapsed_predictions == rescued_predictions
                destination["puzzle_prediction_agreement"] += (
                    (agreement | ~batch_empty_mask).all(dim=1).sum().item()
                )
                destination["empty_cell_prediction_agreement"] += (
                    agreement & batch_empty_mask
                ).sum().item()
                destination["empty_cells"] += batch_empty_mask.sum().item()

            del batch_saved
            log(
                f"Processed {batch_end}/{sample_size} examples in "
                f"{time.time() - started_at:.1f}s"
            )

    summary = {
        "config": {
            "models": {name: path for name, path in model_configs},
            "experiment_module": experiment_module,
            "sample_size": sample_size,
            "examples_per_bucket": examples_per_bucket,
            "bucket_counts": {
                name: bucket_names.count(name) for _, _, name in RATING_BUCKETS
            },
            "batch_size": batch_size,
            "checkpoints": list(checkpoints),
            "seed": seed,
            "precision": "fp32",
        },
        "collapsed_to_rescued_parameter_difference": parameter_difference,
        "models": {},
        "collapsed_vs_rescued": {},
        "elapsed_seconds": time.time() - started_at,
    }

    for name in models:
        log()
        log(name)
        log(
            " Iters | Solved | Ray solved | h RMS | update RMS | "
            "cos(d_in,d_out) | cos(h,d_out) | cos(h,h_2048) | ray min margin"
        )
        summary["models"][name] = {}
        for iteration in checkpoints:
            metrics = model_metrics[name][iteration]
            solved_rate = metrics["solved"] / metrics["count"]
            directional_solved_rate = (
                metrics["directional_solved"] / metrics["count"]
            )
            distributions = {
                metric_name: _quantiles(metrics[metric_name])
                for metric_name in distribution_names
            }
            summary["models"][name][str(iteration)] = {
                "solved_rate": solved_rate,
                "directional_solved_rate": directional_solved_rate,
                **distributions,
            }
            log(
                f"{iteration:6d} | {100 * solved_rate:5.1f}% | "
                f"{100 * directional_solved_rate:9.1f}% | "
                f"{distributions['state_rms']['median']:5.1f} | "
                f"{distributions['update_rms']['median']:10.3g} | "
                f"{distributions['successive_update_cosine']['median']:15.6f} | "
                f"{distributions['state_update_cosine']['median']:12.6f} | "
                f"{distributions['state_to_final_cosine']['median']:14.6f} | "
                f"{distributions['directional_minimum_target_margin']['median']:.5f}"
            )

    log()
    log("collapsed_pre_es vs rescued_post_es")
    log(
        " Iters | state cosine | update cosine | same puzzle prediction | "
        "same empty-cell prediction"
    )
    for iteration in checkpoints:
        metrics = cross_metrics[iteration]
        state_stats = _quantiles(metrics["state_cosine"])
        update_stats = _quantiles(metrics["update_cosine"])
        puzzle_agreement = metrics["puzzle_prediction_agreement"] / sample_size
        cell_agreement = (
            metrics["empty_cell_prediction_agreement"] / metrics["empty_cells"]
        )
        summary["collapsed_vs_rescued"][str(iteration)] = {
            "state_cosine": state_stats,
            "update_cosine": update_stats,
            "puzzle_prediction_agreement": puzzle_agreement,
            "empty_cell_prediction_agreement": cell_agreement,
        }
        log(
            f"{iteration:6d} | {state_stats['median']:12.6f} | "
            f"{update_stats['median']:13.6f} | "
            f"{100 * puzzle_agreement:21.2f}% | {100 * cell_agreement:25.3f}%"
        )

    log(f"\nTotal time: {summary['elapsed_seconds']:.1f}s")
    if output_dir:
        json_path = os.path.join(output_dir, "ray_dynamics_probe.json")
        with open(json_path, "w") as json_file:
            json.dump(summary, json_file, indent=2)
        log(f"Structured results: {json_path}")
        log_file.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable", default=DEFAULT_MODELS[0][1])
    parser.add_argument("--collapsed", default=DEFAULT_MODELS[1][1])
    parser.add_argument("--rescued", default=DEFAULT_MODELS[2][1])
    parser.add_argument("--exp", default="iters.exp_baseline_lr2e3")
    parser.add_argument("--examples-per-bucket", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    arguments = parser.parse_args()
    evaluate(
        model_configs=(
            ("stable_bp", arguments.stable),
            ("collapsed_pre_es", arguments.collapsed),
            ("rescued_post_es", arguments.rescued),
        ),
        experiment_module=arguments.exp,
        examples_per_bucket=arguments.examples_per_bucket,
        batch_size=arguments.batch_size,
        checkpoints=tuple(arguments.checkpoints),
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
    )
