"""Test stronger recurrent damping on stable and collapsed checkpoints.

Earlier intervention sweeps stopped at alpha=0.5. This sweep reaches alpha=1/64,
which keeps the aggregate update budget of 1,024 damped iterations comparable to
the 16 undamped iterations used in training.
"""

import json
import os
import time

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import (
    DEFAULT_MODELS,
    _load_balanced_sample,
    _load_model,
)


DEFAULT_MODEL_CONFIGS = DEFAULT_MODELS[:2]
DEFAULT_ALPHAS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625)
DEFAULT_HORIZONS = (16, 128, 1024)


def relaxed_recurrent_step(
    model,
    hidden_state,
    predictions,
    rope_cos,
    rope_sin,
    alpha,
):
    proposed_state = model.apply_recurrent_updates(
        hidden_state,
        predictions,
        rope_cos,
        rope_sin,
    )
    hidden_state = torch.lerp(hidden_state, proposed_state, alpha)
    return model.normalize_outer_state(hidden_state)


def evaluate_policy(
    model,
    inputs,
    targets,
    empty_mask,
    *,
    alpha,
    horizons,
    warmup_iterations=0,
    batch_size=100,
    bucket_names=None,
):
    if bucket_names is not None and len(bucket_names) != inputs.size(0):
        raise ValueError("bucket_names must match the number of examples")
    bucket_order = tuple(dict.fromkeys(bucket_names or ()))
    device = next(model.parameters()).device
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    horizon_set = set(horizons)
    totals = {
        horizon: {
            "solved": 0,
            "correct_empty_cells": 0,
            "empty_cells": 0,
            "state_rms_sum": 0.0,
            "examples": 0,
            "per_bucket": {
                name: {"solved": 0, "total": 0}
                for name in bucket_order
            },
        }
        for horizon in horizons
    }

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, inputs.size(0), batch_size):
            batch_inputs = inputs[start:start + batch_size].to(device)
            batch_targets = targets[start:start + batch_size].to(device)
            batch_empty_mask = empty_mask[start:start + batch_size].to(device)
            hidden_state = model.initial_encoder(batch_inputs)
            predictions = torch.zeros(
                batch_inputs.size(0),
                81,
                9,
                device=device,
            )

            for iteration in range(1, max(horizons) + 1):
                iteration_alpha = 1.0 if iteration <= warmup_iterations else alpha
                hidden_state = relaxed_recurrent_step(
                    model,
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                    iteration_alpha,
                )
                logits = model.output_head(hidden_state)
                predictions = F.softmax(logits, dim=-1)
                if iteration not in horizon_set:
                    continue

                predicted_digits = logits.argmax(dim=-1)
                correct = predicted_digits.eq(batch_targets)
                solved = (correct | ~batch_empty_mask).all(dim=1)
                result = totals[iteration]
                result["solved"] += solved.sum().item()
                result["correct_empty_cells"] += (
                    correct & batch_empty_mask
                ).sum().item()
                result["empty_cells"] += batch_empty_mask.sum().item()
                result["state_rms_sum"] += (
                    hidden_state.float().square().mean(dim=-1).sqrt().mean().item()
                    * batch_inputs.size(0)
                )
                result["examples"] += batch_inputs.size(0)
                if bucket_names is not None:
                    batch_bucket_names = bucket_names[
                        start:start + batch_inputs.size(0)
                    ]
                    for name in bucket_order:
                        bucket_mask = torch.tensor(
                            [bucket_name == name for bucket_name in batch_bucket_names],
                            device=device,
                            dtype=torch.bool,
                        )
                        bucket_result = result["per_bucket"][name]
                        bucket_result["solved"] += (
                            solved & bucket_mask
                        ).sum().item()
                        bucket_result["total"] += bucket_mask.sum().item()

    return {
        str(horizon): {
            "solved": totals[horizon]["solved"],
            "total": totals[horizon]["examples"],
            "puzzle_accuracy": (
                totals[horizon]["solved"] / totals[horizon]["examples"]
            ),
            "empty_cell_accuracy": (
                totals[horizon]["correct_empty_cells"]
                / totals[horizon]["empty_cells"]
            ),
            "mean_token_rms": (
                totals[horizon]["state_rms_sum"]
                / totals[horizon]["examples"]
            ),
            "per_bucket": {
                name: {
                    "solved": bucket_result["solved"],
                    "total": bucket_result["total"],
                    "accuracy": (
                        bucket_result["solved"] / bucket_result["total"]
                    ),
                }
                for name, bucket_result in totals[horizon]["per_bucket"].items()
            },
        }
        for horizon in horizons
    }


def evaluate(
    model_configs=DEFAULT_MODEL_CONFIGS,
    *,
    examples_per_bucket=20,
    alphas=DEFAULT_ALPHAS,
    horizons=DEFAULT_HORIZONS,
    seed=20_260_720,
    output_dir=".",
    output_prefix="horizon_damping_pilot",
):
    alphas = tuple(float(alpha) for alpha in alphas)
    horizons = tuple(int(horizon) for horizon in horizons)
    if any(alpha <= 0 or alpha > 1 for alpha in alphas):
        raise ValueError("damping alphas must be in (0, 1]")
    if tuple(sorted(set(horizons))) != horizons or not horizons:
        raise ValueError("horizons must be strictly increasing")
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{output_prefix}.log")
    json_path = os.path.join(output_dir, f"{output_prefix}.json")

    with open(log_path, "w") as log_file:
        def log(message):
            print(message)
            log_file.write(message + "\n")
            log_file.flush()

        inputs, targets, empty_mask, _, _, bucket_names = _load_balanced_sample(
            examples_per_bucket,
            seed,
        )
        device = torch.device("cuda")
        results = {
            "metadata": {
                "examples": inputs.size(0),
                "examples_per_bucket": examples_per_bucket,
                "bucket_names": bucket_names,
                "alphas": alphas,
                "horizons": horizons,
                "seed": seed,
            },
            "models": {},
        }
        log(
            f"Horizon damping pilot: examples={inputs.size(0)}, "
            f"alphas={alphas}, horizons={horizons}"
        )

        for model_config in model_configs:
            name = model_config["name"]
            log(f"\n{name}: {model_config['path']}")
            model = _load_model(model_config, device)
            model.requires_grad_(False)
            model_results = {}
            for alpha in alphas:
                started = time.time()
                policy_results = evaluate_policy(
                    model,
                    inputs,
                    targets,
                    empty_mask,
                    alpha=alpha,
                    horizons=horizons,
                )
                model_results[f"{alpha:g}"] = policy_results
                summary = ", ".join(
                    f"h{horizon}={policy_results[str(horizon)]['solved']}"
                    for horizon in horizons
                )
                log(
                    f"alpha={alpha:g} | {summary}/{inputs.size(0)} | "
                    f"{time.time() - started:.1f}s"
                )
            results["models"][name] = model_results
            del model
            torch.cuda.empty_cache()

        with open(json_path, "w") as json_file:
            json.dump(results, json_file, indent=2)
        log(f"\nSaved {json_path}")
    return results


if __name__ == "__main__":
    evaluate()
