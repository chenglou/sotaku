"""Locate which recurrent path causes harmful hidden-state direction changes."""

import json
import os
import re
import time

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import (
    DEFAULT_MODELS,
    _load_balanced_sample,
    _load_model,
)
from looping.eval_update_decomposition import (
    _distribution,
    _minimum_target_margin,
    decompose_update,
)


DEFAULT_MODEL_CONFIGS = DEFAULT_MODELS[:2]
DEFAULT_HORIZONS = (128, 256, 512, 1024, 2048)
DEFAULT_POLICIES = (
    {
        "name": "undamped",
        "warmup_iterations": 0,
        "feedback_tangential_alpha": 1.0,
        "layer_tangential_alpha": 1.0,
    },
    {
        "name": "feedback_tangent_a025_after128",
        "warmup_iterations": 128,
        "feedback_tangential_alpha": 0.25,
        "layer_tangential_alpha": 1.0,
    },
    {
        "name": "layers_tangent_a025_after128",
        "warmup_iterations": 128,
        "feedback_tangential_alpha": 1.0,
        "layer_tangential_alpha": 0.25,
    },
    {
        "name": "all_sources_tangent_a025_after128",
        "warmup_iterations": 128,
        "feedback_tangential_alpha": 0.25,
        "layer_tangential_alpha": 0.25,
    },
)


def _apply_tangential_scale(hidden_state, proposed_state, alpha):
    radial_update, tangential_update = decompose_update(
        hidden_state,
        proposed_state,
    )
    next_state = hidden_state + radial_update + alpha * tangential_update
    return next_state, radial_update, tangential_update


def source_decomposed_recurrent_step(
    model,
    hidden_state,
    predictions,
    rope_cos,
    rope_sin,
    *,
    feedback_tangential_alpha,
    layer_tangential_alpha,
):
    feedback_proposal = (
        hidden_state
        + model.feedback_scale * model.pred_proj(predictions)
    )
    (
        hidden_state,
        feedback_radial,
        feedback_tangential,
    ) = _apply_tangential_scale(
        hidden_state,
        feedback_proposal,
        feedback_tangential_alpha,
    )
    layer_radial_updates = []
    layer_tangential_updates = []
    for layer_index in model.layer_schedule:
        layer_proposal = model.layers[layer_index](
            hidden_state,
            rope_cos,
            rope_sin,
        )
        (
            hidden_state,
            layer_radial,
            layer_tangential,
        ) = _apply_tangential_scale(
            hidden_state,
            layer_proposal,
            layer_tangential_alpha,
        )
        layer_radial_updates.append(layer_radial)
        layer_tangential_updates.append(layer_tangential)
    return (
        model.normalize_outer_state(hidden_state),
        feedback_radial,
        feedback_tangential,
        layer_radial_updates,
        layer_tangential_updates,
    )


def _stacked_update_rms(updates):
    per_layer_mean_squares = torch.stack([
        update.float().flatten(1).square().mean(dim=1)
        for update in updates
    ])
    return per_layer_mean_squares.mean(dim=0).sqrt()


def evaluate_policy(
    model,
    inputs,
    targets,
    empty_mask,
    *,
    policy,
    horizons,
    batch_size,
):
    device = next(model.parameters()).device
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    horizon_set = set(horizons)
    maximum_horizon = max(horizons)
    measurements = {
        horizon: {
            "solved": 0,
            "examples": 0,
            "state_rms": [],
            "local_state_direction_change": [],
            "directional_minimum_target_margin": [],
            "feedback_tangential_update_rms": [],
            "layer_tangential_update_rms": [],
        }
        for horizon in horizons
    }

    with torch.inference_mode(), torch.autocast(
        "cuda",
        dtype=torch.bfloat16,
    ):
        for batch_start in range(0, inputs.size(0), batch_size):
            batch_end = min(batch_start + batch_size, inputs.size(0))
            batch_inputs = inputs[batch_start:batch_end].to(device)
            batch_targets = targets[batch_start:batch_end].to(device)
            batch_empty = empty_mask[batch_start:batch_end].to(device)
            hidden_state = model.initial_encoder(batch_inputs)
            predictions = torch.zeros(
                batch_inputs.size(0),
                81,
                9,
                device=device,
            )

            for iteration in range(1, maximum_horizon + 1):
                previous_state = hidden_state
                after_warmup = iteration > policy["warmup_iterations"]
                feedback_alpha = (
                    policy["feedback_tangential_alpha"]
                    if after_warmup
                    else 1.0
                )
                layer_alpha = (
                    policy["layer_tangential_alpha"]
                    if after_warmup
                    else 1.0
                )
                (
                    hidden_state,
                    _,
                    feedback_tangential,
                    _,
                    layer_tangential_updates,
                ) = source_decomposed_recurrent_step(
                    model,
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                    feedback_tangential_alpha=feedback_alpha,
                    layer_tangential_alpha=layer_alpha,
                )
                logits = model.output_head(hidden_state)
                predictions = F.softmax(logits, dim=-1)
                if iteration not in horizon_set:
                    continue

                predicted_digits = logits.argmax(dim=-1)
                solved = (
                    (predicted_digits == batch_targets) | ~batch_empty
                ).all(dim=1)
                directional_logits = F.linear(
                    F.normalize(hidden_state.float(), dim=-1),
                    model.output_head.weight.float(),
                    bias=None,
                )
                result = measurements[iteration]
                result["solved"] += int(solved.sum().item())
                result["examples"] += batch_inputs.size(0)
                result["state_rms"].append(
                    hidden_state.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt().cpu()
                )
                result["local_state_direction_change"].append(
                    (
                        1
                        - F.cosine_similarity(
                            previous_state.float().flatten(1),
                            hidden_state.float().flatten(1),
                            dim=1,
                            eps=1e-12,
                        ).clamp(-1, 1)
                    ).cpu()
                )
                result["directional_minimum_target_margin"].append(
                    _minimum_target_margin(
                        directional_logits,
                        batch_targets,
                        batch_empty,
                    ).cpu()
                )
                result["feedback_tangential_update_rms"].append(
                    feedback_tangential.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt().cpu()
                )
                result["layer_tangential_update_rms"].append(
                    _stacked_update_rms(
                        layer_tangential_updates,
                    ).cpu()
                )

    return {
        str(horizon): {
            "solved": measurements[horizon]["solved"],
            "total": measurements[horizon]["examples"],
            "puzzle_accuracy": (
                measurements[horizon]["solved"]
                / measurements[horizon]["examples"]
            ),
            **{
                key: _distribution(measurements[horizon][key])
                for key in (
                    "state_rms",
                    "local_state_direction_change",
                    "directional_minimum_target_margin",
                    "feedback_tangential_update_rms",
                    "layer_tangential_update_rms",
                )
            },
        }
        for horizon in horizons
    }


def evaluate(
    model_configs=DEFAULT_MODEL_CONFIGS,
    *,
    policies=DEFAULT_POLICIES,
    horizons=DEFAULT_HORIZONS,
    examples_per_bucket=200,
    batch_size=100,
    seed=42,
    output_dir=".",
    output_prefix="update_source_decomposition_n1000",
):
    horizons = tuple(horizons)
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be strictly increasing")
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")
    for policy in policies:
        if policy["warmup_iterations"] < 0:
            raise ValueError("policy warmup must be non-negative")
        for key in (
            "feedback_tangential_alpha",
            "layer_tangential_alpha",
        ):
            if not 0 < policy[key] <= 1:
                raise ValueError(f"{key} must be in (0, 1]")

    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"{output_prefix}.log")
    json_path = os.path.join(output_dir, f"{output_prefix}.json")
    inputs, targets, empty_mask, _, _, _ = _load_balanced_sample(
        examples_per_bucket,
        seed,
    )
    device = torch.device("cuda")
    result = {
        "config": {
            "models": list(model_configs),
            "policies": list(policies),
            "horizons": list(horizons),
            "examples_per_bucket": examples_per_bucket,
            "sample_size": inputs.size(0),
            "batch_size": batch_size,
            "seed": seed,
            "decomposition": (
                "prediction-feedback versus transformer-layer "
                "tangential updates"
            ),
        },
        "models": {},
    }

    with open(log_path, "w") as log_file:
        def log(message=""):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(
            f"Update-source decomposition: {inputs.size(0)} puzzles, "
            f"models={len(model_configs)}, policies={len(policies)}"
        )
        for model_config in model_configs:
            model_name = model_config["name"]
            log(f"\nMODEL {model_name}: {model_config['path']}")
            model = _load_model(model_config, device)
            model.requires_grad_(False)
            model_results = {}
            for policy in policies:
                started_at = time.time()
                policy_results = evaluate_policy(
                    model,
                    inputs,
                    targets,
                    empty_mask,
                    policy=policy,
                    horizons=horizons,
                    batch_size=batch_size,
                )
                model_results[policy["name"]] = policy_results
                accuracy_text = ", ".join(
                    f"h{horizon}="
                    f"{100 * policy_results[str(horizon)]['puzzle_accuracy']:.1f}%"
                    for horizon in horizons
                )
                log(
                    f"  {policy['name']}: {accuracy_text} "
                    f"({time.time() - started_at:.1f}s)"
                )
            result["models"][model_name] = model_results
            del model
            torch.cuda.empty_cache()

        temporary_path = json_path + ".tmp"
        with open(temporary_path, "w") as result_file:
            json.dump(result, result_file, indent=2)
            result_file.write("\n")
        os.replace(temporary_path, json_path)
        log(f"\nStructured results: {json_path}")
    return result


if __name__ == "__main__":
    evaluate()
