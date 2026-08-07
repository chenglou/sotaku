"""Causally separate recurrent state growth from recurrent direction change."""

import json
import os
import re
import time

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import (
    CLEAN_A_TRAJECTORY_MODELS,
    DEFAULT_MODELS,
    LATE_STATE_MODELS,
    _load_balanced_sample,
    _load_model,
)


DEFAULT_MODEL_CONFIGS = DEFAULT_MODELS[:2]
LATE_SWITCH_MODELS = (
    {
        "name": "late_switch_consistency_best",
        "path": (
            "/outputs/looping/"
            "model_loop_stay_late_switch_consistency_from39k_best_probe.pt"
        ),
        "model_kwargs": {},
    },
)
HEALTHY_METHOD_MODELS = (
    DEFAULT_MODELS[0],
    DEFAULT_MODELS[2],
    LATE_STATE_MODELS[0],
    *LATE_SWITCH_MODELS,
)
MODEL_PRESETS = {
    "reference_models": DEFAULT_MODEL_CONFIGS,
    "clean_a_trajectory": CLEAN_A_TRAJECTORY_MODELS,
    "healthy_methods": HEALTHY_METHOD_MODELS,
}
DEFAULT_HORIZONS = (128, 256, 512, 1024, 2048)
DEFAULT_POLICIES = (
    {
        "name": "undamped",
        "warmup_iterations": 0,
        "radial_alpha": 1.0,
        "tangential_alpha": 1.0,
    },
    {
        "name": "full_a025_after128",
        "warmup_iterations": 128,
        "radial_alpha": 0.25,
        "tangential_alpha": 0.25,
    },
    {
        "name": "radial_a025_after128",
        "warmup_iterations": 128,
        "radial_alpha": 0.25,
        "tangential_alpha": 1.0,
    },
    {
        "name": "tangential_a025_after128",
        "warmup_iterations": 128,
        "radial_alpha": 1.0,
        "tangential_alpha": 0.25,
    },
)
POLICY_PRESETS = {
    "component_ablation": DEFAULT_POLICIES,
    "measure_only": DEFAULT_POLICIES[:1],
}


def decompose_update(hidden_state, proposed_state, epsilon=1e-12):
    update = proposed_state - hidden_state
    denominator = hidden_state.float().square().sum(
        dim=-1,
        keepdim=True,
    ).clamp_min(epsilon)
    radial_coefficient = (
        update.float() * hidden_state.float()
    ).sum(dim=-1, keepdim=True) / denominator
    radial_update = (
        radial_coefficient.to(hidden_state.dtype) * hidden_state
    )
    tangential_update = update - radial_update
    return radial_update, tangential_update


def decomposed_recurrent_step(
    model,
    hidden_state,
    predictions,
    rope_cos,
    rope_sin,
    radial_alpha,
    tangential_alpha,
):
    proposed_state = model.apply_recurrent_updates(
        hidden_state,
        predictions,
        rope_cos,
        rope_sin,
    )
    radial_update, tangential_update = decompose_update(
        hidden_state,
        proposed_state,
    )
    next_state = (
        hidden_state
        + radial_alpha * radial_update
        + tangential_alpha * tangential_update
    )
    return (
        model.normalize_outer_state(next_state),
        radial_update,
        tangential_update,
    )


def _minimum_target_margin(logits, targets, empty_mask):
    target_logits = logits.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    other_logits = logits.masked_fill(
        F.one_hot(targets, num_classes=9).bool(),
        -torch.inf,
    )
    margins = target_logits - other_logits.max(dim=-1).values
    return margins.masked_fill(~empty_mask, torch.inf).min(dim=1).values


def _distribution(chunks):
    values = torch.cat(chunks).float()
    return {
        "mean": values.mean().item(),
        "p10": torch.quantile(values, 0.1).item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, 0.9).item(),
    }


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
            "state_direction_cosine_to_128": [],
            "local_state_direction_change": [],
            "minimum_target_margin": [],
            "directional_minimum_target_margin": [],
            "proposed_radial_update_rms": [],
            "proposed_tangential_update_rms": [],
            "applied_radial_update_rms": [],
            "applied_tangential_update_rms": [],
            "relative_applied_tangential_rms": [],
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
            reference_state = None

            for iteration in range(1, maximum_horizon + 1):
                previous_state = hidden_state
                if iteration <= policy["warmup_iterations"]:
                    radial_alpha = 1.0
                    tangential_alpha = 1.0
                else:
                    radial_alpha = policy["radial_alpha"]
                    tangential_alpha = policy["tangential_alpha"]
                (
                    hidden_state,
                    radial_update,
                    tangential_update,
                ) = decomposed_recurrent_step(
                    model,
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                    radial_alpha,
                    tangential_alpha,
                )
                applied_radial, applied_tangential = decompose_update(
                    previous_state,
                    hidden_state,
                )
                logits = model.output_head(hidden_state)
                predictions = F.softmax(logits, dim=-1)
                if iteration == 128:
                    reference_state = hidden_state.clone()
                if iteration not in horizon_set:
                    continue

                predicted_digits = logits.argmax(dim=-1)
                solved = (
                    (predicted_digits == batch_targets) | ~batch_empty
                ).all(dim=1)
                normalized_hidden = F.normalize(
                    hidden_state.float(),
                    dim=-1,
                )
                directional_logits = F.linear(
                    normalized_hidden,
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
                result["state_direction_cosine_to_128"].append(
                    F.cosine_similarity(
                        reference_state.float().flatten(1),
                        hidden_state.float().flatten(1),
                        dim=1,
                        eps=1e-12,
                    ).clamp(-1, 1).cpu()
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
                result["minimum_target_margin"].append(
                    _minimum_target_margin(
                        logits.float(),
                        batch_targets,
                        batch_empty,
                    ).cpu()
                )
                result["directional_minimum_target_margin"].append(
                    _minimum_target_margin(
                        directional_logits,
                        batch_targets,
                        batch_empty,
                    ).cpu()
                )
                result["proposed_radial_update_rms"].append(
                    radial_update.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt().cpu()
                )
                result["proposed_tangential_update_rms"].append(
                    tangential_update.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt().cpu()
                )
                applied_radial_rms = (
                    applied_radial.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt()
                )
                applied_tangential_rms = (
                    applied_tangential.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt()
                )
                previous_state_rms = (
                    previous_state.float().flatten(1).square().mean(
                        dim=1
                    ).sqrt()
                )
                result["applied_radial_update_rms"].append(
                    applied_radial_rms.cpu()
                )
                result["applied_tangential_update_rms"].append(
                    applied_tangential_rms.cpu()
                )
                result["relative_applied_tangential_rms"].append(
                    (
                        applied_tangential_rms
                        / previous_state_rms.clamp_min(1e-12)
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
                    "state_direction_cosine_to_128",
                    "local_state_direction_change",
                    "minimum_target_margin",
                    "directional_minimum_target_margin",
                    "proposed_radial_update_rms",
                    "proposed_tangential_update_rms",
                    "applied_radial_update_rms",
                    "applied_tangential_update_rms",
                    "relative_applied_tangential_rms",
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
    output_prefix="update_decomposition_n1000",
):
    horizons = tuple(horizons)
    if not horizons or tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be strictly increasing")
    if 128 not in horizons:
        raise ValueError("horizons must include the direction reference at 128")
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")
    for policy in policies:
        if policy["warmup_iterations"] < 0:
            raise ValueError("policy warmup must be non-negative")
        for key in ("radial_alpha", "tangential_alpha"):
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
            "decomposition": "per-token radial and tangential update",
        },
        "models": {},
    }

    with open(log_path, "w") as log_file:
        def log(message=""):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(
            f"Update decomposition: {inputs.size(0)} puzzles, "
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
        with open(temporary_path, "w") as json_file:
            json.dump(result, json_file, indent=2)
            json_file.write("\n")
        os.replace(temporary_path, json_path)
        log(f"\nStructured results: {json_path}")
    return result


if __name__ == "__main__":
    evaluate()
