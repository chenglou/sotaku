"""Test normal recurrent updates followed by strong late-trajectory damping."""

import json
import os
import time

import torch

from looping.eval_horizon_damping import (
    DEFAULT_MODEL_CONFIGS,
    evaluate_policy,
)
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model


DEFAULT_POLICIES = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "constant_a025", "alpha": 0.25, "warmup_iterations": 0},
    {"name": "constant_a0125", "alpha": 0.125, "warmup_iterations": 0},
    {"name": "warm16_a025", "alpha": 0.25, "warmup_iterations": 16},
    {"name": "warm16_a0125", "alpha": 0.125, "warmup_iterations": 16},
    {"name": "warm16_a00625", "alpha": 0.0625, "warmup_iterations": 16},
    {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
    {"name": "warm128_a0125", "alpha": 0.125, "warmup_iterations": 128},
    {"name": "warm128_a00625", "alpha": 0.0625, "warmup_iterations": 128},
)
DEFAULT_HORIZONS = (16, 128, 1024, 2048, 4096)


def evaluate(
    model_configs=DEFAULT_MODEL_CONFIGS,
    *,
    policies=DEFAULT_POLICIES,
    examples_per_bucket=20,
    batch_size=100,
    horizons=DEFAULT_HORIZONS,
    seed=20_260_720,
    output_dir=".",
    output_prefix="delayed_damping_pilot",
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not policies:
        raise ValueError("at least one damping policy is required")

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
        results = {
            "metadata": {
                "examples": inputs.size(0),
                "examples_per_bucket": examples_per_bucket,
                "batch_size": batch_size,
                "bucket_names": list(dict.fromkeys(bucket_names)),
                "policies": policies,
                "horizons": horizons,
                "seed": seed,
            },
            "models": {},
        }
        log(
            f"Delayed damping pilot: examples={inputs.size(0)}, "
            f"policies={len(policies)}, horizons={horizons}"
        )

        device = torch.device("cuda")
        for model_config in model_configs:
            name = model_config["name"]
            log(f"\n{name}: {model_config['path']}")
            model = _load_model(model_config, device)
            model.requires_grad_(False)
            model_results = {}
            for policy in policies:
                started = time.time()
                policy_results = evaluate_policy(
                    model,
                    inputs,
                    targets,
                    empty_mask,
                    alpha=policy["alpha"],
                    warmup_iterations=policy["warmup_iterations"],
                    horizons=horizons,
                    batch_size=batch_size,
                    bucket_names=bucket_names,
                )
                model_results[policy["name"]] = policy_results
                summary = ", ".join(
                    f"h{horizon}={policy_results[str(horizon)]['solved']}"
                    for horizon in horizons
                )
                log(
                    f"{policy['name']} | {summary}/{inputs.size(0)} | "
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
