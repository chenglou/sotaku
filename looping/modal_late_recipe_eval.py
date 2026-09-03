"""Run the recommended late-state evaluation on a Modal checkpoint."""

import re

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-late-recipe-eval")

CHECKPOINT_KINDS = ("best", "final")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=PROJECT_IGNORE,
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_evaluation(
    run_name: str,
    checkpoint_kind: str,
    examples_per_bucket: int,
    alpha: float,
    warmup_iterations: int,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    from looping.eval_late_recipe import evaluate_checkpoint, make_damping_policy

    if checkpoint_kind == "best":
        model_path = f"{output_dir}/model_{run_name}_best_probe.pt"
        model_suffix = "best"
        checkpoint_prefix = run_name
    elif checkpoint_kind == "final":
        model_path = f"{output_dir}/model_{run_name}.pt"
        model_suffix = "final"
        checkpoint_prefix = f"{run_name}_final"
    else:
        raise ValueError(f"unknown checkpoint kind: {checkpoint_kind!r}")
    policy = make_damping_policy(alpha, warmup_iterations)
    if policy["name"] == "warm512_a025":
        output_prefix = f"{checkpoint_prefix}_recommended_damped"
    else:
        output_prefix = f"{checkpoint_prefix}_{policy['name']}_damped"
    try:
        return evaluate_checkpoint(
            model_path,
            model_name=f"{run_name}_{model_suffix}",
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
            output_prefix=output_prefix,
            alpha=alpha,
            warmup_iterations=warmup_iterations,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    run_name: str = "loop_late_random_replace_trial0",
    checkpoint_kind: str = "final",
    examples_per_bucket: int = 5000,
    alpha: float = 0.25,
    warmup_iterations: int = 512,
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")
    if checkpoint_kind not in CHECKPOINT_KINDS:
        choices = ", ".join(CHECKPOINT_KINDS)
        raise ValueError(
            f"unknown checkpoint kind {checkpoint_kind!r}; choose one of: {choices}"
        )
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be non-negative")
    call = run_evaluation.spawn(
        run_name,
        checkpoint_kind,
        examples_per_bucket,
        alpha,
        warmup_iterations,
    )
    print(f"Spawned recommended late-state evaluation: {call.object_id}")
    checkpoint_prefix = run_name if checkpoint_kind == "best" else f"{run_name}_final"
    alpha_label = f"{alpha:g}".replace(".", "")
    policy_name = f"warm{warmup_iterations}_a{alpha_label}"
    if policy_name == "warm512_a025":
        prefix = f"{checkpoint_prefix}_recommended_damped"
    else:
        prefix = f"{checkpoint_prefix}_{policy_name}_damped"
    print(
        f"Poll looping/{prefix}.log "
        "on sudoku-outputs for progress."
    )
