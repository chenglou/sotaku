"""Run the recurrent update decomposition on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-update-decomposition")

PRESET_NAMES = (
    "reference_models",
    "clean_a_trajectory",
    "healthy_methods",
)
POLICY_PRESET_NAMES = (
    "component_ablation",
    "measure_only",
)

hf_cache_volume = modal.Volume.from_name(
    "sudoku-hf-cache",
    create_if_missing=True,
)
outputs_volume = modal.Volume.from_name(
    "sudoku-outputs",
    create_if_missing=True,
)

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
def run_diagnostics(
    output_prefix: str,
    examples_per_bucket: int,
    batch_size: int,
    model_preset: str,
    policy_preset: str,
):
    import datetime
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    import torch

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
    ).stdout
    driver_line = next(
        (
            line.strip()
            for line in smi.splitlines()
            if "Driver Version" in line
        ),
        "nvidia-smi unavailable",
    )
    print(
        f"{timestamp.isoformat()} | {output_prefix} | "
        f"torch {torch.__version__}, cuda {torch.version.cuda} | "
        f"{driver_line}"
    )

    from looping.eval_update_decomposition import (
        MODEL_PRESETS,
        POLICY_PRESETS,
        evaluate,
    )

    try:
        return evaluate(
            model_configs=MODEL_PRESETS[model_preset],
            policies=POLICY_PRESETS[policy_preset],
            examples_per_bucket=examples_per_bucket,
            batch_size=batch_size,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    output_prefix: str = "update_decomposition_n1000_v1",
    examples_per_bucket: int = 200,
    batch_size: int = 100,
    model_preset: str = "reference_models",
    policy_preset: str = "component_ablation",
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_preset = model_preset.replace("-", "_")
    if model_preset not in PRESET_NAMES:
        choices = ", ".join(PRESET_NAMES)
        raise ValueError(
            f"unknown model preset {model_preset!r}; choose one of: {choices}"
        )
    policy_preset = policy_preset.replace("-", "_")
    if policy_preset not in POLICY_PRESET_NAMES:
        choices = ", ".join(POLICY_PRESET_NAMES)
        raise ValueError(
            f"unknown policy preset {policy_preset!r}; "
            f"choose one of: {choices}"
        )
    call = run_diagnostics.spawn(
        output_prefix,
        examples_per_bucket,
        batch_size,
        model_preset,
        policy_preset,
    )
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(f"Poll looping/{output_prefix}.log on sudoku-outputs for progress.")
