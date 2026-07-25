"""Run the loop-gradient and local Jacobian diagnostics on Modal."""

import modal


app = modal.App("sudoku-loop-diagnostics")

PRESET_NAMES = (
    "reference_models",
    "clean_a_trajectory",
    "late_state_models",
)

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "venv/", "__pycache__/", "*.pyc", ".git/", "logs/", "runs/",
            "runs_modal/", "*.pt", "*.log",
        ],
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
    model_preset: str,
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
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    driver_line = next(
        (line.strip() for line in smi.splitlines() if "Driver Version" in line),
        "nvidia-smi unavailable",
    )
    environment_line = (
        f"{timestamp.isoformat()} | {output_prefix} | torch {torch.__version__}, "
        f"cuda {torch.version.cuda} | {driver_line}"
    )
    print(environment_line)
    os.makedirs("/outputs/env_runs", exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    with open(
        f"/outputs/env_runs/{stamp}_{output_prefix}.log",
        "w",
    ) as environment_log:
        environment_log.write(environment_line + "\n")

    from looping.eval_loop_diagnostics import MODEL_PRESETS, evaluate

    try:
        return evaluate(
            model_configs=MODEL_PRESETS[model_preset],
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    output_prefix: str = "loop_diagnostics_v1",
    examples_per_bucket: int = 4,
    model_preset: str = "reference_models",
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    model_preset = model_preset.replace("-", "_")
    if model_preset not in PRESET_NAMES:
        choices = ", ".join(PRESET_NAMES)
        raise ValueError(
            f"unknown model preset {model_preset!r}; choose one of: {choices}"
        )
    call = run_diagnostics.spawn(
        output_prefix=output_prefix,
        examples_per_bucket=examples_per_bucket,
        model_preset=model_preset,
    )
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(f"Poll looping/{output_prefix}.log on sudoku-outputs for progress.")
