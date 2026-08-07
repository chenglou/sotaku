"""Run margin-floor calibration on Modal."""

import modal


app = modal.App("sudoku-margin-floor-calibration")

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
    batch_size: int,
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

    from looping.eval_margin_floor_calibration import evaluate

    try:
        return evaluate(
            examples_per_bucket=examples_per_bucket,
            batch_size=batch_size,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    output_prefix: str = "margin_floor_calibration_n1000_v1",
    examples_per_bucket: int = 200,
    batch_size: int = 100,
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    call = run_diagnostics.spawn(
        output_prefix,
        examples_per_bucket,
        batch_size,
    )
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(f"Poll looping/{output_prefix}.log on sudoku-outputs for progress.")
