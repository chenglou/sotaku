"""Confirm the two strongest delayed-damping policies on 1,000 puzzles."""

import modal


app = modal.App("sudoku-delayed-damping-confirm")

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
def run_confirmation(output_prefix: str, examples_per_bucket: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    from looping.eval_delayed_damping import evaluate

    policies = (
        {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
        {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
        {"name": "warm128_a0125", "alpha": 0.125, "warmup_iterations": 128},
    )
    try:
        return evaluate(
            policies=policies,
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    output_prefix: str = "delayed_damping_confirm_n1000",
    examples_per_bucket: int = 200,
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    call = run_confirmation.spawn(output_prefix, examples_per_bucket)
    print(f"Spawned delayed-damping confirmation: {call.object_id}")
    print(f"Poll looping/{output_prefix}.log on sudoku-outputs for progress.")
