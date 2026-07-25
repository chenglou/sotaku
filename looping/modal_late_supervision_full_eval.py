"""Run a full 25K evaluation of one late-supervision checkpoint on Modal."""

import re

import modal


app = modal.App("sudoku-late-supervision-full-eval")

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
def run_evaluation(
    run_name: str,
    arm: str,
    outer_state_norm: bool,
    examples_per_bucket: int,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    from looping.eval_late_supervision_full import evaluate_run

    try:
        return evaluate_run(
            run_name,
            arm=arm.replace("-", "_") or None,
            outer_state_norm=outer_state_norm,
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    run_name: str = "loop_late_random_replace_trial0",
    arm: str = "",
    outer_state_norm: bool = False,
    examples_per_bucket: int = 5000,
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    call = run_evaluation.spawn(
        run_name,
        arm,
        outer_state_norm,
        examples_per_bucket,
    )
    print(f"Spawned full late-supervision evaluation: {call.object_id}")
    print(
        f"Poll looping/{run_name}_best_full_horizon.log "
        "on sudoku-outputs for progress."
    )
