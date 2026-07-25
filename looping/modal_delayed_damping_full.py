"""Run full delayed-damping evaluation for one checkpoint on Modal."""

import modal


app = modal.App("sudoku-delayed-damping-full")

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
def run_evaluation(model_name: str, examples_per_bucket: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    from looping.eval_delayed_damping_full import evaluate_model

    try:
        return evaluate_model(
            model_name,
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = "collapsed_unbounded",
    examples_per_bucket: int = 5000,
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    call = run_evaluation.spawn(model_name, examples_per_bucket)
    print(f"Spawned full delayed-damping evaluation: {call.object_id}")
    print(
        f"Poll looping/delayed-damping-full-{model_name}.log "
        "on sudoku-outputs for progress."
    )
