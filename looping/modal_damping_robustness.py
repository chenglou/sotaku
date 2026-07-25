"""Launch one delayed-damping robustness experiment on Modal."""

import modal


app = modal.App("sudoku-damping-robustness")


def normalize_experiment_name(experiment_name):
    return experiment_name.replace("-", "_")

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
def run_remote(experiment_name: str, examples_per_bucket: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    from looping.eval_damping_robustness import run_experiment

    try:
        return run_experiment(
            experiment_name,
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(experiment_name: str = "trajectory", examples_per_bucket: int = 200):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    experiment_name = normalize_experiment_name(experiment_name)
    call = run_remote.spawn(experiment_name, examples_per_bucket)
    print(f"Spawned damping {experiment_name}: {call.object_id}")
    print("Poll looping/delayed-damping-*.log on sudoku-outputs for progress.")
