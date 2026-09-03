"""Run the strong recurrent-damping pilot on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-horizon-damping")

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
def run_pilot(output_prefix: str, examples_per_bucket: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    from looping.eval_horizon_damping import evaluate

    try:
        return evaluate(
            examples_per_bucket=examples_per_bucket,
            output_dir=output_dir,
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    output_prefix: str = "horizon_damping_pilot",
    examples_per_bucket: int = 20,
):
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    call = run_pilot.spawn(output_prefix, examples_per_bucket)
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(f"Poll looping/{output_prefix}.log on sudoku-outputs for progress.")
