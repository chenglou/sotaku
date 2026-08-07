"""Run recurrent trajectory-geometry diagnostics on Modal."""

import modal


app = modal.App("sudoku-trajectory-geometry")

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
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_analysis(examples_per_bucket: int, output_prefix: str):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    from looping.eval_trajectory_geometry import evaluate

    try:
        return evaluate(
            examples_per_bucket=examples_per_bucket,
            device="cuda",
            output_dir="/outputs/trajectory_geometry",
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 10,
    output_prefix: str = "trajectory_geometry_v1",
):
    call = run_analysis.spawn(examples_per_bucket, output_prefix)
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(
        "Poll trajectory_geometry/"
        f"{output_prefix}.log on sudoku-outputs for progress."
    )
