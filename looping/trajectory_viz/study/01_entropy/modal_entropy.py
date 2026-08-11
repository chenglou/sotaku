"""Run the held-out uncertainty-coordinate study on Modal."""

import modal


app = modal.App("sudoku-trajectory-entropy-study")
outputs_volume = modal.Volume.from_name(
    "sudoku-outputs",
    create_if_missing=True,
)
hf_cache_volume = modal.Volume.from_name(
    "sudoku-hf-cache",
    create_if_missing=True,
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.9.4")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "venv/",
            "__pycache__/",
            "*.pyc",
            ".git/",
            "*.pt",
            "*.log",
            "*.png",
            "*.html",
            "*.npz",
            "*.csv",
            "looping/trajectory_viz/study/**",
            "!looping/trajectory_viz/study/01_entropy/**",
        ],
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=16.0,
    memory=65536,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/outputs": outputs_volume,
        "/hf_cache": hf_cache_volume,
    },
)
def analyze(
    examples_per_bucket: int,
    control_count: int,
    seed: int,
    run_name: str,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root/project")
    sys.path.insert(
        0,
        "/root/project/looping/trajectory_viz/study/01_entropy",
    )

    outputs_volume.reload()
    from run_entropy_study import run

    try:
        return run(
            output_root="/outputs/trajectory_viz/study/01_entropy",
            run_name=run_name,
            examples_per_bucket=examples_per_bucket,
            control_count=control_count,
            seed=seed,
            device="cuda",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 12,
    control_count: int = 64,
    seed: int = 20260811,
    run_name: str = "entropy_v1_20260811",
):
    call = analyze.spawn(
        examples_per_bucket,
        control_count,
        seed,
        run_name,
    )
    print(f"Spawned entropy study {run_name}: {call.object_id}")
    print(
        "Poll trajectory_viz/study/01_entropy/"
        f"{run_name}/run.log on sudoku-outputs for progress."
    )
