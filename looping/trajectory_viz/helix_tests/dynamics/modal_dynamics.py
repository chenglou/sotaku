"""Run held-out per-cell trajectory dynamics on Modal."""

import modal


app = modal.App("sudoku-per-cell-helix-dynamics")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.10.5")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=[
            "venv/",
            "__pycache__/",
            "*.pyc",
            ".git/",
            "logs/",
            "runs/",
            "runs_modal/",
            "*.pt",
            "*.log",
            "*.png",
            "*.json",
            "*.npz",
            "*.html",
        ],
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,
    memory=32768,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/outputs": outputs_volume, "/hf_cache": hf_cache_volume},
)
def analyze(examples_per_bucket: int, seed: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root/project")
    outputs_volume.reload()
    from looping.trajectory_viz.helix_tests.dynamics.analyze_dynamics import run

    try:
        return run(
            "/outputs/trajectory_viz/helix_tests/dynamics",
            examples_per_bucket=examples_per_bucket,
            seed=seed,
            device="cuda",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(examples_per_bucket: int = 10, seed: int = 20260807):
    call = analyze.spawn(examples_per_bucket, seed)
    print(f"Spawned per-cell helix dynamics: {call.object_id}")
    print(
        "Use `modal container list`, then `modal container exec <id> cat "
        "/outputs/trajectory_viz/helix_tests/dynamics/progress.log` for progress."
    )
