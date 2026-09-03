"""Run temporal-mode analysis on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-trajectory-study-08-temporal-modes")
outputs = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
cache = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.10.5")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=PROJECT_IGNORE,
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10),
    volumes={"/outputs": outputs, "/hf_cache": cache},
)
def analyze():
    import importlib
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    outputs.reload()
    run = importlib.import_module(
        "looping.trajectory_viz.study.08_temporal_modes.analysis"
    ).run

    try:
        return run("/outputs/trajectory_viz/study/08_temporal_modes", device="cuda")
    finally:
        outputs.commit()


@app.local_entrypoint()
def main():
    call = analyze.spawn()
    print(f"Spawned temporal-mode study: {call.object_id}")
