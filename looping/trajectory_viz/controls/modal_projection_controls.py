"""Run projection-artifact controls on Modal."""

import modal


app = modal.App("sudoku-trajectory-projection-controls")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.9.4")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "*.pt", "*.log"],
    )
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/outputs": outputs_volume, "/hf_cache": hf_cache_volume},
)
def analyze():
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    outputs_volume.reload()
    from looping.trajectory_viz.controls.analyze_projection_controls import run

    try:
        return run("/outputs/trajectory_viz/controls", device="cuda")
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main():
    call = analyze.spawn()
    print(f"Spawned projection controls: {call.object_id}")
