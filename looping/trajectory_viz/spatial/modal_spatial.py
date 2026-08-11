"""Run spatial trajectory rendering on Modal."""

import modal

app = modal.App("sudoku-trajectory-spatial-viz")
outputs = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
cache = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.10.5")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", ".git/", "*.pt"])
)


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10),
    volumes={"/outputs": outputs, "/hf_cache": cache},
)
def run(examples_per_bucket: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    outputs.reload()
    from looping.trajectory_viz.spatial.render_spatial import render

    try:
        return render(
            "/outputs/trajectory_viz/spatial",
            examples_per_bucket=examples_per_bucket,
        )
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(examples_per_bucket: int = 2):
    call = run.spawn(examples_per_bucket)
    print(f"Spawned spatial visualization: {call.object_id}")
