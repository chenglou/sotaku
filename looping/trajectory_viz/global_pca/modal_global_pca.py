"""Run global recurrent-trajectory PCA on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-trajectory-global-pca")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.10.5")
    .add_local_dir(".", remote_path="/root/project", ignore=PROJECT_IGNORE)
)


@app.function(
    image=image,
    gpu="H200",
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/outputs": outputs_volume, "/hf_cache": hf_cache_volume},
)
def analyze(examples_per_bucket: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    outputs_volume.reload()
    from looping.trajectory_viz.global_pca.analyze_global_pca import run

    try:
        return run(
            "/outputs/trajectory_viz/global_pca",
            examples_per_bucket=examples_per_bucket,
            device="cuda",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(examples_per_bucket: int = 2):
    call = analyze.spawn(examples_per_bucket)
    print(f"Spawned global PCA: {call.object_id}")
