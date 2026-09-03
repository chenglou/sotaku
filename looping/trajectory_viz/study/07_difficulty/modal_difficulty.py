"""Run difficulty-geometry study arm 07 on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-geometry-study-07-difficulty")
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
    cpu=8.0,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10),
    volumes={"/outputs": outputs, "/hf_cache": cache},
)
def analyze(per_split_bucket: int, seed: int, output_name: str):
    import importlib
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    outputs.reload()
    run = importlib.import_module(
        "looping.trajectory_viz.study.07_difficulty.analyze_difficulty"
    ).run

    try:
        return run(
            f"/outputs/trajectory_viz/study/07_difficulty/{output_name}",
            per_split_bucket=per_split_bucket,
            seed=seed,
            device="cuda",
        )
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(per_split_bucket: int = 4, seed: int = 7027, output_name: str = "protocol_v1"):
    call = analyze.spawn(per_split_bucket, seed, output_name)
    print(f"Spawned study 07: {call.object_id}")
    print(f"Poll trajectory_viz/study/07_difficulty/{output_name} on sudoku-outputs")
