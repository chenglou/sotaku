"""Run the cell-role geometry study on Modal."""

import re

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-study-cell-roles")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.9.4")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=PROJECT_IGNORE + ["looping/trajectory_viz/study/06_cell_roles/artifacts/"],
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
def analyze(
    run_name: str,
    examples_per_bucket: int,
    seed: int,
    split_seed: int,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = (
        "/outputs/trajectory_viz/study/06_cell_roles/.mplconfig"
    )
    sys.path.insert(0, "/root/project")
    sys.path.insert(
        0,
        "/root/project/looping/trajectory_viz/study/06_cell_roles",
    )
    outputs_volume.reload()
    from run_analysis import run

    output_dir = f"/outputs/trajectory_viz/study/06_cell_roles/{run_name}"
    try:
        return run(
            output_dir,
            examples_per_bucket=examples_per_bucket,
            seed=seed,
            split_seed=split_seed,
            device="cuda",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    run_name: str = "cell_roles_v1",
    examples_per_bucket: int = 12,
    seed: int = 20260811,
    split_seed: int = 20260812,
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")
    call = analyze.spawn(run_name, examples_per_bucket, seed, split_seed)
    print(f"Spawned {run_name}: {call.object_id}")
    print(
        "Poll trajectory_viz/study/06_cell_roles/"
        f"{run_name}/run.log on sudoku-outputs."
    )
