"""Run the Sudoku-constraint trajectory collection on Modal."""

import re

import modal


app = modal.App("sudoku-trajectory-study-constraints")

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
def run_collection(
    examples_per_split_bucket: int,
    seed: int,
    output_name: str,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    study_directory = "/outputs/trajectory_study/03_constraints"
    sys.path.insert(0, "/root/project/looping/trajectory_viz/study/03_constraints")
    from collect_constraints import collect

    try:
        return collect(
            os.path.join(study_directory, f"{output_name}.pt"),
            examples_per_split_bucket=examples_per_split_bucket,
            seed=seed,
            device="cuda",
        )["config"]
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_split_bucket: int = 4,
    seed: int = 20260811,
    output_name: str = "constraints_states_v1",
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_name):
        raise ValueError(f"unsafe output name: {output_name!r}")
    call = run_collection.spawn(
        examples_per_split_bucket,
        seed,
        output_name,
    )
    print(f"Spawned {output_name}: {call.object_id}")
    print(
        "Poll trajectory_study/03_constraints/"
        f"{output_name}.log on sudoku-outputs for progress."
    )
