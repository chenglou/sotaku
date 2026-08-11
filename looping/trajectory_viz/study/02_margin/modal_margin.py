"""Detached Modal entrypoint for ARM 02.

Launch this wrapper only as a detached command from the repository root:

    modal run --detach looping/trajectory_viz/study/02_margin/modal_margin.py

The local entrypoint makes exactly one ``spawn`` call, so the worker survives
after the detached entrypoint exits.  Aggregate artifacts are committed to the
``sudoku-outputs`` volume under ``trajectory_study/02_margin``.
"""

import modal


app = modal.App("sudoku-trajectory-study-margin")
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
    examples_per_split_bucket: int,
    controls: int,
    iteration_permutations: int,
    bootstrap: int,
    seed: int,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root/project")
    sys.path.insert(
        0, "/root/project/looping/trajectory_viz/study/02_margin"
    )

    outputs_volume.reload()
    from analyze_margin import run

    try:
        metrics = run(
            "/outputs/trajectory_viz/study/02_margin/ordered_probe_v1",
            examples_per_split_bucket=examples_per_split_bucket,
            label_and_projection_repetitions=controls,
            iteration_permutations=iteration_permutations,
            bootstrap_repetitions=bootstrap,
            seed=seed,
            device="cuda",
        )
        return {
            "selected_target": metrics["selection"]["selected_target"],
            "elapsed_seconds": metrics["elapsed_seconds"],
        }
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_split_bucket: int = 4,
    controls: int = 99,
    iteration_permutations: int = 199,
    bootstrap: int = 500,
    seed: int = 20260811,
):
    call = analyze.spawn(
        examples_per_split_bucket,
        controls,
        iteration_permutations,
        bootstrap,
        seed,
    )
    print(f"Spawned ARM 02 margin analysis: {call.object_id}")
    print(
        "Poll trajectory_viz/study/02_margin/ordered_probe_v1/run.log "
        "on sudoku-outputs; "
        "the detached local entrypoint finishing does not imply worker completion."
    )
