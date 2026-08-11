"""Run held-out number-geometry statistics on Modal."""

import modal


app = modal.App("sudoku-number-helix-statistics")
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
def analyze(examples_per_bucket: int, permutations: int, seed: int):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root/project")

    outputs_volume.reload()
    from looping.trajectory_viz.helix_tests.statistics.analyze_number_helix import (
        run,
    )

    try:
        return run(
            output_dir=(
                "/outputs/trajectory_viz/helix_tests/statistics"
            ),
            examples_per_bucket=examples_per_bucket,
            permutations=permutations,
            seed=seed,
            device="cuda",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 20,
    permutations: int = 9999,
    seed: int = 20260807,
):
    call = analyze.spawn(examples_per_bucket, permutations, seed)
    print(f"Spawned number-helix statistics: {call.object_id}")
    print(
        "Poll trajectory_viz/helix_tests/statistics/run.log "
        "on sudoku-outputs for progress."
    )
