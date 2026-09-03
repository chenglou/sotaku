"""Run the held-out progress-coordinate study on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-progress-coordinate-study")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.10.5")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=PROJECT_IGNORE + ["*.png", "*.json", "*.csv", "*.npz", "*.html"],
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
    examples_per_bucket: int,
    seed: int,
    control_repetitions: int,
    random_directions: int,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root/project")
    sys.path.insert(
        0, "/root/project/looping/trajectory_viz/study/04_progress"
    )
    outputs_volume.reload()
    from analyze_progress import run

    try:
        return run(
            "/outputs/trajectory_viz/study/04_progress",
            examples_per_bucket=examples_per_bucket,
            seed=seed,
            device="cuda",
            control_repetitions=control_repetitions,
            random_directions=random_directions,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 12,
    seed: int = 20260811,
    control_repetitions: int = 32,
    random_directions: int = 64,
):
    call = analyze.spawn(
        examples_per_bucket,
        seed,
        control_repetitions,
        random_directions,
    )
    print(f"Spawned progress-coordinate study: {call.object_id}")
    print(
        "Use `modal container list`, then `modal container exec <id> cat "
        "/outputs/trajectory_viz/study/04_progress/progress.log` for progress."
    )
