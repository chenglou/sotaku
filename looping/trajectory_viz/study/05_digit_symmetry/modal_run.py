"""Run study arm 05 on Modal using one detached spawned worker."""

import modal


app = modal.App("sudoku-study-05-digit-symmetry")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.9.4")
    .add_local_file(
        "checkpoint_utils.py", remote_path="/root/project/checkpoint_utils.py"
    )
    .add_local_file(
        "iters/state_norm.py", remote_path="/root/project/iters/state_norm.py"
    )
    .add_local_file(
        "stabilize/exp_testbed_20k.py",
        remote_path="/root/project/stabilize/exp_testbed_20k.py",
    )
    .add_local_file(
        "looping/eval_loop_diagnostics.py",
        remote_path="/root/project/looping/eval_loop_diagnostics.py",
    )
    .add_local_file(
        "looping/eval_trajectory_geometry.py",
        remote_path="/root/project/looping/eval_trajectory_geometry.py",
    )
    .add_local_file(
        "looping/trajectory_viz/helix_tests/statistics/trajectory_data.py",
        remote_path=(
            "/root/project/looping/trajectory_viz/helix_tests/statistics/"
            "trajectory_data.py"
        ),
    )
    .add_local_dir(
        "looping/trajectory_viz/study/05_digit_symmetry",
        remote_path=(
            "/root/project/looping/trajectory_viz/study/05_digit_symmetry"
        ),
        ignore=["__pycache__/", "*.pyc", "*.log", "*.png", "*.html", "*.json"],
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
    examples_per_bucket: int,
    seed: int,
    shuffle_count: int,
    random_subspace_count: int,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    sys.path.insert(0, "/root/project")
    sys.path.insert(
        0, "/root/project/looping/trajectory_viz/study/05_digit_symmetry"
    )

    outputs_volume.reload()
    from run_analysis import run

    try:
        return run(
            output_dir="/outputs/trajectory_viz/study/05_digit_symmetry",
            examples_per_bucket=examples_per_bucket,
            seed=seed,
            shuffle_count=shuffle_count,
            random_subspace_count=random_subspace_count,
            device="cuda",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 12,
    seed: int = 20260811,
    shuffle_count: int = 199,
    random_subspace_count: int = 99,
):
    call = analyze.spawn(
        examples_per_bucket,
        seed,
        shuffle_count,
        random_subspace_count,
    )
    print(f"Spawned digit-symmetry study: {call.object_id}")
    print(
        "Poll trajectory_viz/study/05_digit_symmetry/run.log "
        "on sudoku-outputs for progress."
    )
