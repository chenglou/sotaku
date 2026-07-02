"""
Modal wrapper for running experiments on GPU.

Usage:
    modal run --detach modal_run.py --exp iters.exp_baseline_lr2e3

Outputs (checkpoints, logs) are saved to a Modal volume.
"""

import modal

app = modal.App("sudoku-solver")

# Volume for HuggingFace cache (persists across runs, avoids re-downloading)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)

# Volume for outputs (checkpoints, logs)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", "__pycache__/", "*.pyc", ".git/", "logs/", "runs/", "runs_modal/", "*.pt", "*.log"])
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,  # more cores for data loading with mp.Pool
    timeout=24 * 60 * 60,  # 24 hours (max)
    # Restart on worker loss instead of dying; experiments auto-resume from their
    # latest checkpoint in /outputs at function start.
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_training(
    exp_name: str,
):
    import os
    import sys
    import importlib

    # Point HuggingFace to the cached volume
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"

    sys.path.insert(0, "/root/project")

    # A reused container (e.g. a retry after worker loss) needs an explicit refresh
    # before find_latest_checkpoint() inspects the volume.
    outputs_volume.reload()
    try:
        # Dynamically import the experiment module
        exp_module = importlib.import_module(exp_name)

        result = exp_module.train(output_dir="/outputs")

        print(f"\nResult: {result}")
        print("\nOutputs saved to 'sudoku-outputs' volume.")
        print("Run 'modal volume ls sudoku-outputs' to see files.")
        print("Run 'modal volume get sudoku-outputs <filename>' to download.")

        return result
    finally:
        # Persist the latest checkpoint/log writes promptly, including on errors.
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    exp: str = "iters.exp_baseline_lr2e3",
):
    # spawn (fire-and-forget), not .remote(): a .remote() client holds a live connection
    # for the whole run, and when that connection breaks (laptop sleep, network blip)
    # the exiting client CANCELS the input — even under `modal run --detach`. Three
    # training runs died this way on 2026-07-02. With spawn, no client needs to stay up.
    print(f"Running experiment: {exp}")
    call = run_training.spawn(exp_name=exp)
    print(f"Spawned function call: {call.object_id}")
    print("Training continues server-side; poll the experiment's .log on the sudoku-outputs volume for progress.")
