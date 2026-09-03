"""Run the full cap-1 recurrent-state evaluation on Modal.

Usage:
    modal run --detach stabilize/modal_testbed_outer_cap1_eval.py
"""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-testbed-outer-cap1-eval")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(
        ".",
        remote_path="/root/project",
        ignore=PROJECT_IGNORE,
    )
)


@app.function(
    image=image,
    gpu="H200",
    timeout=4 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_evaluation(output_prefix: str):
    import datetime
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    import torch

    outputs_volume.reload()
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    driver_line = next(
        (line.strip() for line in smi.splitlines() if "Driver Version" in line),
        "nvidia-smi unavailable",
    )
    environment_line = (
        f"{timestamp.isoformat()} | {output_prefix} | torch {torch.__version__}, "
        f"cuda {torch.version.cuda} | {driver_line}"
    )
    print(environment_line)
    os.makedirs("/outputs/env_runs", exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    with open(f"/outputs/env_runs/{stamp}_{output_prefix}.log", "w") as log_file:
        log_file.write(environment_line + "\n")

    from stabilize.eval_testbed_outer_cap1 import evaluate

    try:
        return evaluate(output_dir="/outputs", output_prefix=output_prefix)
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(output_prefix: str = "testbed_outer_cap1_trial0_full_horizon"):
    call = run_evaluation.spawn(output_prefix=output_prefix)
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(f"Poll {output_prefix}.log on the sudoku-outputs volume for progress.")
