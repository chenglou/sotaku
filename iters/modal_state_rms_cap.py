"""Run the recurrent-state RMS-cap diagnostic on Modal.

Usage:
    modal run --detach iters/modal_state_rms_cap.py
"""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-state-rms-cap")

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
def run_probe(
    examples_per_bucket: int,
    batch_size: int,
    caps_csv: str,
    output_prefix: str,
):
    import datetime
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    from iters.eval_state_rms_cap import evaluate

    caps = tuple(float(value) for value in caps_csv.split(","))

    outputs_volume.reload()
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    driver_line = next(
        (line.strip() for line in smi.splitlines() if "Driver Version" in line),
        "nvidia-smi unavailable",
    )
    import torch
    environment_line = (
        f"{timestamp.isoformat()} | {output_prefix} | torch {torch.__version__}, "
        f"cuda {torch.version.cuda} | {driver_line}"
    )
    print(environment_line)
    os.makedirs("/outputs/env_runs", exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    with open(f"/outputs/env_runs/{stamp}_{output_prefix}.log", "w") as log_file:
        log_file.write(environment_line + "\n")
    try:
        return evaluate(
            examples_per_bucket=examples_per_bucket,
            batch_size=batch_size,
            caps=(None, *caps),
            device="cuda",
            output_dir="/outputs",
            output_prefix=output_prefix,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 200,
    batch_size: int = 250,
    caps: str = "32,64,128",
    output_prefix: str = "state_rms_cap_probe",
):
    call = run_probe.spawn(
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
        caps_csv=caps,
        output_prefix=output_prefix,
    )
    print(f"Spawned {output_prefix}: {call.object_id}")
    print(f"Poll {output_prefix}.log on the sudoku-outputs volume for progress.")
