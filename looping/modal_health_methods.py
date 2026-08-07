"""Launch one standalone trajectory-health method on Modal."""

import modal


app = modal.App("sudoku-health-methods")

ARM_NAMES = (
    "vanilla",
    "rmsnorm",
    "late_state_ce",
    "consistency_only",
    "margin_only",
    "margin_cap80",
    "margin_cap128",
    "margin_cap192",
)

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
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_trial(arm: str, run_name: str, random_seed: int):
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
    smi = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
    ).stdout
    driver_line = next(
        (
            line.strip()
            for line in smi.splitlines()
            if "Driver Version" in line
        ),
        "nvidia-smi unavailable",
    )
    environment_line = (
        f"{timestamp.isoformat()} | {run_name} | torch {torch.__version__}, "
        f"cuda {torch.version.cuda} | {driver_line}"
    )
    print(environment_line)
    os.makedirs("/outputs/env_runs", exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    with open(
        f"/outputs/env_runs/{stamp}_{run_name}.log",
        "w",
    ) as environment_log:
        environment_log.write(environment_line + "\n")

    from looping.exp_health_methods import train

    try:
        return train(
            output_dir="/outputs/looping",
            arm=arm,
            run_name=run_name,
            random_seed=random_seed,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(arm: str = "vanilla", trial: int = 0):
    arm = arm.replace("-", "_")
    if arm not in ARM_NAMES:
        choices = ", ".join(ARM_NAMES)
        raise ValueError(f"unknown arm {arm!r}; choose one of: {choices}")
    if trial < 0:
        raise ValueError("trial must be non-negative")
    run_name = f"loop_health_standalone_v1_{arm}_50k_trial{trial}"
    random_seed = 20_260_730 + trial
    call = run_trial.spawn(arm, run_name, random_seed)
    print(f"Spawned {run_name}: {call.object_id}")
    print(f"Poll looping/{run_name}.log on sudoku-outputs for progress.")
