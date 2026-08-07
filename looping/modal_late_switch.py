"""Branch a 50K run at step 39K and compare its late training objective."""

import modal


app = modal.App("sudoku-late-switch")

SOURCE_CHECKPOINT = (
    "/outputs/looping/"
    "loop_stay_control_50k_trial0_checkpoint_step39000.pt"
)
SOURCE_STEP = 39000
MODES = ("plain", "consistency", "margin_floor5")
MODE_ARMS = {
    "plain": "control",
    "consistency": "stay_consistency",
    "margin_floor5": "stay_margin_floor",
}

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
def run_branch(mode: str, run_name: str):
    import datetime
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    import torch

    outputs_volume.reload()
    if not os.path.isfile(SOURCE_CHECKPOINT):
        raise FileNotFoundError(SOURCE_CHECKPOINT)

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

    from looping.exp_stay_solved import train

    arm = MODE_ARMS[mode]
    try:
        return train(
            output_dir="/outputs/looping",
            arm=arm,
            run_name=run_name,
            random_seed=20_260_724,
            full_50k=True,
            branch_checkpoint_path=SOURCE_CHECKPOINT,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(mode: str = "plain"):
    mode = mode.replace("-", "_")
    if mode not in MODES:
        choices = ", ".join(MODES)
        raise ValueError(f"unknown mode {mode!r}; choose one of: {choices}")
    run_name = f"loop_stay_late_switch_{mode}_from{SOURCE_STEP // 1000}k"
    call = run_branch.spawn(mode, run_name)
    print(f"Spawned {run_name}: {call.object_id}")
    print(f"Poll looping/{run_name}.log on sudoku-outputs for progress.")
