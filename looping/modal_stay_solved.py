"""Launch one stay-solved or staged late-state experiment on Modal."""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-stay-solved")

ARM_NAMES = (
    "control",
    "late_state_ce",
    "ramp_after_2k",
    "curriculum_after_2k",
    "clean_curriculum",
    "stay_recheck",
    "stay_consistency",
    "stay_consistency_strong",
    "stay_margin_floor",
    "clean_stay",
    "clean_rmsnorm",
)
SCREEN_SUFFIX = "_healthy_screen"
FULL_50K_SUFFIX = "_50k"

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
    cpu=8.0,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_trial(
    arm: str,
    run_name: str,
    random_seed: int,
    screen: bool,
    full_50k: bool,
):
    import datetime
    import os
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    import torch

    outputs_volume.reload()
    output_dir = "/outputs/looping"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    driver_line = next(
        (line.strip() for line in smi.splitlines() if "Driver Version" in line),
        "nvidia-smi unavailable",
    )
    environment_line = (
        f"{timestamp.isoformat()} | {run_name} | torch {torch.__version__}, "
        f"cuda {torch.version.cuda} | {driver_line}"
    )
    print(environment_line)
    os.makedirs("/outputs/env_runs", exist_ok=True)
    stamp = timestamp.strftime("%Y%m%dT%H%M%SZ")
    with open(f"/outputs/env_runs/{stamp}_{run_name}.log", "w") as environment_log:
        environment_log.write(environment_line + "\n")

    from looping.exp_stay_solved import train

    try:
        return train(
            output_dir=output_dir,
            arm=arm,
            run_name=run_name,
            random_seed=random_seed,
            screen=screen,
            full_50k=full_50k,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    arm: str = "late_state_ce",
    trial: int = 0,
    screen: bool = False,
    full_50k: bool = False,
    seed: int = -1,
    name: str = "",
):
    arm = arm.replace("-", "_")
    if arm not in ARM_NAMES:
        choices = ", ".join(ARM_NAMES)
        raise ValueError(f"unknown arm {arm!r}; choose one of: {choices}")
    if trial < 0:
        raise ValueError("trial must be non-negative")
    if full_50k:
        screen = False
        suffix = FULL_50K_SUFFIX
    else:
        suffix = SCREEN_SUFFIX if screen else ""
    run_name = name or f"loop_stay_{arm}{suffix}_trial{trial}"
    random_seed = 20_260_724 + trial if seed == -1 else seed
    if random_seed < 0:
        raise ValueError("seed must be non-negative")
    call = run_trial.spawn(
        arm,
        run_name,
        random_seed,
        screen,
        full_50k,
    )
    print(f"Spawned {run_name}: {call.object_id}")
    print(f"Poll looping/{run_name}.log on sudoku-outputs for progress.")
