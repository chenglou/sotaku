"""Launch matched paired-vs-independent ES trials on Modal.

Examples:
    modal run --detach es/modal_es_sampling_ablation.py --stage rescuable --trial 0
    modal run --detach es/modal_es_sampling_ablation.py --stage rescuable --sampling-mode paired --trial 0
"""

import modal

from es.es_sampling import DEFAULT_SAMPLING_MODE


app = modal.App("sudoku-es-sampling-ablation")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

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


STAGES = {
    "rescuable": {
        "seed_path": "/outputs/model_baseline_lr2e3_clean_a.pt",
        "fitness_dense": False,
    },
    "hard": {
        "seed_path": "/outputs/baseline_lr2e3_cohort_d_checkpoint_step45000.pt",
        "fitness_dense": True,
    },
}


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
    run_name: str,
    seed_path: str,
    sampling_mode: str,
    es_random_seed: int,
    fitness_dense: bool,
    generations: int,
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

    from es.exp_es_sampling_ablation import train_sampling_ablation

    try:
        return train_sampling_ablation(
            output_dir="/outputs",
            run_name=run_name,
            seed_path=seed_path,
            sampling_mode=sampling_mode,
            es_random_seed=es_random_seed,
            fitness_dense=fitness_dense,
            total_generations=generations,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    stage: str = "rescuable",
    sampling_mode: str = DEFAULT_SAMPLING_MODE,
    trial: int = 0,
    generations: int = 60,
    run_prefix: str = "sampling_ablation",
):
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {sorted(STAGES)}")
    if sampling_mode not in {"paired", "independent"}:
        raise ValueError("sampling_mode must be 'paired' or 'independent'")
    if trial < 0:
        raise ValueError("trial must be non-negative")

    stage_config = STAGES[stage]
    es_random_seed = 20_260_719 + trial
    run_name = f"{run_prefix}_{stage}_{sampling_mode}_trial{trial}"
    # A detached local entrypoint only preserves its last spawned call. Keep exactly
    # one call per invocation; launch matched arms with separate `modal run` commands.
    call = run_trial.spawn(
        run_name=run_name,
        seed_path=stage_config["seed_path"],
        sampling_mode=sampling_mode,
        es_random_seed=es_random_seed,
        fitness_dense=stage_config["fitness_dense"],
        generations=generations,
    )
    print(f"Spawned {run_name}: {call.object_id}")
    print("The trial continues server-side; poll its .log file on sudoku-outputs.")
