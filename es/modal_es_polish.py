"""Launch copy-only independent-sampling settledness ES on a volume model."""

import os
import re

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-es-settledness-polish")

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
        ignore=PROJECT_IGNORE,
    )
)


def resolve_run_paths(model_name, run_name=None):
    normalized_model = os.path.normpath(model_name)
    if (
        os.path.isabs(normalized_model)
        or normalized_model == ".."
        or normalized_model.startswith(f"..{os.sep}")
    ):
        raise ValueError(f"unsafe model path: {model_name!r}")
    if not normalized_model.endswith(".pt"):
        raise ValueError("model path must end in .pt")

    if run_name is None:
        model_stem = os.path.basename(normalized_model)[:-3]
        run_name = f"{model_stem}_settle_independent"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")

    return (
        os.path.join("/outputs", normalized_model),
        os.path.join("/outputs/es_polish", run_name),
        run_name,
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
def run_polish(model_name: str, run_name: str, generations: int):
    import datetime
    import subprocess
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    import torch

    outputs_volume.reload()
    seed_model_path, output_dir, resolved_run_name = resolve_run_paths(
        model_name,
        run_name,
    )
    if not os.path.isfile(seed_model_path):
        raise FileNotFoundError(seed_model_path)
    os.makedirs(output_dir, exist_ok=True)

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
        f"{timestamp.isoformat()} | {resolved_run_name} | "
        f"seed={model_name} | generations={generations} | "
        f"torch {torch.__version__}, "
        f"cuda {torch.version.cuda} | {driver_line}"
    )
    print(environment_line)
    with open(
        os.path.join(output_dir, "environment.log"),
        "w",
    ) as environment_log:
        environment_log.write(environment_line + "\n")

    from es.exp_es_settle_independent import train

    try:
        return train(
            output_dir=output_dir,
            seed_model_path=seed_model_path,
            generations=generations,
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(model: str, run_name: str = "", generations: int = 60):
    if generations <= 0:
        raise ValueError("generations must be positive")
    _, _, resolved_run_name = resolve_run_paths(
        model,
        run_name or None,
    )
    call = run_polish.spawn(model, resolved_run_name, generations)
    print(f"Spawned {resolved_run_name}: {call.object_id}")
    print(f"Generations: {generations}")
    print(
        "Poll es_polish/"
        f"{resolved_run_name}/exp_es_settle_independent.log "
        "on sudoku-outputs."
    )
