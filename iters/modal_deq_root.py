"""Run the DEQ root probe on Modal.

Usage:
    modal run --detach iters/modal_deq_root.py
"""

import modal

from modal_config import PROJECT_IGNORE


app = modal.App("sudoku-deq-root")

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
    max_solver_steps: int,
    precision: str,
):
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")

    from iters.eval_deq_root import evaluate

    outputs_volume.reload()
    try:
        return evaluate(
            "/outputs/model_baseline_lr2e3.pt",
            examples_per_bucket=examples_per_bucket,
            batch_size=batch_size,
            max_solver_steps=max_solver_steps,
            precision=precision,
            device="cuda",
            output_dir="/outputs",
        )
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    examples_per_bucket: int = 200,
    batch_size: int = 100,
    max_solver_steps: int = 64,
    precision: str = "fp32",
):
    call = run_probe.spawn(
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
        max_solver_steps=max_solver_steps,
        precision=precision,
    )
    print(f"Spawned DEQ root probe: {call.object_id}")
    print(
        "Poll model_baseline_lr2e3_deq_root_probe.log on the "
        "sudoku-outputs volume for progress."
    )
