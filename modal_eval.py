"""
Modal wrapper for running eval_more_iters on GPU.

Usage:
    modal run --detach modal_eval.py --model model_baseline_lr2e3.pt --iters 128,1024,2048,4096
    modal run --detach modal_eval.py --model looping/model_late_ce_50k.pt --precision fp32
"""

import modal

app = modal.App("sudoku-eval")

hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=["venv/", ".venv/", "__pycache__/", "*.pyc", ".git/", ".claude/", ".codex/", "logs/", "runs/", "runs_modal/", "*.pt", "*.log", "temp-side-convo.txt", "release/validation/", "release/v2/*.zip"])
)


@app.function(
    image=image,
    gpu="H200",
    cpu=8.0,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={
        "/hf_cache": hf_cache_volume,
        "/outputs": outputs_volume,
    },
)
def run_eval(exp_name: str, model_name: str, iter_counts_str: str, output_name: str,
             manifest: str = "", precision: str = "fp32", batch_size: int = 256,
             compiled: bool = False, legacy_defaults: bool = False):
    import contextlib
    import os
    import sys
    import traceback

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")

    from iters.eval_more_iters import evaluate
    from runtime_utils import Tee, output_subdirectory

    outputs_volume.reload()
    from pathlib import Path
    if Path(model_name).is_absolute() or ".." in Path(model_name).parts:
        raise ValueError("model must be relative to the output volume")
    model_path = os.path.join("/outputs", model_name)
    iter_counts = [int(x) for x in iter_counts_str.split(",")]

    output_dir = output_subdirectory("/outputs/evaluations", output_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_dir / "worker.log", "a", buffering=1) as handle:
            with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(Tee(sys.stderr, handle)):
                try:
                    return evaluate(
                        model_path, exp_module=exp_name or None, iter_counts=iter_counts, device='cuda',
                        output_dir=output_dir,
                        manifest_path=manifest or None, benchmark_path="release/benchmark_25k.json",
                        precision=precision, batch_size=batch_size, compiled=compiled,
                        matmul_precision="highest" if precision == "fp32" else "high",
                        legacy_defaults=legacy_defaults,
                    )
                except BaseException:
                    traceback.print_exc()
                    raise
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(
    exp: str = "",
    model: str = "model_baseline_lr2e3.pt",
    iters: str = "16,32,64,128,256,512,1024",
    name: str = "",
    manifest: str = "",
    precision: str = "fp32",
    batch_size: int = 256,
    compiled: bool = False,
    legacy_defaults: bool = False,
):
    import uuid
    from runtime_utils import output_subdirectory
    if precision not in ("fp32", "bf16") or batch_size <= 0:
        raise ValueError("Use fp32/bf16 and a positive batch size")
    name = name or f"eval_{uuid.uuid4().hex[:12]}"
    output_subdirectory("/outputs/evaluations", name)
    print(f"Evaluating {model} with exp={exp}, iters={iters}")
    call = run_eval.spawn(
        exp_name=exp, model_name=model, iter_counts_str=iters, output_name=name,
        manifest=manifest, precision=precision, batch_size=batch_size,
        compiled=compiled, legacy_defaults=legacy_defaults,
    )
    print(f"Spawned eval call: {call.object_id}")
    print(f"Eval continues server-side; results: sudoku-outputs/evaluations/{name}/")
