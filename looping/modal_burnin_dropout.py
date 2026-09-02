"""Launch one detached, copy-only dropout continuation per invocation."""

import modal

app = modal.App("sotaku-burnin-dropout")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=[
        "venv/", ".venv/", "__pycache__/", "*.pyc", ".git/", ".claude/", ".codex/",
        "logs/", "runs/", "runs_modal/", "*.pt", "*.log", "temp-side-convo.txt",
        "release/validation/", "release/v2/*.zip",
    ])
)


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/outputs": outputs_volume, "/hf_cache": hf_cache_volume},
)
def run(seed: int, dropout: bool, name: str, stop_after_step: int):
    import contextlib
    import os
    import sys
    import traceback

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    from runtime_utils import Tee, output_subdirectory
    from looping.exp_burnin_dropout import train

    outputs_volume.reload()
    output_dir = output_subdirectory("/outputs/release_validation", name)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_dir / "worker.log", "a", buffering=1) as handle:
            with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(Tee(sys.stderr, handle)):
                try:
                    return train(
                        output_dir, source_root="/outputs", seed=seed,
                        dropout_enabled=dropout, stop_after_step=stop_after_step,
                    )
                except BaseException:
                    traceback.print_exc()
                    raise
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(seed: int = 20260724, dropout: bool = True, name: str = "", stop_after_step: int = 43000):
    from runtime_utils import output_subdirectory
    if not name:
        raise ValueError("Pass a unique --name for this seed and dropout setting")
    output_subdirectory("/outputs/release_validation", name)
    call = run.spawn(seed, dropout, name, stop_after_step)
    print(f"Spawned {call.object_id}; log: release_validation/{name}/worker.log")
