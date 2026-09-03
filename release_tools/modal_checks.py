"""One detached release-validation job per invocation."""

import modal

from modal_config import PROJECT_IGNORE

app = modal.App("sotaku-release-validation")
outputs_volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=True)
hf_cache_volume = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=PROJECT_IGNORE)
)


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=3, initial_delay=10.0),
    volumes={"/outputs": outputs_volume, "/hf_cache": hf_cache_volume},
)
def check(kind: str, name: str, model: str):
    import contextlib
    import os
    import sys
    import traceback

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    from checkpoint_utils import atomic_json_save
    from runtime_utils import Tee, output_subdirectory, runtime_manifest

    outputs_volume.reload()
    directory = output_subdirectory("/outputs/release_validation", name)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with open(directory / "worker.log", "a", buffering=1) as handle:
            with contextlib.redirect_stdout(Tee(sys.stdout, handle)), contextlib.redirect_stderr(Tee(sys.stderr, handle)):
                try:
                    atomic_json_save(runtime_manifest((
                        "stabilize/exp_testbed_20k.py", "checkpoint_utils.py", "requirements-modal.txt",
                    )), directory / "environment.json")
                    print(f"Release check kind={kind}, model={model}, output={directory}", flush=True)
                    if kind in ("smoke", "smoke-eager"):
                        from release_tools.smoke_training import run
                        return run(directory, backend="eager" if kind == "smoke-eager" else "inductor")
                    from release_tools.validation import run
                    return run(kind, model, directory)
                except BaseException:
                    traceback.print_exc()
                    raise
    finally:
        outputs_volume.commit()


@app.local_entrypoint()
def main(kind: str = "smoke", name: str = "", model: str = "late_state_ce"):
    from runtime_utils import output_subdirectory
    if kind not in ("smoke", "smoke-eager", "full", "precision"):
        raise ValueError("kind must be smoke, smoke-eager, full, or precision")
    output_subdirectory("/outputs/release_validation", name)
    call = check.spawn(kind, name, model)
    print(f"Spawned {call.object_id}; log: release_validation/{name}/worker.log")
