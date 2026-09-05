"""One detached training-and-evaluation worker per invocation."""

import modal

app = modal.App("sotaku-window-selection")
outputs = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
cache = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements("requirements-modal.txt")
if modal.is_local():
    from looping.window_selection.common import SOURCE_PATHS
    for source in (*SOURCE_PATHS, "looping/__init__.py", "looping/weight_tying/__init__.py",
                   "looping/window_selection/__init__.py", "stabilize/__init__.py", "iters/__init__.py",
                   "release/benchmark_25k.json"):
        image = image.add_local_file(source, remote_path=f"/root/project/{source}")


def configure():
    import os
    import sys
    from pathlib import Path
    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    outputs.reload()
    from looping.window_selection.common import protocol
    settings = protocol()
    return Path("/outputs") / settings["study_id"], Path(settings["data_directory"])


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=7200,
              retries=modal.Retries(max_retries=1, initial_delay=30),
              volumes={"/outputs": outputs, "/hf_cache": cache})
def smoke():
    import contextlib
    import sys
    import uuid
    root, data = configure()
    from checkpoint_utils import atomic_json_save
    from runtime_utils import Tee
    from looping.window_selection.test_selection import gpu_smoke
    directory = root / "preflight" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with (directory / "smoke.log").open("a") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                print(f"PREFLIGHT_DIRECTORY {directory}", flush=True)
                result = gpu_smoke(data, directory)
                result["directory"] = str(directory)
                atomic_json_save(result, root / "preflight_passed.json")
                return result
    finally:
        outputs.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=24 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30),
              volumes={"/outputs": outputs, "/hf_cache": cache})
def train(selector: str, seed: int):
    root, data = configure()
    import json
    from checkpoint_utils import atomic_json_save, validate_config
    from looping.window_selection.common import SOURCE_PATHS, run_name
    from looping.window_selection.train import evaluate_run, train_run
    from runtime_utils import runtime_manifest
    preflight = json.loads((root / "preflight_passed.json").read_text())
    if preflight["status"] != "passed":
        raise ValueError("CUDA preflight did not pass")
    validate_config(preflight["source_sha256"], runtime_manifest(SOURCE_PATHS)["source_sha256"])
    directory = root / "runs" / run_name(selector, seed)
    try:
        result = train_run(data, directory, selector, seed, checkpoint_callback=outputs.commit)
        outputs.commit()
        evaluation = evaluate_run(directory, checkpoint_callback=outputs.commit)
        summary = {"training_status": result["status"], "evaluation": evaluation}
        atomic_json_save(summary, directory / "completed.json")
        return summary
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(action: str = "smoke", selector: str = "random", seed: int = 20260904):
    from looping.window_selection.common import run_config
    if action == "smoke":
        call = smoke.spawn()
    elif action == "train":
        run_config(selector, seed)
        call = train.spawn(selector, seed)
    else:
        raise ValueError("Action must be smoke or train")
    print(f"Spawned {action}: {call.object_id}")
