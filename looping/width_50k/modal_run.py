"""One detached width160 preflight or training/evaluation worker per invocation."""

import modal

app = modal.App("sotaku-width160-50k")
outputs = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.11").pip_install_from_requirements("requirements-modal.txt")
if modal.is_local():
    from looping.width_50k.common import SOURCE_PATHS

    for source in (*SOURCE_PATHS, "looping/__init__.py", "looping/hyperloop/__init__.py",
                   "looping/window_selection/__init__.py", "looping/weight_tying/__init__.py",
                   "stabilize/__init__.py", "iters/__init__.py"):
        image = image.add_local_file(source, remote_path=f"/root/project/{source}")


def configure():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    outputs.reload()
    from looping.width_50k.common import protocol, verify_inherited_preflight
    verify_inherited_preflight()
    settings = protocol()
    return Path("/outputs") / settings["study_id"], Path(settings["data_directory"])


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=3 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=30), volumes={"/outputs": outputs})
def smoke():
    import contextlib
    import sys
    import uuid

    root, data = configure()
    from checkpoint_utils import atomic_json_save
    from looping.width_50k.preflight import gpu_preflight
    from runtime_utils import Tee
    directory = root / "preflight" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    try:
        with (directory / "preflight.log").open("a") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                print(f"PREFLIGHT_DIRECTORY {directory}", flush=True)
                result = gpu_preflight(data, directory)
                result["directory"] = str(directory)
                atomic_json_save(result, root / "preflight_passed.json")
                return result
    finally:
        outputs.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=24 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=30), volumes={"/outputs": outputs})
def train(arm: str, seed: int):
    import json

    root, data = configure()
    from checkpoint_utils import atomic_json_save, validate_config
    from looping.width_50k.common import SOURCE_PATHS, protocol, run_name, validate_data, verify_completed_pair
    from looping.width_50k.evaluate import evaluate_run
    from looping.width_50k.train import train_run
    from runtime_utils import runtime_manifest
    preflight = json.loads((root / "preflight_passed.json").read_text())
    if preflight["status"] != "passed":
        raise ValueError("GPU preflight did not pass")
    validate_config(preflight["source_sha256"], runtime_manifest(SOURCE_PATHS)["source_sha256"])
    validate_config(preflight["protocol"], protocol())
    validate_config(preflight["data_sha256"], validate_data(data))
    if not preflight["inherited_full_batch_verified"]:
        raise ValueError("Missing full-batch verification")
    directory = root / "runs" / run_name(arm, seed)
    try:
        result = train_run(data, directory, arm, seed, checkpoint_callback=outputs.commit)
        evaluations = {}
        if result["status"] == "complete":
            verify_completed_pair(result)
            for selection in ("final", "best_validation"):
                evaluations[selection] = evaluate_run(data, directory, selection, checkpoint_callback=outputs.commit)
        completed = {"training_status": result["status"], "evaluations": evaluations}
        atomic_json_save(completed, directory / "completed.json")
        return completed
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(action: str = "smoke", arm: str = "width160", seed: int = 20260910):
    from looping.width_50k.common import run_config
    if action == "smoke":
        call = smoke.spawn()
    elif action == "train":
        run_config(arm, seed)
        call = train.spawn(arm, seed)
    else:
        raise ValueError("Action must be smoke or train")
    print(f"Spawned {action}: {call.object_id}")
