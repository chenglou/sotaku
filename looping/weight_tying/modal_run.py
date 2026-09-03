"""One detached worker per invocation for the preregistered weight-tying study."""

import modal

app = modal.App("sotaku-weight-tying-study")
outputs = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
cache = modal.Volume.from_name("sudoku-hf-cache", create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("qqwing")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=[
        "venv/", ".venv/", ".git/", ".claude/", ".codex/", "__pycache__/", "*.pyc",
        "*.pt", "*.log", "logs/", "runs/", "runs_modal/", "data/", "temp-side-convo.txt",
        "release/validation/", "release/v2/*.zip", "looping/trajectory_viz/", "viz/output/",
    ])
)


def configure():
    import os
    import sys
    from pathlib import Path

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    outputs.reload()
    from looping.weight_tying.common import protocol
    return Path("/outputs") / protocol()["study_id"]


@app.function(image=image, cpu=8, memory=32768, timeout=4 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30),
              volumes={"/outputs": outputs, "/hf_cache": cache})
def prepare():
    root = configure()
    import contextlib
    import sys
    from looping.weight_tying.data import prepare_data
    from runtime_utils import Tee
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / "prepare.log").open("a") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                return prepare_data(root / "data")
    finally:
        outputs.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=8 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30),
              volumes={"/outputs": outputs, "/hf_cache": cache})
def train(architecture: str, regime: str, seed: int):
    root = configure()
    import json
    from checkpoint_utils import validate_config
    from looping.weight_tying.common import SOURCE_PATHS
    from runtime_utils import file_sha256, runtime_manifest
    passed = json.loads((root / "smoke/passed.json").read_text())
    if file_sha256(passed["path"]) != passed["sha256"]:
        raise ValueError("CUDA preflight result checksum mismatch")
    validate_config(passed["source_sha256"], runtime_manifest(SOURCE_PATHS)["source_sha256"])
    from looping.weight_tying.common import run_name
    from looping.weight_tying.train import train_run
    try:
        return train_run(root / "data", root / "runs" / run_name(architecture, regime, seed),
                         architecture, regime, seed, checkpoint_callback=outputs.commit)
    finally:
        outputs.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=2 * 3600,
              retries=modal.Retries(max_retries=1, initial_delay=30),
              volumes={"/outputs": outputs, "/hf_cache": cache})
def smoke():
    root = configure()
    import contextlib
    import sys
    from looping.weight_tying.test_study import gpu_smoke
    from runtime_utils import Tee
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / "smoke.log").open("a") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                return gpu_smoke(root / "smoke")
    finally:
        outputs.commit()


@app.function(image=image, gpu="H200", cpu=8, memory=16384, timeout=4 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30),
              volumes={"/outputs": outputs, "/hf_cache": cache})
def evaluate(architecture: str, regime: str, seed: int, selection: str):
    root = configure()
    from looping.weight_tying.evaluate import evaluate_run
    try:
        return evaluate_run(root, architecture, regime, seed, selection)
    finally:
        outputs.commit()


@app.function(image=image, cpu=1, timeout=600, volumes={"/outputs": outputs, "/hf_cache": cache})
def seal():
    root = configure()
    from looping.weight_tying.evaluate import seal_cohort
    try:
        return seal_cohort(root)
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(action: str = "smoke", architecture: str = "tied", regime: str = "late",
         seed: int = 20260902, selection: str = "final"):
    from looping.weight_tying.common import run_config
    if action == "prepare":
        call = prepare.spawn()
    elif action == "smoke":
        call = smoke.spawn()
    elif action == "seal":
        call = seal.spawn()
    elif action == "train":
        run_config(architecture, regime, seed)
        call = train.spawn(architecture, regime, seed)
    elif action == "evaluate":
        run_config(architecture, regime, seed)
        if selection not in ("final", "best_validation"):
            raise ValueError("Invalid checkpoint selection")
        call = evaluate.spawn(architecture, regime, seed, selection)
    else:
        raise ValueError("Action must be prepare, smoke, train, seal, or evaluate")
    print(f"Spawned {action}: {call.object_id}")
