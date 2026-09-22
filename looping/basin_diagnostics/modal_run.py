"""One detached, resumable analysis worker per invocation."""

import modal

app = modal.App("sotaku-basin-diagnostics")
outputs = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt")
         .pip_install("matplotlib==3.10.5"))
if modal.is_local():
    from pathlib import Path
    from looping.basin_diagnostics.common import SOURCE_PATHS

    sources = set(SOURCE_PATHS)
    for source in tuple(sources):
        for parent in Path(source).parents:
            initializer = parent / "__init__.py"
            if initializer.is_file():
                sources.add(str(initializer))
    for source in sorted(sources):
        image = image.add_local_file(source, remote_path=f"/root/project/{source}")


def configure():
    import os
    import sys
    from pathlib import Path

    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    outputs.reload()
    from looping.basin_diagnostics.common import protocol
    return Path("/outputs") / protocol()["study_id"]


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30), volumes={"/outputs": outputs})
def analyze(key: str, smoke: bool = False):
    import contextlib
    import json
    import sys

    root = configure()
    from checkpoint_utils import atomic_json_save, validate_config
    from looping.basin_diagnostics.analysis import prepare_selection, run_model
    from looping.basin_diagnostics.common import SOURCE_PATHS
    from runtime_utils import Tee, runtime_manifest
    folder = root / ("smoke" if smoke else "models") / key
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with (folder / "analysis.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(f"WORKER key={key} smoke={smoke} directory={folder}", flush=True)
                if smoke:
                    import torch
                    from looping.basin_diagnostics.test_diagnostics import gpu_preflight
                    torch.set_float32_matmul_precision("highest")
                    prepare_selection(root)
                    preflight = gpu_preflight()
                    preflight["source_sha256"] = runtime_manifest(SOURCE_PATHS)["source_sha256"]
                else:
                    preflight = json.loads((root / "preflight_passed.json").read_text())
                    if preflight["status"] != "passed":
                        raise ValueError("Preflight did not pass")
                    validate_config(preflight["source_sha256"], runtime_manifest(SOURCE_PATHS)["source_sha256"])
                result = run_model(key, root, smoke=smoke, checkpoint_callback=outputs.commit)
                if smoke:
                    checks = result["maps"][0]["checks"]
                    if checks["zero_control_board_mismatches"] or checks["zero_control_max_hidden_error"]:
                        raise ValueError("Zero perturbations changed the reference trajectory")
                    atomic_json_save(preflight, root / "preflight_passed.json")
                return {"status": result["status"], "directory": str(folder), "seconds": result["elapsed_seconds"]}
    finally:
        outputs.commit()


@app.local_entrypoint()
def main(key: str = "20k_20260908", smoke: bool = False):
    from looping.basin_diagnostics.common import protocol
    if key not in protocol()["models"]:
        raise ValueError("Unknown fixed checkpoint")
    call = analyze.spawn(key, smoke)
    print(f"Spawned {key} smoke={smoke}: {call.object_id}")
