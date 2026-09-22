"""Separate detached workers for framing and each recursive visualization."""

import modal

app = modal.App("sotaku-recursive-viz")
volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt").pip_install("matplotlib==3.10.5"))
if modal.is_local():
    from pathlib import Path
    from looping.basin_diagnostics.recursive.run import SOURCE_PATHS
    sources = set(SOURCE_PATHS)
    for name in tuple(sources):
        for parent in Path(name).parents:
            initializer = parent / "__init__.py"
            if initializer.is_file():
                sources.add(str(initializer))
    for name in sorted(sources):
        image = image.add_local_file(name, remote_path=f"/root/project/{name}")


@app.function(image=image, gpu="H200", cpu=8, memory=32768, timeout=6 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30), volumes={"/outputs": volume})
def worker(key="prepare"):
    import contextlib
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
    volume.reload()
    from looping.basin_diagnostics.recursive.run import prepare, protocol, render_model
    from runtime_utils import Tee
    settings = protocol()
    if key not in ["prepare", *settings["models"]]:
        raise ValueError("Unknown visualization job")
    root = Path("/outputs") / settings["study_id"]
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / f"{key}.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                print(f"WORKER {key} root={root}", flush=True)
                return prepare(root, volume.commit) if key == "prepare" else render_model(key, root, volume.commit)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(key: str = "prepare"):
    call = worker.spawn(key)
    print(f"Spawned {key}: {call.object_id}")
