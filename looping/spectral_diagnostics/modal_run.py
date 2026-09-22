"""Launch exactly one detached, resumable Jacobian diagnostic per invocation."""

import modal

app = modal.App("sotaku-jacobian-fp64")
volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install_from_requirements("requirements-modal.txt").pip_install("scipy==1.13.1"))
if modal.is_local():
    from pathlib import Path
    from looping.spectral_diagnostics.run import SOURCE_PATHS
    sources = set(SOURCE_PATHS)
    for name in tuple(sources):
        for parent in Path(name).parents:
            initializer = parent / "__init__.py"
            if initializer.is_file():
                sources.add(str(initializer))
    for name in sorted(sources):
        image = image.add_local_file(name, remote_path=f"/root/project/{name}")


@app.function(image=image, gpu="H200", cpu=4, memory=16384, timeout=4 * 3600,
              retries=modal.Retries(max_retries=2, initial_delay=30), volumes={"/outputs": volume})
def worker(key, smoke=False):
    import contextlib
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")
    volume.reload()
    from looping.spectral_diagnostics.run import protocol, run
    from runtime_utils import Tee
    root = Path("/outputs") / protocol()["study_id"]
    root.mkdir(parents=True, exist_ok=True)
    try:
        with (root / f"{'smoke_' if smoke else ''}{key}.log").open("a", buffering=1) as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
                return run(key, root, callback=volume.commit, smoke=smoke)
    finally:
        volume.commit()


@app.local_entrypoint()
def main(key: str = "old_stable", smoke: bool = False):
    if key not in ("old_stable", "old_collapsing", "v2"):
        raise ValueError("Unknown checkpoint")
    call = worker.spawn(key, smoke)
    print(f"Spawned key={key} smoke={smoke}: {call.object_id}")
