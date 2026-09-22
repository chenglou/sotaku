"""Download committed recursive-map artifacts; never start GPU jobs."""

import argparse
import hashlib
import json
from pathlib import Path

import modal


def collect(destination, logs_only=False):
    settings = json.loads(Path(__file__).with_name("protocol.json").read_text())
    volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
    root = settings["study_id"]
    destination = Path(destination)

    def download(remote, local, checksum=None):
        if checksum and local.exists() and hashlib.sha256(local.read_bytes()).hexdigest() == checksum:
            return
        content = b"".join(volume.read_file(remote))
        if checksum and hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"Checksum mismatch: {remote}")
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(local.name + ".download")
        temporary.write_bytes(content)
        temporary.replace(local)

    entries = {Path(entry.path).name: entry for entry in volume.iterdir(root, recursive=False)}
    for name in entries:
        if Path(name).suffix not in ((".log",) if logs_only else (".log", ".json", ".png")):
            continue
        download(f"{root}/{name}", destination / name)
        if name.endswith(".log"):
            lines = (destination / name).read_text().splitlines()
            progress = [line for line in lines if line.startswith(("ROWS", "LEVEL", "COMPLETE", "WORKER"))]
            print(name, *progress[-2:], sep="\n", flush=True)
    if logs_only:
        return
    for key in settings["models"]:
        if key not in entries:
            continue
        folder = f"{root}/{key}"
        names = [Path(entry.path).name for entry in volume.iterdir(folder, recursive=False)]
        checksums = {}
        if "completed.json" in names:
            download(f"{folder}/completed.json", destination / key / "completed.json")
            checksums = json.loads((destination / key / "completed.json").read_text())["sha256"]
        for name in names:
            if name == "completed.json" or Path(name).suffix not in (".json", ".png", ".npz"):
                continue
            download(f"{folder}/{name}", destination / key / name, checksums.get(name))
        levels = len(list((destination / key).glob("level_*.npz")))
        print(f"{key}: {levels}/3 zoom levels, {'complete' if checksums else 'running'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).with_name("results_1024"))
    parser.add_argument("--logs-only", action="store_true")
    args = parser.parse_args()
    collect(args.destination, args.logs_only)
