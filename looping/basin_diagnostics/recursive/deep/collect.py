"""Read committed results and checksums without launching any GPU work."""

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

    entries = {Path(entry.path).name for entry in volume.iterdir(root, recursive=False)}
    for key in settings["models"]:
        if f"{key}.log" in entries:
            path = destination / f"{key}.log"
            download(f"{root}/{key}.log", path)
            lines = [line for line in path.read_text().splitlines() if line.startswith(("PREFLIGHT", "ROWS", "LEVEL", "AUDIT", "COMPLETE"))]
            print(key, *lines[-3:], sep="\n", flush=True)
        if logs_only or key not in entries:
            continue
        names = {Path(entry.path).name for entry in volume.iterdir(f"{root}/{key}", recursive=False)}
        checksums = {}
        if "completed.json" in names:
            download(f"{root}/{key}/completed.json", destination / key / "completed.json")
            result = json.loads((destination / key / "completed.json").read_text())
            if result["status"] != "complete" or len(result["levels"]) != settings["levels"]:
                raise ValueError("Incomplete completion record")
            checksums = result["sha256"]
        for name in sorted(names):
            if name == "completed.json" or Path(name).suffix not in (".json", ".png", ".npz"):
                continue
            download(f"{root}/{key}/{name}", destination / key / name, checksums.get(name))
        print(f"{key}: {'complete, verified' if checksums else 'still running'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--logs-only", action="store_true")
    args = parser.parse_args()
    collect(args.destination, args.logs_only)
