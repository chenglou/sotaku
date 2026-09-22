"""Download committed diagnostic summaries and figures without launching workers."""

import argparse
import hashlib
import json
from pathlib import Path

import modal


def collect(destination, keys=None, include_map_arrays=False):
    settings = json.loads(Path(__file__).with_name("protocol.json").read_text())
    volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)
    prefix = settings["study_id"]
    keys = keys or list(settings["models"])
    if not set(keys) <= set(settings["models"]):
        raise ValueError("Unknown checkpoint key")
    destination = Path(destination)

    def download(remote, local, checksum=None):
        if local.exists() and checksum and hashlib.sha256(local.read_bytes()).hexdigest() == checksum:
            return
        content = b"".join(volume.read_file(remote))
        if checksum and hashlib.sha256(content).hexdigest() != checksum:
            raise ValueError(f"Downloaded artifact failed its checksum: {remote}")
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_name(local.name + ".download")
        temporary.write_bytes(content)
        temporary.replace(local)

    for filename in ("selection.json", "preflight_passed.json"):
        download(f"{prefix}/{filename}", destination / filename)
    for key in keys:
        folder = f"{prefix}/models/{key}"
        entries = list(volume.iterdir(folder, recursive=False))
        by_name = {Path(entry.path).name: entry for entry in entries}
        local_dir = destination / "models" / key
        checksums = {}
        if "completed.json" in by_name:
            download(f"{folder}/completed.json", local_dir / "completed.json")
            checksums = json.loads((local_dir / "completed.json").read_text())["artifact_sha256"]
        for name in sorted(by_name):
            if name == "completed.json":
                continue
            if name.startswith("map_") and name.endswith(".npz") and not include_map_arrays:
                continue
            if Path(name).suffix not in (".json", ".png", ".npz", ".log"):
                continue
            download(f"{folder}/{name}", local_dir / name, checksums.get(name))
        count = len(list(local_dir.glob("map_*.png")))
        print(f"{key}: {'complete' if checksums else 'in progress'}, {count}/16 maps downloaded", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).with_name("results"))
    parser.add_argument("--key", action="append")
    parser.add_argument("--include-map-arrays", action="store_true")
    args = parser.parse_args()
    collect(args.destination, args.key, args.include_map_arrays)
