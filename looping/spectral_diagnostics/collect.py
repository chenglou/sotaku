"""Read committed results without creating or resubmitting GPU jobs."""

import argparse
import hashlib
import json
from pathlib import Path

import modal


def collect(keys):
    directory = Path(__file__).parent
    settings = json.loads((directory / "protocol.json").read_text())
    root = settings["study_id"]
    destination = directory / "results"
    volume = modal.Volume.from_name("sudoku-outputs", create_if_missing=False)

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

    entries = {Path(entry.path).name for entry in volume.iterdir(root)}
    for key in keys:
        if f"{key}.log" in entries:
            log_path = destination / f"{key}.log"
            download(f"{root}/{key}.log", log_path)
            print(key, *log_path.read_text().splitlines()[-6:], sep="\n", flush=True)
        if key not in entries:
            continue
        names = {Path(entry.path).name for entry in volume.iterdir(f"{root}/{key}")}
        checksums = {}
        if "completed.json" in names:
            local = destination / key / "completed.json"
            download(f"{root}/{key}/completed.json", local)
            result = json.loads(local.read_text())
            if result["status"] != "complete":
                raise ValueError("Incomplete result")
            checksums = result["sha256"]
        for name in sorted(names):
            if name.endswith(".json") and name != "completed.json":
                download(f"{root}/{key}/{name}", destination / key / name, checksums.get(name))
        if checksums:
            print(f"{key}: complete; {len(checksums)} artifact checksums verified", flush=True)
        else:
            print(f"{key}: not yet complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", choices=("old_stable", "old_collapsing", "v2", "smoke_old_stable"))
    args = parser.parse_args()
    collect([args.key] if args.key else ["old_stable", "old_collapsing", "v2"])
