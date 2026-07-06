"""One-command daisy-chaining for Viridian training runs.

Usage (from the repo root, venv active):
    python viridian/train/chain.py               # scan all job dirs, chain what needs it
    python viridian/train/chain.py --dry-run     # report states, submit nothing
    python viridian/train/chain.py --only <job-dir-name substring>

For every viridian/train/jobs/<name>/manifest.safe.json this classifies the run:

  complete     final model is on R2 — nothing to do
  running      the job is still alive per the platform — leave it alone
  chain        job is terminal, final model missing, and a hash-verified checkpoint
               exists on R2 — resubmit from the newest one (the verification guards
               against the submit-before-upload race that produced a converged-score-0
               zombie on 2026-07-04)
  no-material  job is terminal with no usable checkpoint — investigate by hand
  legacy       manifest predates the extra_modules/job_id fields — chain by hand

Chaining replays the original submit: same experiment, same extra modules, resume
keys pointing at the newest verified checkpoint. The new job records its own
manifest, so a re-scan picks up where this one left off.
"""

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER_DIR = Path(__file__).resolve().parent
JOBS_DIR = WRAPPER_DIR / "jobs"
VD_BIN = REPO_ROOT / "viridian" / "bin" / "vd"


def rclone_list(bucket, prefix):
    completed = subprocess.run(
        ["rclone", "lsjson", f"r2:{bucket}/{prefix.rstrip('/')}/outputs/"],
        text=True, capture_output=True, check=False,
    )
    if completed.returncode != 0:
        return None
    return {entry["Name"] for entry in json.loads(completed.stdout)}


def job_status(job_id):
    completed = subprocess.run([str(VD_BIN), "job", job_id], text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        return "unknown"
    try:
        return json.loads(completed.stdout).get("status", "unknown")
    except json.JSONDecodeError:
        return "unknown"


def classify(manifest):
    """Returns (state, detail). States: complete/running/chain/no-material/legacy."""
    if "extra_modules" not in manifest or "job_id" not in manifest:
        return "legacy", "manifest predates chainable fields"

    sys.path.insert(0, str(REPO_ROOT))
    exp_module = importlib.import_module(manifest["exp"])
    prefix = exp_module.CHECKPOINT_PREFIX
    final_model = f"model_{prefix[: -len('_checkpoint_step')]}.pt"

    names = rclone_list(manifest["bucket"], manifest["prefix"])
    if names is None:
        return "no-material", "R2 listing failed"
    if final_model in names:
        return "complete", final_model

    status = job_status(manifest["job_id"])
    if status in ("running", "pending"):
        return "running", status

    # Terminal without a final model: find the newest checkpoint whose hash
    # sidecar also exists (sidecar presence means the checkpoint object is complete).
    steps = []
    for name in names:
        if name.startswith(prefix) and name.endswith(".pt") and f"{name}.sha256" in names:
            try:
                steps.append(int(name[len(prefix):-len(".pt")]))
            except ValueError:
                continue
    if not steps:
        return "no-material", f"job {status}, no verified checkpoint on R2"
    return "chain", f"{prefix}{max(steps)}.pt"


def chain(manifest, checkpoint_name):
    key = f"{manifest['prefix'].rstrip('/')}/outputs/{checkpoint_name}"
    command = [
        sys.executable, str(WRAPPER_DIR / "submit.py"),
        "--exp", manifest["exp"],
        "--gpu-tier", manifest.get("gpu_tier", "b200"),
        "--bucket", manifest["bucket"],
        "--resume-checkpoint-key", key,
        "--resume-checkpoint-sha-key", f"{key}.sha256",
    ]
    for module in manifest.get("extra_modules", []):
        command.extend(["--extra-module", module])
    completed = subprocess.run(command, text=True, capture_output=True, check=False, cwd=str(REPO_ROOT))
    print(completed.stdout.strip())
    if completed.returncode != 0:
        print(f"  CHAIN FAILED: {completed.stderr.strip()[-500:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="classify only, submit nothing")
    parser.add_argument("--only", default="", help="restrict to job dirs whose name contains this substring")
    args = parser.parse_args()

    job_dirs = sorted(d for d in JOBS_DIR.iterdir() if (d / "manifest.safe.json").is_file())
    if args.only:
        job_dirs = [d for d in job_dirs if args.only in d.name]

    # Chain each experiment at most once per scan, from its NEWEST job dir only —
    # older dirs of the same experiment are superseded by their own chains.
    newest_by_exp = {}
    manifests = {}
    for d in job_dirs:
        manifest = json.loads((d / "manifest.safe.json").read_text())
        manifests[d.name] = manifest
        if "extra_modules" in manifest:
            newest_by_exp[manifest["exp"]] = d.name

    chained = 0
    for d in job_dirs:
        manifest = manifests[d.name]
        state, detail = classify(manifest)
        superseded = "extra_modules" in manifest and newest_by_exp.get(manifest["exp"]) != d.name
        tag = " (superseded)" if superseded and state == "chain" else ""
        print(f"{state:12s} {d.name}  [{detail}]{tag}")
        if state == "chain" and not superseded and not args.dry_run:
            if chain(manifest, detail):
                chained += 1
    print(f"\n{'(dry run) ' if args.dry_run else ''}chained: {chained}")


if __name__ == "__main__":
    main()
