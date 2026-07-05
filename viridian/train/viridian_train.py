"""Viridian training entrypoint: run the canonical experiment train() on a Viridian GPU.

This is the Viridian counterpart of modal_run.py. It never reimplements training: it
imports the experiment module (e.g. iters.exp_baseline_lr2e3) and calls its train()
with a local output_dir, exactly like the Modal wrapper. Everything else here is
Viridian/R2 plumbing outside the training loop: dependency install, resume-checkpoint
staging, and a background thread that uploads the experiment's log/checkpoints/final
model to R2 as they appear.

Preemption: when Viridian restarts the job, the restart auto-resumes from the newest
SHA-verified checkpoint the previous attempt uploaded (see prepare_resume_checkpoint).
An explicitly presigned --resume-checkpoint-get-url, used for resuming across jobs,
takes precedence over the auto-resume scan.

This file runs inside the job, next to r2_io.py, with the experiment code packaged by
viridian/train/submit.py from the real repo files (no vendored copies).
"""

import argparse
import importlib
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

from r2_io import (
    append_event,
    configure_hf_cache,
    download_file,
    download_text,
    ensure_import,
    file_sha256,
    load_artifact_manifest,
    run_command,
    upload_file,
    upload_report,
    upload_status,
    upload_text,
)


def upload_known_outputs(output_dir, uploads, uploaded_sha, args, report_path, *, force=False):
    uploaded = []
    now = time.time()
    for upload in uploads:
        rel_path = upload["path"]
        path = output_dir / rel_path
        if not path.is_file():
            continue
        # Skip files modified within the last 2s: the trainer may still be writing them.
        if not force and now - path.stat().st_mtime < 2:
            continue

        sha256 = file_sha256(path)
        if uploaded_sha.get(rel_path) == sha256:
            continue

        result = upload_file(path, upload["put_url"])
        sha_result = None
        if upload.get("sha_put_url"):
            sha_result = upload_text(sha256 + "\n", upload["sha_put_url"])
        uploaded_sha[rel_path] = sha256

        latest = {
            "bytes": path.stat().st_size,
            "event": "artifact_uploaded",
            "object": upload["object_name"],
            "path": rel_path,
            "sha256": sha256,
        }
        upload_text(json.dumps(latest, sort_keys=True) + "\n", args.latest_put_url)
        append_event(
            report_path,
            {
                **latest,
                "result": result,
                "sha_result": sha_result,
            },
        )
        uploaded.append(latest)
    return uploaded


def start_sync_thread(output_dir, args, report_path, status_path):
    stop_event = threading.Event()
    uploaded_sha = {}
    uploads = getattr(args, "artifact_uploads", [])

    def sync_once(force=False):
        uploaded = upload_known_outputs(output_dir, uploads, uploaded_sha, args, report_path, force=force)
        upload_report(report_path, args.report_put_url)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "sync",
                "mode": "canonical_train",
                "report_bytes": report_path.stat().st_size,
                "uploaded_count": len(uploaded),
            },
        )
        return uploaded

    def sync_loop():
        while not stop_event.wait(args.upload_every_s):
            try:
                sync_once()
            except Exception as error:
                append_event(report_path, {"event": "sync_failed", "error": repr(error)})

    thread = threading.Thread(target=sync_loop, name="artifact-sync", daemon=True)
    thread.start()
    return stop_event, thread, sync_once


def parse_checkpoint_step(file_name, prefix):
    if not (file_name.startswith(prefix) and file_name.endswith(".pt")):
        return None
    try:
        return int(file_name[len(prefix):-len(".pt")])
    except ValueError:
        return None


def place_verified_checkpoint(download_path, expected_sha, exp_module, output_dir, report_path, source):
    """SHA-verify a downloaded checkpoint and rename it so the experiment's own
    find_latest_checkpoint() picks it up — resume then goes through the canonical path."""
    import torch

    actual_sha = file_sha256(download_path)
    append_event(
        report_path,
        {
            "actual_sha256": actual_sha,
            "event": "resume_checkpoint_downloaded",
            "expected_sha256": expected_sha,
            "source": source,
        },
    )
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("downloaded resume checkpoint SHA-256 did not match")

    checkpoint = torch.load(download_path, map_location="cpu", weights_only=False)
    step = int(checkpoint["step"])
    checkpoint_path = output_dir / f"{exp_module.CHECKPOINT_PREFIX}{step}.pt"
    download_path.replace(checkpoint_path)
    append_event(report_path, {"event": "resume_checkpoint_prepared", "path": checkpoint_path.name, "step": step})
    return step


def auto_resume_candidates(args, prefix):
    """Checkpoint objects from this job's uploads, newest step first. When Viridian
    restarts a dead job, the previous attempt's checkpoints are still on R2, and the
    manifest's GET URLs let the restart pick up where it left off."""
    candidates = []
    for upload in getattr(args, "artifact_uploads", []):
        step = parse_checkpoint_step(upload["path"], prefix)
        if step is None or not upload.get("get_url"):
            continue
        candidates.append((step, upload))
    candidates.sort(key=lambda item: -item[0])
    return candidates


def prepare_resume_checkpoint(args, exp_module, output_dir, report_path):
    """Stage the checkpoint to resume from: an explicitly presigned one
    (--resume-checkpoint-get-url) wins; otherwise scan this job's uploads for the
    newest SHA-verified checkpoint a previous (e.g. preempted) attempt uploaded."""
    import urllib.error
    import urllib.request

    download_path = output_dir / "resume_download.pt"

    if args.resume_checkpoint_get_url:
        download_file(args.resume_checkpoint_get_url, download_path)
        expected_sha = args.resume_checkpoint_sha256
        if args.resume_checkpoint_sha_get_url:
            expected_sha = urllib.request.urlopen(args.resume_checkpoint_sha_get_url, timeout=120).read().decode().strip()
        return place_verified_checkpoint(
            download_path, expected_sha, exp_module, output_dir, report_path, source="explicit"
        )

    for step, upload in auto_resume_candidates(args, exp_module.CHECKPOINT_PREFIX):
        try:
            # SHA sidecar first: it is uploaded right after the checkpoint, so its
            # presence means the checkpoint object is complete.
            expected_sha_text = download_text(upload.get("sha_get_url", ""))
            if expected_sha_text is None:
                continue
            download_file(upload["get_url"], download_path)
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                append_event(report_path, {"event": "auto_resume_candidate_missing", "step": step, "code": error.code})
                continue
            raise
        try:
            return place_verified_checkpoint(
                download_path,
                expected_sha_text.strip(),
                exp_module,
                output_dir,
                report_path,
                source="auto_resume",
            )
        except (ValueError, RuntimeError, OSError) as error:
            # ValueError: sha sidecar and checkpoint bytes disagree (mid-upload race).
            # RuntimeError/OSError: torch.load on a checkpoint that was truncated at
            # save time — its sha matches its own truncated bytes, so only the load
            # fails. Either way, try the next (older) candidate rather than failing
            # the whole attempt (and every retry after it) on one bad object.
            append_event(report_path, {"event": "auto_resume_candidate_bad", "step": step, "error": repr(error)})
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="iters.exp_baseline_lr2e3")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed torch/numpy/random before exp.train(). Labels the run's RNG draw; "
                        "GPU kernel nondeterminism still prevents bit-exact replay.")
    parser.add_argument("--artifact-manifest-get-url", required=True)
    parser.add_argument("--upload-every-s", type=float, default=60)
    parser.add_argument("--resume-checkpoint-get-url", default="")
    parser.add_argument("--resume-checkpoint-sha-get-url", default="")
    parser.add_argument("--resume-checkpoint-sha256", default="")
    parser.add_argument("--metric-step", type=float, default=50000)
    args = parser.parse_args()

    # Defaults for the attributes load_artifact_manifest() populates, so the except
    # handler below can still report a failure (empty URL means the upload no-ops)
    # when loading the manifest itself is what raised.
    args.artifact_uploads = []
    args.status_put_url = ""
    args.report_put_url = ""
    args.latest_put_url = ""

    output_dir = Path("run_output")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "report.jsonl"
    status_path = output_dir / "status.json"

    append_event(
        report_path,
        {
            "event": "start",
            "mode": "canonical_train",
            "exp": args.exp,
            "cwd": os.getcwd(),
            "python": sys.version,
        },
    )
    stop_event = None
    sync_thread = None
    sync_once = None
    try:
        load_artifact_manifest(args, report_path)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "started",
                "mode": "canonical_train",
                "report_bytes": report_path.stat().st_size,
            },
        )
        append_event(report_path, {"event": "nvidia_smi", "result": run_command(["nvidia-smi"])})
        configure_hf_cache(Path("hf_cache"), report_path)
        ensure_import("numpy", "numpy", report_path)
        ensure_import("datasets", ["datasets==4.4.1", "pyarrow==22.0.0", "pandas==2.3.3"], report_path)

        import torch

        exp_module = importlib.import_module(args.exp)

        append_event(
            report_path,
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
                "event": "torch_status",
                "torch_version": torch.__version__,
            },
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for canonical Viridian training")

        resume_step = prepare_resume_checkpoint(args, exp_module, output_dir, report_path)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "runtime_ready",
                "mode": "canonical_train",
                "report_bytes": report_path.stat().st_size,
                "resume_step": resume_step,
            },
        )
        upload_report(report_path, args.report_put_url)

        if args.seed is not None:
            import random

            import numpy as np

            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed_all(args.seed)
            append_event(report_path, {"event": "seeded", "seed": args.seed})

        stop_event, sync_thread, sync_once = start_sync_thread(output_dir, args, report_path, status_path)
        append_event(report_path, {"event": "canonical_train_start", "exp": args.exp, "output_dir": str(output_dir)})
        result = exp_module.train(output_dir=str(output_dir))
        append_event(report_path, {"event": "canonical_train_returned", "result": repr(result)})

        stop_event.set()
        sync_thread.join(timeout=30)
        uploaded = sync_once(force=True)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "finished",
                "mode": "canonical_train",
                "report_bytes": report_path.stat().st_size,
                "uploaded_count": len(uploaded),
            },
        )
        upload_report(report_path, args.report_put_url)
        print(f"METRIC: {args.metric_step}", flush=True)
        return 0
    except Exception as error:
        if stop_event is not None:
            stop_event.set()
        if sync_thread is not None:
            sync_thread.join(timeout=30)
        if sync_once is not None:
            try:
                sync_once(force=True)
            except Exception as sync_error:
                append_event(report_path, {"event": "sync_after_failure_failed", "error": repr(sync_error)})
        error_report = {"event": "failed", "error": repr(error), "traceback": traceback.format_exc()[-6000:]}
        append_event(report_path, error_report)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "failed",
                "mode": "canonical_train",
                "error": repr(error),
                "traceback_tail": error_report["traceback"][-2000:],
                "report_bytes": report_path.stat().st_size,
            },
        )
        upload_report(report_path, args.report_put_url)
        print("METRIC: 0", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
