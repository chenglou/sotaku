import argparse
import json
import os
import threading
import time
import traceback
from pathlib import Path

from eval import (
    append_event,
    claim_artifact_slot,
    configure_hf_cache,
    download_file,
    ensure_import,
    file_sha256,
    run_command,
    should_skip_duplicate_attempt,
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
            "artifact_attempt_id": args.artifact_attempt_id,
            "artifact_slot": args.artifact_slot,
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
                "artifact_attempt_id": args.artifact_attempt_id,
                "artifact_slot": args.artifact_slot,
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


def prepare_resume_checkpoint(args, output_dir, report_path):
    if not args.resume_checkpoint_get_url:
        return None

    import torch
    from iters import exp_baseline_lr2e3 as exp

    download_path = output_dir / "resume_download.pt"
    download_result = download_file(args.resume_checkpoint_get_url, download_path)
    expected_sha = args.resume_checkpoint_sha256
    if args.resume_checkpoint_sha_get_url:
        import urllib.request

        expected_sha = urllib.request.urlopen(args.resume_checkpoint_sha_get_url, timeout=120).read().decode().strip()
    actual_sha = file_sha256(download_path)
    append_event(
        report_path,
        {
            "actual_sha256": actual_sha,
            "event": "resume_checkpoint_downloaded",
            "expected_sha256": expected_sha,
            "result": download_result,
        },
    )
    if expected_sha and actual_sha != expected_sha:
        raise ValueError("downloaded resume checkpoint SHA-256 did not match")

    checkpoint = torch.load(download_path, map_location="cpu", weights_only=False)
    step = int(checkpoint["step"])
    checkpoint_path = output_dir / f"{exp.CHECKPOINT_PREFIX}{step}.pt"
    download_path.replace(checkpoint_path)
    append_event(report_path, {"event": "resume_checkpoint_prepared", "path": checkpoint_path.name, "step": step})
    return step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-manifest-get-url", required=True)
    parser.add_argument("--upload-every-s", type=float, default=60)
    parser.add_argument("--duplicate-primary-fresh-s", type=float, default=300)
    parser.add_argument("--resume-checkpoint-get-url", default="")
    parser.add_argument("--resume-checkpoint-sha-get-url", default="")
    parser.add_argument("--resume-checkpoint-sha256", default="")
    parser.add_argument("--metric-step", type=float, default=50000)
    args = parser.parse_args()

    output_dir = Path("run_output")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "report.jsonl"
    status_path = output_dir / "status.json"

    append_event(
        report_path,
        {
            "event": "start",
            "mode": "canonical_train",
            "cwd": os.getcwd(),
            "python": __import__("sys").version,
        },
    )
    stop_event = None
    sync_thread = None
    sync_once = None
    try:
        claim_artifact_slot(args, report_path)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "artifact_attempt_id": args.artifact_attempt_id,
                "artifact_slot": args.artifact_slot,
                "event": "started",
                "mode": "canonical_train",
                "report_bytes": report_path.stat().st_size,
            },
        )
        if should_skip_duplicate_attempt(args, report_path):
            upload_status(
                status_path,
                args.status_put_url,
                {
                    "artifact_attempt_id": args.artifact_attempt_id,
                    "artifact_slot": args.artifact_slot,
                    "event": "duplicate_attempt_skipped",
                    "mode": "canonical_train",
                    "report_bytes": report_path.stat().st_size,
                },
            )
            upload_report(report_path, args.report_put_url)
            print("METRIC: 0", flush=True)
            return 0

        append_event(report_path, {"event": "nvidia_smi", "result": run_command(["nvidia-smi"])})
        configure_hf_cache(Path("hf_cache"), report_path)
        ensure_import("numpy", "numpy", report_path)
        ensure_import("datasets", ["datasets==4.4.1", "pyarrow==22.0.0", "pandas==2.3.3"], report_path)

        import torch
        from iters import exp_baseline_lr2e3 as exp

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

        resume_step = prepare_resume_checkpoint(args, output_dir, report_path)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "artifact_attempt_id": args.artifact_attempt_id,
                "artifact_slot": args.artifact_slot,
                "event": "runtime_ready",
                "mode": "canonical_train",
                "report_bytes": report_path.stat().st_size,
                "resume_step": resume_step,
            },
        )
        upload_report(report_path, args.report_put_url)

        stop_event, sync_thread, sync_once = start_sync_thread(output_dir, args, report_path, status_path)
        append_event(report_path, {"event": "canonical_train_start", "output_dir": str(output_dir)})
        result = exp.train(output_dir=str(output_dir))
        append_event(report_path, {"event": "canonical_train_returned", "result": repr(result)})

        stop_event.set()
        sync_thread.join(timeout=30)
        uploaded = sync_once(force=True)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "artifact_attempt_id": args.artifact_attempt_id,
                "artifact_slot": args.artifact_slot,
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
