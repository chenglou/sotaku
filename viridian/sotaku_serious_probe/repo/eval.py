import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


def append_event(path, event):
    row = {"time": time.time(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"EVENT: {row.get('event', 'event')}", flush=True)


def write_json(path, payload):
    path.write_text(json.dumps({"time": time.time(), **payload}, sort_keys=True) + "\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upload_bytes(data, url, content_type="application/octet-stream", extra_headers=None):
    if not url:
        return {"skipped": True}
    headers = {"content-type": content_type}
    if extra_headers:
        headers.update(extra_headers)
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return {"status": response.status, "bytes": len(data)}


def upload_file(path, url, content_type="application/octet-stream"):
    if not url:
        return {"skipped": True}
    return upload_bytes(path.read_bytes(), url, content_type=content_type)


def upload_text(text, url):
    return upload_bytes(text.encode(), url, content_type="application/json")


def download_file(url, path):
    if not url:
        return {"skipped": True}
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": file_sha256(path)}


def download_json(url):
    if not url:
        return None
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode())


def run_command(args, timeout=120):
    try:
        completed = subprocess.run(args, text=True, capture_output=True, check=False, timeout=timeout)
    except FileNotFoundError:
        return {"ok": False, "error": f"{args[0]} not found"}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def ensure_import(module_name, pip_packages, report_path):
    packages = pip_packages if isinstance(pip_packages, list) else [pip_packages]
    force_reinstall = False
    try:
        __import__(module_name)
        append_event(report_path, {"event": "dependency_present", "module": module_name})
        return
    except ModuleNotFoundError:
        append_event(report_path, {"event": "dependency_missing", "module": module_name})
    except Exception as error:
        force_reinstall = True
        append_event(report_path, {"event": "dependency_broken", "module": module_name, "error": repr(error)})
    command = [sys.executable, "-m", "pip", "install"]
    if force_reinstall:
        command.extend(["--upgrade", "--force-reinstall"])
    command.extend(packages)
    result = run_command(command, timeout=600)
    append_event(report_path, {"event": "dependency_install", "module": module_name, "packages": packages, "result": result})
    __import__(module_name)


def configure_hf_cache(cache_dir, report_path):
    cache_dir.mkdir(parents=True, exist_ok=True)
    datasets_cache = cache_dir / "datasets"
    datasets_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir.resolve())
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache.resolve())
    append_event(
        report_path,
        {
            "event": "hf_cache_configured",
            "hf_home": os.environ["HF_HOME"],
            "hf_datasets_cache": os.environ["HF_DATASETS_CACHE"],
        },
    )


def get_lr(step, *, lr, warmup_steps, total_steps, lr_min_ratio):
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine_decay = 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress))
    return lr * (lr_min_ratio + (1 - lr_min_ratio) * cosine_decay)


def build_config(args, train_data_sha256):
    return {
        "experiment": "exp_baseline_lr2e3",
        "probe": "viridian_sotaku_serious",
        "hf_dataset": "sapientinc/sudoku-extreme",
        "train_size": args.train_size,
        "batch_size": args.batch_size,
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "lr": args.lr,
        "lr_min_ratio": args.lr_min_ratio,
        "compile": args.compile,
        "train_data_sha256": train_data_sha256,
        "model_config": {
            "d_model": 128,
            "d_ff": 512,
            "n_layers": 4,
            "n_iterations": 16,
        },
    }


def verify_config(checkpoint, expected_config):
    saved_config = checkpoint.get("config")
    if saved_config != expected_config:
        raise ValueError(f"checkpoint config mismatch: saved={saved_config}, expected={expected_config}")


def save_checkpoint(path, model, optimizer, step, config):
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in model.state_dict().items()}
    import torch

    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def upload_status(status_path, status_url, status):
    write_json(status_path, status)
    return upload_file(status_path, status_url, content_type="application/json")


def upload_report(report_path, report_url):
    return upload_file(report_path, report_url, content_type="application/json")


def upload_checkpoint_set(checkpoint_path, checkpoint_sha_path, latest_path, args, step, checkpoint_sha):
    checkpoint_uploads = getattr(args, "checkpoint_uploads", None)
    if checkpoint_uploads is not None:
        upload_index = args.checkpoint_upload_index
        if upload_index >= len(checkpoint_uploads):
            raise RuntimeError(f"no checkpoint upload URL left for upload index {upload_index}")
        upload = checkpoint_uploads[upload_index]
        args.checkpoint_upload_index += 1
        latest = {
            "artifact_attempt_id": args.artifact_attempt_id,
            "artifact_slot": args.artifact_slot,
            "checkpoint": upload["checkpoint_name"],
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_sha256_file": upload["checkpoint_sha_name"],
            "step": step,
            "upload_index": upload_index,
        }
        latest_path.write_text(json.dumps(latest, sort_keys=True) + "\n")
        return {
            "checkpoint": upload_file(checkpoint_path, upload["checkpoint_put_url"]),
            "checkpoint_sha": upload_file(checkpoint_sha_path, upload["checkpoint_sha_put_url"]),
            "latest": upload_file(latest_path, args.latest_put_url, content_type="application/json"),
            "upload_index": upload_index,
        }

    latest = {
        "checkpoint": "latest_checkpoint.pt",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_sha256_file": "latest_checkpoint.pt.sha256",
        "step": step,
    }
    latest_path.write_text(json.dumps(latest, sort_keys=True) + "\n")
    return {
        "checkpoint": upload_file(checkpoint_path, args.checkpoint_put_url),
        "checkpoint_sha": upload_file(checkpoint_sha_path, args.checkpoint_sha_put_url),
        "latest": upload_file(latest_path, args.latest_put_url, content_type="application/json"),
    }


def claim_artifact_slot(args, report_path):
    manifest = download_json(args.artifact_manifest_get_url)
    if manifest is None:
        args.artifact_attempt_id = ""
        args.artifact_slot = None
        args.primary_status_get_url = ""
        args.checkpoint_uploads = None
        args.artifact_uploads = []
        args.checkpoint_upload_index = 0
        return

    args.primary_status_get_url = manifest["slots"][0].get("status_get_url", "")
    claim_base = {
        "cwd": os.getcwd(),
        "event": "artifact_slot_claim",
        "pid": os.getpid(),
        "platform": platform.platform(),
        "python": sys.version,
        "time": time.time(),
    }
    for slot in manifest["slots"]:
        attempt_id = f"slot-{slot['slot']:03d}-{int(claim_base['time'])}-{claim_base['pid']}"
        claim = {**claim_base, "attempt_id": attempt_id, "slot": slot["slot"]}
        data = json.dumps(claim, sort_keys=True).encode()
        try:
            result = upload_bytes(
                data,
                slot["lease_put_url"],
                content_type="application/json",
                extra_headers={"If-None-Match": "*"},
            )
        except urllib.error.HTTPError as error:
            if error.code == 412:
                append_event(report_path, {"event": "artifact_slot_taken", "slot": slot["slot"]})
                continue
            raise
        args.artifact_attempt_id = attempt_id
        args.artifact_slot = slot["slot"]
        args.status_put_url = slot["status_put_url"]
        args.report_put_url = slot["report_put_url"]
        args.latest_put_url = slot["latest_put_url"]
        args.checkpoint_uploads = slot["checkpoints"]
        args.artifact_uploads = slot.get("uploads", [])
        args.checkpoint_upload_index = 0
        append_event(
            report_path,
            {
                "event": "artifact_slot_claimed",
                "attempt_id": attempt_id,
                "result": result,
                "slot": slot["slot"],
            },
        )
        return
    raise RuntimeError("no free artifact slot available")


def should_skip_duplicate_attempt(args, report_path):
    if args.artifact_slot in (None, 0):
        return False
    if args.duplicate_primary_fresh_s <= 0:
        return False
    if not args.primary_status_get_url:
        append_event(
            report_path,
            {
                "event": "duplicate_guard_no_primary_status_url",
                "artifact_slot": args.artifact_slot,
            },
        )
        return False

    try:
        primary_status = download_json(args.primary_status_get_url)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            append_event(report_path, {"event": "duplicate_guard_primary_status_missing"})
            return False
        raise

    if not primary_status:
        append_event(report_path, {"event": "duplicate_guard_primary_status_empty"})
        return False

    primary_event = primary_status.get("event")
    primary_age_s = None
    if isinstance(primary_status.get("time"), (int, float)):
        primary_age_s = time.time() - primary_status["time"]
    primary_is_fresh = primary_age_s is not None and primary_age_s <= args.duplicate_primary_fresh_s
    primary_is_finished = primary_event == "finished"
    primary_is_failed = primary_event == "failed"
    should_skip = not primary_is_failed and (primary_is_finished or primary_is_fresh)

    append_event(
        report_path,
        {
            "event": "duplicate_guard_checked_primary",
            "artifact_slot": args.artifact_slot,
            "primary_event": primary_event,
            "primary_age_s": round(primary_age_s, 3) if primary_age_s is not None else None,
            "primary_step": primary_status.get("step"),
            "should_skip": should_skip,
        },
    )
    return should_skip


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    parser.add_argument("--duration-s", type=float, default=3600)
    parser.add_argument("--train-size", type=int, default=2_700_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--total-steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=1400)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-min-ratio", type=float, default=0.01)
    parser.add_argument("--progress-every-s", type=float, default=60)
    parser.add_argument("--upload-every-s", type=float, default=60)
    parser.add_argument("--first-checkpoint-s", type=float, default=120)
    parser.add_argument("--checkpoint-every-s", type=float, default=300)
    parser.add_argument("--max-report-bytes", type=int, default=10_000_000)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-get-url", default="")
    parser.add_argument("--checkpoint-sha-get-url", default="")
    parser.add_argument("--checkpoint-put-url", default="")
    parser.add_argument("--checkpoint-sha-put-url", default="")
    parser.add_argument("--latest-put-url", default="")
    parser.add_argument("--report-put-url", default="")
    parser.add_argument("--status-put-url", default="")
    parser.add_argument("--artifact-manifest-get-url", default="")
    parser.add_argument("--duplicate-primary-fresh-s", type=float, default=0)
    args = parser.parse_args()

    output_dir = Path("run_output")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "report.jsonl"
    status_path = output_dir / "status.json"
    checkpoint_path = output_dir / "latest_checkpoint.pt"
    checkpoint_sha_path = output_dir / "latest_checkpoint.pt.sha256"
    latest_path = output_dir / "latest.json"
    cache_dir = Path("hf_cache")
    start_step = 0

    append_event(
        report_path,
        {
            "event": "start",
            "mode": args.mode,
            "cwd": os.getcwd(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    )
    try:
        claim_artifact_slot(args, report_path)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "artifact_attempt_id": args.artifact_attempt_id,
                "artifact_slot": args.artifact_slot,
                "event": "started",
                "mode": args.mode,
                "step": None,
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
                    "mode": args.mode,
                    "step": 0,
                    "report_bytes": report_path.stat().st_size,
                },
            )
            upload_report(report_path, args.report_put_url)
            print("METRIC: 0", flush=True)
            return 0
        append_event(report_path, {"event": "nvidia_smi", "result": run_command(["nvidia-smi"])})
        configure_hf_cache(cache_dir, report_path)
        ensure_import("numpy", "numpy", report_path)
        ensure_import("datasets", ["datasets==4.4.1", "pyarrow", "pandas"], report_path)

        import numpy as np
        import torch
        import torch.nn.functional as F
        from datasets import Features, Value, load_dataset
        from iters import exp_baseline_lr2e3 as exp

        append_event(
            report_path,
            {
                "event": "torch_status",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            },
        )
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this Viridian run")
        device = torch.device(args.device)
        device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        append_event(report_path, {"event": "device", "name": device_name})
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "runtime_ready",
                "mode": args.mode,
                "step": start_step,
                "device": device_name,
                "report_bytes": report_path.stat().st_size,
            },
        )

        data_started = time.monotonic()
        append_event(report_path, {"event": "dataset_load_start", "dataset": "sapientinc/sudoku-extreme"})
        sudoku_features = Features(
            {
                "question": Value("string"),
                "answer": Value("string"),
                "rating": Value("int32"),
            }
        )
        train_dataset = load_dataset("sapientinc/sudoku-extreme", split="train", features=sudoku_features)
        train_size = min(args.train_size, len(train_dataset))
        append_event(
            report_path,
            {
                "event": "dataset_loaded",
                "available_train": len(train_dataset),
                "train_size": train_size,
                "elapsed_s": round(time.monotonic() - data_started, 3),
            },
        )
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "dataset_loaded",
                "mode": args.mode,
                "step": start_step,
                "train_size": train_size,
                "elapsed_s": round(time.monotonic() - data_started, 3),
                "report_bytes": report_path.stat().st_size,
            },
        )

        encode_started = time.monotonic()
        append_event(report_path, {"event": "encode_start", "train_size": train_size})
        train_rows = train_dataset[:train_size]
        ratings = np.asarray(train_rows["rating"], dtype=np.int16)
        puzzles_all = train_rows["question"]
        solutions_all = train_rows["answer"]
        train_digest = __import__("hashlib").sha256(
            ("".join(puzzles_all[:1024]) + "".join(solutions_all[:1024])).encode()
        ).hexdigest()
        x_all = exp.encode_puzzles(puzzles_all)
        targets_all = exp.encode_solutions(solutions_all)
        del puzzles_all, solutions_all, train_rows, train_dataset

        train_data = {}
        for min_rating, max_rating, name in exp.RATING_BUCKETS:
            idx = np.where((ratings >= min_rating) & (ratings <= max_rating))[0]
            if idx.size == 0:
                continue
            train_data[(min_rating, max_rating)] = {
                "idx": torch.from_numpy(idx),
                "size": int(idx.size),
                "name": name,
            }
        phase_buckets = {}
        for _start, _end, min_rating, phase_name in exp.PHASES:
            buckets = [bucket for bucket in train_data if bucket[0] >= min_rating]
            phase_buckets[min_rating] = buckets
        append_event(
            report_path,
            {
                "event": "encoded",
                "elapsed_s": round(time.monotonic() - encode_started, 3),
                "x_shape": list(x_all.shape),
                "targets_shape": list(targets_all.shape),
                "bucket_sizes": {train_data[bucket]["name"]: train_data[bucket]["size"] for bucket in train_data},
                "train_digest_first_1024": train_digest,
            },
        )
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "encoded",
                "mode": args.mode,
                "step": start_step,
                "elapsed_s": round(time.monotonic() - encode_started, 3),
                "report_bytes": report_path.stat().st_size,
            },
        )
        upload_report(report_path, args.report_put_url)

        config = build_config(args, train_digest)
        model = exp.SudokuTransformer().to(device)
        optimizer_state = None

        if args.mode == "resume":
            download_result = download_file(args.checkpoint_get_url, checkpoint_path)
            expected_sha = urllib.request.urlopen(args.checkpoint_sha_get_url, timeout=120).read().decode().strip()
            actual_sha = file_sha256(checkpoint_path)
            append_event(
                report_path,
                {
                    "event": "downloaded_checkpoint",
                    "result": download_result,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                },
            )
            if expected_sha != actual_sha:
                raise ValueError("downloaded checkpoint SHA-256 did not match sidecar")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            verify_config(checkpoint, config)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = checkpoint["optimizer_state_dict"]
            start_step = int(checkpoint["step"])
            append_event(report_path, {"event": "resumed", "step": start_step})
            upload_status(
                status_path,
                args.status_put_url,
                {"event": "resumed", "mode": args.mode, "step": start_step, "report_bytes": report_path.stat().st_size},
            )

        if args.compile:
            model = torch.compile(model)
            append_event(report_path, {"event": "compiled"})
            upload_status(
                status_path,
                args.status_put_url,
                {"event": "compiled", "mode": args.mode, "step": start_step, "report_bytes": report_path.stat().st_size},
            )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

        x_all = x_all.to(device)
        targets_all = targets_all.to(device, dtype=torch.long)
        for bucket_data in train_data.values():
            bucket_data["idx"] = bucket_data["idx"].to(device)
        generator = torch.Generator(device=device)
        generator.manual_seed(20260701 + start_step)

        def active_buckets_for_step(step):
            for phase_start, phase_end, min_rating, phase_name in exp.PHASES:
                if phase_start <= step < phase_end:
                    return phase_buckets[min_rating], phase_name
            return phase_buckets[0], "Phase 4: All"

        def sample_batch(active_buckets):
            sizes = np.array([train_data[bucket]["size"] for bucket in active_buckets], dtype=np.int64)
            probabilities = sizes / sizes.sum()
            counts = np.random.multinomial(args.batch_size, probabilities)
            x_parts = []
            target_parts = []
            for bucket, count in zip(active_buckets, counts):
                if count == 0:
                    continue
                bucket_idx = train_data[bucket]["idx"]
                selected = bucket_idx[
                    torch.randint(0, train_data[bucket]["size"], (count,), device=device, generator=generator)
                ]
                x_parts.append(x_all[selected])
                target_parts.append(targets_all[selected])
            x_batch = torch.cat(x_parts, dim=0)
            target_batch = torch.cat(target_parts, dim=0)
            permutation = torch.randperm(x_batch.size(0), device=device)
            return x_batch[permutation], target_batch[permutation]

        model.train()
        train_started = time.monotonic()
        deadline = train_started + args.duration_s
        next_progress = train_started
        next_upload = train_started
        next_checkpoint = train_started + args.first_checkpoint_s
        current_step = start_step
        last_loss = None
        last_acc = None
        current_phase = None

        while time.monotonic() < deadline and current_step < args.total_steps:
            active_buckets, phase_name = active_buckets_for_step(current_step)
            if phase_name != current_phase:
                current_phase = phase_name
                append_event(report_path, {"event": "phase", "step": current_step, "phase": phase_name})
            lr = get_lr(
                current_step,
                lr=args.lr,
                warmup_steps=args.warmup_steps,
                total_steps=args.total_steps,
                lr_min_ratio=args.lr_min_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            x_batch, target_batch = sample_batch(active_buckets)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits_by_iter = model(x_batch, return_all=True)
                mask = x_batch[:, :, 0].to(dtype=torch.float32)
                loss = 0
                for logits in logits_by_iter:
                    per_cell = F.cross_entropy(logits.reshape(-1, 9), target_batch.reshape(-1), reduction="none")
                    per_cell = per_cell.view(target_batch.size(0), 81)
                    loss = loss + (per_cell * mask).sum() / mask.sum()
                loss = loss / len(logits_by_iter)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                predictions = logits_by_iter[-1].argmax(dim=-1)
                correct = (predictions == target_batch) & (mask > 0)
                last_acc = float(correct.sum().detach().cpu()) / float(mask.sum().detach().cpu())
                last_loss = float(loss.detach().cpu())
            current_step += 1
            now = time.monotonic()

            if now >= next_progress:
                elapsed_s = now - train_started
                steps_done = current_step - start_step
                append_event(
                    report_path,
                    {
                        "event": "progress",
                        "step": current_step,
                        "elapsed_s": round(elapsed_s, 3),
                        "steps_per_sec": round(steps_done / max(elapsed_s, 1e-6), 4),
                        "lr": lr,
                        "loss": last_loss,
                        "train_acc": last_acc,
                        "phase": current_phase,
                        "report_bytes": report_path.stat().st_size,
                    },
                )
                next_progress = now + args.progress_every_s

            if report_path.stat().st_size > args.max_report_bytes:
                append_event(
                    report_path,
                    {
                        "event": "report_size_limit_exceeded",
                        "step": current_step,
                        "report_bytes": report_path.stat().st_size,
                        "max_report_bytes": args.max_report_bytes,
                    },
                )
                break

            if now >= next_upload:
                upload_status(
                    status_path,
                    args.status_put_url,
                    {
                        "event": "running",
                        "mode": args.mode,
                        "step": current_step,
                        "elapsed_s": round(now - train_started, 3),
                        "loss": last_loss,
                        "train_acc": last_acc,
                        "phase": current_phase,
                        "report_bytes": report_path.stat().st_size,
                    },
                )
                upload_report(report_path, args.report_put_url)
                next_upload = now + args.upload_every_s

            if now >= next_checkpoint:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                save_checkpoint(checkpoint_path, model, optimizer, current_step, config)
                checkpoint_sha = file_sha256(checkpoint_path)
                checkpoint_sha_path.write_text(checkpoint_sha + "\n")
                upload_results = upload_checkpoint_set(
                    checkpoint_path,
                    checkpoint_sha_path,
                    latest_path,
                    args,
                    current_step,
                    checkpoint_sha,
                )
                append_event(
                    report_path,
                    {
                        "event": "checkpoint_uploaded",
                        "step": current_step,
                        "checkpoint_bytes": checkpoint_path.stat().st_size,
                        "checkpoint_sha256": checkpoint_sha,
                        "results": upload_results,
                    },
                )
                next_checkpoint = now + args.checkpoint_every_s

        if device.type == "cuda":
            torch.cuda.synchronize()
        save_checkpoint(checkpoint_path, model, optimizer, current_step, config)
        checkpoint_sha = file_sha256(checkpoint_path)
        checkpoint_sha_path.write_text(checkpoint_sha + "\n")
        upload_results = upload_checkpoint_set(checkpoint_path, checkpoint_sha_path, latest_path, args, current_step, checkpoint_sha)
        append_event(
            report_path,
            {
                "event": "finished",
                "step": current_step,
                "loss": last_loss,
                "train_acc": last_acc,
                "report_bytes": report_path.stat().st_size,
                "checkpoint_sha256": checkpoint_sha,
                "upload_results": upload_results,
            },
        )
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "finished",
                "mode": args.mode,
                "step": current_step,
                "loss": last_loss,
                "train_acc": last_acc,
                "report_bytes": report_path.stat().st_size,
                "checkpoint_sha256": checkpoint_sha,
            },
        )
        upload_report(report_path, args.report_put_url)
        print(f"METRIC: {current_step}", flush=True)
        return 0
    except Exception as error:
        error_report = {"event": "failed", "error": repr(error), "traceback": traceback.format_exc()[-6000:]}
        append_event(report_path, error_report)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "failed",
                "mode": args.mode,
                "step": None,
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
