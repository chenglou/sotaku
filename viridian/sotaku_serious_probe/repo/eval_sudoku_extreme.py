import argparse
import json
import os
import time
import traceback
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from eval import (
    append_event,
    claim_artifact_slot,
    configure_hf_cache,
    download_file,
    download_json,
    ensure_import,
    file_sha256,
    run_command,
    should_skip_duplicate_attempt,
    upload_report,
    upload_status,
    upload_text,
)


RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]


def run_with_iters(model, x, n_iters, exp):
    batch_size = x.size(0)
    device = x.device
    rope_cos = exp.ROPE_COS.to(device)
    rope_sin = exp.ROPE_SIN.to(device)

    h_prev = model.initial_encoder(x)
    preds = torch.zeros(batch_size, 81, 9, device=device)

    for _ in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        logits = model.output_head(h)
        preds = F.softmax(logits, dim=-1)
    return logits


def select_test_rows(test_dataset, max_test):
    import random

    bucket_indices = {}
    for row_index in range(len(test_dataset)):
        rating = test_dataset[row_index]["rating"]
        for min_rating, max_rating, name in RATING_BUCKETS:
            if min_rating <= rating <= max_rating:
                bucket_indices.setdefault(name, []).append(row_index)
                break

    random.seed(42)
    for name, indices in bucket_indices.items():
        if len(indices) > max_test:
            bucket_indices[name] = random.sample(indices, max_test)

    puzzles = []
    answers = []
    bucket_names = []
    bucket_order = [bucket[2] for bucket in RATING_BUCKETS]
    for name in bucket_order:
        for row_index in bucket_indices.get(name, []):
            row = test_dataset[row_index]
            puzzles.append(row["question"])
            answers.append(row["answer"])
            bucket_names.append(name)
    return puzzles, answers, bucket_names


def evaluate(args, report_path, status_path):
    ensure_import("numpy", "numpy", report_path)
    ensure_import("datasets", ["datasets==4.4.1", "pyarrow", "pandas"], report_path)

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
        raise RuntimeError("CUDA is required for this Viridian eval")
    device = torch.device(args.device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    append_event(report_path, {"event": "device", "name": device_name})

    checkpoint_path = Path("run_output") / "model.pt"
    download_result = download_file(args.checkpoint_get_url, checkpoint_path)
    with urllib.request.urlopen(args.checkpoint_sha_get_url, timeout=120) as response:
        expected_sha = response.read().decode().strip()
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
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model = exp.SudokuTransformer().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    checkpoint_step = checkpoint.get("step") if isinstance(checkpoint, dict) else None
    append_event(
        report_path,
        {
            "event": "model_loaded",
            "checkpoint_step": checkpoint_step,
            "checkpoint_keys": sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else None,
        },
    )

    upload_status(
        status_path,
        args.status_put_url,
        {
            "event": "model_loaded",
            "mode": "eval",
            "checkpoint_step": checkpoint_step,
            "report_bytes": report_path.stat().st_size,
        },
    )

    test_features = Features(
        {
            "question": Value("string"),
            "answer": Value("string"),
            "rating": Value("int32"),
        }
    )
    data_started = time.monotonic()
    append_event(report_path, {"event": "dataset_load_start", "dataset": "sapientinc/sudoku-extreme", "split": "test"})
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test", features=test_features)
    puzzles, answers, bucket_names = select_test_rows(test_dataset, args.max_test)
    append_event(
        report_path,
        {
            "event": "dataset_loaded",
            "available_test": len(test_dataset),
            "selected_test": len(puzzles),
            "elapsed_s": round(time.monotonic() - data_started, 3),
            "bucket_counts": {name: bucket_names.count(name) for _, _, name in RATING_BUCKETS},
        },
    )

    x_all = exp.encode_puzzles(puzzles).to(device)
    solution_targets = torch.tensor([[int(answer[j]) - 1 for j in range(81)] for answer in answers], dtype=torch.long)
    empty_masks = torch.tensor([[puzzle[j] == "." for j in range(81)] for puzzle in puzzles])
    n_total = len(puzzles)

    append_event(report_path, {"event": "encoded", "x_shape": list(x_all.shape), "n_total": n_total})
    upload_report(report_path, args.report_put_url)

    bucket_order = [bucket[2] for bucket in RATING_BUCKETS]
    results = []
    best_accuracy = 0.0
    use_autocast = device.type == "cuda"
    progress_every_s = args.progress_every_s

    for n_iters in args.iters:
        iter_started = time.monotonic()
        next_progress = iter_started + progress_every_s
        predictions = []
        context = torch.autocast(device.type, dtype=torch.bfloat16) if use_autocast else torch.no_grad()
        with torch.no_grad(), context:
            for start in range(0, n_total, args.batch_size):
                end = min(start + args.batch_size, n_total)
                logits = run_with_iters(model, x_all[start:end], n_iters, exp)
                predictions.append(logits.argmax(dim=-1).cpu())
                now = time.monotonic()
                if now >= next_progress:
                    upload_status(
                        status_path,
                        args.status_put_url,
                        {
                            "event": "running",
                            "mode": "eval",
                            "checkpoint_step": checkpoint_step,
                            "iters": n_iters,
                            "evaluated": end,
                            "n_total": n_total,
                            "elapsed_s": round(now - iter_started, 3),
                            "report_bytes": report_path.stat().st_size,
                        },
                    )
                    next_progress = now + progress_every_s

        elapsed_s = time.monotonic() - iter_started
        all_predictions = torch.cat(predictions, dim=0)
        correct = (all_predictions == solution_targets) & empty_masks
        solved = correct.sum(dim=1) == empty_masks.sum(dim=1)
        total_solved = int(solved.sum().item())
        accuracy = total_solved / n_total
        best_accuracy = max(best_accuracy, accuracy)

        bucket_results = {}
        for name in bucket_order:
            mask = torch.tensor([bucket_name == name for bucket_name in bucket_names])
            bucket_total = int(mask.sum().item())
            bucket_solved = int((solved & mask).sum().item())
            bucket_results[name] = {
                "solved": bucket_solved,
                "total": bucket_total,
                "accuracy": bucket_solved / bucket_total if bucket_total else None,
            }

        result = {
            "iters": n_iters,
            "solved": total_solved,
            "total": n_total,
            "accuracy": accuracy,
            "elapsed_s": round(elapsed_s, 3),
            "bucket_results": bucket_results,
        }
        results.append(result)
        append_event(report_path, {"event": "eval_result", **result})
        upload_report(report_path, args.report_put_url)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "eval_result",
                "mode": "eval",
                "checkpoint_step": checkpoint_step,
                "iters": n_iters,
                "accuracy": accuracy,
                "solved": total_solved,
                "total": n_total,
                "elapsed_s": round(elapsed_s, 3),
                "report_bytes": report_path.stat().st_size,
            },
        )

    summary = {
        "artifact_attempt_id": args.artifact_attempt_id,
        "artifact_slot": args.artifact_slot,
        "checkpoint_sha256": actual_sha,
        "checkpoint_step": checkpoint_step,
        "event": "finished",
        "max_test": args.max_test,
        "results": results,
    }
    upload_text(json.dumps(summary, sort_keys=True) + "\n", args.latest_put_url)
    upload_status(
        status_path,
        args.status_put_url,
        {
            "event": "finished",
            "mode": "eval",
            "checkpoint_step": checkpoint_step,
            "best_accuracy": best_accuracy,
            "best_accuracy_percent": best_accuracy * 100,
            "report_bytes": report_path.stat().st_size,
        },
    )
    upload_report(report_path, args.report_put_url)
    print(f"METRIC: {best_accuracy * 100}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-get-url", required=True)
    parser.add_argument("--checkpoint-sha-get-url", required=True)
    parser.add_argument("--artifact-manifest-get-url", required=True)
    parser.add_argument("--iters", type=int, nargs="+", required=True)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--progress-every-s", type=float, default=60)
    parser.add_argument("--duplicate-primary-fresh-s", type=float, default=300)
    args = parser.parse_args()

    output_dir = Path("run_output")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "report.jsonl"
    status_path = output_dir / "status.json"

    append_event(
        report_path,
        {
            "event": "start",
            "mode": "eval",
            "cwd": os.getcwd(),
            "python": __import__("sys").version,
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
                "mode": "eval",
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
                    "mode": "eval",
                    "report_bytes": report_path.stat().st_size,
                },
            )
            upload_report(report_path, args.report_put_url)
            print("METRIC: 0", flush=True)
            return 0
        append_event(report_path, {"event": "nvidia_smi", "result": run_command(["nvidia-smi"])})
        configure_hf_cache(Path("hf_cache"), report_path)
        evaluate(args, report_path, status_path)
        return 0
    except Exception as error:
        error_report = {"event": "failed", "error": repr(error), "traceback": traceback.format_exc()[-6000:]}
        append_event(report_path, error_report)
        upload_status(
            status_path,
            args.status_put_url,
            {
                "event": "failed",
                "mode": "eval",
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
