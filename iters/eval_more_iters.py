"""Evaluate supported loop checkpoints with saved settings and per-puzzle records."""

import argparse
import json
import os
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from checkpoint_utils import atomic_json_save, validate_config
from dataset_utils import (
    DATASET_NAME, DATASET_REVISION, RATING_BUCKETS, benchmark_manifest, validate_benchmark,
)
from inference import RecurrentRunner, validate_iterations
from model_io import load_model
from runtime_utils import file_sha256, runtime_manifest
from stabilize.exp_testbed_20k import encode_puzzles, encode_solutions


def evaluate(model_path, exp_module=None, iter_counts=(16, 32, 64, 128),
             max_test=5000, device="cuda", output_dir=None, *, manifest_path=None,
             benchmark_path=None, precision=None, batch_size=256, compiled=False,
             matmul_precision=None, track_solutions=False, legacy_defaults=False,
             dataset=None):
    iterations = validate_iterations(iter_counts)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; pass --device cpu explicitly")
    precision = precision or "fp32"
    if precision not in ("fp32", "bf16") or (precision == "bf16" and device.type != "cuda"):
        raise ValueError("Use fp32 on CPU, or fp32/bf16 on CUDA")
    matmul_precision = matmul_precision or ("highest" if precision == "fp32" else "high")
    if matmul_precision not in ("high", "highest"):
        raise ValueError("matmul_precision must be high or highest")
    model, model_manifest = load_model(
        model_path, manifest_path=manifest_path, legacy_exp=exp_module,
        legacy_defaults=legacy_defaults, device=device,
    )
    torch.set_float32_matmul_precision(matmul_precision)
    if dataset is None:
        dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="test")
    if benchmark_path:
        benchmark = json.loads(Path(benchmark_path).read_text())
    else:
        benchmark = benchmark_manifest(dataset, per_bucket=max_test)
    rows = validate_benchmark(dataset, benchmark)
    if output_dir is None:
        output_dir = Path("runs") / f"eval_{Path(model_path).stem}_{uuid.uuid4().hex[:10]}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = runtime_manifest((
        "model_io.py", "inference.py", "iters/eval_more_iters.py",
        "stabilize/exp_testbed_20k.py", "dataset_utils.py", "iters/state_norm.py",
    ))
    identity = {
        "weights_sha256": file_sha256(model_path), "model": model_manifest["model"],
        "benchmark_rows_sha256": benchmark["rows_sha256"], "iterations": list(iterations),
        "precision": precision, "matmul_precision": matmul_precision,
        "batch_size": batch_size, "compiled": compiled, "chunk_iterations": 16,
        "track_solutions_every_iteration": track_solutions, "device_type": device.type,
        "implementation_sha256": environment["source_sha256"],
        "runtime": {key: environment.get(key) for key in (
            "python", "packages", "installed_packages", "inductor", "cuda", "gpu", "gpu_capability", "cudnn", "driver_version",
            "tf32_matmul", "tf32_cudnn", "sdpa_flash", "sdpa_memory_efficient", "sdpa_math",
            "bf16_reduced_precision_reduction",
        )},
    }
    identity_path = output_dir / "identity.json"
    if identity_path.exists():
        validate_config(json.loads(identity_path.read_text()), identity)
        if (output_dir / "result.json").exists():
            result = json.loads((output_dir / "result.json").read_text())
            validate_config(result["identity"], identity)
            if file_sha256(output_dir / "per_puzzle.npz") != result["per_puzzle_sha256"]:
                raise ValueError("Saved evaluation arrays do not match their checksum")
            print(f"Reusing completed evaluation in {output_dir}")
            return result
    else:
        atomic_json_save(identity, identity_path)
    atomic_json_save(benchmark, output_dir / "benchmark.json")
    atomic_json_save(environment, output_dir / "environment.json")
    questions, answers = rows["question"], rows["answer"]
    inputs = encode_puzzles(questions)
    targets = encode_solutions(answers).long()
    empty_masks = inputs[:, :, 0].bool()
    predictions_by_horizon = {horizon: [] for horizon in iterations}
    diagnostics_parts = {}
    runner = RecurrentRunner(model, compiled=compiled, track_solutions=track_solutions)
    start_time = time.monotonic()
    log_path = output_dir / f"{Path(model_path).stem}_eval.log"
    with log_path.open("a", buffering=1) as log_file:
        def log(message):
            print(message, flush=True)
            log_file.write(message + "\n")

        log_settings = {key: value for key, value in identity.items()
                        if key not in ("runtime", "implementation_sha256")}
        log(f"Evaluation: {json.dumps(log_settings, sort_keys=True)}")
        log("Complete runtime and source hashes: environment.json and identity.json")
        log(f"Total test puzzles: {len(rows)}; output: {output_dir}")
        for start in range(0, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16")
            with autocast:
                outputs, diagnostics = runner.run_batch(
                    inputs[start:end].to(device), iterations,
                    targets=targets[start:end].to(device), empty_mask=empty_masks[start:end].to(device),
                )
            for horizon, logits in outputs.items():
                predictions_by_horizon[horizon].append(logits.argmax(-1).to(torch.uint8).cpu())
            if diagnostics:
                for key, values in diagnostics.items():
                    diagnostics_parts.setdefault(key, []).append(values)
            if end == len(rows) or (start // batch_size + 1) % 10 == 0:
                log(f"Completed {end}/{len(rows)} puzzles through iteration {iterations[-1]}")
        arrays = {"indices": np.asarray(benchmark["indices"], dtype=np.int64)}
        summary, previous_solved = {}, None
        for horizon in iterations:
            predictions = torch.cat(predictions_by_horizon[horizon], dim=0)
            solved = ((predictions == targets) | ~empty_masks).all(-1)
            count = int(solved.sum())
            bucket_results = {}
            for _, _, name in RATING_BUCKETS:
                selected = torch.tensor([value == name for value in benchmark["bucket_names"]])
                bucket_results[name] = {"solved": int(solved[selected].sum()), "total": int(selected.sum())}
            row = {"solved": count, "total": len(rows), "accuracy_percent": 100 * count / len(rows), "buckets": bucket_results}
            if previous_solved is not None:
                row["lost_since_previous_recorded_horizon"] = int((previous_solved & ~solved).sum())
                row["gained_since_previous_recorded_horizon"] = int((~previous_solved & solved).sum())
            previous_solved = solved
            summary[str(horizon)] = row
            arrays[f"predictions_{horizon}"] = predictions.numpy()
            arrays[f"solved_{horizon}"] = solved.numpy()
            log(f"{horizon:5d} | {count}/{len(rows)} | {100 * count / len(rows):.3f}%")
        for key, parts in diagnostics_parts.items():
            arrays[key] = torch.cat(parts).numpy()
        temporary_arrays = output_dir / "per_puzzle.npz.tmp"
        with temporary_arrays.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary_arrays, output_dir / "per_puzzle.npz")
        result = {
            "identity": identity, "scores": summary,
            "seconds_including_compilation": time.monotonic() - start_time,
            "per_puzzle_file": "per_puzzle.npz", "per_puzzle_sha256": file_sha256(output_dir / "per_puzzle.npz"),
            "benchmark": {key: value for key, value in benchmark.items() if key not in ("indices", "bucket_names")},
        }
        if track_solutions:
            result["solution_tracking"] = {
                "ever_solved": int(arrays["ever_solved"].sum()),
                "stayed_solved_after_first": int(arrays["stayed_solved_after_first"].sum()),
                "puzzles_with_a_regression": int((arrays["regression_count"] > 0).sum()),
                "observation_interval": 1,
            }
        atomic_json_save(result, output_dir / "result.json")
        log(f"Saved per-puzzle records and result.json in {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("--manifest")
    parser.add_argument("--exp", help="Legacy plain model module, only with --legacy-defaults")
    parser.add_argument("--legacy-defaults", action="store_true")
    parser.add_argument("--iters", type=int, nargs="+", default=[128, 1024, 2048, 4096])
    parser.add_argument("--max-test", type=int, default=5000, help="Puzzles per rating bucket")
    parser.add_argument("--benchmark", help="Frozen benchmark JSON; overrides --max-test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--matmul-precision", choices=("high", "highest"),
                        help="Defaults to highest for FP32, high for BF16")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--track-solutions", action="store_true")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    evaluate(
        args.model_path, args.exp, args.iters, args.max_test, args.device, args.output_dir,
        manifest_path=args.manifest, benchmark_path=args.benchmark, precision=args.precision,
        matmul_precision=args.matmul_precision, batch_size=args.batch_size,
        compiled=args.compiled, track_solutions=args.track_solutions, legacy_defaults=args.legacy_defaults,
    )


if __name__ == "__main__":
    main()
