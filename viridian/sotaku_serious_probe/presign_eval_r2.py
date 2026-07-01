import argparse
import json
from pathlib import Path

from presign_r2 import (
    build_attempt_slot_manifest,
    presign_url,
    read_r2_config,
    shell_quote,
    upload_bytes,
)


def build_spec(digest, eval_cmd, timeout, budget):
    return {
        "customer": "self",
        "baseline": {
            "stack": {
                "stack": [
                    {
                        "gen": 0,
                        "job": "seed",
                        "bytes": 0,
                        "parent": None,
                        "digest": digest,
                    }
                ]
            }
        },
        "contract": {
            "eval_cmd": eval_cmd,
            "metric": "accuracy",
            "direction": "max",
            "locked_paths": [
                "eval_sudoku_extreme.py",
                "eval.py",
                "iters/exp_baseline_lr2e3.py",
                "checkpoint_utils.py",
            ],
        },
        "gpu_tier": "b200",
        "agents": 0,
        "margin": 0.0001,
        "patience": 1,
        "limits": {
            "max_gens": 1,
            "budget": budget,
            "turn_timeout_s": timeout,
            "gpu_timeout_s": timeout,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bucket", default="sotaku-viridian")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--remote", default="r2")
    parser.add_argument("--expires", type=int, default=12 * 60 * 60)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument("--checkpoint-key", required=True)
    parser.add_argument("--checkpoint-sha-key", required=True)
    parser.add_argument("--iters", type=int, nargs="+", required=True)
    parser.add_argument("--max-test", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--progress-every-s", type=int, default=60)
    parser.add_argument("--attempt-slots", type=int, default=8)
    parser.add_argument("--duplicate-primary-fresh-s", type=int, default=300)
    args = parser.parse_args()

    config = read_r2_config(args.remote)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    private_manifest_key = f"{args.prefix.rstrip('/')}/private/presigned_artifact_manifest.json"
    private_manifest = build_attempt_slot_manifest(
        config,
        args.bucket,
        args.prefix,
        args.expires,
        args.attempt_slots,
        1,
    )
    private_manifest_url = presign_url(config, "PUT", args.bucket, private_manifest_key, args.expires)
    upload_bytes(json.dumps(private_manifest, sort_keys=True).encode(), private_manifest_url)
    manifest_get_url = presign_url(config, "GET", args.bucket, private_manifest_key, args.expires)

    eval_parts = [
        "python3 eval_sudoku_extreme.py",
        "--checkpoint-get-url",
        shell_quote(presign_url(config, "GET", args.bucket, args.checkpoint_key, args.expires)),
        "--checkpoint-sha-get-url",
        shell_quote(presign_url(config, "GET", args.bucket, args.checkpoint_sha_key, args.expires)),
        "--artifact-manifest-get-url",
        shell_quote(manifest_get_url),
        "--iters",
        *[str(value) for value in args.iters],
        "--max-test",
        str(args.max_test),
        "--batch-size",
        str(args.batch_size),
        "--device cuda",
        "--progress-every-s",
        str(args.progress_every_s),
        "--duplicate-primary-fresh-s",
        str(args.duplicate_primary_fresh_s),
    ]
    eval_cmd = " ".join(eval_parts)

    spec_path = out_dir / "job.json"
    spec_path.write_text(json.dumps(build_spec(args.digest, eval_cmd, args.timeout, args.budget), indent=2))
    (out_dir / "manifest.safe.json").write_text(
        json.dumps(
            {
                "bucket": args.bucket,
                "prefix": args.prefix,
                "spec": str(spec_path),
                "gpu_tier": "b200",
                "checkpoint_key": args.checkpoint_key,
                "checkpoint_sha_key": args.checkpoint_sha_key,
                "iters": args.iters,
                "max_test": args.max_test,
                "batch_size": args.batch_size,
                "attempt_slots": args.attempt_slots,
                "duplicate_primary_fresh_s": args.duplicate_primary_fresh_s,
                "private_manifest_key": private_manifest_key,
                "timeout": args.timeout,
                "budget": args.budget,
            },
            indent=2,
        )
    )
    print(spec_path)


if __name__ == "__main__":
    main()
