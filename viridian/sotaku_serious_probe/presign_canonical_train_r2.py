import argparse
import json
from pathlib import Path

from presign_r2 import presign_url, read_r2_config, shell_quote, upload_bytes


CANONICAL_CHECKPOINT_STEPS = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 49999]
CANONICAL_FILES = [
    "exp_baseline_lr2e3.log",
    "model_baseline_lr2e3.pt",
    *[f"baseline_lr2e3_checkpoint_step{step}.pt" for step in CANONICAL_CHECKPOINT_STEPS],
]


def build_manifest(config, bucket, prefix, expires, attempt_slots):
    clean_prefix = prefix.rstrip("/")

    def url(method, name):
        return presign_url(config, method, bucket, f"{clean_prefix}/{name}", expires)

    slots = []
    for slot_index in range(attempt_slots):
        slot_name = f"attempts/slot-{slot_index:03d}"
        uploads = []
        for rel_path in CANONICAL_FILES:
            object_name = f"{slot_name}/outputs/{rel_path}"
            sha_name = f"{object_name}.sha256"
            uploads.append(
                {
                    "object_name": object_name,
                    "path": rel_path,
                    "put_url": url("PUT", object_name),
                    "sha_name": sha_name,
                    "sha_put_url": url("PUT", sha_name),
                }
            )
        slots.append(
            {
                "slot": slot_index,
                "lease_name": f"{slot_name}/lease.json",
                "lease_get_url": url("GET", f"{slot_name}/lease.json"),
                "lease_put_url": url("PUT", f"{slot_name}/lease.json"),
                "status_name": f"{slot_name}/status.json",
                "status_get_url": url("GET", f"{slot_name}/status.json"),
                "status_put_url": url("PUT", f"{slot_name}/status.json"),
                "report_name": f"{slot_name}/report.jsonl",
                "report_put_url": url("PUT", f"{slot_name}/report.jsonl"),
                "latest_name": f"{slot_name}/latest.json",
                "latest_get_url": url("GET", f"{slot_name}/latest.json"),
                "latest_put_url": url("PUT", f"{slot_name}/latest.json"),
                "checkpoints": [],
                "uploads": uploads,
            }
        )
    return {
        "attempt_slots": attempt_slots,
        "bucket": bucket,
        "layout": "canonical-train-slots-v1",
        "prefix": clean_prefix,
        "slots": slots,
        "uploads": CANONICAL_FILES,
    }


def build_spec(digest, eval_cmd, timeout, budget):
    return {
        "customer": "self",
        "baseline": {
            "stack": {
                "stack": [
                    {
                        "bytes": 0,
                        "digest": digest,
                        "gen": 0,
                        "job": "seed",
                        "parent": None,
                    }
                ]
            }
        },
        "contract": {
            "eval_cmd": eval_cmd,
            "locked_paths": [
                "train_canonical.py",
                "eval.py",
                "iters/exp_baseline_lr2e3.py",
                "checkpoint_utils.py",
            ],
            "metric": "training_step",
            "direction": "max",
        },
        "gpu_tier": "b200",
        "agents": 0,
        "margin": 0.0001,
        "patience": 1,
        "limits": {
            "budget": budget,
            "gpu_timeout_s": timeout,
            "max_gens": 1,
            "turn_timeout_s": timeout,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bucket", default="sotaku-viridian")
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--remote", default="r2")
    parser.add_argument("--expires", type=int, default=24 * 60 * 60)
    parser.add_argument("--timeout", type=int, default=24 * 60 * 60)
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument("--attempt-slots", type=int, default=4)
    parser.add_argument("--upload-every-s", type=int, default=60)
    parser.add_argument("--duplicate-primary-fresh-s", type=int, default=300)
    parser.add_argument("--resume-checkpoint-key", default="")
    parser.add_argument("--resume-checkpoint-sha-key", default="")
    parser.add_argument("--resume-checkpoint-sha256", default="")
    args = parser.parse_args()

    config = read_r2_config(args.remote)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    private_manifest_key = f"{args.prefix.rstrip('/')}/private/presigned_artifact_manifest.json"
    private_manifest = build_manifest(config, args.bucket, args.prefix, args.expires, args.attempt_slots)
    private_manifest_url = presign_url(config, "PUT", args.bucket, private_manifest_key, args.expires)
    upload_bytes(json.dumps(private_manifest, sort_keys=True).encode(), private_manifest_url)
    manifest_get_url = presign_url(config, "GET", args.bucket, private_manifest_key, args.expires)

    eval_parts = [
        "python3 train_canonical.py",
        "--artifact-manifest-get-url",
        shell_quote(manifest_get_url),
        "--upload-every-s",
        str(args.upload_every_s),
        "--duplicate-primary-fresh-s",
        str(args.duplicate_primary_fresh_s),
    ]
    if args.resume_checkpoint_key:
        eval_parts.extend(
            [
                "--resume-checkpoint-get-url",
                shell_quote(presign_url(config, "GET", args.bucket, args.resume_checkpoint_key, args.expires)),
            ]
        )
        if args.resume_checkpoint_sha_key:
            eval_parts.extend(
                [
                    "--resume-checkpoint-sha-get-url",
                    shell_quote(presign_url(config, "GET", args.bucket, args.resume_checkpoint_sha_key, args.expires)),
                ]
            )
        if args.resume_checkpoint_sha256:
            eval_parts.extend(["--resume-checkpoint-sha256", args.resume_checkpoint_sha256])
    eval_cmd = " ".join(eval_parts)

    spec_path = out_dir / "job.json"
    spec_path.write_text(json.dumps(build_spec(args.digest, eval_cmd, args.timeout, args.budget), indent=2))
    (out_dir / "manifest.safe.json").write_text(
        json.dumps(
            {
                "attempt_slots": args.attempt_slots,
                "bucket": args.bucket,
                "gpu_tier": "b200",
                "private_manifest_key": private_manifest_key,
                "prefix": args.prefix,
                "resume_checkpoint_key": args.resume_checkpoint_key,
                "resume_checkpoint_sha_key": args.resume_checkpoint_sha_key,
                "spec": str(spec_path),
                "timeout": args.timeout,
                "upload_every_s": args.upload_every_s,
                "uploads": CANONICAL_FILES,
            },
            indent=2,
        )
    )
    print(spec_path)


if __name__ == "__main__":
    main()
