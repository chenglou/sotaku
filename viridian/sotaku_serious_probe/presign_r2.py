import argparse
import configparser
import datetime as dt
import hashlib
import hmac
import json
import urllib.request
import urllib.parse
from pathlib import Path


def shell_quote(value):
    return "'" + value.replace("'", "'\"'\"'") + "'"


def read_r2_config(remote):
    config_path = Path.home() / ".config" / "rclone" / "rclone.conf"
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if remote not in parser:
        raise SystemExit(f"missing rclone remote: {remote}")
    section = parser[remote]
    return {
        "access_key": section["access_key_id"],
        "secret_key": section["secret_access_key"],
        "endpoint": section["endpoint"].rstrip("/"),
    }


def signing_key(secret_key, date_stamp, region, service):
    key_date = hmac.new(("AWS4" + secret_key).encode(), date_stamp.encode(), hashlib.sha256).digest()
    key_region = hmac.new(key_date, region.encode(), hashlib.sha256).digest()
    key_service = hmac.new(key_region, service.encode(), hashlib.sha256).digest()
    return hmac.new(key_service, b"aws4_request", hashlib.sha256).digest()


def presign_url(config, method, bucket, key, expires):
    region = "auto"
    service = "s3"
    now = dt.datetime.utcnow()
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")
    host = urllib.parse.urlparse(config["endpoint"]).netloc
    credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
    credential = f"{config['access_key']}/{credential_scope}"
    encoded_key = "/".join(urllib.parse.quote(part, safe="") for part in key.split("/"))
    canonical_uri = f"/{bucket}/{encoded_key}"
    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": credential,
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}"
        for k, v in sorted(query.items())
    )
    canonical_headers = f"host:{host}\n"
    signed_headers = "host"
    payload_hash = "UNSIGNED-PAYLOAD"
    canonical_request = "\n".join([method, canonical_uri, canonical_query, canonical_headers, signed_headers, payload_hash])
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(signing_key(config["secret_key"], date_stamp, region, service), string_to_sign.encode(), hashlib.sha256).hexdigest()
    return f"{config['endpoint']}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"


def upload_bytes(data, url):
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return {"status": response.status, "bytes": len(data)}


def build_attempt_slot_manifest(config, bucket, prefix, expires, attempt_slots, max_checkpoint_uploads):
    clean_prefix = prefix.rstrip("/")

    def url(method, name):
        return presign_url(config, method, bucket, f"{clean_prefix}/{name}", expires)

    slots = []
    for slot_index in range(attempt_slots):
        slot_name = f"attempts/slot-{slot_index:03d}"
        checkpoints = []
        for upload_index in range(max_checkpoint_uploads):
            checkpoint_name = f"{slot_name}/checkpoints/checkpoint-{upload_index:05d}.pt"
            checkpoint_sha_name = f"{checkpoint_name}.sha256"
            checkpoints.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_put_url": url("PUT", checkpoint_name),
                    "checkpoint_sha_name": checkpoint_sha_name,
                    "checkpoint_sha_put_url": url("PUT", checkpoint_sha_name),
                    "upload_index": upload_index,
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
                "checkpoints": checkpoints,
            }
        )
    return {
        "bucket": bucket,
        "prefix": clean_prefix,
        "layout": "attempt-slots-v1",
        "attempt_slots": attempt_slots,
        "max_checkpoint_uploads": max_checkpoint_uploads,
        "slots": slots,
    }


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
            "metric": "training_step",
            "direction": "max",
            "locked_paths": ["eval.py", "iters/exp_baseline_lr2e3.py", "checkpoint_utils.py"],
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
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument("--budget", type=int, default=10_000_000)
    parser.add_argument("--duration-s", type=int, default=3600)
    parser.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    parser.add_argument("--resume-checkpoint-key", default="")
    parser.add_argument("--resume-checkpoint-sha-key", default="")
    parser.add_argument("--train-size", type=int, default=2_700_000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--progress-every-s", type=int, default=60)
    parser.add_argument("--upload-every-s", type=int, default=60)
    parser.add_argument("--first-checkpoint-s", type=int, default=120)
    parser.add_argument("--checkpoint-every-s", type=int, default=300)
    parser.add_argument("--max-report-bytes", type=int, default=10_000_000)
    parser.add_argument("--artifact-layout", choices=["attempt-slots", "fixed"], default="attempt-slots")
    parser.add_argument("--attempt-slots", type=int, default=8)
    parser.add_argument("--max-checkpoint-uploads", type=int, default=128)
    parser.add_argument("--duplicate-primary-fresh-s", type=int, default=300)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    config = read_r2_config(args.remote)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    compile_flag = "--no-compile" if args.no_compile else "--compile"
    if args.mode == "resume" and (not args.resume_checkpoint_key or not args.resume_checkpoint_sha_key):
        raise SystemExit("--mode resume requires --resume-checkpoint-key and --resume-checkpoint-sha-key")

    eval_parts = [
        "python3 eval.py",
        f"--mode {args.mode}",
        f"--duration-s {args.duration_s}",
        f"--train-size {args.train_size}",
        f"--batch-size {args.batch_size}",
        f"--progress-every-s {args.progress_every_s}",
        f"--upload-every-s {args.upload_every_s}",
        f"--first-checkpoint-s {args.first_checkpoint_s}",
        f"--checkpoint-every-s {args.checkpoint_every_s}",
        f"--max-report-bytes {args.max_report_bytes}",
        "--device cuda",
        compile_flag,
    ]
    if args.mode == "resume":
        eval_parts.extend(
            [
                "--checkpoint-get-url",
                shell_quote(presign_url(config, "GET", args.bucket, args.resume_checkpoint_key, args.expires)),
                "--checkpoint-sha-get-url",
                shell_quote(presign_url(config, "GET", args.bucket, args.resume_checkpoint_sha_key, args.expires)),
            ]
        )

    private_manifest_key = None
    if args.artifact_layout == "attempt-slots":
        private_manifest_key = f"{args.prefix.rstrip('/')}/private/presigned_artifact_manifest.json"
        private_manifest = build_attempt_slot_manifest(
            config,
            args.bucket,
            args.prefix,
            args.expires,
            args.attempt_slots,
            args.max_checkpoint_uploads,
        )
        private_manifest_url = presign_url(config, "PUT", args.bucket, private_manifest_key, args.expires)
        upload_bytes(json.dumps(private_manifest, sort_keys=True).encode(), private_manifest_url)
        manifest_get_url = presign_url(config, "GET", args.bucket, private_manifest_key, args.expires)
        eval_parts.extend(["--artifact-manifest-get-url", shell_quote(manifest_get_url)])
        if args.duplicate_primary_fresh_s > 0:
            eval_parts.extend(["--duplicate-primary-fresh-s", str(args.duplicate_primary_fresh_s)])
    else:
        def url(method, name):
            return presign_url(config, method, args.bucket, f"{args.prefix.rstrip('/')}/{name}", args.expires)

        urls = {
            "checkpoint_put": url("PUT", "latest_checkpoint.pt"),
            "checkpoint_sha_put": url("PUT", "latest_checkpoint.pt.sha256"),
            "latest_put": url("PUT", "latest.json"),
            "report_put": url("PUT", "report.jsonl"),
            "status_put": url("PUT", "status.json"),
        }
        eval_parts.extend(
            [
                "--checkpoint-put-url",
                shell_quote(urls["checkpoint_put"]),
                "--checkpoint-sha-put-url",
                shell_quote(urls["checkpoint_sha_put"]),
                "--latest-put-url",
                shell_quote(urls["latest_put"]),
                "--report-put-url",
                shell_quote(urls["report_put"]),
                "--status-put-url",
                shell_quote(urls["status_put"]),
            ]
        )

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
                "duration_s": args.duration_s,
                "mode": args.mode,
                "resume_checkpoint_key": args.resume_checkpoint_key if args.mode == "resume" else None,
                "resume_checkpoint_sha_key": args.resume_checkpoint_sha_key if args.mode == "resume" else None,
                "train_size": args.train_size,
                "batch_size": args.batch_size,
                "progress_every_s": args.progress_every_s,
                "upload_every_s": args.upload_every_s,
                "first_checkpoint_s": args.first_checkpoint_s,
                "checkpoint_every_s": args.checkpoint_every_s,
                "max_report_bytes": args.max_report_bytes,
                "compile": not args.no_compile,
                "artifact_layout": args.artifact_layout,
                "attempt_slots": args.attempt_slots if args.artifact_layout == "attempt-slots" else None,
                "max_checkpoint_uploads": args.max_checkpoint_uploads if args.artifact_layout == "attempt-slots" else None,
                "duplicate_primary_fresh_s": args.duplicate_primary_fresh_s if args.artifact_layout == "attempt-slots" else None,
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
