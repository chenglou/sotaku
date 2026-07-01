import argparse
import configparser
import datetime as dt
import hashlib
import hmac
import json
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
    canonical_request = "\n".join(
        [method, canonical_uri, canonical_query, canonical_headers, signed_headers, payload_hash]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(
        signing_key(config["secret_key"], date_stamp, region, service),
        string_to_sign.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{config['endpoint']}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"


def build_spec(digest, eval_cmd, metric, timeout, budget):
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
            "metric": metric,
            "direction": "max",
            "locked_paths": ["eval.py", "iters/exp_baseline_lr2e3.py", "checkpoint_utils.py", "data/tiny_sudoku.csv"],
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
    parser.add_argument("--expires", type=int, default=6 * 60 * 60)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--budget", type=int, default=5_000_000)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=0)
    parser.add_argument("--progress-every-s", type=float, default=60)
    parser.add_argument("--upload-every-s", type=float, default=60)
    parser.add_argument("--max-report-bytes", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    config = read_r2_config(args.remote)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def url(method, name):
        return presign_url(config, method, args.bucket, f"{args.prefix.rstrip('/')}/{name}", args.expires)

    urls = {
        "fresh_checkpoint_put": url("PUT", "fresh_checkpoint.pt"),
        "fresh_checkpoint_sha_put": url("PUT", "fresh_checkpoint.pt.sha256"),
        "fresh_report_put": url("PUT", "fresh_report.jsonl"),
        "fresh_status_put": url("PUT", "fresh_status.json"),
        "fresh_checkpoint_get": url("GET", "fresh_checkpoint.pt"),
        "fresh_checkpoint_sha_get": url("GET", "fresh_checkpoint.pt.sha256"),
        "resume_checkpoint_put": url("PUT", "resume_checkpoint.pt"),
        "resume_checkpoint_sha_put": url("PUT", "resume_checkpoint.pt.sha256"),
        "resume_report_put": url("PUT", "resume_report.jsonl"),
        "resume_status_put": url("PUT", "resume_status.json"),
    }
    compile_flag = "--no-compile" if args.no_compile else "--compile"
    common = (
        f"--steps {args.steps} "
        f"--duration-s {args.duration_s} "
        f"--progress-every-s {args.progress_every_s} "
        f"--upload-every-s {args.upload_every_s} "
        f"--max-report-bytes {args.max_report_bytes} "
        f"--batch-size {args.batch_size} "
        f"--device cuda {compile_flag}"
    )
    fresh_cmd = " ".join(
        [
            "python3 eval.py",
            "--mode fresh",
            common,
            "--checkpoint-put-url",
            shell_quote(urls["fresh_checkpoint_put"]),
            "--checkpoint-sha-put-url",
            shell_quote(urls["fresh_checkpoint_sha_put"]),
            "--report-put-url",
            shell_quote(urls["fresh_report_put"]),
            "--status-put-url",
            shell_quote(urls["fresh_status_put"]),
        ]
    )
    resume_cmd = " ".join(
        [
            "python3 eval.py",
            "--mode resume",
            common,
            "--checkpoint-get-url",
            shell_quote(urls["fresh_checkpoint_get"]),
            "--checkpoint-sha-get-url",
            shell_quote(urls["fresh_checkpoint_sha_get"]),
            "--checkpoint-put-url",
            shell_quote(urls["resume_checkpoint_put"]),
            "--checkpoint-sha-put-url",
            shell_quote(urls["resume_checkpoint_sha_put"]),
            "--report-put-url",
            shell_quote(urls["resume_report_put"]),
            "--status-put-url",
            shell_quote(urls["resume_status_put"]),
        ]
    )
    fresh_path = out_dir / "fresh.job.json"
    resume_path = out_dir / "resume.job.json"
    fresh_path.write_text(json.dumps(build_spec(args.digest, fresh_cmd, "fresh_step", args.timeout, args.budget), indent=2))
    resume_path.write_text(json.dumps(build_spec(args.digest, resume_cmd, "resume_step", args.timeout, args.budget), indent=2))
    (out_dir / "manifest.safe.json").write_text(
        json.dumps(
            {
                "bucket": args.bucket,
                "prefix": args.prefix,
                "fresh_spec": str(fresh_path),
                "resume_spec": str(resume_path),
                "gpu_tier": "b200",
                "steps": args.steps,
                "duration_s": args.duration_s,
                "progress_every_s": args.progress_every_s,
                "upload_every_s": args.upload_every_s,
                "max_report_bytes": args.max_report_bytes,
                "batch_size": args.batch_size,
                "compile": not args.no_compile,
                "timeout": args.timeout,
                "budget": args.budget,
            },
            indent=2,
        )
    )
    print(fresh_path)
    print(resume_path)


if __name__ == "__main__":
    main()
