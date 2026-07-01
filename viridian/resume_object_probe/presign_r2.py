from __future__ import annotations

import argparse
import configparser
import datetime as dt
import hashlib
import hmac
import json
import os
import urllib.parse
from pathlib import Path


REGION = "auto"
SERVICE = "s3"


def quote(value: str) -> str:
    return urllib.parse.quote(value, safe="-_.~")


def signing_key(secret_key: str, date: str) -> bytes:
    date_key = hmac.new(("AWS4" + secret_key).encode(), date.encode(), hashlib.sha256).digest()
    region_key = hmac.new(date_key, REGION.encode(), hashlib.sha256).digest()
    service_key = hmac.new(region_key, SERVICE.encode(), hashlib.sha256).digest()
    return hmac.new(service_key, b"aws4_request", hashlib.sha256).digest()


def canonical_query(params: dict[str, str]) -> str:
    return "&".join(f"{quote(key)}={quote(value)}" for key, value in sorted(params.items()))


def presign_url(
    *,
    method: str,
    endpoint: str,
    access_key_id: str,
    secret_access_key: str,
    bucket: str,
    object_name: str,
    expires: int,
    now: dt.datetime,
) -> str:
    endpoint = endpoint.rstrip("/")
    parsed_endpoint = urllib.parse.urlparse(endpoint)
    host = parsed_endpoint.netloc
    date = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%dT%H%M%SZ")
    credential_scope = f"{date}/{REGION}/{SERVICE}/aws4_request"
    path = "/" + "/".join(quote(part) for part in [bucket, *object_name.split("/")])
    params = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{access_key_id}/{credential_scope}",
        "X-Amz-Date": timestamp,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    query = canonical_query(params)
    canonical_request = "\n".join(
        [
            method,
            path,
            query,
            f"host:{host}\n",
            "host",
            "UNSIGNED-PAYLOAD",
        ]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            timestamp,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    signature = hmac.new(signing_key(secret_access_key, date), string_to_sign.encode(), hashlib.sha256).hexdigest()
    return f"{endpoint}{path}?{query}&X-Amz-Signature={signature}"


def read_rclone_remote(config_path: Path, remote_name: str) -> dict[str, str]:
    config = configparser.ConfigParser()
    config.read(config_path)
    if remote_name not in config:
        raise SystemExit(f"missing rclone remote: {remote_name}")
    remote = config[remote_name]
    required = ["access_key_id", "secret_access_key", "endpoint"]
    missing = [key for key in required if key not in remote]
    if missing:
        raise SystemExit(f"missing rclone config keys: {', '.join(missing)}")
    return {key: remote[key] for key in required}


def build_spec(*, digest: str, eval_cmd: str, metric: str, timeout: int) -> dict:
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
        },
        "gpu_tier": "l4",
        "agents": 0,
        "margin": 0.0001,
        "limits": {
            "max_gens": 1,
            "budget": 1000000,
            "turn_timeout_s": timeout,
            "gpu_timeout_s": timeout,
        },
    }


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--digest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--bucket", default="sotaku-viridian")
    parser.add_argument("--prefix", default="viridian-runner-probes")
    parser.add_argument("--remote", default="r2")
    parser.add_argument("--config", default=str(Path.home() / ".config/rclone/rclone.conf"))
    parser.add_argument("--expires", type=int, default=6 * 60 * 60)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--matrix-size", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    remote = read_rclone_remote(Path(args.config), args.remote)
    now = dt.datetime.now(dt.timezone.utc)
    object_prefix = f"{args.prefix}/runs/{args.run_id}"

    def url(method: str, name: str) -> str:
        return presign_url(
            method=method,
            endpoint=remote["endpoint"],
            access_key_id=remote["access_key_id"],
            secret_access_key=remote["secret_access_key"],
            bucket=args.bucket,
            object_name=f"{object_prefix}/{name}",
            expires=args.expires,
            now=now,
        )

    urls = {
        "checkpoint_put": url("PUT", "checkpoint.pt"),
        "checksum_put": url("PUT", "checkpoint.pt.sha256"),
        "fresh_report_put": url("PUT", "fresh_report.jsonl"),
        "checkpoint_get": url("GET", "checkpoint.pt"),
        "checksum_get": url("GET", "checkpoint.pt.sha256"),
        "resume_checkpoint_put": url("PUT", "resume_checkpoint.pt"),
        "resume_checksum_put": url("PUT", "resume_checkpoint.pt.sha256"),
        "resume_report_put": url("PUT", "resume_report.jsonl"),
    }

    fresh_cmd = " ".join(
        [
            "python3 eval.py",
            "--mode fresh",
            f"--steps {args.steps}",
            f"--matrix-size {args.matrix_size}",
            "--checkpoint-put-url",
            shell_quote(urls["checkpoint_put"]),
            "--checksum-put-url",
            shell_quote(urls["checksum_put"]),
            "--fresh-report-put-url",
            shell_quote(urls["fresh_report_put"]),
        ]
    )
    resume_cmd = " ".join(
        [
            "python3 eval.py",
            "--mode resume",
            f"--steps {args.steps}",
            f"--matrix-size {args.matrix_size}",
            "--checkpoint-get-url",
            shell_quote(urls["checkpoint_get"]),
            "--checksum-get-url",
            shell_quote(urls["checksum_get"]),
            "--resume-checkpoint-put-url",
            shell_quote(urls["resume_checkpoint_put"]),
            "--resume-checksum-put-url",
            shell_quote(urls["resume_checksum_put"]),
            "--resume-report-put-url",
            shell_quote(urls["resume_report_put"]),
        ]
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fresh_path = out_dir / "fresh.job.json"
    resume_path = out_dir / "resume.job.json"
    manifest_path = out_dir / "manifest.safe.json"
    fresh_path.write_text(json.dumps(build_spec(digest=args.digest, eval_cmd=fresh_cmd, metric="fresh_step", timeout=args.timeout), indent=2))
    resume_path.write_text(json.dumps(build_spec(digest=args.digest, eval_cmd=resume_cmd, metric="resume_step", timeout=args.timeout), indent=2))
    manifest_path.write_text(
        json.dumps(
            {
                "bucket": args.bucket,
                "object_prefix": object_prefix,
                "run_id": args.run_id,
                "expires_at": (now + dt.timedelta(seconds=args.expires)).isoformat(),
                "fresh_job": str(fresh_path),
                "resume_job": str(resume_path),
            },
            indent=2,
        )
    )
    os.chmod(fresh_path, 0o600)
    os.chmod(resume_path, 0o600)
    print(json.dumps({"fresh_job": str(fresh_path), "resume_job": str(resume_path), "manifest": str(manifest_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
