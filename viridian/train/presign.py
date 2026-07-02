"""R2 presigning for Viridian training jobs (runs locally, not inside the job).

Reads R2 credentials from the rclone config (~/.config/rclone/rclone.conf) and builds
SigV4 presigned URLs, plus the job manifest that viridian_train.py loads its upload
URLs from. Viridian jobs have no R2 credentials; they only ever see presigned URLs.
"""

import configparser
import datetime as dt
import hashlib
import hmac
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
    encoded_key = urllib.parse.quote(key, safe="/")
    canonical_uri = f"/{bucket}/{encoded_key}"

    query = {
        "X-Amz-Algorithm": "AWS4-HMAC-SHA256",
        "X-Amz-Credential": f"{config['access_key']}/{credential_scope}",
        "X-Amz-Date": amz_date,
        "X-Amz-Expires": str(expires),
        "X-Amz-SignedHeaders": "host",
    }
    canonical_query = "&".join(
        f"{urllib.parse.quote(k, safe='')}={urllib.parse.quote(v, safe='')}" for k, v in sorted(query.items())
    )
    canonical_request = "\n".join(
        [
            method,
            canonical_uri,
            canonical_query,
            f"host:{host}\n",
            "host",
            "UNSIGNED-PAYLOAD",
        ]
    )
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            credential_scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    key = signing_key(config["secret_key"], date_stamp, region, service)
    signature = hmac.new(key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    return f"{config['endpoint']}{canonical_uri}?{canonical_query}&X-Amz-Signature={signature}"


def build_train_manifest(config, bucket, prefix, expires, upload_paths):
    """Manifest with presigned URLs for one training job. Each expected output file
    gets PUT and GET URLs plus SHA sidecar URLs; the GET URLs let a restarted job
    download its own newest verified checkpoint and resume instead of training from
    step 0. Viridian runs one evaluator per job (the historical duplicate-evaluator
    bug is fixed platform-side), so there is a single output prefix — no attempt
    slots or lease claiming."""
    clean_prefix = prefix.rstrip("/")

    def url(method, name):
        return presign_url(config, method, bucket, f"{clean_prefix}/{name}", expires)

    uploads = []
    for rel_path in upload_paths:
        object_name = f"outputs/{rel_path}"
        sha_name = f"{object_name}.sha256"
        uploads.append(
            {
                "object_name": object_name,
                "path": rel_path,
                "put_url": url("PUT", object_name),
                "get_url": url("GET", object_name),
                "sha_name": sha_name,
                "sha_put_url": url("PUT", sha_name),
                "sha_get_url": url("GET", sha_name),
            }
        )
    return {
        "bucket": bucket,
        "layout": "canonical-train-v3",
        "prefix": clean_prefix,
        "status_put_url": url("PUT", "status.json"),
        "report_put_url": url("PUT", "report.jsonl"),
        "latest_put_url": url("PUT", "latest.json"),
        "uploads": uploads,
    }
