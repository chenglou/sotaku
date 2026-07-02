"""R2/status/report plumbing for Viridian training jobs.

These helpers run inside the Viridian job. They handle durable state on R2:
status.json, report.jsonl, and artifact uploads with SHA-256 sidecars, all through
presigned URLs from the job manifest.

Extracted from the sotaku_serious_probe wrapper so the training entrypoint does not
have to import the (retired) recreated trainer. Training logic does not belong here.
The probe era also had attempt-slot claiming and a duplicate-attempt guard as a
defense against Viridian spawning overlapping duplicate evaluators; that platform bug
is fixed, so the defense is gone.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request


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


def upload_bytes(data, url, content_type="application/octet-stream"):
    if not url:
        return {"skipped": True}
    headers = {"content-type": content_type}
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


def download_text(url):
    if not url:
        return None
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode()


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


def upload_status(status_path, status_url, status):
    write_json(status_path, status)
    return upload_file(status_path, status_url, content_type="application/json")


def upload_report(report_path, report_url):
    return upload_file(report_path, report_url, content_type="application/json")


def load_artifact_manifest(args, report_path):
    """Load the presigned manifest and populate args with the job's R2 URLs. A missing
    manifest (empty --artifact-manifest-get-url) means "run without R2": every upload
    helper no-ops on an empty URL."""
    manifest = download_json(args.artifact_manifest_get_url)
    if manifest is None:
        args.artifact_uploads = []
        args.status_put_url = ""
        args.report_put_url = ""
        args.latest_put_url = ""
        return

    args.status_put_url = manifest["status_put_url"]
    args.report_put_url = manifest["report_put_url"]
    args.latest_put_url = manifest["latest_put_url"]
    args.artifact_uploads = manifest["uploads"]
    append_event(report_path, {"event": "artifact_manifest_loaded", "uploads": len(args.artifact_uploads)})
