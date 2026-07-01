import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def run_command(args):
    try:
        completed = subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError:
        return {"ok": False, "error": f"{args[0]} not found"}
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
    }


def upload_file(path, url):
    if not url:
        return {"skipped": True}
    data = path.read_bytes()
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"content-type": "application/octet-stream"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return {"status": response.status, "bytes": len(data)}


def download_file(url, path):
    if not url:
        return {"skipped": True}
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except urllib.error.HTTPError as error:
        return {"ok": False, "status": error.code, "error": str(error)}
    path.write_bytes(data)
    return {"ok": True, "bytes": len(data)}


def append_event(log_path, event):
    event = {"time": time.time(), **event}
    with log_path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    print(json.dumps(event, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--checkpoint-every", type=int, default=60)
    parser.add_argument("--matrix-size", type=int, default=1024)
    parser.add_argument("--checkpoint-download-url", default="")
    parser.add_argument("--checkpoint-upload-url", default="")
    parser.add_argument("--log-upload-url", default="")
    args = parser.parse_args()

    output_dir = Path("probe_output")
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.pt"
    log_path = output_dir / "report.jsonl"

    append_event(
        log_path,
        {
            "event": "start",
            "cwd": os.getcwd(),
            "python": sys.version,
            "platform": platform.platform(),
            "seconds": args.seconds,
            "matrix_size": args.matrix_size,
        },
    )
    append_event(log_path, {"event": "nvidia_smi", "result": run_command(["nvidia-smi"])})

    try:
        import torch
    except Exception as error:
        append_event(log_path, {"event": "torch_import_failed", "error": repr(error)})
        upload_file(log_path, args.log_upload_url)
        print("METRIC: 0", flush=True)
        return 0

    append_event(
        log_path,
        {
            "event": "torch_status",
            "torch_version": torch.__version__,
            "torch_cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        },
    )

    if not torch.cuda.is_available():
        upload_file(log_path, args.log_upload_url)
        print("METRIC: 0", flush=True)
        return 0

    device = torch.device("cuda")
    append_event(log_path, {"event": "device", "name": torch.cuda.get_device_name(0)})

    resume_result = download_file(args.checkpoint_download_url, checkpoint_path)
    append_event(log_path, {"event": "checkpoint_download", "result": resume_result})

    size = args.matrix_size
    torch.manual_seed(0)
    input_matrix = torch.randn(size, size, device=device)
    weights = torch.randn(size, size, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weights], lr=1e-4)
    step = 0

    if checkpoint_path.exists() and resume_result.get("ok"):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        weights.data.copy_(checkpoint["weights"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        append_event(log_path, {"event": "resumed", "step": step})

    deadline = time.monotonic() + args.seconds
    next_checkpoint = time.monotonic() + args.checkpoint_every
    last_loss = None
    checkpoints_written = 0

    while time.monotonic() < deadline:
        output = input_matrix @ weights
        loss = output.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        last_loss = float(loss.detach().cpu())

        now = time.monotonic()
        if now >= next_checkpoint:
            torch.save(
                {
                    "step": step,
                    "loss": last_loss,
                    "weights": weights.detach().cpu(),
                    "optimizer": optimizer.state_dict(),
                    "saved_at": time.time(),
                },
                checkpoint_path,
            )
            checkpoints_written += 1
            append_event(log_path, {"event": "checkpoint", "step": step, "loss": last_loss})
            append_event(
                log_path,
                {
                    "event": "checkpoint_upload",
                    "result": upload_file(checkpoint_path, args.checkpoint_upload_url),
                },
            )
            upload_file(log_path, args.log_upload_url)
            next_checkpoint = now + args.checkpoint_every

    torch.cuda.synchronize()
    torch.save(
        {
            "step": step,
            "loss": last_loss,
            "weights": weights.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "saved_at": time.time(),
        },
        checkpoint_path,
    )
    checkpoints_written += 1
    checkpoint_size = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
    log_size = log_path.stat().st_size if log_path.exists() else 0
    append_event(
        log_path,
        {
            "event": "finish",
            "step": step,
            "loss": last_loss,
            "checkpoints_written": checkpoints_written,
            "checkpoint_size": checkpoint_size,
            "log_size": log_size,
        },
    )
    if step <= 0 or checkpoints_written <= 0 or checkpoint_size <= 0 or log_size <= 0:
        append_event(
            log_path,
            {
                "event": "validation_failed",
                "step": step,
                "checkpoints_written": checkpoints_written,
                "checkpoint_size": checkpoint_size,
                "log_size": log_size,
            },
        )
        print("METRIC: 0", flush=True)
        return 0
    append_event(
        log_path,
        {
            "event": "final_checkpoint_upload",
            "result": upload_file(checkpoint_path, args.checkpoint_upload_url),
        },
    )
    final_log_upload = upload_file(log_path, args.log_upload_url)
    append_event(log_path, {"event": "final_log_upload", "result": final_log_upload})
    upload_file(log_path, args.log_upload_url)
    print(f"METRIC: {step}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
