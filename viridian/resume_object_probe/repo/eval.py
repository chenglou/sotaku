from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_event(report_path: Path, event: dict) -> None:
    record = {"time": time.time(), **event}
    with report_path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(record, sort_keys=True), flush=True)


def put_url(url: str, data: bytes) -> dict:
    request = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(request, timeout=120) as response:
        response.read()
        return {"status": response.status, "bytes": len(data)}


def get_url(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def upload_file(url: str, path: Path) -> dict:
    return put_url(url, path.read_bytes())


def build_state(size: int):
    import torch

    device = torch.device("cuda")
    torch.manual_seed(0)
    input_matrix = torch.randn(size, size, device=device)
    weights = torch.randn(size, size, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weights], lr=1e-4)
    return device, input_matrix, weights, optimizer


def train_steps(input_matrix, weights, optimizer, steps: int, start_step: int) -> tuple[int, float]:
    last_loss = 0.0
    step = start_step
    for _ in range(steps):
        output = input_matrix @ weights
        loss = output.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        last_loss = float(loss.detach().cpu())
    return step, last_loss


def save_checkpoint(path: Path, *, step: int, loss: float, weights, optimizer, config: dict) -> None:
    import torch

    torch.save(
        {
            "step": step,
            "loss": loss,
            "weights": weights.detach().cpu(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "saved_at": time.time(),
        },
        path,
    )


def report_url(args: argparse.Namespace) -> str:
    return args.fresh_report_put_url if args.mode == "fresh" else args.resume_report_put_url


def upload_failure_report(args: argparse.Namespace, report_path: Path) -> None:
    url = report_url(args)
    if url and report_path.exists():
        try:
            upload_file(url, report_path)
        except Exception as error:
            append_event(report_path, {"event": "failure_report_upload_failed", "error": repr(error)})


def run(args: argparse.Namespace, report_path: Path, checkpoint_path: Path, checksum_path: Path) -> int:
    import torch

    test_data_path = Path("test_data/test.txt")
    test_data = test_data_path.read_bytes()
    test_data_sha256 = hashlib.sha256(test_data).hexdigest()
    config = {
        "matrix_size": args.matrix_size,
        "repo_test_data_sha256": test_data_sha256,
        "probe": "resume_object_probe",
    }

    append_event(
        report_path,
        {
            "event": "start",
            "mode": args.mode,
            "python": sys.version,
            "platform": platform.platform(),
            "test_data_bytes": len(test_data),
            "test_data_sha256": test_data_sha256,
        },
    )
    append_event(
        report_path,
        {
            "event": "torch_status",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
        },
    )
    if not torch.cuda.is_available():
        append_event(report_path, {"event": "cuda_unavailable"})
        upload_failure_report(args, report_path)
        print("METRIC: 0")
        return 0

    device, input_matrix, weights, optimizer = build_state(args.matrix_size)
    start_step = 0

    if args.mode == "resume":
        checkpoint_bytes = get_url(args.checkpoint_get_url)
        expected_sha256 = get_url(args.checksum_get_url).decode("utf-8").strip()
        checkpoint_path.write_bytes(checkpoint_bytes)
        actual_sha256 = sha256_file(checkpoint_path)
        append_event(
            report_path,
            {
                "event": "downloaded_checkpoint",
                "checkpoint_bytes": len(checkpoint_bytes),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            },
        )
        if actual_sha256 != expected_sha256:
            append_event(report_path, {"event": "checksum_mismatch"})
            upload_failure_report(args, report_path)
            print("METRIC: 0")
            return 0

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint["config"] != config:
            append_event(
                report_path,
                {"event": "config_mismatch", "checkpoint_config": checkpoint["config"], "expected_config": config},
            )
            upload_failure_report(args, report_path)
            print("METRIC: 0")
            return 0
        weights.data.copy_(checkpoint["weights"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        append_event(report_path, {"event": "resumed", "step": start_step})

    step, loss = train_steps(input_matrix, weights, optimizer, args.steps, start_step)
    torch.cuda.synchronize()
    save_checkpoint(checkpoint_path, step=step, loss=loss, weights=weights, optimizer=optimizer, config=config)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checksum_path.write_text(checkpoint_sha256 + "\n")
    append_event(
        report_path,
        {
            "event": "trained",
            "mode": args.mode,
            "start_step": start_step,
            "step": step,
            "loss": loss,
            "checkpoint_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": checkpoint_sha256,
        },
    )

    if args.mode == "fresh":
        upload_results = [
            upload_file(args.checkpoint_put_url, checkpoint_path),
            upload_file(args.checksum_put_url, checksum_path),
        ]
        append_event(report_path, {"event": "uploaded_outputs", "results": upload_results})
        upload_file(args.fresh_report_put_url, report_path)
    else:
        upload_results = [
            upload_file(args.resume_checkpoint_put_url, checkpoint_path),
            upload_file(args.resume_checksum_put_url, checksum_path),
        ]
        append_event(report_path, {"event": "uploaded_outputs", "results": upload_results})
        upload_file(args.resume_report_put_url, report_path)

    print(f"METRIC: {step}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fresh", "resume"], required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--matrix-size", type=int, default=256)
    parser.add_argument("--checkpoint-put-url", default="")
    parser.add_argument("--checksum-put-url", default="")
    parser.add_argument("--fresh-report-put-url", default="")
    parser.add_argument("--checkpoint-get-url", default="")
    parser.add_argument("--checksum-get-url", default="")
    parser.add_argument("--resume-checkpoint-put-url", default="")
    parser.add_argument("--resume-checksum-put-url", default="")
    parser.add_argument("--resume-report-put-url", default="")
    args = parser.parse_args()

    output_dir = Path("probe_output")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{args.mode}_report.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    checksum_path = output_dir / "checkpoint.pt.sha256"

    try:
        return run(args, report_path, checkpoint_path, checksum_path)
    except Exception as error:
        append_event(report_path, {"event": "unhandled_error", "error": repr(error)})
        upload_failure_report(args, report_path)
        print("METRIC: 0")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
