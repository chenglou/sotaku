import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path


def append_event(path, event):
    row = {"time": time.time(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"EVENT: {row.get('event', 'event')}", flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    with urllib.request.urlopen(request, timeout=120) as response:
        return {"status": response.status, "bytes": len(data)}


def upload_text(text, url):
    if not url:
        return {"skipped": True}
    data = text.encode()
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return {"status": response.status, "bytes": len(data)}


def download_file(url, path):
    if not url:
        return {"skipped": True}
    with urllib.request.urlopen(url, timeout=120) as response:
        data = response.read()
    path.write_bytes(data)
    return {"bytes": len(data), "sha256": file_sha256(path)}


def run_command(args):
    try:
        completed = subprocess.run(args, text=True, capture_output=True, check=False, timeout=120)
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


def ensure_import(module_name, pip_package, report_path):
    try:
        __import__(module_name)
        append_event(report_path, {"event": "dependency_present", "module": module_name})
        return
    except ModuleNotFoundError:
        append_event(report_path, {"event": "dependency_missing", "module": module_name})
    result = run_command([sys.executable, "-m", "pip", "install", pip_package])
    append_event(report_path, {"event": "dependency_install", "module": module_name, "result": result})
    __import__(module_name)


def install_datasets_stub(report_path):
    module = types.ModuleType("datasets")

    def load_dataset(*_args, **_kwargs):
        raise RuntimeError("datasets.load_dataset is not used by the packaged-data Viridian probe")

    module.load_dataset = load_dataset
    sys.modules["datasets"] = module
    append_event(report_path, {"event": "datasets_stubbed_for_packaged_data"})


def load_rows(path):
    puzzles = []
    solutions = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            puzzle = row["puzzle"].strip()
            solution = row["solution"].strip()
            if len(puzzle) != 81 or len(solution) != 81:
                raise ValueError(f"bad sudoku row length: {len(puzzle)}, {len(solution)}")
            puzzles.append(puzzle)
            solutions.append(solution)
    if not puzzles:
        raise ValueError("tiny sudoku shard is empty")
    return puzzles, solutions


def build_config(args, data_sha256):
    return {
        "experiment": "exp_baseline_lr2e3",
        "probe": "viridian_sotaku_tiny",
        "batch_size": args.batch_size,
        "compile": args.compile,
        "data_sha256": data_sha256,
        "model_config": {
            "d_model": 128,
            "d_ff": 512,
            "n_layers": 4,
            "n_iterations": 16,
            "lr": 2e-3,
        },
    }


def verify_config(checkpoint, expected_config):
    saved_config = checkpoint.get("config")
    if saved_config != expected_config:
        raise ValueError(f"checkpoint config mismatch: saved={saved_config}, expected={expected_config}")


def save_checkpoint(path, model, optimizer, step, config):
    state_dict = {key.replace("_orig_mod.", ""): value for key, value in model.state_dict().items()}
    import torch

    torch.save(
        {
            "step": step,
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        },
        path,
    )


def write_status(path, status):
    path.write_text(json.dumps({"time": time.time(), **status}, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["fresh", "resume"], default="fresh")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--duration-s", type=float, default=0)
    parser.add_argument("--progress-every-s", type=float, default=60)
    parser.add_argument("--upload-every-s", type=float, default=60)
    parser.add_argument("--max-report-bytes", type=int, default=10_000_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--checkpoint-get-url", default="")
    parser.add_argument("--checkpoint-sha-get-url", default="")
    parser.add_argument("--checkpoint-put-url", default="")
    parser.add_argument("--checkpoint-sha-put-url", default="")
    parser.add_argument("--report-put-url", default="")
    parser.add_argument("--status-put-url", default="")
    args = parser.parse_args()

    output_dir = Path("probe_output")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / f"{args.mode}_report.jsonl"
    status_path = output_dir / "status.json"
    checkpoint_path = output_dir / "sotaku_tiny_checkpoint.pt"
    checkpoint_sha_path = output_dir / "sotaku_tiny_checkpoint.pt.sha256"
    data_path = Path("data/tiny_sudoku.csv")
    data_sha256 = file_sha256(data_path)
    start_step = 0

    append_event(
        report_path,
        {
            "event": "start",
            "mode": args.mode,
            "cwd": os.getcwd(),
            "platform": platform.platform(),
            "python": sys.version,
            "data_bytes": data_path.stat().st_size,
            "data_sha256": data_sha256,
        },
    )
    write_status(
        status_path,
        {
            "event": "started",
            "mode": args.mode,
            "step": None,
            "report_bytes": report_path.stat().st_size,
            "data_sha256": data_sha256,
        },
    )
    upload_file(status_path, args.status_put_url)
    append_event(report_path, {"event": "nvidia_smi", "result": run_command(["nvidia-smi"])})

    try:
        ensure_import("numpy", "numpy", report_path)
        install_datasets_stub(report_path)

        import torch
        import torch.nn.functional as F
        from iters import exp_baseline_lr2e3 as exp

        append_event(
            report_path,
            {
                "event": "torch_status",
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "device_count": torch.cuda.device_count(),
            },
        )

        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this Viridian probe")

        device = torch.device(args.device)
        if device.type == "cuda":
            append_event(report_path, {"event": "device", "name": torch.cuda.get_device_name(0)})
        write_status(
            status_path,
            {
                "event": "runtime_ready",
                "mode": args.mode,
                "step": start_step,
                "report_bytes": report_path.stat().st_size,
                "cuda_available": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            },
        )
        upload_file(status_path, args.status_put_url)

        puzzles, solutions = load_rows(data_path)
        x_all = exp.encode_puzzles(puzzles)
        targets_all = exp.encode_solutions(solutions)
        config = build_config(args, data_sha256)

        model = exp.SudokuTransformer().to(device)
        optimizer_state = None

        if args.mode == "resume":
            download_result = download_file(args.checkpoint_get_url, checkpoint_path)
            expected_sha = urllib.request.urlopen(args.checkpoint_sha_get_url, timeout=120).read().decode().strip()
            actual_sha = file_sha256(checkpoint_path)
            append_event(
                report_path,
                {
                    "event": "downloaded_checkpoint",
                    "result": download_result,
                    "expected_sha256": expected_sha,
                    "actual_sha256": actual_sha,
                },
            )
            if expected_sha != actual_sha:
                raise ValueError("downloaded checkpoint SHA-256 did not match sidecar")
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            verify_config(checkpoint, config)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = checkpoint["optimizer_state_dict"]
            start_step = int(checkpoint["step"])
            append_event(report_path, {"event": "resumed", "step": start_step})
            write_status(
                status_path,
                {
                    "event": "resumed",
                    "mode": args.mode,
                    "step": start_step,
                    "report_bytes": report_path.stat().st_size,
                },
            )
            upload_file(status_path, args.status_put_url)

        if args.compile:
            model = torch.compile(model)
            append_event(report_path, {"event": "compiled"})
            write_status(
                status_path,
                {
                    "event": "compiled",
                    "mode": args.mode,
                    "step": start_step,
                    "report_bytes": report_path.stat().st_size,
                },
            )
            upload_file(status_path, args.status_put_url)

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95))
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

        x_all = x_all.to(device)
        targets_all = targets_all.to(device, dtype=torch.long)
        generator = torch.Generator(device=device)
        generator.manual_seed(1234 + start_step)
        model.train()
        last_loss = None
        last_acc = None
        train_started = time.monotonic()
        deadline = train_started + args.duration_s if args.duration_s > 0 else None
        next_progress = train_started
        next_upload = train_started
        current_step = start_step

        while True:
            if args.duration_s > 0:
                if deadline is not None and time.monotonic() >= deadline and current_step > start_step:
                    break
            elif current_step >= start_step + args.steps:
                break

            step = current_step
            selected = torch.randint(0, x_all.size(0), (args.batch_size,), generator=generator, device=device)
            x_batch = x_all[selected]
            targets = targets_all[selected]
            for group in optimizer.param_groups:
                group["lr"] = exp.get_lr(step)

            optimizer.zero_grad(set_to_none=True)
            autocast_enabled = device.type == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
                logits_by_iter = model(x_batch, return_all=True)
                mask = x_batch[:, :, 0].to(dtype=torch.float32)
                loss = 0
                for logits in logits_by_iter:
                    per_cell = F.cross_entropy(logits.reshape(-1, 9), targets.reshape(-1), reduction="none")
                    per_cell = per_cell.view(targets.size(0), 81)
                    loss = loss + (per_cell * mask).sum() / mask.sum()
                loss = loss / len(logits_by_iter)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits_by_iter[-1].argmax(dim=-1)
                correct = (preds == targets) & (mask > 0)
                last_acc = float(correct.sum().detach().cpu()) / float(mask.sum().detach().cpu())
                last_loss = float(loss.detach().cpu())
            current_step = step + 1
            now = time.monotonic()
            should_log_progress = args.duration_s <= 0 or now >= next_progress
            should_upload = now >= next_upload
            if should_log_progress:
                report_bytes = report_path.stat().st_size
                append_event(
                    report_path,
                    {
                        "event": "progress",
                        "step": current_step,
                        "elapsed_s": round(now - train_started, 3),
                        "lr": exp.get_lr(step),
                        "loss": last_loss,
                        "train_acc": last_acc,
                        "report_bytes": report_bytes,
                    },
                )
                next_progress = now + args.progress_every_s
            if report_path.stat().st_size > args.max_report_bytes:
                append_event(
                    report_path,
                    {
                        "event": "report_size_limit_exceeded",
                        "step": current_step,
                        "report_bytes": report_path.stat().st_size,
                        "max_report_bytes": args.max_report_bytes,
                    },
                )
                break
            if should_upload:
                write_status(
                    status_path,
                    {
                        "event": "running",
                        "mode": args.mode,
                        "step": current_step,
                        "elapsed_s": round(now - train_started, 3),
                        "loss": last_loss,
                        "train_acc": last_acc,
                        "report_bytes": report_path.stat().st_size,
                    },
                )
                upload_file(status_path, args.status_put_url)
                upload_file(report_path, args.report_put_url)
                next_upload = now + args.upload_every_s

        if device.type == "cuda":
            torch.cuda.synchronize()
        final_step = current_step
        save_checkpoint(checkpoint_path, model, optimizer, final_step, config)
        checkpoint_sha = file_sha256(checkpoint_path)
        checkpoint_sha_path.write_text(checkpoint_sha + "\n")
        append_event(
            report_path,
            {
                "event": "checkpoint_saved",
                "step": final_step,
                "checkpoint_bytes": checkpoint_path.stat().st_size,
                "checkpoint_sha256": checkpoint_sha,
                "loss": last_loss,
                    "train_acc": last_acc,
                    "report_bytes": report_path.stat().st_size,
                },
            )
        upload_results = {
            "checkpoint": upload_file(checkpoint_path, args.checkpoint_put_url),
            "checkpoint_sha": upload_file(checkpoint_sha_path, args.checkpoint_sha_put_url),
        }
        append_event(report_path, {"event": "uploaded_checkpoint", "results": upload_results})
        write_status(
            status_path,
            {
                "event": "finished",
                "mode": args.mode,
                "step": final_step,
                "loss": last_loss,
                "train_acc": last_acc,
                "report_bytes": report_path.stat().st_size,
                "checkpoint_sha256": checkpoint_sha,
            },
        )
        upload_file(status_path, args.status_put_url)
        report_upload = upload_file(report_path, args.report_put_url)
        append_event(report_path, {"event": "uploaded_report", "result": report_upload})
        upload_file(report_path, args.report_put_url)
        print(f"METRIC: {final_step}", flush=True)
        return 0
    except Exception as error:
        append_event(report_path, {"event": "failed", "error": repr(error)})
        write_status(
            status_path,
            {
                "event": "failed",
                "mode": args.mode,
                "step": None,
                "error": repr(error),
                "report_bytes": report_path.stat().st_size,
            },
        )
        try:
            upload_file(status_path, args.status_put_url)
            upload_file(report_path, args.report_put_url)
        finally:
            print("METRIC: 0", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
