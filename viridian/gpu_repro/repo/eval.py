import sys


def emit(value):
    print(f"METRIC: {value}", flush=True)


try:
    import torch
except Exception as exc:
    print(f"torch_import_error: {type(exc).__name__}: {exc}", file=sys.stderr)
    emit(-2)
    raise SystemExit(0)

try:
    if not torch.cuda.is_available():
        print(f"torch_version: {torch.__version__}", flush=True)
        print("cuda_available: false", flush=True)
        emit(0)
        raise SystemExit(0)

    device = torch.device("cuda")
    x = torch.randn((128, 128), device=device)
    y = x @ x
    torch.cuda.synchronize()
    print(f"torch_version: {torch.__version__}", flush=True)
    print(f"cuda_device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"checksum: {float(y[0, 0].detach().cpu()):.6f}", flush=True)
    emit(1)
except SystemExit:
    raise
except Exception as exc:
    print(f"cuda_runtime_error: {type(exc).__name__}: {exc}", file=sys.stderr)
    emit(-1)
