"""Small provenance helpers shared by release checks and training experiments."""

import hashlib
import importlib.metadata
import platform
import subprocess
import sys
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_manifest(source_paths=()):
    from torch._inductor import config as compiler_config
    versions = {}
    for name in ("torch", "numpy", "datasets", "pandas", "pyarrow", "huggingface_hub"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "inductor": {
            "emulate_precision_casts": compiler_config.emulate_precision_casts,
            "use_fast_math": compiler_config.use_fast_math,
            "max_autotune": compiler_config.max_autotune,
        },
        "source_sha256": {str(path): file_sha256(Path(__file__).parent / path) for path in source_paths},
        "installed_packages": dict(sorted(
            (distribution.metadata["Name"], distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata["Name"]
        )),
    }
    if torch.cuda.is_available():
        result.update({
            "gpu": torch.cuda.get_device_name(),
            "gpu_capability": list(torch.cuda.get_device_capability()),
            "cudnn": torch.backends.cudnn.version(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math": torch.backends.cuda.math_sdp_enabled(),
        })
        try:
            result["nvidia_smi"] = subprocess.run(
                ["nvidia-smi"], capture_output=True, text=True,
            ).stdout
            result["driver_version"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
        except FileNotFoundError:
            result["nvidia_smi"] = None
    return result


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def output_subdirectory(root, name):
    if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for character in name):
        raise ValueError("Output name must contain only letters, digits, underscores, dots, or hyphens")
    if name in (".", ".."):
        raise ValueError("Output name must identify a subdirectory")
    return Path(root) / name
