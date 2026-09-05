"""Frozen experiment identity and data checks."""

import hashlib
import json
from pathlib import Path

from looping.weight_tying.common import SOURCE_PATHS as PREVIOUS_SOURCES


DIRECTORY = Path(__file__).parent
SOURCE_PATHS = tuple(dict.fromkeys((*PREVIOUS_SOURCES,
    "looping/window_selection/common.py", "looping/window_selection/selection.py",
    "looping/window_selection/train.py", "looping/window_selection/test_selection.py",
    "looping/window_selection/modal_run.py", "looping/window_selection/analyze.py",
    "looping/window_selection/protocol.json", "runtime_utils.py", "model_io.py",
    "inference.py", "iters/eval_more_iters.py", "iters/state_norm.py")))


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def run_config(selector, seed, *, smoke=False):
    settings = protocol()
    if selector not in settings["selectors"] or seed not in settings["seeds"]:
        raise ValueError("Selector or seed is not in the fixed protocol")
    digest = hashlib.sha256((DIRECTORY / "protocol.json").read_bytes()).hexdigest()
    return {"selector": selector, "seed": seed, "smoke": smoke,
            "protocol_sha256": digest, "protocol": settings}


def run_name(selector, seed):
    run_config(selector, seed)
    return f"{selector}_seed{seed}"


def validate_data(directory, *, smoke=False):
    from runtime_utils import file_sha256
    directory = Path(directory)
    if smoke:
        return {name: file_sha256(directory / name) for name in ("train.npz", "validation.npz")}
    expected = protocol()["data_files"]
    actual = {name: file_sha256(directory / name) for name in expected}
    if actual != expected:
        raise ValueError("Prepared arrays do not match the fixed data checksums")
    return actual


def state_sha256(state):
    """Hash tensor contents, independent of torch.save archive filenames."""
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        description = json.dumps([name, str(tensor.dtype), list(tensor.shape)])
        digest.update(description.encode() + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
