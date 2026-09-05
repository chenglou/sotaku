"""Identity, serialization, and locked settings for the weight-tying study."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from runtime_utils import file_sha256


DIRECTORY = Path(__file__).parent
PROTOCOL_PATH = DIRECTORY / "protocol.json"
SOURCE_PATHS = (
    "looping/weight_tying/model.py", "looping/weight_tying/common.py",
    "looping/weight_tying/data.py", "looping/weight_tying/train.py",
    "looping/weight_tying/evaluate.py", "looping/weight_tying/modal_run.py",
    "looping/weight_tying/test_study.py",
    "looping/weight_tying/protocol.json", "stabilize/exp_testbed_20k.py",
    "checkpoint_utils.py", "dataset_utils.py", "requirements-modal.txt",
)


def protocol():
    return json.loads(PROTOCOL_PATH.read_text())


def protocol_sha256():
    return file_sha256(PROTOCOL_PATH)


def run_config(architecture, regime, seed, *, smoke=False):
    settings = protocol()
    if architecture not in settings["architectures"] or regime not in settings["regimes"]:
        raise ValueError("Unknown architecture or training regime")
    if seed not in settings["seeds"]:
        raise ValueError("Seed is not in the preregistered cohort")
    return {"study_id": settings["study_id"], "protocol_sha256": protocol_sha256(),
            "architecture": architecture, "model": settings["architectures"][architecture],
            "regime": regime, "regime_settings": settings["regimes"][regime],
            "seed": seed, "training": settings["training"], "smoke": smoke}


def run_name(architecture, regime, seed):
    return f"{architecture}_{regime}_seed{seed}"


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".arrays-", suffix=".tmp")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def update_sample_digest(previous, indices, horizon):
    digest = hashlib.sha256(bytes.fromhex(previous))
    digest.update(np.asarray(indices, dtype="<i8").tobytes())
    digest.update(int(horizon).to_bytes(8, "little"))
    return digest.hexdigest()
