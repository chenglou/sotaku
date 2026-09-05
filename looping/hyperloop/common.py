"""Frozen configuration and source/data identities for the Hyperloop study."""

import json
from pathlib import Path

from looping.weight_tying.common import SOURCE_PATHS as SHARED_SOURCES
from looping.window_selection.common import state_sha256
from runtime_utils import file_sha256

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = tuple(dict.fromkeys((*SHARED_SOURCES,
    "runtime_utils.py", "model_io.py", "inference.py", "iters/state_norm.py", "looping/window_selection/common.py",
    "looping/hyperloop/common.py", "looping/hyperloop/model.py", "looping/hyperloop/train.py",
    "looping/hyperloop/evaluate.py", "looping/hyperloop/analyze.py", "looping/hyperloop/test_hyperloop.py",
    "looping/hyperloop/modal_run.py", "looping/hyperloop/protocol.json", "release/benchmark_25k.json")))


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def run_config(arm, seed, *, smoke=False):
    settings = protocol()
    if arm not in settings["arms"] or seed not in settings["seeds"]:
        raise ValueError("Arm and seed must belong to the fixed protocol")
    if smoke:
        settings["training"].update({"steps": 4, "batch_size": 2, "warmup_steps": 1,
                                    "probe_every": 2, "phases": [[0, 4, 0]], "probe_iterations": [16]})
        settings["burnin_probability"] = 1.0
        settings["burnin_iterations"] = [16, 32]
    return {"arm": arm, "seed": seed, "smoke": smoke, "protocol": settings,
            "protocol_sha256": file_sha256(DIRECTORY / "protocol.json")}


def build_model(config):
    from looping.hyperloop.model import HyperloopTransformer
    settings = config["protocol"]
    model = settings["model"]
    return HyperloopTransformer(settings["arms"][config["arm"]], gate_scale=model["gate_scale"],
                               retention_bias=model["retention_bias"], gate_epsilon=model["gate_epsilon"])


def run_name(arm, seed):
    run_config(arm, seed)
    return f"{arm}_seed{seed}"


def validate_data(directory, *, smoke=False, names=("train.npz", "validation.npz")):
    actual = {name: file_sha256(Path(directory) / name) for name in names}
    if not smoke and actual != {name: protocol()["data_files"][name] for name in names}:
        raise ValueError("Prepared arrays do not match the fixed data checksums")
    return actual
