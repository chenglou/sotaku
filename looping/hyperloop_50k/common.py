"""Separate identities for fresh 50K runs; the completed 20K sources stay frozen."""

import json
from pathlib import Path

from looping.hyperloop.common import SOURCE_PATHS as ORIGINAL_SOURCES
from looping.hyperloop.common import build_model, state_sha256
from runtime_utils import file_sha256

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = (*ORIGINAL_SOURCES, "looping/hyperloop/results/preflight.json",
                "looping/exp_stay_solved.py", "looping/hyperloop_50k/__init__.py",
                "looping/hyperloop_50k/common.py", "looping/hyperloop_50k/train.py",
                "looping/hyperloop_50k/evaluate.py", "looping/hyperloop_50k/analyze.py",
                "looping/hyperloop_50k/preflight.py", "looping/hyperloop_50k/test_confirmation.py",
                "looping/hyperloop_50k/modal_run.py", "looping/hyperloop_50k/protocol.json")


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def run_config(arm, seed, *, smoke=False):
    settings = protocol()
    if arm not in settings["arms"] or seed not in settings["seeds"]:
        raise ValueError("Arm and seed must belong to the fixed 50K protocol")
    if smoke:
        settings["training"].update({"steps": 4, "batch_size": 2, "warmup_steps": 1,
                                    "probe_every": 2, "phases": [[0, 4, 0]], "probe_iterations": [16]})
        settings["burnin_probability"] = 1.0
        settings["burnin_iterations"] = [16, 32]
    return {"arm": arm, "seed": seed, "smoke": smoke, "protocol": settings,
            "protocol_sha256": file_sha256(DIRECTORY / "protocol.json")}


def run_name(arm, seed):
    run_config(arm, seed)
    return f"{arm}_seed{seed}"


def validate_data(directory, *, smoke=False, names=("train.npz", "validation.npz")):
    actual = {name: file_sha256(Path(directory) / name) for name in names}
    if not smoke and actual != {name: protocol()["data_files"][name] for name in names}:
        raise ValueError("Prepared arrays do not match the fixed data checksums")
    return actual


def verify_inherited_preflight():
    """Reuse full-batch evidence only for identical model and numerical code."""
    root, spec = DIRECTORY.parents[1], protocol()
    path = root / spec["preflight"]["inherited_full_batch_report"]
    if file_sha256(path) != spec["preflight"]["inherited_full_batch_sha256"]:
        raise ValueError("Inherited full-batch preflight report changed")
    report = json.loads(path.read_text())
    if report["status"] != "passed" or set(report["source_sha256"]) != set(ORIGINAL_SOURCES):
        raise ValueError("Incomplete inherited preflight")
    for source, checksum in report["source_sha256"].items():
        if file_sha256(root / source) != checksum:
            raise ValueError(f"Previously tested source changed: {source}")
    original = json.loads((root / "looping/hyperloop/protocol.json").read_text())
    if any(streams != original["arms"].get(arm) for arm, streams in spec["arms"].items()):
        raise ValueError("50K arms changed the tested number of states")
    for key in ("model", "window_length", "burnin_probability", "burnin_iterations", "data_files", "evaluation"):
        if spec[key] != original[key]:
            raise ValueError(f"50K setup changed previously tested {key}")
    for key in ("batch_size", "train_rows", "learning_rate", "adam_betas", "weight_decay"):
        if spec["training"][key] != original["training"][key]:
            raise ValueError(f"50K setup changed previously tested {key}")
    if (report["batch_size"] != spec["training"]["batch_size"]
            or report["supervised_iterations"] != spec["window_length"]
            or report["prefix_iterations"] != max(spec["burnin_iterations"])):
        raise ValueError("Inherited preflight used a different batch or iteration count")
    tested = {arm["arm"] for arm in report["arms"] if arm["populated_optimizer_resume_exact"]}
    if not set(spec["arms"]) <= tested:
        raise ValueError("An arm lacks a full-batch resume check")
    for filename in ("train.py", "evaluate.py"):
        expected = (root / "looping/hyperloop" / filename).read_text()
        expected = expected.replace("looping.hyperloop.common", "looping.hyperloop_50k.common")
        expected = expected.replace("looping.hyperloop.evaluate", "looping.hyperloop_50k.evaluate")
        if filename == "evaluate.py":
            expected = expected.replace("sotaku-hyperloop-study-inference", "sotaku-hyperloop-50k-inference")
        if (DIRECTORY / filename).read_text() != expected:
            raise ValueError(f"50K numerical code differs from the full-batch preflight: {filename}")
    return report
