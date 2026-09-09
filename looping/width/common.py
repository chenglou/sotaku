"""Fixed widths, paired sampling, and source identities for the width study."""

import json
from pathlib import Path

from checkpoint_utils import validate_config
from looping.hyperloop.common import SOURCE_PATHS as ORIGINAL_SOURCES
from looping.hyperloop.common import state_sha256
from runtime_utils import file_sha256

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = (*ORIGINAL_SOURCES, "looping/width/__init__.py", "looping/width/common.py",
                "looping/width/model.py", "looping/width/train.py", "looping/width/evaluate.py",
                "looping/width/preflight.py", "looping/width/test_width.py", "looping/width/modal_run.py",
                "looping/width/protocol.json", "looping/width/reference.json")


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def run_config(arm, seed, *, smoke=False):
    settings = protocol()
    if arm not in settings["arms"] or seed not in settings["seeds"]:
        raise ValueError("Arm and seed must belong to the fixed width protocol")
    if smoke:
        settings["training"].update({"steps": 4, "batch_size": 2, "warmup_steps": 1,
                                    "probe_every": 2, "phases": [[0, 4, 0]], "probe_iterations": [16]})
        settings["burnin_probability"] = 1.0
        settings["burnin_iterations"] = [16, 32]
    return {"arm": arm, "seed": seed, "smoke": smoke, "protocol": settings,
            "protocol_sha256": file_sha256(DIRECTORY / "protocol.json")}


def build_model(config):
    from looping.width.model import WidthTransformer
    settings = config["protocol"]
    return WidthTransformer(settings["arms"][config["arm"]], dropout=settings["model"]["dropout"])


def run_name(arm, seed):
    run_config(arm, seed)
    return f"{arm}_seed{seed}"


def validate_data(directory, *, smoke=False, names=("train.npz", "validation.npz")):
    actual = {name: file_sha256(Path(directory) / name) for name in names}
    if not smoke and actual != {name: protocol()["data_files"][name] for name in names}:
        raise ValueError("Prepared arrays do not match the fixed data checksums")
    return actual


def verify_reference():
    """Reuse all three controls only while their training and scoring code is unchanged."""
    root, spec = DIRECTORY.parents[1], protocol()
    reference = json.loads((DIRECTORY / "reference.json").read_text())
    original = json.loads((root / "looping/hyperloop/protocol.json").read_text())
    for key in ("seeds", "training", "burnin_iterations", "burnin_probability", "window_length",
                "data_directory", "data_files", "evaluation"):
        validate_config({key: spec[key]}, {key: original[key]})
    validate_config(spec["model"], {"blocks": 4, "heads": 4, "dropout": 0.1, "feedforward_ratio": 4})
    for filename in ("train.py", "evaluate.py"):
        expected = (root / "looping/hyperloop" / filename).read_text()
        expected = expected.replace("looping.hyperloop.common", "looping.width.common")
        expected = expected.replace("looping.hyperloop.evaluate", "looping.width.evaluate")
        if filename == "evaluate.py":
            expected = expected.replace("sotaku-hyperloop-study-inference", "sotaku-width-study-inference")
            expected = expected.replace("Not a Hyperloop study inference export", "Not a width study inference export")
        if (DIRECTORY / filename).read_text() != expected:
            raise ValueError(f"Width study changed training or evaluation calculations: {filename}")
    for seed in spec["seeds"]:
        row = reference["runs"][str(seed)]
        if row["status"] != "complete" or row["updates"] != spec["training"]["steps"]:
            raise ValueError("Incomplete reference control")
        if set(row["source_sha256"]) != set(ORIGINAL_SOURCES):
            raise ValueError("Reference sources do not match the archived study")
        for source, checksum in row["source_sha256"].items():
            if file_sha256(root / source) != checksum:
                raise ValueError(f"Reference source changed: {source}")
    return reference


def verify_completed_pair(result):
    reference = verify_reference()["runs"][str(result["config"]["seed"])]
    for key in ("sample_digest", "horizon_counts", "work_counts", "data_sha256", "updates", "status"):
        validate_config({key: result[key]}, {key: reference[key]})
