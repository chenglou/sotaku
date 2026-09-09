"""Isolated 50K width confirmation with frozen controls and full-batch evidence."""

import json
from pathlib import Path

from checkpoint_utils import validate_config
from looping.hyperloop_50k.common import SOURCE_PATHS as CONTROL_SOURCES
from looping.width.common import SOURCE_PATHS as WIDTH_SOURCES
from looping.width.common import build_model, state_sha256
from runtime_utils import file_sha256

DIRECTORY = Path(__file__).parent
SOURCE_PATHS = tuple(dict.fromkeys((*CONTROL_SOURCES, *WIDTH_SOURCES,
    "looping/width/results/preflight_width160.json",
    "looping/width_50k/__init__.py", "looping/width_50k/common.py",
    "looping/width_50k/train.py", "looping/width_50k/evaluate.py",
    "looping/width_50k/preflight.py", "looping/width_50k/modal_run.py",
    "looping/width_50k/test_confirmation.py", "looping/width_50k/protocol.json",
    "looping/width_50k/reference.json")))


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def run_config(arm, seed, *, smoke=False):
    settings = protocol()
    if arm not in settings["arms"] or seed not in settings["seeds"]:
        raise ValueError("Arm and seed must belong to the fixed width160 50K protocol")
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


def verify_reference():
    root, spec = DIRECTORY.parents[1], protocol()
    reference = json.loads((DIRECTORY / "reference.json").read_text())
    original = json.loads((root / "looping/hyperloop_50k/protocol.json").read_text())
    for key in ("seeds", "training", "burnin_iterations", "burnin_probability",
                "window_length", "data_directory", "data_files", "evaluation"):
        validate_config({key: spec[key]}, {key: original[key]})
    for seed in spec["seeds"]:
        row = reference["runs"][str(seed)]
        if row["status"] != "complete" or row["updates"] != spec["training"]["steps"]:
            raise ValueError("Incomplete reference control")
        if row["config"]["arm"] != "baseline" or row["config"]["seed"] != seed or row["config"]["smoke"]:
            raise ValueError("Reference is not the expected fresh baseline")
        validate_config(row["config"]["protocol"], original)
        if set(row["source_sha256"]) != set(CONTROL_SOURCES):
            raise ValueError("Reference sources do not match the archived 50K study")
        for source, checksum in row["source_sha256"].items():
            if file_sha256(root / source) != checksum:
                raise ValueError(f"Reference source changed: {source}")
        for selection in ("final", "best_validation"):
            evaluation = row["evaluations"][selection]
            if set(evaluation["scores"]) != {str(h) for h in spec["evaluation"]["iterations"]}:
                raise ValueError("Incomplete reference evaluation horizons")
            for score in evaluation["scores"].values():
                if score["total"] != 25000 or score["nonfinite"] or score["accuracy"] != score["solved"] / 25000:
                    raise ValueError("Incomplete or nonfinite reference evaluation")
            validate_config(evaluation["identity"]["config"], row["config"])
            for key, expected in spec["evaluation"].items():
                validate_config({key: evaluation["identity"][key]}, {key: expected})
    return reference


def verify_inherited_preflight():
    root, spec = DIRECTORY.parents[1], protocol()
    path = root / spec["preflight"]["inherited_full_batch_report"]
    if file_sha256(path) != spec["preflight"]["inherited_full_batch_sha256"]:
        raise ValueError("Inherited full-batch preflight report changed")
    report = json.loads(path.read_text())
    if report["status"] != "passed" or not report["compiled"] or set(report["source_sha256"]) != set(WIDTH_SOURCES):
        raise ValueError("Incomplete inherited full-batch preflight")
    for source, checksum in report["source_sha256"].items():
        if file_sha256(root / source) != checksum:
            raise ValueError(f"Previously tested source changed: {source}")
    original = json.loads((root / "looping/width/protocol.json").read_text())
    if spec["arms"] != {"width160": original["arms"]["width160"]}:
        raise ValueError("Confirmation must use the tested width160 architecture")
    for key in ("model", "window_length", "burnin_probability", "burnin_iterations", "data_files", "evaluation"):
        validate_config({key: spec[key]}, {key: original[key]})
    for key in ("batch_size", "train_rows", "learning_rate", "adam_betas", "weight_decay"):
        validate_config({key: spec["training"][key]}, {key: original["training"][key]})
    if (report["batch_size"] != spec["training"]["batch_size"]
            or report["supervised_iterations"] != spec["window_length"]
            or report["prefix_iterations"] != max(spec["burnin_iterations"])):
        raise ValueError("Inherited preflight used different batch or iteration counts")
    if not any(row["arm"] == "width160" and row["width"] == 160 and row["populated_optimizer_resume_exact"]
               for row in report["arms"]):
        raise ValueError("Width160 lacks a full-batch resume check")
    for filename in ("train.py", "evaluate.py"):
        expected = (root / "looping/width" / filename).read_text()
        expected = expected.replace("looping.width.common", "looping.width_50k.common")
        expected = expected.replace("looping.width.evaluate", "looping.width_50k.evaluate")
        if filename == "evaluate.py":
            expected = expected.replace("sotaku-width-study-inference", "sotaku-width-50k-inference")
            expected = expected.replace("Not a width study inference export", "Not a width 50K inference export")
        if (DIRECTORY / filename).read_text() != expected:
            raise ValueError(f"50K numerical code differs from tested width code: {filename}")
    verify_reference()
    return report


def verify_completed_pair(result):
    reference = verify_reference()["runs"][str(result["config"]["seed"])]
    for key in ("sample_digest", "horizon_counts", "work_counts", "data_sha256", "updates", "status"):
        validate_config({key: result[key]}, {key: reference[key]})
