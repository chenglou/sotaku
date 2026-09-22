"""Fixed identities for copy-only analysis; never modify training artifacts."""

import json
from pathlib import Path

from looping.hyperloop_50k.common import SOURCE_PATHS as MODEL_SOURCES
from runtime_utils import file_sha256

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).parent
SOURCE_PATHS = tuple(dict.fromkeys((*MODEL_SOURCES,
    "looping/basin_diagnostics/__init__.py", "looping/basin_diagnostics/common.py",
    "looping/basin_diagnostics/protocol.json", "looping/basin_diagnostics/analysis.py",
    "looping/basin_diagnostics/plots.py", "looping/basin_diagnostics/modal_run.py",
    "looping/basin_diagnostics/test_diagnostics.py", "looping/width/reference.json",
    "looping/width_50k/reference.json")))


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def checkpoint_spec(key, volume_root="/outputs"):
    settings = protocol()
    spec = settings["models"][key]
    reference = json.loads((ROOT / spec["reference"]).read_text())
    record = reference["runs"][str(spec["seed"])]
    path = Path(volume_root) / reference["study_id"] / "runs" / f"baseline_seed{spec['seed']}" / "final.pt"
    return {**spec, "key": key, "path": str(path), "updates": record["updates"],
            "weights_sha256": record["evaluations"]["final"]["identity"]["weights_sha256"],
            "archived_scores": {step: value["accuracy"] for step, value in
                                record["evaluations"]["final"]["scores"].items()}}


def load_model(key, device="cuda", volume_root="/outputs"):
    spec = checkpoint_spec(key, volume_root)
    if spec["cohort"] == "20k":
        from looping.hyperloop.evaluate import load_export
    else:
        from looping.hyperloop_50k.evaluate import load_export
    model, manifest = load_export(spec["path"], device=device)
    if manifest["weights_sha256"] != spec["weights_sha256"] or manifest["updates"] != spec["updates"]:
        raise ValueError("Checkpoint does not match the archived final model")
    if model.streams != 0 or model.initial_encoder.out_features != 128:
        raise ValueError("This experiment is restricted to ordinary width-128 models")
    for source in ("stabilize/exp_testbed_20k.py", "looping/hyperloop/model.py"):
        if file_sha256(ROOT / source) != manifest["source_sha256"][source]:
            raise ValueError(f"Model implementation changed: {source}")
    return model, spec
