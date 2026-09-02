"""Matched continuations that change only dropout during detached state preparation."""

from pathlib import Path

import torch

from checkpoint_utils import atomic_json_save, validate_config
from looping.exp_stay_solved import BASE_REPLACEMENT, FULL_50K_SCHEDULE
from runtime_utils import file_sha256, runtime_manifest
from stabilize.exp_testbed_20k import train as train_testbed

SOURCES = {
    20260724: (
        "looping/loop_stay_control_50k_trial0_checkpoint_step39000.pt",
        "3e1e86dd40fe75fade2dec17c2d3916ad8959b5b12f1170c5dbbb5e3673e4a98",
    ),
    20260730: (
        "looping/loop_health_standalone_v1_late_state_ce_50k_trial0_checkpoint_step39000.pt",
        "8f7e82678ecaf900de27d1403f783a4b414dd3ccce600dd9a0fb858339ad9725",
    ),
}
SOURCE_STEP = 39000
ALLOWED_CHANGES = {"experiment", "run_name", "branch_source_checkpoint", "detached_burnin_dropout", "record_sample_digest"}
SOURCE_FILES = (
    "stabilize/exp_testbed_20k.py", "checkpoint_utils.py", "dataset_utils.py",
    "looping/exp_burnin_dropout.py", "looping/exp_stay_solved.py",
    "requirements-modal.txt",
)


def train(output_dir, *, source_root, seed, dropout_enabled, stop_after_step=43000):
    if seed not in SOURCES:
        raise ValueError(f"Unknown predeclared seed {seed}")
    if not isinstance(dropout_enabled, bool):
        raise ValueError("dropout_enabled must be a boolean")
    if not SOURCE_STEP < stop_after_step < 50000:
        raise ValueError("The continuation must stop between steps 39001 and 49999")
    relative_path, expected_hash = SOURCES[seed]
    source = Path(source_root) / relative_path
    source_hash = file_sha256(source)
    if source_hash != expected_hash:
        raise ValueError(f"Source checkpoint checksum mismatch: {source}")
    checkpoint = torch.load(source, weights_only=False, map_location="cpu")
    if checkpoint["step"] != SOURCE_STEP:
        raise ValueError("Wrong source step")
    if checkpoint["config"]["random_seed"] != seed:
        raise ValueError("Wrong source seed")
    for key in ("optimizer_state_dict", "rng_state"):
        if key not in checkpoint:
            raise ValueError(f"Source lacks {key}")
    for key in ("python", "numpy", "torch", "cuda"):
        if key not in checkpoint["rng_state"]:
            raise ValueError(f"Source lacks {key} RNG state")
    run_name = f"burnin_dropout_{'on' if dropout_enabled else 'off'}_seed{seed}"
    identity = {
        "seed": seed, "source": relative_path, "source_sha256": source_hash,
        "source_step": SOURCE_STEP, "detached_burnin_dropout": dropout_enabled,
        "supervised_dropout": True, "run_name": run_name,
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    identity_path = destination / "identity.json"
    if identity_path.exists():
        import json
        validate_config(json.loads(identity_path.read_text()), identity)
    else:
        atomic_json_save(identity, identity_path)
    atomic_json_save(runtime_manifest(SOURCE_FILES), destination / f"environment_to{stop_after_step}.json")
    print(f"Verified experiment: {identity}", flush=True)
    print(f"Original 50K schedule; run through {stop_after_step}, restoring optimizer and all RNG states", flush=True)
    return train_testbed(
        output_dir=str(destination), experiment_name="exp_burnin_dropout",
        run_name=run_name, random_seed=seed, checkpoint_on_probe=True,
        schedule=FULL_50K_SCHEDULE, branch_checkpoint_path=str(source),
        expected_branch_step=SOURCE_STEP, branch_config_changes=ALLOWED_CHANGES,
        detached_burnin_dropout=dropout_enabled, stop_after_step=stop_after_step,
        record_sample_digest=True,
        **BASE_REPLACEMENT,
    )
