"""Exercise the real compiled CUDA trainer, save/resume, and dropout toggle."""

from pathlib import Path
from unittest.mock import patch
from contextlib import nullcontext

import torch
from datasets import Dataset

from checkpoint_utils import atomic_json_save
from stabilize import exp_testbed_20k as trainer


def run(output_dir, backend="inductor"):
    output_dir = Path(output_dir)
    solution = "534678912672195348198342567859761423426853791713924856961537284287419635345286179"
    dataset = Dataset.from_dict({
        "question": ["." + solution[1:]] * 5,
        "answer": [solution] * 5,
        "rating": [0, 1, 3, 11, 51],
    })
    schedule = {
        "total_steps": 4, "warmup_steps": 1, "eval_every": 4,
        "probe_every": 1000, "phases": ((0, 4, 0, "Smoke fixture"),),
    }
    settings = {
        "run_name": "smoke", "random_seed": 101, "schedule": schedule,
        "run_batch_size": 2, "checkpoint_on_probe": True,
        "late_supervision_horizons": (16,), "late_supervision_probability": 1.0,
        "late_supervision_mix": 1.0,
    }
    original_compile = torch.compile
    compile_context = nullcontext() if backend == "inductor" else patch.object(
        torch, "compile", side_effect=lambda model, **kwargs: original_compile(model, backend=backend, **kwargs),
    )
    with compile_context, patch.object(trainer, "load_dataset", return_value=dataset):
        first = trainer.train(str(output_dir / "resume"), stop_after_step=1, **settings)
        before = torch.load(output_dir / "resume/smoke_checkpoint_step1.pt", weights_only=False, map_location="cpu")
        resumed = trainer.train(str(output_dir / "resume"), **settings)
        after = torch.load(output_dir / "resume/smoke_checkpoint_step3.pt", weights_only=False, map_location="cpu")
        off = trainer.train(str(output_dir / "dropout_off"), detached_burnin_dropout=False, stop_after_step=1, **settings)
    if first["last_step"] != 1 or resumed["last_step"] != 3:
        raise AssertionError("Training did not resume at the requested step")
    if after["late_horizon_counts"] != {"16": 4}:
        raise AssertionError("Resume repeated or skipped optimizer updates")
    for key, state in before["optimizer_state_dict"]["state"].items():
        new_state = after["optimizer_state_dict"]["state"][key]
        if new_state["step"].item() != state["step"].item() + 2:
            raise AssertionError("Optimizer step state was not preserved")
    for tensor in after["model_state_dict"].values():
        if not torch.isfinite(tensor).all():
            raise AssertionError("Non-finite model after resume")
    if off["config"]["detached_burnin_dropout"] is not False:
        raise AssertionError("Dropout intervention was not recorded")
    result = {
        "passed": True, "torch_compile_backend": backend, "cuda_training": True,
        "resume_from_step": 1, "resume_to_step": 3,
        "resumed_late_horizon_counts": after["late_horizon_counts"],
        "dropout_off_updates": 2, "architecture": "full four-layer, 16 supervised iterations",
        "fixture": "five copies of an almost-filled board; mechanics only, not a score experiment",
    }
    atomic_json_save(result, output_dir / "smoke_result.json")
    return result
