"""Paired old/v2 checkpoint diagnostics; all outputs are separate from weights."""

import copy
import json
import time
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from checkpoint_utils import atomic_json_save
from model_io import DEFAULT_SETTINGS, build_model, model_settings
from runtime_utils import file_sha256, runtime_manifest
from stabilize.exp_testbed_20k import ROPE_COS, ROPE_SIN
from looping.spectral_diagnostics.core import (
    JacobianOperator, derivative_checks, estimate_eigenvalues,
    finite_difference_power, unit_directions,
)

DIRECTORY = Path(__file__).parent
ROOT = DIRECTORY.parents[1]
SOURCE_PATHS = (
    "requirements-modal.txt", "checkpoint_utils.py", "dataset_utils.py", "runtime_utils.py",
    "model_io.py", "iters/state_norm.py", "iters/exp_baseline_lr2e3.py",
    "iters/exp_baseline_lr3e3.py", "stabilize/exp_testbed_20k.py",
    "release/v2/model_late_state_ce.pt.json",
    *[f"looping/spectral_diagnostics/{name}" for name in
      ("__init__.py", "protocol.json", "core.py", "run.py", "modal_run.py", "test_spectral.py")],
)


def protocol():
    return json.loads((DIRECTORY / "protocol.json").read_text())


def precision(high=False):
    torch.set_float32_matmul_precision("high" if high else "highest")
    torch.backends.cuda.matmul.allow_tf32 = high
    torch.backends.cudnn.allow_tf32 = False


def load_model(key, volume_root):
    spec = protocol()["models"][key]
    path = Path(volume_root) / spec["path"]
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    settings = dict(DEFAULT_SETTINGS)
    if key == "v2":
        manifest = json.loads((ROOT / "release/v2/model_late_state_ce.pt.json").read_text())
        if manifest["weights"]["sha256"] != spec["sha256"]:
            raise ValueError("v2 source does not match released weights")
        settings = manifest["model"]
    model = build_model(settings)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Expected tensor-only model weights")
    model.load_state_dict(state, strict=True)
    model.eval().requires_grad_(False)
    if model_settings(model) != DEFAULT_SETTINGS:
        raise ValueError("This comparison requires the unchanged plain architecture")
    return model


def recurrent_function(model, dtype, device):
    cosine, sine = ROPE_COS.to(device=device, dtype=dtype), ROPE_SIN.to(device=device, dtype=dtype)

    def step(hidden):
        predictions = model.output_head(hidden).softmax(-1)
        return model.recurrent_step(hidden, predictions, cosine, sine)

    return step


def select_sample(arrays, settings):
    rng = np.random.default_rng(settings["seed"])
    rows = []
    for bucket in ("0", "1-2", "3-10", "11-50", "51+"):
        choices = np.flatnonzero(arrays["labels"] == bucket)[settings["exclude_first_per_bucket"]:]
        rows.extend(sorted(rng.choice(choices, settings["puzzles_per_bucket"], replace=False).tolist()))
    return np.asarray(rows)


@torch.no_grad()
def trace(model, inputs, iterations):
    hidden = model.initial_encoder(inputs)
    predictions = inputs.new_zeros(len(inputs), 81, 9)
    cosine, sine = ROPE_COS.to(inputs), ROPE_SIN.to(inputs)
    saved = {}
    for iteration in range(1, max(iterations) + 1):
        hidden = model.recurrent_step(hidden, predictions, cosine, sine)
        predictions = model.output_head(hidden).softmax(-1)
        if iteration in iterations:
            if not torch.isfinite(hidden).all():
                raise ValueError(f"Nonfinite trajectory at iteration {iteration}")
            saved[iteration] = hidden.clone()
    return saved


def measure_point(model, hidden, target, given_mask, settings):
    started = time.perf_counter()
    function = recurrent_function(model, hidden.dtype, hidden.device)
    operator = JacobianOperator(function, hidden)
    directions = unit_directions(hidden, settings["random_directions"], settings["seed"])
    with torch.no_grad():
        next_hidden = function(hidden)
        board = model.output_head(hidden).argmax(-1)
        update = next_hidden - hidden
    gains = [float(operator.apply(direction).norm()) for direction in directions]
    update_gain = float(operator.apply(update / update.norm()).norm()) if update.norm() > 0 else None
    linear_error = float((operator.apply(directions[0] + directions[1])
                          - operator.apply(directions[0]) - operator.apply(directions[1])).norm())
    checks = derivative_checks(function, hidden, directions, settings["finite_difference_epsilons"])
    # Each sampled direction must have at least one independently agreeing FP64 difference.
    stride = len(settings["finite_difference_epsilons"])
    if linear_error > 1e-8 or any(min(row["central_relative_error"] for row in checks[i:i + stride]) > 1e-3
                                for i in range(0, len(checks), stride)):
        raise ValueError("Automatic derivative failed linearity or FP64 finite-difference validation")
    solvers = [estimate_eigenvalues(operator, seed=seed, k=settings["eigenvalues"],
                                   ncv=settings["arnoldi_ncv"], maxiter=settings["arnoldi_maxiter"],
                                   tolerance=settings["arnoldi_tolerance"],
                                   residual_tolerance=settings["residual_tolerance"])
               for seed in settings["solver_seeds"]]
    radii = [row["radius"] for row in solvers]
    agreement = all(radius is not None for radius in radii) and bool(np.isclose(
        radii[0], radii[1], rtol=settings["seed_agreement_rtol"], atol=1e-6))
    result = {"state_rms": float(hidden.square().mean().sqrt()),
              "update_norm": float(update.norm()), "relative_update": float(update.norm() / hidden.norm()),
              "solved": bool(((board == target) | given_mask).all()),
              "raw_board_solved": bool((board == target).all()),
              "random_direction_gains": gains, "update_direction_gain": update_gain,
              "jvp_linearity_absolute_error": linear_error,
              "fp64_derivative_checks": checks, "eigensolvers": solvers,
              "solver_seed_agreement": agreement,
              "spectral_radius": max(radii) if agreement else None}
    audits = {}
    for mode in ("fp64", "fp32_highest", "fp32_high"):
        audit_model = model if mode == "fp64" else copy.deepcopy(model).float()
        state = hidden if mode == "fp64" else hidden.float()
        precision(mode == "fp32_high")
        audit_function = recurrent_function(audit_model, state.dtype, state.device)
        audit_directions = [direction.to(state) for direction in directions]
        audits[mode] = {
            "matmul_precision": torch.get_float32_matmul_precision(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "derivative_checks": derivative_checks(audit_function, state, audit_directions,
                                                    settings["finite_difference_epsilons"]),
            "historical_power": finite_difference_power(audit_function, state, seed=settings["seed"],
                                                          steps=settings["historical_power_iterations"]),
        }
        if mode != "fp64":
            del audit_model
    precision()
    result.update({"precision_audits": audits, "seconds": time.perf_counter() - started})
    return result


def run(key, output_root, volume_root="/outputs", callback=lambda: None, smoke=False):
    settings = protocol()
    directory = Path(output_root) / (f"smoke_{key}" if smoke else key)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    torch.set_num_threads(4)
    precision()
    data_path = Path(volume_root) / settings["data_path"]
    if file_sha256(data_path) != settings["data_sha256"]:
        raise ValueError("Prepared puzzle sample checksum mismatch")
    with np.load(data_path, allow_pickle=False) as arrays:
        rows = select_sample(arrays, settings)
        digits, targets = arrays["digits"][rows], arrays["targets"][rows]
        sample = {"rows": rows.tolist(), "indices": arrays["indices"][rows].tolist(),
                  "buckets": arrays["labels"][rows].tolist()}
    identity = {"settings": settings, "key": key, "sample": sample, "smoke": smoke,
                "source_sha256": {name: file_sha256(ROOT / name) for name in SOURCE_PATHS}}
    identity_path = directory / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Refusing resume with different source, model or sample")
    atomic_json_save(identity, identity_path)
    if (directory / "completed.json").exists():
        return json.loads((directory / "completed.json").read_text())
    if smoke:
        suite = unittest.defaultTestLoader.loadTestsFromName("looping.spectral_diagnostics.test_spectral")
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise ValueError("GPU-image tests failed")
        for model_key in settings["models"]:
            load_model(model_key, volume_root)
    model = load_model(key, volume_root).to(device="cuda", dtype=torch.float64)
    if any(value.dtype != torch.float64 for value in model.parameters()):
        raise ValueError("Non-FP64 parameter")
    precision()
    inputs = F.one_hot(torch.as_tensor(digits, device="cuda").long(), 10).double()
    targets = torch.as_tensor(targets, device="cuda").long()
    iterations = [16] if smoke else settings["iterations"]
    activation_dtypes = set()

    def observe(_module, _args, output):
        if isinstance(output, torch.Tensor):
            activation_dtypes.add(str(output.dtype))

    handles = [module.register_forward_hook(observe) for module in model.modules()]
    with sdpa_kernel(SDPBackend.MATH):
        saved = trace(model, inputs, iterations)
    for handle in handles:
        handle.remove()
    if activation_dtypes != {"torch.float64"}:
        raise ValueError(f"Unexpected activation dtypes: {activation_dtypes}")
    environment = runtime_manifest(SOURCE_PATHS)
    environment.update({"activation_dtypes": sorted(activation_dtypes),
                        "attention_backend": "math", "model_module": str(__import__(
                            "stabilize.exp_testbed_20k", fromlist=["__file__"]).__file__)})
    atomic_json_save(environment, directory / "environment.json")
    callback()
    print(f"VERIFIED {key} sha256={settings['models'][key]['sha256']} sample={sample['indices']} "
          f"FP64 activations={sorted(activation_dtypes)} math_SDPA", flush=True)
    records = []
    with sdpa_kernel(SDPBackend.MATH):
        for iteration in iterations:
            for index in range(1 if smoke else len(rows)):
                path = directory / f"iter{iteration}_puzzle{sample['indices'][index]}.json"
                if path.exists():
                    record = json.loads(path.read_text())
                else:
                    record = measure_point(model, saved[iteration][index:index + 1], targets[index:index + 1],
                                           inputs[index:index + 1, :, 0] == 0, settings)
                    record.update({"iteration": iteration, "puzzle_index": sample["indices"][index],
                                   "bucket": sample["buckets"][index]})
                    atomic_json_save(record, path)
                    callback()
                records.append(record)
                print(f"POINT {key} iter={iteration} puzzle={record['puzzle_index']} "
                      f"solved={record['solved']} rho={record['spectral_radius']} "
                      f"oldstyle_fp32={record['precision_audits']['fp32_high']['historical_power']['last_gain']:.4f} "
                      f"seconds={record['seconds']:.1f}", flush=True)
    completed = {"status": "complete", "identity": identity, "records": records,
                 "validated_points": sum(row["spectral_radius"] is not None for row in records),
                 "seconds_this_invocation": time.perf_counter() - started,
                 "sha256": {str(path.relative_to(directory)): file_sha256(path)
                            for path in directory.glob("*.json") if path.name != "completed.json"}}
    atomic_json_save(completed, directory / "completed.json")
    callback()
    print(f"COMPLETE {key} validated={completed['validated_points']}/{len(records)} "
          f"seconds={completed['seconds_this_invocation']:.1f}", flush=True)
    return {"directory": str(directory), "validated_points": completed["validated_points"]}
