"""Modal jobs for the attention-residual and message variants.

Every app and volume name starts with `cheng-`. Launch each job as its own detached
invocation from the repository root, for example:

    modal run --detach looping/attention_residuals/modal_run.py::preflight
    modal run --detach looping/attention_residuals/modal_run.py::train --arm attnres --seed 20261123
    modal run --detach looping/attention_residuals/modal_run.py::evaluate --arm attnres --seed 20261123

Each worker writes `status/<job>.json` on the volume when it starts, finishes, or fails.
After a failed launch, stop the App (`modal app stop <app-id>`) before relaunching.
"""

import sys

import modal

# Inside a worker this file is imported from /root, while the project lives in /root/project.
sys.path.insert(0, "/root/project")
from modal_config import PROJECT_IGNORE  # noqa: E402

APP_NAME = "cheng-sotaku-attnres"
OUTPUT_VOLUME = "cheng-sotaku-outputs"
CACHE_VOLUME = "cheng-sotaku-hf-cache"
RUN_DIRECTORY = "attnres20k"

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .add_local_dir(".", remote_path="/root/project", ignore=PROJECT_IGNORE)
)
VOLUMES = {"/hf_cache": hf_cache_volume, "/outputs": outputs_volume}


def _enter_project():
    import os
    import sys

    os.environ["HF_HOME"] = "/hf_cache"
    os.environ["HF_DATASETS_CACHE"] = "/hf_cache/datasets"
    if "/root/project" not in sys.path:
        sys.path.insert(0, "/root/project")
    os.chdir("/root/project")


def _write_status(job, state, **details):
    import datetime
    import json
    import os

    os.makedirs("/outputs/status", exist_ok=True)
    record = {
        "job": job,
        "state": state,
        "time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **details,
    }
    path = f"/outputs/status/{job}.json"
    history = []
    if os.path.exists(path):
        with open(path) as handle:
            history = json.load(handle).get("history", [])
    history.append({"state": state, "time_utc": record["time_utc"]})
    record["history"] = history
    with open(path + ".tmp", "w") as handle:
        json.dump(record, handle, indent=2)
    os.replace(path + ".tmp", path)
    outputs_volume.commit()


@app.function(image=image, gpu="H200", cpu=8.0, timeout=30 * 60, volumes=VOLUMES)
def run_preflight(tag: str):
    """One training-sized step per arm: checks imports and reports peak GPU memory."""
    import time
    import traceback

    _enter_project()
    import torch
    import torch.nn.functional as F

    from looping.attention_residuals.model import LoopVariantTransformer
    from stabilize.exp_testbed_20k import SudokuTransformer, build_detached_advance

    outputs_volume.reload()
    job = f"preflight_attnres_{tag}"
    _write_status(job, "started", torch=torch.__version__)
    report = {}
    try:
        generator = torch.Generator().manual_seed(0)
        digits = torch.randint(0, 10, (2048, 81), generator=generator)
        inputs = F.one_hot(digits, 10).float().cuda()
        targets = torch.randint(0, 9, (2048, 81), generator=generator).cuda()
        def training_step(model):
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                advance = build_detached_advance(model, compile_chunk=False)
                start = (model.initial_encoder(inputs), torch.zeros(2048, 81, 9, device="cuda"))
                state = advance(start, 64)
                logits = model(inputs, return_all=True, initial_state=state)
                loss = torch.stack([
                    F.cross_entropy(step.flatten(0, 1).float(), targets.flatten()) for step in logits
                ]).mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            return time.perf_counter() - started, float(loss)

        arms = [("baseline", {}), ("attnres", {}), ("message", {}), ("attnres_message", {}),
                ("message", {"separate_qkv": True}), ("attnres_message", {"separate_qkv": True})]
        for arm, settings in arms:
            name = arm + ("_separate_qkv" if settings else "")
            torch.manual_seed(0)
            model = (SudokuTransformer() if arm == "baseline"
                     else LoopVariantTransformer(arm, **settings)).cuda().train()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                steps = [training_step(model) for _ in range(3)]
                report[name] = {
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
                    "step_seconds_uncompiled": min(seconds for seconds, _ in steps),
                    "loss": steps[-1][1],
                }
            except torch.OutOfMemoryError:
                report[name] = {"out_of_memory": True,
                                "peak_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(name, report[name], flush=True)
            del model
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:],
                      report=report)
        raise
    _write_status(job, "finished", report=report)
    return report


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_training(arm: str, random_seed: int, tag: str, separate_qkv: bool):
    import traceback

    _enter_project()
    import torch

    from looping.attention_residuals.train import run_name, train_arm

    outputs_volume.reload()
    name = run_name(arm, random_seed, tag)
    job = f"train_{name}"
    variant_settings = {} if arm in ("baseline", "attnres") else {"separate_qkv": separate_qkv}
    _write_status(job, "started", arm=arm, seed=random_seed, torch=torch.__version__, **variant_settings)
    try:
        result = train_arm(f"/outputs/{RUN_DIRECTORY}", arm=arm, random_seed=random_seed, tag=tag,
                           **variant_settings)
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(
        job, "finished", arm=arm, seed=random_seed, final_probe_128=result["final_probe_128"],
        final_probe_1024=result["final_probe_1024"], best_probe=result["best_probe"],
        **variant_settings,
    )
    return result


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=8 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_evaluation(arm: str, random_seed: int, tag: str, separate_qkv: bool, iterations: str,
                   best_probe: bool):
    import traceback

    _enter_project()
    from looping.attention_residuals.evaluate import evaluate_arm
    from looping.attention_residuals.train import run_name
    from runtime_utils import output_subdirectory

    outputs_volume.reload()
    name = run_name(arm, random_seed, tag) + ("_best_probe" if best_probe else "")
    job = f"eval_{name}"
    variant_settings = {} if arm in ("baseline", "attnres") else {"separate_qkv": separate_qkv}
    _write_status(job, "started", arm=arm, seed=random_seed, iterations=iterations)
    try:
        result = evaluate_arm(
            f"/outputs/{RUN_DIRECTORY}/model_{name}.pt",
            output_subdirectory("/outputs/evaluations", name), arm=arm,
            iterations=[int(value) for value in iterations.split(",")], **variant_settings,
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(job, "finished", arm=arm, seed=random_seed, scores={
        horizon: row["accuracy_percent"] for horizon, row in result["scores"].items()
    })
    return result["scores"]


@app.local_entrypoint()
def preflight(tag: str = "v1"):
    call = run_preflight.spawn(tag)
    print(f"Spawned preflight {tag}: {call.object_id}")


@app.local_entrypoint()
def train(arm: str, seed: int, tag: str = "", separate_qkv: bool = False):
    call = run_training.spawn(arm, seed, tag, separate_qkv)
    print(f"Spawned training {arm} seed {seed}: {call.object_id}")


@app.local_entrypoint()
def evaluate(arm: str, seed: int, tag: str = "", separate_qkv: bool = False,
             iters: str = "16,128,1024,4096", best_probe: bool = False):
    call = run_evaluation.spawn(arm, seed, tag, separate_qkv, iters, best_probe)
    print(f"Spawned evaluation {arm} seed {seed}: {call.object_id}")
