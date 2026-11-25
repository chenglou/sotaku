"""Modal jobs for the paper: a fresh paired training cohort, evaluations, and figure data.

Every app and volume name starts with `cheng-`. Launch each job as its own detached
invocation from the repository root, for example:

    modal run --detach paper/modal_jobs.py::prepare
    modal run --detach paper/modal_jobs.py::train --arm late_state_ce --seed 20261119
    modal run --detach paper/modal_jobs.py::evaluate --model paper_cohort/model_paper_late_state_ce_seed20261119.pt
    modal run --detach paper/modal_jobs.py::analyze --model release/model_late_state_ce.pt --name v2

Each worker writes `status/<job>.json` on the volume when it starts, finishes, or fails.
"""

import sys

import modal

# Inside a worker this file is imported from /root, while the project lives in /root/project.
sys.path.insert(0, "/root/project")
from modal_config import PROJECT_IGNORE  # noqa: E402

APP_NAME = "cheng-sotaku-paper"
OUTPUT_VOLUME = "cheng-sotaku-outputs"
CACHE_VOLUME = "cheng-sotaku-hf-cache"
DEFAULT_ITERATIONS = "1,2,4,8,16,32,64,128,256,512,1024,2048,4096"

app = modal.App(APP_NAME)
hf_cache_volume = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
outputs_volume = modal.Volume.from_name(OUTPUT_VOLUME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements-modal.txt")
    .pip_install("matplotlib==3.9.4")
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


@app.function(image=image, cpu=4.0, memory=16384, timeout=60 * 60, volumes=VOLUMES)
def prepare_data():
    """Download the pinned dataset once so later GPU jobs start from a warm cache."""
    _enter_project()
    from datasets import load_dataset

    from dataset_utils import DATASET_NAME, DATASET_REVISION

    sizes = {}
    for split in ("train", "test"):
        sizes[split] = len(load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=split))
    hf_cache_volume.commit()
    _write_status("prepare_data", "finished", sizes=sizes)
    return sizes


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_training(arm: str, run_name: str, random_seed: int):
    import subprocess
    import traceback

    _enter_project()
    import torch

    outputs_volume.reload()
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True,
    ).stdout.strip()
    job = f"train_{run_name}"
    _write_status(job, "started", arm=arm, seed=random_seed, gpu=driver,
                  torch=torch.__version__, cuda=torch.version.cuda)

    from looping.exp_health_methods import train as train_method

    try:
        result = train_method(
            output_dir="/outputs/paper_cohort", arm=arm, run_name=run_name,
            random_seed=random_seed,
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(
        job, "finished", final_probe_128=result["final_probe_128"],
        final_probe_1024=result["final_probe_1024"], best_probe=result["best_probe"],
    )
    return result


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_small_training(arm: str, run_name: str, random_seed: int, puzzles: int,
                       augmentations: int, subset_seed: int, weight_decay: float,
                       stop_after_step: int, mixed_difficulty: bool = False,
                       learning_rate: float = -1.0, total_steps: int = -1):
    """The 50K-step recipe on a small training set (see paper/small_data.py)."""
    import traceback

    _enter_project()
    import torch

    from paper.small_data import train_small as train_on_small_set

    outputs_volume.reload()
    job = f"train_{run_name}"
    settings = {
        "arm": arm, "seed": random_seed, "puzzles": puzzles, "augmentations": augmentations,
        "subset_seed": subset_seed, "weight_decay": weight_decay if weight_decay >= 0 else 0.01,
        "mixed_difficulty": mixed_difficulty,
        "learning_rate": learning_rate if learning_rate > 0 else 2e-3,
        "total_steps": total_steps if total_steps > 0 else 50000,
    }
    _write_status(job, "started", torch=torch.__version__, **settings)
    try:
        result = train_on_small_set(
            "/outputs/paper_small", arm=arm, run_name=run_name, random_seed=random_seed,
            puzzles=puzzles, augmentations=augmentations, subset_seed=subset_seed,
            weight_decay=weight_decay if weight_decay >= 0 else None,
            stop_after_step=stop_after_step if stop_after_step >= 0 else None,
            mixed_difficulty=mixed_difficulty,
            learning_rate=learning_rate if learning_rate > 0 else None,
            total_steps=total_steps if total_steps > 0 else None,
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(
        job, "finished", final_probe_128=result["final_probe_128"],
        final_probe_1024=result["final_probe_1024"], best_probe=result["best_probe"], **settings,
    )
    return result


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=8 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_evaluation(model_path: str, output_name: str, iterations: str, precision: str,
                   legacy_exp: str, track_solutions: bool, full_test: bool = False):
    import traceback

    _enter_project()
    from iters.eval_more_iters import evaluate as evaluate_model
    from runtime_utils import output_subdirectory

    outputs_volume.reload()
    job = f"eval_{output_name}"
    _write_status(job, "started", model=model_path, iterations=iterations, precision=precision)
    output_dir = output_subdirectory("/outputs/evaluations", output_name)
    try:
        result = evaluate_model(
            f"/outputs/{model_path}", exp_module=legacy_exp or None,
            iter_counts=[int(value) for value in iterations.split(",")], device="cuda",
            output_dir=output_dir,
            # `full_test` scores every puzzle of the test split instead of the frozen 25K sample.
            benchmark_path=None if full_test else "release/benchmark_25k.json",
            max_test=10 ** 9 if full_test else 5000,
            precision=precision, batch_size=256, compiled=False,
            matmul_precision="highest" if precision == "fp32" else "high",
            track_solutions=track_solutions, legacy_defaults=bool(legacy_exp),
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(job, "finished", scores={
        horizon: row["accuracy_percent"] for horizon, row in result["scores"].items()
    })
    return result["scores"]


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=4 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_analysis(model_path: str, output_name: str, legacy_exp: str, max_iterations: int,
                 puzzles_per_bucket: int):
    import traceback

    _enter_project()
    from paper.scripts.trajectory_stats import collect

    outputs_volume.reload()
    job = f"analysis_{output_name}"
    _write_status(job, "started", model=model_path)
    try:
        summary = collect(
            f"/outputs/{model_path}", f"/outputs/analysis/{output_name}",
            legacy_exp=legacy_exp or None, max_iterations=max_iterations,
            puzzles_per_bucket=puzzles_per_bucket, device="cuda",
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(job, "finished", summary=summary)
    return summary


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_checkpoint_curve(run_name: str, iterations: str, puzzles_per_bucket: int):
    """Re-score every saved training checkpoint of one run with FP32 inference."""
    import traceback

    _enter_project()
    from paper.scripts.checkpoint_curve import score_checkpoints

    outputs_volume.reload()
    job = f"curve_{run_name}"
    _write_status(job, "started", run=run_name)
    try:
        summary = score_checkpoints(
            "/outputs/paper_cohort", run_name, f"/outputs/curves/{run_name}.json",
            iterations=[int(value) for value in iterations.split(",")],
            puzzles_per_bucket=puzzles_per_bucket, device="cuda",
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(job, "finished", checkpoints=summary["checkpoints"])
    return summary


@app.function(
    image=image, gpu="H200", cpu=8.0, timeout=2 * 60 * 60,
    retries=modal.Retries(max_retries=2, initial_delay=30.0),
    volumes=VOLUMES,
)
def run_precision(model_path: str, output_name: str, legacy_exp: str):
    """Compare FP32, BF16, and BF16 with an FP32 carried state on the same puzzles."""
    import traceback

    _enter_project()
    from paper.scripts.precision_state import compare

    outputs_volume.reload()
    job = f"precision_{output_name}"
    _write_status(job, "started", model=model_path)
    try:
        result = compare(
            f"/outputs/{model_path}", f"/outputs/precision/{output_name}.json",
            legacy_exp=legacy_exp or None,
        )
    except BaseException as error:
        _write_status(job, "failed", error=repr(error), traceback=traceback.format_exc()[-4000:])
        raise
    finally:
        outputs_volume.commit()
    _write_status(job, "finished", solved_4096={
        variant: rows["4096"]["solved"] for variant, rows in result["variants"].items()
    })
    return result


@app.local_entrypoint()
def prepare():
    call = prepare_data.spawn()
    print(f"Spawned prepare_data: {call.object_id}")


@app.local_entrypoint()
def train(arm: str = "late_state_ce", seed: int = 20261119):
    arm = arm.replace("-", "_")
    if arm not in ("vanilla", "late_state_ce"):
        raise ValueError("The paper cohort compares only vanilla and late_state_ce")
    run_name = f"paper_{arm}_seed{seed}"
    call = run_training.spawn(arm, run_name, seed)
    print(f"Spawned {run_name}: {call.object_id}")


@app.local_entrypoint()
def train_small(arm: str = "late_state_ce", seed: int = 20261120, puzzles: int = 1000,
                augmentations: int = 1000, subset_seed: int = 0, weight_decay: float = -1.0,
                stop_after_step: int = -1, tag: str = "", mixed: bool = False,
                learning_rate: float = -1.0, total_steps: int = -1):
    """Train on a small subset of the training split with symmetry augmentation.

    The default is the setting of HRM and TRM: 1,000 puzzles, each with 1,000 augmented copies.
    A negative `weight_decay` or `learning_rate` keeps the recipe's value (0.01 and 2e-3).
    `mixed` samples all difficulties from the first step. `total_steps` shortens the whole
    schedule and needs `mixed`. `stop_after_step` is for smoke tests.
    """
    arm = arm.replace("-", "_")
    if arm not in ("vanilla", "late_state_ce"):
        raise ValueError("choose vanilla or late_state_ce")
    name = f"small{puzzles}_{arm}"
    if weight_decay >= 0:
        name += f"_wd{weight_decay:g}".replace(".", "p")
    if mixed:
        name += "_mixed"
    if learning_rate > 0:
        name += f"_lr{learning_rate:g}".replace(".", "p").replace("-", "m")
    if total_steps > 0:
        if not mixed:
            raise ValueError("--total-steps needs --mixed")
        name += f"_steps{total_steps}"
    run_name = f"{name}{'_' + tag if tag else ''}_seed{seed}"
    call = run_small_training.spawn(
        arm, run_name, seed, puzzles, augmentations, subset_seed, weight_decay, stop_after_step,
        mixed, learning_rate, total_steps,
    )
    print(f"Spawned {run_name}: {call.object_id}")


@app.local_entrypoint()
def evaluate(model: str, name: str = "", iters: str = DEFAULT_ITERATIONS,
             precision: str = "fp32", legacy_exp: str = "", track: bool = True,
             full_test: bool = False):
    from pathlib import Path

    output_name = name or Path(model).stem
    call = run_evaluation.spawn(
        model, output_name, iters, precision, legacy_exp, track and not full_test, full_test,
    )
    print(f"Spawned evaluation {output_name}: {call.object_id}")


@app.local_entrypoint()
def analyze(model: str, name: str, legacy_exp: str = "", max_iterations: int = 4096,
            puzzles_per_bucket: int = 200):
    call = run_analysis.spawn(model, name, legacy_exp, max_iterations, puzzles_per_bucket)
    print(f"Spawned analysis {name}: {call.object_id}")


@app.local_entrypoint()
def curve(run: str, iters: str = "128,1024", puzzles_per_bucket: int = 200):
    call = run_checkpoint_curve.spawn(run, iters, puzzles_per_bucket)
    print(f"Spawned checkpoint curve {run}: {call.object_id}")


@app.local_entrypoint()
def precision(model: str, name: str, legacy_exp: str = ""):
    call = run_precision.spawn(model, name, legacy_exp)
    print(f"Spawned precision comparison {name}: {call.object_id}")
