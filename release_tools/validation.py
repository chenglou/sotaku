"""Fixed-weight GPU evaluation and the preregistered numerical sensitivity matrix."""

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from checkpoint_utils import atomic_json_save
from dataset_utils import DATASET_NAME, DATASET_REVISION, manifest_for_indices, validate_benchmark
from inference import RecurrentRunner
from iters.eval_more_iters import evaluate
from model_io import DEFAULT_SETTINGS, V1_SHA256, load_model, write_model_manifest
from runtime_utils import file_sha256
from stabilize.exp_testbed_20k import encode_puzzles

ITERATIONS = (128, 1024, 2048, 4096)
REFERENCE_MODELS = {
    "late_state_ce": (
        "looping/model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt",
        "a12508bd32263596b9d87cd5a2f315c0b2d6ded7e66b7f684a05e46397f51198",
    ),
    "v1": ("model_baseline_lr2e3.pt", V1_SHA256),
}


def prepare_model(name, directory):
    if name in REFERENCE_MODELS:
        relative_path, expected_hash = REFERENCE_MODELS[name]
    else:
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("Custom model paths must be relative to the output volume")
        relative_path, expected_hash = name, None
    source = Path("/outputs") / relative_path
    checksum = file_sha256(source)
    if expected_hash is not None and checksum != expected_hash:
        raise ValueError("The reference weight file has changed")
    destination = directory / ("model_late_state_ce.pt" if name == "late_state_ce" else source.name)
    if destination.exists():
        if file_sha256(destination) != checksum:
            raise ValueError("Refusing to replace a different evaluation model")
    else:
        shutil.copyfile(source, destination)
    if name == "late_state_ce":
        manifest_source = Path("release/v2/model_late_state_ce.pt.json")
    elif name == "v1":
        model, _ = load_model(destination)
        write_model_manifest(destination, model, provenance={"published_v1_sha256": V1_SHA256})
        manifest_source = None
    else:
        manifest_source = Path(str(source) + ".json")
    if manifest_source is not None:
        shutil.copyfile(manifest_source, str(destination) + ".json")
    load_model(destination)
    return destination


def verify_legacy_equivalence(weights_path, dataset, benchmark, directory):
    from iters import exp_baseline_lr2e3 as legacy
    model, metadata = load_model(weights_path, device="cuda")
    settings = metadata["model"]
    if settings != DEFAULT_SETTINGS:
        return {"skipped": "not a plain reference model"}
    original = legacy.SudokuTransformer().cuda().eval()
    original.load_state_dict(torch.load(weights_path, weights_only=True, map_location="cpu"))
    inputs = encode_puzzles(dataset.select(benchmark["indices"][:8])["question"]).cuda()
    comparisons = {}
    old_iterations = legacy.n_iterations
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            current, _ = RecurrentRunner(model).run_batch(inputs, (16, 128, 1024))
            for horizon in (16, 128, 1024):
                legacy.n_iterations = horizon
                expected = original(inputs)
                actual = current[horizon]
                comparisons[str(horizon)] = {
                    "equal": bool(torch.equal(expected, actual)),
                    "max_logit_difference": float((expected.float() - actual.float()).abs().max()),
                }
    finally:
        legacy.n_iterations = old_iterations
    atomic_json_save(comparisons, directory / "legacy_equivalence.json")
    if not all(value["equal"] for value in comparisons.values()):
        raise AssertionError(f"Public recurrence differs from the historical forward: {comparisons}")
    print(f"Historical forward matches exactly at 16/128/1024: {comparisons}", flush=True)
    return comparisons


def compare_predictions(baseline_path, variant_path, empty_masks):
    comparisons = {}
    with np.load(baseline_path, allow_pickle=False) as baseline, np.load(variant_path, allow_pickle=False) as variant:
        if not np.array_equal(baseline["indices"], variant["indices"]):
            raise ValueError("Cannot compare different puzzle indices")
        for horizon in ITERATIONS:
            before, after = baseline[f"solved_{horizon}"], variant[f"solved_{horizon}"]
            changed = ((baseline[f"predictions_{horizon}"] != variant[f"predictions_{horizon}"]) & empty_masks).any(-1)
            comparisons[str(horizon)] = {
                "baseline_solved": int(before.sum()), "variant_solved": int(after.sum()),
                "baseline_only_solved": int((before & ~after).sum()),
                "variant_only_solved": int((~before & after).sum()),
                "changed_blank_cell_boards": int(changed.sum()),
            }
    return comparisons


def run(kind, model_name, directory):
    directory = Path(directory)
    weights_path = prepare_model(model_name, directory)
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="test")
    full_benchmark = json.loads(Path("release/benchmark_25k.json").read_text())
    validate_benchmark(dataset, full_benchmark)
    verify_legacy_equivalence(weights_path, dataset, full_benchmark, directory)
    if kind == "full":
        return evaluate(
            weights_path, iter_counts=ITERATIONS, output_dir=directory / "full",
            benchmark_path="release/benchmark_25k.json", precision="bf16", batch_size=256,
            device="cuda", matmul_precision="high", dataset=dataset,
        )
    if kind != "precision" or model_name not in REFERENCE_MODELS:
        raise ValueError("Precision checks require one of the two reference checkpoints")
    indices, names = [], []
    for label in ("0", "1-2", "3-10", "11-50", "51+"):
        chosen = [index for index, bucket in zip(full_benchmark["indices"], full_benchmark["bucket_names"]) if bucket == label][:200]
        indices.extend(chosen)
        names.extend([label] * len(chosen))
    benchmark = manifest_for_indices(dataset, indices, names)
    benchmark_path = directory / "precision_benchmark.json"
    atomic_json_save(benchmark, benchmark_path)
    empty_masks = encode_puzzles(dataset.select(indices)["question"])[:, :, 0].bool().numpy()
    results = {}
    baseline = directory / "bf16_eager_bs256" / "per_puzzle.npz"
    for precision in ("bf16", "fp32"):
        for compiled in (False, True):
            for batch_size in (256, 32):
                name = f"{precision}_{'compiled' if compiled else 'eager'}_bs{batch_size}"
                if compiled:
                    torch._dynamo.reset()
                result = evaluate(
                    weights_path, iter_counts=ITERATIONS, output_dir=directory / name,
                    benchmark_path=benchmark_path, precision=precision, batch_size=batch_size,
                    compiled=compiled, matmul_precision="highest" if precision == "fp32" else "high",
                    device="cuda", dataset=dataset,
                )
                results[name] = {
                    "scores": result["scores"],
                    "paired_against_bf16_eager_bs256": compare_predictions(baseline, directory / name / "per_puzzle.npz", empty_masks),
                }
                atomic_json_save(results, directory / "precision_results.json")
                torch.cuda.empty_cache()
    for precision in ("bf16", "fp32"):
        name = f"{precision}_trajectory_bs256"
        result = evaluate(
            weights_path, iter_counts=ITERATIONS, output_dir=directory / name,
            benchmark_path=benchmark_path, precision=precision, batch_size=256,
            matmul_precision="highest" if precision == "fp32" else "high",
            track_solutions=True, device="cuda", dataset=dataset,
        )
        comparisons = compare_predictions(
            directory / f"{precision}_eager_bs256" / "per_puzzle.npz",
            directory / name / "per_puzzle.npz", empty_masks,
        )
        if any(value["changed_blank_cell_boards"] for value in comparisons.values()):
            raise AssertionError("Recording the eager trajectory changed its predictions")
        results[name] = {"scores": result["scores"], "solution_tracking": result["solution_tracking"]}
        atomic_json_save(results, directory / "precision_results.json")
    atomic_json_save({"complete": True, "conditions": len(results)}, directory / "completed.json")
    return results
