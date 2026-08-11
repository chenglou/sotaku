"""Analyze per-cell digit-cycle and helix-like geometry with held-out puzzles."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS
from looping.trajectory_viz.helix_tests.visual.helix_geometry import (
    NATURAL_ORDER,
    analyze_periodic_readout,
    decoder_row_basis,
    downsample_indices,
    effective_rank_from_centroids,
    fit_pca,
    helix_code,
    intrinsic_digit_geometry,
    pairwise_distances,
    pca_metrics,
    periodic_code,
    per_iteration_periodic_metrics,
    remove_subspace,
    stratified_split_indices,
    unit_normalize,
)
from looping.trajectory_viz.helix_tests.visual.render_helix import (
    render_attribute_atlas,
    render_centroid_diagnostics,
    render_geometry_3d,
    render_model_comparison,
    render_selected_trajectories,
)


DEFAULT_ITERATIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
RATING_BUCKETS = (
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
)
PRIMARY_REPRESENTATIONS = ("unit_state", "unit_update")
ANALYZED_REPRESENTATIONS = (
    "raw_state",
    "unit_state",
    "unit_state_no_decoder",
    "raw_update",
    "unit_update",
    "unit_update_no_decoder",
)


def _safe_name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"unsafe model name: {name!r}")
    return name


def _load_dataset_split(dataset_arrow: str | None):
    if dataset_arrow:
        return Dataset.from_file(dataset_arrow)
    return load_dataset("sapientinc/sudoku-extreme", split="test")


def load_balanced_sample(
    examples_per_bucket: int,
    seed: int,
    *,
    dataset_arrow: str | None = None,
):
    dataset = _load_dataset_split(dataset_arrow)
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, rating in enumerate(dataset["rating"]):
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= rating <= maximum:
                bucket_indices[name].append(index)
                break
    generator = random.Random(seed)
    selected_indices = []
    bucket_names = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        if len(candidates) < examples_per_bucket:
            raise ValueError(
                f"bucket {name!r} has only {len(candidates)} examples"
            )
        chosen = generator.sample(candidates, examples_per_bucket)
        selected_indices.extend(chosen)
        bucket_names.extend([name] * examples_per_bucket)
    selected = dataset.select(selected_indices)
    puzzles = list(selected["question"])
    solutions = list(selected["answer"])
    inputs = model_module.encode_puzzles(puzzles)
    targets = model_module.encode_solutions(solutions).long()
    empty_mask = inputs[:, :, 0].bool()
    return {
        "inputs": inputs,
        "targets": targets,
        "empty_mask": empty_mask,
        "puzzles": puzzles,
        "solutions": solutions,
        "ratings": list(selected["rating"]),
        "bucket_names": bucket_names,
        "source_indices": selected_indices,
    }


def collect_snapshots(
    model,
    inputs: torch.Tensor,
    iterations: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Collect h_t, the applied one-step update, and logits at requested t."""

    requested = set(iterations)
    maximum = max(iterations)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    states = []
    updates = []
    logits_at_state = []
    autocast_enabled = inputs.device.type == "cuda"
    with torch.inference_mode(), torch.autocast(
        device_type=inputs.device.type,
        dtype=torch.bfloat16 if autocast_enabled else torch.float32,
        enabled=autocast_enabled,
    ):
        hidden = model.initial_encoder(inputs)
        feedback = torch.zeros(
            inputs.size(0),
            81,
            9,
            device=inputs.device,
        )
        for iteration in range(maximum + 1):
            logits = model.output_head(hidden)
            if iteration in requested:
                states.append(hidden.detach().float().cpu())
                logits_at_state.append(logits.detach().float().cpu())
            next_hidden = model.recurrent_step(
                hidden,
                feedback,
                rope_cos,
                rope_sin,
            )
            if iteration in requested:
                updates.append((next_hidden - hidden).detach().float().cpu())
            if iteration == maximum:
                break
            hidden = next_hidden
            feedback = F.softmax(model.output_head(hidden), dim=-1)
    return {
        "states": torch.stack(states, dim=1),
        "updates": torch.stack(updates, dim=1),
        "logits": torch.stack(logits_at_state, dim=1),
        "iterations": torch.tensor(iterations, dtype=torch.long),
    }


def flatten_blank_split(
    capture: dict[str, torch.Tensor],
    targets: torch.Tensor,
    empty_mask: torch.Tensor,
    puzzle_indices: list[int],
    puzzle_bucket_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    puzzle_index = torch.tensor(puzzle_indices, dtype=torch.long)
    states = capture["states"][puzzle_index]
    updates = capture["updates"][puzzle_index]
    logits = capture["logits"][puzzle_index]
    puzzle_count, time_count, cell_count, _ = states.shape
    blank = empty_mask[puzzle_index][:, None, :].expand(
        puzzle_count, time_count, cell_count
    )
    split_targets = targets[puzzle_index][:, None, :].expand_as(blank)
    iterations = capture["iterations"][None, :, None].expand_as(blank)
    cells = torch.arange(cell_count)[None, None, :].expand_as(blank)
    puzzles = puzzle_index[:, None, None].expand_as(blank)
    if puzzle_bucket_ids is None:
        puzzle_bucket_ids = torch.zeros(
            empty_mask.size(0), dtype=torch.long
        )
    buckets = puzzle_bucket_ids[puzzle_index][:, None, None].expand_as(blank)
    selected_logits = logits[blank]
    probabilities = F.softmax(selected_logits, dim=-1)
    predicted = probabilities.argmax(dim=-1)
    confidence = probabilities.max(dim=-1).values
    true_digit = split_targets[blank]
    true_logits = selected_logits.gather(1, true_digit[:, None]).squeeze(1)
    wrong_logits = selected_logits.masked_fill(
        F.one_hot(true_digit, num_classes=9).bool(),
        -torch.inf,
    )
    target_margin = true_logits - wrong_logits.max(dim=1).values
    selected_cells = cells[blank]
    return {
        "state": states[blank],
        "update": updates[blank],
        "true_digit": true_digit,
        "predicted_digit": predicted,
        "confidence": confidence,
        "target_margin": target_margin,
        "correct": predicted == true_digit,
        "iteration": iterations[blank],
        "cell": selected_cells,
        "row": selected_cells // 9,
        "column": selected_cells % 9,
        "puzzle": puzzles[blank],
        "bucket": buckets[blank],
    }


def build_representations(
    flattened: dict[str, torch.Tensor],
    output_basis: torch.Tensor,
) -> dict[str, torch.Tensor]:
    state = flattened["state"].float()
    update = flattened["update"].float()
    state_without_decoder = remove_subspace(state, output_basis)
    update_without_decoder = remove_subspace(update, output_basis)
    return {
        "raw_state": state,
        "unit_state": unit_normalize(state),
        "unit_state_no_decoder": unit_normalize(state_without_decoder),
        "raw_update": update,
        "unit_update": unit_normalize(update),
        "unit_update_no_decoder": unit_normalize(update_without_decoder),
    }


def _metadata(flattened: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in flattened.items()
        if key not in ("state", "update")
    }


def _horizon_intrinsic(
    train_values: torch.Tensor,
    train_metadata: dict[str, torch.Tensor],
    test_values: torch.Tensor,
    test_metadata: dict[str, torch.Tensor],
) -> dict:
    result = {}
    for iteration in torch.unique(test_metadata["iteration"]).tolist():
        train_mask = train_metadata["iteration"] == iteration
        test_mask = test_metadata["iteration"] == iteration
        geometry = intrinsic_digit_geometry(
            train_values[train_mask],
            train_metadata["true_digit"][train_mask],
            test_values[test_mask],
            test_metadata["true_digit"][test_mask],
        )
        # Centroid coordinates are already saved in the aggregate result and
        # would dominate the JSON if repeated for every horizon.
        geometry.pop("train_centroids", None)
        geometry.pop("test_centroids", None)
        result[str(int(iteration))] = geometry
    return result


def analyze_representation(
    name: str,
    train_values: torch.Tensor,
    validation_values: torch.Tensor,
    test_values: torch.Tensor,
    train_metadata: dict[str, torch.Tensor],
    validation_metadata: dict[str, torch.Tensor],
    test_metadata: dict[str, torch.Tensor],
    *,
    seed: int,
) -> tuple[dict, dict]:
    projection = fit_pca(train_values, component_count=16)
    geometry = intrinsic_digit_geometry(
        train_values,
        train_metadata["true_digit"],
        test_values,
        test_metadata["true_digit"],
    )
    train_shortest_cycle = tuple(
        digit - 1 for digit in geometry["train_shortest_cycle"]
    )
    periodic, readout, test_scores = analyze_periodic_readout(
        train_values,
        train_metadata["true_digit"],
        validation_values,
        validation_metadata["true_digit"],
        test_values,
        test_metadata["true_digit"],
        test_metadata["puzzle"],
        test_metadata["bucket"],
        train_shortest_cycle,
        seed=seed,
    )
    periodic["test_natural_cycle_by_iteration"] = (
        per_iteration_periodic_metrics(
            test_scores,
            test_metadata["true_digit"],
            test_metadata["iteration"],
        )
    )
    metrics = {
        "sample_counts": {
            "train": int(train_values.size(0)),
            "validation": int(validation_values.size(0)),
            "test": int(test_values.size(0)),
        },
        "pca": pca_metrics(
            projection,
            train_values,
            validation_values,
            test_values,
        ),
        "intrinsic_digit_geometry": geometry,
        "periodic_readout": periodic,
    }
    if name in PRIMARY_REPRESENTATIONS:
        metrics["intrinsic_by_iteration"] = _horizon_intrinsic(
            train_values,
            train_metadata,
            test_values,
            test_metadata,
        )
    coordinates = {
        "pca_test": projection.project(test_values)[:, :3],
        "periodic_test": test_scores @ periodic_code(),
        "helix_test": test_scores @ helix_code(),
        "test_scores": test_scores,
        "pca_projection": projection,
        "readout": readout,
        "train_shortest_cycle": train_shortest_cycle,
    }
    return metrics, coordinates


def output_head_geometry(model) -> dict:
    weights = unit_normalize(model.output_head.weight.detach().float().cpu())
    labels = torch.arange(9)
    geometry = intrinsic_digit_geometry(weights, labels, weights, labels)
    geometry.pop("train_centroids", None)
    geometry.pop("test_centroids", None)
    geometry["centered_spectrum"] = effective_rank_from_centroids(weights)
    geometry["pairwise_distances"] = pairwise_distances(weights).tolist()
    return geometry


def _serialize_sample(
    output_dir: str,
    model_name: str,
    metadata: dict[str, torch.Tensor],
    coordinate_sets: dict[str, dict],
    *,
    seed: int,
) -> str:
    indices = downsample_indices(
        metadata["true_digit"],
        metadata["iteration"],
        6000,
        seed=seed,
    )
    arrays = {
        key: value[indices].numpy()
        for key, value in metadata.items()
    }
    for representation in PRIMARY_REPRESENTATIONS:
        coordinates = coordinate_sets[representation]
        arrays[f"{representation}_pca"] = coordinates["pca_test"][indices].numpy()
        arrays[f"{representation}_periodic"] = coordinates["periodic_test"][indices].numpy()
        arrays[f"{representation}_helix"] = coordinates["helix_test"][indices].numpy()
    filename = f"{model_name}_projection_samples.npz"
    np.savez_compressed(os.path.join(output_dir, filename), **arrays)
    return filename


def _render_model(
    output_dir: str,
    model_name: str,
    metrics: dict,
    metadata: dict[str, torch.Tensor],
    coordinate_sets: dict[str, dict],
    *,
    seed: int,
) -> list[str]:
    artifacts = []
    for representation in PRIMARY_REPRESENTATIONS:
        coordinates = coordinate_sets[representation]
        pca_filename = f"{model_name}_{representation}_pca_atlas.png"
        render_attribute_atlas(
            coordinates["pca_test"][:, :2],
            metadata,
            os.path.join(output_dir, pca_filename),
            title=(
                f"{model_name} · {representation.replace('_', ' ')} · "
                "train-fit PCA on held-out blank cells"
            ),
            axis_labels=("PC1", "PC2"),
            seed=seed,
            target_ring=False,
        )
        artifacts.append(pca_filename)
        periodic_filename = f"{model_name}_{representation}_periodic_atlas.png"
        render_attribute_atlas(
            coordinates["periodic_test"],
            metadata,
            os.path.join(output_dir, periodic_filename),
            title=(
                f"{model_name} · {representation.replace('_', ' ')} · "
                "train-fit, target-imposed natural-cycle readout on held-out blanks"
            ),
            axis_labels=("cosine readout", "sine readout"),
            seed=seed,
            target_ring=True,
        )
        artifacts.append(periodic_filename)
    diagnostics_filename = f"{model_name}_centroid_diagnostics.png"
    render_centroid_diagnostics(
        model_name,
        metrics,
        coordinate_sets,
        os.path.join(output_dir, diagnostics_filename),
    )
    artifacts.append(diagnostics_filename)
    geometry_filename = f"{model_name}_geometry_3d.png"
    render_geometry_3d(
        model_name,
        metadata,
        coordinate_sets,
        os.path.join(output_dir, geometry_filename),
        seed=seed,
    )
    artifacts.append(geometry_filename)
    trajectory_filename = f"{model_name}_selected_cell_trajectories.png"
    render_selected_trajectories(
        model_name,
        metadata,
        coordinate_sets["unit_state"],
        os.path.join(output_dir, trajectory_filename),
    )
    artifacts.append(trajectory_filename)
    return artifacts


def run(
    output_dir: str,
    *,
    examples_per_bucket: int = 20,
    seed: int = 20260807,
    device: str = "cuda",
    model_configs=DEFAULT_MODELS,
    iterations: tuple[int, ...] = DEFAULT_ITERATIONS,
    dataset_arrow: str | None = None,
) -> dict:
    if examples_per_bucket < 4:
        raise ValueError("examples_per_bucket must be at least 4 for train/val/test")
    if tuple(sorted(set(iterations))) != tuple(iterations):
        raise ValueError("iterations must be strictly increasing")
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(
        device if device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    sample = load_balanced_sample(
        examples_per_bucket,
        seed,
        dataset_arrow=dataset_arrow,
    )
    splits = stratified_split_indices(sample["bucket_names"], seed=seed + 1)
    bucket_to_id = {
        bucket: index
        for index, bucket in enumerate(dict.fromkeys(sample["bucket_names"]))
    }
    puzzle_bucket_ids = torch.tensor([
        bucket_to_id[bucket] for bucket in sample["bucket_names"]
    ])
    summary = {
        "config": {
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(sample["puzzles"]),
            "seed": seed,
            "device": str(resolved_device),
            "inference_dtype": (
                "bfloat16 autocast" if resolved_device.type == "cuda" else "float32"
            ),
            "iterations": list(iterations),
            "models": list(model_configs),
            "primary_population": "originally blank cells only",
            "split_unit": "whole puzzle, stratified within rating bucket",
            "periodic_target": "natural digit order 1→2→…→9→1",
            "cycle_null_count": 20160,
        },
        "sample": {
            "puzzles": sample["puzzles"],
            "solutions": sample["solutions"],
            "ratings": sample["ratings"],
            "buckets": sample["bucket_names"],
            "source_indices": sample["source_indices"],
            "splits": splits,
        },
        "models": {},
        "artifacts": [],
    }
    log_path = os.path.join(output_dir, "analysis.log")
    started = time.time()
    comparison_data = {}
    with open(log_path, "w") as log_file:
        def log(message=""):
            print(message, flush=True)
            log_file.write(message + "\n")
            log_file.flush()

        log(
            f"Helix tests: {len(sample['puzzles'])} puzzles, "
            f"splits={{{', '.join(f'{key}:{len(value)}' for key, value in splits.items())}}}, "
            f"models={len(model_configs)}, device={resolved_device}"
        )
        inputs = sample["inputs"].to(resolved_device)
        for model_index, model_config in enumerate(model_configs):
            model_name = _safe_name(model_config["name"])
            model_started = time.time()
            log(f"\nMODEL {model_name}: {model_config['path']}")
            model = _load_model(model_config, resolved_device)
            model.requires_grad_(False)
            capture = collect_snapshots(model, inputs, iterations)
            output_basis = decoder_row_basis(
                model.output_head.weight.detach().float().cpu()
            )
            flattened = {
                split_name: flatten_blank_split(
                    capture,
                    sample["targets"],
                    sample["empty_mask"],
                    indices,
                    puzzle_bucket_ids,
                )
                for split_name, indices in splits.items()
            }
            metadata = {
                split_name: _metadata(values)
                for split_name, values in flattened.items()
            }
            representations = {
                split_name: build_representations(values, output_basis)
                for split_name, values in flattened.items()
            }
            model_metrics = {
                "model_config": model_config,
                "output_head_geometry": output_head_geometry(model),
                "representations": {},
                "elapsed_seconds": None,
                "artifacts": [],
            }
            coordinate_sets = {}
            for representation_index, representation in enumerate(
                ANALYZED_REPRESENTATIONS
            ):
                log(f"  analyzing {representation}")
                representation_metrics, coordinates = analyze_representation(
                    representation,
                    representations["train"][representation],
                    representations["validation"][representation],
                    representations["test"][representation],
                    metadata["train"],
                    metadata["validation"],
                    metadata["test"],
                    seed=seed + model_index * 100 + representation_index,
                )
                model_metrics["representations"][representation] = (
                    representation_metrics
                )
                if representation in PRIMARY_REPRESENTATIONS:
                    coordinate_sets[representation] = coordinates
                natural = representation_metrics["periodic_readout"][
                    "test_natural_cycle"
                ]
                intrinsic = representation_metrics["intrinsic_digit_geometry"]
                log(
                    f"    periodic sector={100 * natural['sector_accuracy']:.1f}%, "
                    f"cos={natural['mean_cosine_alignment']:.3f}; "
                    f"natural cycle shorter percentile="
                    f"{100 * intrinsic['natural_cycle_shorter_percentile']:.1f}%"
                )
            model_metrics["artifacts"] = _render_model(
                output_dir,
                model_name,
                model_metrics,
                metadata["test"],
                coordinate_sets,
                seed=seed + model_index,
            )
            sample_file = _serialize_sample(
                output_dir,
                model_name,
                metadata["test"],
                coordinate_sets,
                seed=seed + model_index,
            )
            model_metrics["artifacts"].append(sample_file)
            model_metrics["elapsed_seconds"] = time.time() - model_started
            summary["models"][model_name] = model_metrics
            summary["artifacts"].extend(model_metrics["artifacts"])
            comparison_data[model_name] = model_metrics
            log(f"  completed in {model_metrics['elapsed_seconds']:.1f}s")
            del model, capture, flattened, representations, coordinate_sets
            if resolved_device.type == "cuda":
                torch.cuda.empty_cache()

        comparison_filename = "model_comparison.png"
        render_model_comparison(
            comparison_data,
            os.path.join(output_dir, comparison_filename),
        )
        summary["artifacts"].append(comparison_filename)
        summary["artifacts"].append("analysis.log")
        summary["elapsed_seconds"] = time.time() - started
        metrics_path = os.path.join(output_dir, "helix_metrics.json")
        temporary_path = metrics_path + ".tmp"
        with open(temporary_path, "w") as result_file:
            json.dump(summary, result_file, indent=2)
            result_file.write("\n")
        os.replace(temporary_path, metrics_path)
        log(f"\nStructured results: {metrics_path}")
        log(f"Total time: {summary['elapsed_seconds']:.1f}s")
    return summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument("--examples-per-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dataset-arrow")
    parser.add_argument(
        "--local-stable",
        action="store_true",
        help="Analyze only the locally available stable baseline checkpoint.",
    )
    parser.add_argument(
        "--maximum-iteration",
        type=int,
        default=1024,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_args()
    if arguments.local_stable:
        configs = (
            {
                "name": "stable_plain",
                "path": "model_baseline_lr2e3.pt",
                "model_kwargs": {},
            },
        )
    else:
        configs = DEFAULT_MODELS
    selected_iterations = tuple(
        iteration
        for iteration in DEFAULT_ITERATIONS
        if iteration <= arguments.maximum_iteration
    )
    run(
        arguments.output_dir,
        examples_per_bucket=arguments.examples_per_bucket,
        seed=arguments.seed,
        device=arguments.device,
        model_configs=configs,
        iterations=selected_iterations,
        dataset_arrow=arguments.dataset_arrow,
    )
