"""Collect target-free early features and separately define late collapse labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from datasets import Dataset

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import RATING_BUCKETS, _load_model

from core import (
    EARLY_ITERATIONS,
    FEATURE_MANIFEST,
    GEOMETRY_FEATURES,
    assign_balanced_splits,
    define_collapse_labels,
    extract_early_features,
)


DEFAULT_DATASET_ARROW = os.path.expanduser(
    "~/.cache/huggingface/datasets/sapientinc___sudoku-extreme/default/0.0.0/"
    "58942f96baeb572ca3127e2a9e9c70f330783d6b/sudoku-extreme-test.arrow"
)


def canonical_model_configs(model_directory):
    return (
        {
            "name": "stable_plain",
            "family": "plain_stable",
            "path": os.path.abspath("model_baseline_lr2e3.pt"),
            "model_kwargs": {},
        },
        {
            "name": "collapsed_plain",
            "family": "plain_clean_a",
            "path": os.path.join(model_directory, "model_baseline_lr2e3_clean_a.pt"),
            "model_kwargs": {},
        },
        {
            "name": "late_state_ce",
            "family": "late_state_ce",
            "path": os.path.join(
                model_directory,
                "model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt",
            ),
            "model_kwargs": {},
        },
        {
            "name": "combined_margin",
            "family": "combined_margin",
            "path": os.path.join(
                model_directory,
                "model_loop_stay_late_switch_margin_floor5_from39k.pt",
            ),
            "model_kwargs": {},
        },
    )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_balanced_arrow_sample(dataset_arrow, examples_per_bucket, seed):
    """Load the cached canonical test Arrow file without creating cache locks."""

    dataset = Dataset.from_file(dataset_arrow)
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, rating in enumerate(dataset["rating"]):
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= rating <= maximum:
                bucket_indices[name].append(index)
                break
    generator = random.Random(seed)
    selected_indices = []
    bucket_names = []
    rating_values = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        selected = generator.sample(candidates, examples_per_bucket)
        selected_indices.extend(selected)
        bucket_names.extend([name] * examples_per_bucket)
        rating_values.extend(int(dataset[index]["rating"]) for index in selected)
    rows = [dataset[index] for index in selected_indices]
    puzzles = [row["question"] for row in rows]
    solutions = [row["answer"] for row in rows]
    inputs = model_module.encode_puzzles(puzzles)
    targets = model_module.encode_solutions(solutions).long()
    blank_mask = inputs[:, :, 0].bool()
    return inputs, targets, blank_mask, puzzles, bucket_names, rating_values


def collect_batch(model, inputs, iteration_permutations):
    """Extract early features before returning the iteration-1024 prediction."""

    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    selected = set(EARLY_ITERATIONS)
    states = []
    updates = []
    logits = []
    prediction_128 = None
    prediction_1024 = None
    with torch.inference_mode():
        hidden = model.initial_encoder(inputs)
        predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
        for iteration in range(1, 1025):
            next_hidden = model.recurrent_step(
                hidden,
                predictions,
                rope_cos,
                rope_sin,
            )
            current_logits = model.output_head(next_hidden)
            if iteration in selected:
                states.append(next_hidden.float())
                updates.append((next_hidden - hidden).float())
                logits.append(current_logits.float())
            if iteration == 128:
                prediction_128 = current_logits.argmax(dim=-1).cpu()
            if iteration == 1024:
                prediction_1024 = current_logits.argmax(dim=-1).cpu()
            hidden = next_hidden
            predictions = F.softmax(current_logits, dim=-1)

        # Targets are intentionally not an argument to this function.  Feature
        # extraction completes before the caller defines any collapse label.
        blank_mask = inputs[:, :, 0].bool()
        stacked_states = torch.stack(states, dim=1)
        stacked_updates = torch.stack(updates, dim=1)
        stacked_logits = torch.stack(logits, dim=1)
        features = extract_early_features(
            EARLY_ITERATIONS,
            stacked_states,
            stacked_updates,
            stacked_logits,
            blank_mask,
        )
        permutations = iteration_permutations.to(inputs.device)
        state_index = permutations[:, :, None, None].expand_as(stacked_states)
        logit_index = permutations[:, :, None, None].expand_as(stacked_logits)
        shuffled_features = extract_early_features(
            EARLY_ITERATIONS,
            stacked_states.gather(1, state_index),
            stacked_updates.gather(1, state_index),
            stacked_logits.gather(1, logit_index),
            blank_mask,
        )
        features.update({
            f"shuffled_iteration_{name}": shuffled_features[name]
            for name in GEOMETRY_FEATURES
        })
    return (
        {name: values.cpu() for name, values in features.items()},
        prediction_128,
        prediction_1024,
    )


def collect(
    output_path,
    *,
    model_directory,
    dataset_arrow=DEFAULT_DATASET_ARROW,
    examples_per_split_bucket=4,
    seed=20260811,
    batch_size=60,
    device="cpu",
):
    if examples_per_split_bucket < 4:
        raise ValueError("the protocol requires at least 20 puzzles per split")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    resolved_device = torch.device(device)
    model_configs = canonical_model_configs(os.path.abspath(model_directory))
    for config in model_configs:
        if not os.path.isfile(config["path"]):
            raise FileNotFoundError(config["path"])
    examples_per_bucket = examples_per_split_bucket * 3
    inputs, targets, blank_mask, puzzles, buckets, ratings = load_balanced_arrow_sample(
        dataset_arrow,
        examples_per_bucket,
        seed,
    )
    split_names = assign_balanced_splits(buckets, examples_per_split_bucket)
    shuffle_generator = torch.Generator().manual_seed(seed + 1281024)
    iteration_permutations = torch.stack([
        torch.randperm(len(EARLY_ITERATIONS), generator=shuffle_generator)
        for _ in range(len(inputs))
    ])
    puzzle_hashes = [
        hashlib.sha256(puzzle.encode("ascii")).hexdigest()[:16]
        for puzzle in puzzles
    ]
    bucket_order = [name for _, _, name in RATING_BUCKETS]
    rows = []
    started_at = time.time()

    for model_config in model_configs:
        model_started_at = time.time()
        model = _load_model(model_config, resolved_device)
        model_features = {name: [] for name in FEATURE_MANIFEST["geometry"]}
        model_features.update({name: [] for name in FEATURE_MANIFEST["output_baseline"]})
        model_features.update({
            name: [] for name in FEATURE_MANIFEST["shuffled_iteration_control"]
        })
        predictions_128 = []
        predictions_1024 = []
        for start in range(0, len(inputs), batch_size):
            stop = min(start + batch_size, len(inputs))
            features, prediction_128, prediction_1024 = collect_batch(
                model,
                inputs[start:stop].to(resolved_device),
                iteration_permutations[start:stop],
            )
            for name, values in features.items():
                model_features[name].append(values)
            predictions_128.append(prediction_128)
            predictions_1024.append(prediction_1024)
        model_features = {
            name: torch.cat(chunks).tolist()
            for name, chunks in model_features.items()
        }
        late_labels = define_collapse_labels(
            torch.cat(predictions_128),
            torch.cat(predictions_1024),
            targets,
            blank_mask,
        )
        for puzzle_index in range(len(inputs)):
            eligible = bool(late_labels["eligible"][puzzle_index])
            row = {
                "puzzle_index": puzzle_index,
                "puzzle_hash": puzzle_hashes[puzzle_index],
                "split": split_names[puzzle_index],
                "rating_bucket": buckets[puzzle_index],
                "rating": ratings[puzzle_index],
                "rating_ordinal": bucket_order.index(buckets[puzzle_index]),
                "clue_fraction": float((~blank_mask[puzzle_index]).float().mean()),
                "checkpoint": model_config["name"],
                "checkpoint_family": model_config["family"],
                "solved_128": bool(late_labels["solved_128"][puzzle_index]),
                "solved_1024": bool(late_labels["solved_1024"][puzzle_index]),
                "eligible": eligible,
                "collapse": int(late_labels["collapse"][puzzle_index]) if eligible else None,
            }
            row.update({name: float(values[puzzle_index]) for name, values in model_features.items()})
            rows.append(row)
        elapsed = time.time() - model_started_at
        print(
            f"{model_config['name']}: eligible={int(late_labels['eligible'].sum())}, "
            f"collapse={int(late_labels['collapse'].sum())}, elapsed={elapsed:.1f}s",
            flush=True,
        )
        del model

    payload = {
        "format_version": 1,
        "config": {
            "seed": seed,
            "examples_per_split_bucket": examples_per_split_bucket,
            "puzzles_per_split": examples_per_split_bucket * len(RATING_BUCKETS),
            "sample_size": len(inputs),
            "dataset_arrow": os.path.abspath(dataset_arrow),
            "early_iterations": list(EARLY_ITERATIONS),
            "late_outcome_iteration": 1024,
            "device": str(resolved_device),
            "models": [
                {
                    "name": config["name"],
                    "family": config["family"],
                    "filename": os.path.basename(config["path"]),
                    "sha256": _sha256(config["path"]),
                }
                for config in model_configs
            ],
            "iteration_shuffle_seed": seed + 1281024,
            "iteration_shuffle_permutations": iteration_permutations.tolist(),
            "elapsed_seconds": time.time() - started_at,
        },
        "feature_manifest": FEATURE_MANIFEST,
        "label_definition": {
            "eligible": "all originally blank cells correct at iteration 128",
            "collapse": "eligible puzzle has at least one wrong originally blank cell at iteration 1024",
        },
        "rows": rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary_path = output_path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, output_path)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-directory", required=True)
    parser.add_argument("--dataset-arrow", default=DEFAULT_DATASET_ARROW)
    parser.add_argument("--examples-per-split-bucket", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    collect(
        arguments.output,
        model_directory=arguments.model_directory,
        dataset_arrow=arguments.dataset_arrow,
        examples_per_split_bucket=arguments.examples_per_split_bucket,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
        device=arguments.device,
    )


if __name__ == "__main__":
    main()
