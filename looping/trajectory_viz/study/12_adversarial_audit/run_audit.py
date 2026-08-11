"""Run the preregistered adversarial audit of recurrent trajectory geometry."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPOSITORY_ROOT))

from audit_core import (  # noqa: E402
    REPRESENTATION_NAMES,
    aggregate_metric,
    benjamini_hochberg,
    build_representations,
    dynamic_variance_fraction,
    file_sha256,
    json_ready,
    load_json,
    local_pca_coordinates,
    make_development_resplits,
    make_three_way_split,
    percentile_against_controls,
    per_puzzle_metrics,
    project,
    puzzle_sha256,
    random_orthonormal_coefficients,
    select_plane,
    synthetic_selection_experiment,
    temporal_center,
    time_shuffle_test,
    write_json,
)
from looping.eval_loop_diagnostics import RATING_BUCKETS, _load_model  # noqa: E402
import stabilize.exp_testbed_20k as model_module  # noqa: E402


CHECKPOINT_NAMES = (
    "stable_plain",
    "collapsed_plain",
    "late_state_ce",
    "combined_margin",
)
DEFAULT_CHECKPOINTS = {
    "stable_plain": REPOSITORY_ROOT / "model_baseline_lr2e3.pt",
    "collapsed_plain": Path(
        "/tmp/sotaku-arm12-checkpoints/model_baseline_lr2e3_clean_a.pt"
    ),
    "late_state_ce": Path(
        "/tmp/sotaku-arm12-checkpoints/"
        "model_loop_health_standalone_v1_late_state_ce_50k_trial0.pt"
    ),
    "combined_margin": Path(
        "/tmp/sotaku-arm12-checkpoints/"
        "model_loop_stay_late_switch_margin_floor5_from39k.pt"
    ),
}
DEFAULT_TEST_ARROW = Path(
    "/Users/chenglou/.cache/huggingface/datasets/"
    "sapientinc___sudoku-extreme/default/0.0.0/"
    "58942f96baeb572ca3127e2a9e9c70f330783d6b/"
    "sudoku-extreme-test.arrow"
)
PRIMARY_REPRESENTATION = "normalized_state"
DEVELOPMENT_PERMUTATIONS = 99
DEVELOPMENT_BOOTSTRAPS = 300


def log(message):
    print(message, flush=True)


def stable_seed(base_seed, *parts):
    text = "::".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "little")


def load_balanced_sample_from_arrow(arrow_path, examples_per_bucket, seed):
    """Mirror the canonical loader without trying to write a cache lock."""

    from datasets import Dataset

    dataset = Dataset.from_file(str(arrow_path))
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, example in enumerate(dataset):
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= example["rating"] <= maximum:
                bucket_indices[name].append(index)
                break
    generator = random.Random(seed)
    selected_indices = []
    bucket_names = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        if len(candidates) < examples_per_bucket:
            raise ValueError(f"rating bucket {name!r} has too few examples")
        selected_indices.extend(generator.sample(candidates, examples_per_bucket))
        bucket_names.extend([name] * examples_per_bucket)
    puzzles = [dataset[index]["question"] for index in selected_indices]
    solutions = [dataset[index]["answer"] for index in selected_indices]
    inputs = model_module.encode_puzzles(puzzles)
    targets = model_module.encode_solutions(solutions).long()
    originally_blank = inputs[:, :, 0].bool()
    return {
        "inputs": inputs,
        "targets": targets,
        "originally_blank": originally_blank,
        "puzzles": tuple(puzzles),
        "solutions": tuple(solutions),
        "rating_buckets": tuple(bucket_names),
        "dataset_indices": tuple(selected_indices),
    }


def prepare_sample(criteria, arrow_path):
    per_split = criteria["sample"]["puzzles_per_bucket_per_split"]
    sample = load_balanced_sample_from_arrow(
        arrow_path,
        examples_per_bucket=3 * per_split,
        seed=criteria["seed"],
    )
    splits = make_three_way_split(
        sample["rating_buckets"],
        seed=criteria["seed"] + 1,
        puzzles_per_bucket_per_split=per_split,
    )
    manifest_rows = []
    split_by_index = {
        int(index): split_name
        for split_name, indices in splits.items()
        for index in indices
    }
    for index, (puzzle, bucket, dataset_index) in enumerate(
        zip(sample["puzzles"], sample["rating_buckets"], sample["dataset_indices"])
    ):
        manifest_rows.append(
            {
                "sample_index": index,
                "dataset_index": dataset_index,
                "rating_bucket": bucket,
                "split": split_by_index[index],
                "puzzle_sha256": puzzle_sha256(puzzle),
            }
        )
    return sample, splits, manifest_rows


def load_model(model_name, checkpoint_path, device):
    return _load_model(
        {
            "name": model_name,
            "path": str(checkpoint_path),
            "model_kwargs": {},
        },
        device,
    )


def collect_trajectory(model, sample, indices, iterations, device):
    indices = torch.as_tensor(indices, dtype=torch.long)
    inputs = sample["inputs"][indices].to(device)
    targets = sample["targets"][indices].to(device)
    originally_blank = sample["originally_blank"][indices].to(device)
    selected = set(iterations)
    states = []
    updates = []
    solved_fractions = []
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    with torch.inference_mode():
        hidden = model.initial_encoder(inputs)
        logits = model.output_head(hidden)
        predictions = torch.zeros(
            len(inputs), 81, 9, device=device, dtype=hidden.dtype
        )
        for iteration in range(iterations[-1] + 1):
            next_hidden = model.recurrent_step(
                hidden, predictions, rope_cos, rope_sin
            )
            if iteration in selected:
                states.append(hidden.detach().float().cpu())
                updates.append((next_hidden - hidden).detach().float().cpu())
                predicted_digits = logits.argmax(dim=-1)
                solved = (
                    (predicted_digits == targets) | ~originally_blank
                ).all(dim=1)
                solved_fractions.append(float(solved.float().mean().item()))
            hidden = next_hidden
            if iteration < iterations[-1]:
                logits = model.output_head(hidden)
                predictions = F.softmax(logits, dim=-1)
    return {
        "states": torch.stack(states, dim=1).numpy(),
        "updates": torch.stack(updates, dim=1).numpy(),
        "solved_fraction": solved_fractions,
    }


def fit_pca_dual(values, rank):
    """Exact PCA through the smaller sample-space covariance matrix."""

    centered = temporal_center(values)
    matrix = torch.from_numpy(centered.reshape(-1, centered.shape[-1])).double()
    mean = matrix.mean(dim=0)
    matrix = matrix - mean
    scale = matrix.abs().max().clamp_min(torch.finfo(matrix.dtype).tiny)
    scaled_matrix = matrix / scale
    gram = scaled_matrix @ scaled_matrix.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    eigenvalues = eigenvalues[order]
    tolerance = eigenvalues[0].clamp_min(1e-30) * 1e-12
    if int((eigenvalues > tolerance).sum().item()) < rank:
        raise RuntimeError(
            f"discovery trajectory has numerical rank below requested rank {rank}"
        )
    left_vectors = eigenvectors[:, order]
    basis = scaled_matrix.T @ (left_vectors / eigenvalues.sqrt()[None, :])
    basis, _ = torch.linalg.qr(basis, mode="reduced")
    return {
        "mean": mean.numpy(),
        "basis": basis.numpy(),
        "singular_values": (eigenvalues.sqrt() * scale).numpy(),
        "training_energy": float(matrix.square().sum().item()),
    }


def checkpoint_metadata(checkpoints):
    metadata = {}
    for name, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing checkpoint for {name}: {path}")
        metadata[name] = {
            "path_at_run_time": str(path),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "canonical": True,
        }
    return metadata


def choose_random_planes(coordinates32, coefficients, metric):
    planes = []
    scores = []
    for coefficient in coefficients:
        random_coordinates = coordinates32 @ coefficient
        plane, score = select_plane(random_coordinates, metric)
        planes.append(plane)
        scores.append(score)
    return np.asarray(planes, dtype=np.int64), np.asarray(scores, dtype=np.float64)


def development_split_robustness(
    representation,
    development_indices,
    rating_buckets,
    criteria,
    model_name,
):
    repeats = criteria["resampling"]["development_split_repeats"]
    resplits = make_development_resplits(
        rating_buckets,
        development_indices,
        seed=criteria["seed"] + 37,
        repeats=repeats,
    )
    local_by_original = {
        int(original): local
        for local, original in enumerate(development_indices)
    }
    rows = []
    for repeat_index, split in enumerate(resplits):
        discovery_local = np.array(
            [local_by_original[int(index)] for index in split["discovery"]]
        )
        validation_local = np.array(
            [local_by_original[int(index)] for index in split["validation"]]
        )
        fit = fit_pca_dual(representation[discovery_local], rank=3)
        validation_coordinates = project(
            representation[validation_local], fit["mean"], fit["basis"]
        )
        row = {"repeat": repeat_index, "metrics": {}}
        for metric in ("arc", "loop", "helix"):
            plane, _ = select_plane(validation_coordinates, metric)
            result = time_shuffle_test(
                validation_coordinates,
                metric,
                plane=plane,
                permutations=DEVELOPMENT_PERMUTATIONS,
                bootstraps=DEVELOPMENT_BOOTSTRAPS,
                seed=stable_seed(criteria["seed"], model_name, repeat_index, metric),
            )
            result["passes_development_repeat"] = (
                result["time_shuffle_p"] <= 0.05
                and result["ordered_minus_shuffle_effect"]
                >= criteria["criteria"]["single_cell_common"][
                    "minimum_ordered_minus_shuffle_effect"
                ]
                and result["puzzle_bootstrap_95_percent_effect_interval"][0] > 0
            )
            row["metrics"][metric] = result
        rows.append(row)
    fractions = {
        metric: float(
            np.mean(
                [row["metrics"][metric]["passes_development_repeat"] for row in rows]
            )
        )
        for metric in ("arc", "loop", "helix")
    }
    return {"repeats": rows, "pass_fraction": fractions}


def run_development(output_dir, checkpoints, arrow_path):
    criteria_path = SCRIPT_DIR / "acceptance_criteria.json"
    criteria = load_json(criteria_path)
    criteria_hash = file_sha256(criteria_path)
    sample, splits, manifest = prepare_sample(criteria, arrow_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split_manifest.json", {"rows": manifest})
    discovery_indices = splits["discovery"]
    validation_indices = splits["validation"]
    development_indices = np.concatenate([discovery_indices, validation_indices])
    discovery_local = np.arange(len(discovery_indices))
    validation_local = np.arange(len(discovery_indices), len(development_indices))
    iterations = tuple(criteria["trajectory"]["iterations"])
    random_count = criteria["projection"]["matched_rank_random_projections"]
    ambient_rank = criteria["projection"]["ambient_pca_rank"]
    checkpoint_info = checkpoint_metadata(checkpoints)
    device = torch.device("cpu")
    frozen_arrays = {}
    choices = {}
    metrics = {
        "phase": "development",
        "criteria_sha256": criteria_hash,
        "config": {
            "device": str(device),
            "development_time_shuffles": DEVELOPMENT_PERMUTATIONS,
            "development_bootstraps": DEVELOPMENT_BOOTSTRAPS,
            "checkpoint_metadata": checkpoint_info,
            "test_arrow_path": str(arrow_path),
        },
        "models": {},
    }
    started = time.time()
    for model_index, model_name in enumerate(CHECKPOINT_NAMES):
        log(f"development: collecting {model_name}")
        model = load_model(model_name, checkpoints[model_name], device)
        collected = collect_trajectory(
            model, sample, development_indices, iterations, device
        )
        representations = build_representations(
            collected["states"], collected["updates"]
        )
        model_result = {
            "solved_fraction_by_iteration": collected["solved_fraction"],
            "representations": {},
        }
        choices[model_name] = {}
        for representation_index, representation_name in enumerate(
            REPRESENTATION_NAMES
        ):
            log(f"  fitting {representation_name}")
            values = representations[representation_name]
            fit = fit_pca_dual(values[discovery_local], ambient_rank)
            prefix = f"{model_name}__{representation_name}"
            frozen_arrays[f"{prefix}__mean"] = fit["mean"]
            frozen_arrays[f"{prefix}__basis"] = fit["basis"]
            validation_coordinates32 = project(
                values[validation_local], fit["mean"], fit["basis"]
            )
            validation_coordinates3 = validation_coordinates32[:, :, :3]
            random_coefficients = random_orthonormal_coefficients(
                ambient_rank,
                criteria["projection"]["reported_rank"],
                random_count,
                seed=stable_seed(
                    criteria["seed"], model_name, representation_name, "random"
                ),
            )
            frozen_arrays[f"{prefix}__random_coefficients"] = random_coefficients
            representation_choices = {}
            representation_result = {
                "heldout_name": "validation",
                "rank3_dynamic_variance_fraction": dynamic_variance_fraction(
                    values[validation_local], validation_coordinates3
                ),
                "global_tests": {},
                "favorable_local_scores": {},
                "random_projection_validation": {},
            }
            local_coordinates = local_pca_coordinates(values[validation_local], rank=3)
            for metric in ("arc", "loop", "helix"):
                plane, _ = select_plane(validation_coordinates3, metric)
                representation_choices[f"{metric}_plane"] = list(plane)
                representation_result["global_tests"][metric] = time_shuffle_test(
                    validation_coordinates3,
                    metric,
                    plane=plane,
                    permutations=DEVELOPMENT_PERMUTATIONS,
                    bootstraps=DEVELOPMENT_BOOTSTRAPS,
                    seed=stable_seed(
                        criteria["seed"], model_name, representation_name, metric
                    ),
                )
                local_plane, local_score = select_plane(local_coordinates, metric)
                representation_result["favorable_local_scores"][metric] = {
                    "score": local_score,
                    "plane": list(local_plane),
                }
                random_planes, random_scores = choose_random_planes(
                    validation_coordinates32, random_coefficients, metric
                )
                frozen_arrays[f"{prefix}__random_{metric}_planes"] = random_planes
                representation_result["random_projection_validation"][metric] = {
                    "score_median": float(np.median(random_scores)),
                    "score_p95": float(np.quantile(random_scores, 0.95)),
                    "pca_percentile": percentile_against_controls(
                        representation_result["global_tests"][metric][
                            "observed_median"
                        ],
                        random_scores,
                    ),
                }
            choices[model_name][representation_name] = representation_choices
            model_result["representations"][representation_name] = representation_result
        model_result["split_robustness"] = development_split_robustness(
            representations[PRIMARY_REPRESENTATION],
            development_indices,
            sample["rating_buckets"],
            criteria,
            model_name,
        )
        metrics["models"][model_name] = model_result
        del model, collected, representations
    log("development: generating deceptive synthetic nulls")
    metrics["synthetic_nulls"] = synthetic_selection_experiment(
        seed=criteria["seed"] + 509,
        candidates=256,
        time_count=len(iterations),
        feature_count=64,
        local_rank=8,
    )
    metrics["elapsed_seconds"] = time.time() - started
    np.savez_compressed(output_dir / "frozen_projection_choices.npz", **frozen_arrays)
    write_json(
        output_dir / "frozen_choices.json",
        {
            "criteria_sha256": criteria_hash,
            "checkpoint_metadata": checkpoint_info,
            "choices": choices,
            "frozen_projection_file_sha256": file_sha256(
                output_dir / "frozen_projection_choices.npz"
            ),
        },
    )
    write_json(output_dir / "development_metrics.json", metrics)
    log(f"development complete in {metrics['elapsed_seconds']:.1f}s")
    return metrics


def evaluate_projection(
    coordinates3,
    coordinates32,
    values,
    local_coordinates,
    random_coefficients,
    random_planes,
    selected_planes,
    criteria,
    seed_parts,
):
    result = {
        "rank3_dynamic_variance_fraction": dynamic_variance_fraction(
            values, coordinates3
        ),
        "tests": {},
        "favorable_local_scores": {},
        "random_projection_controls": {},
    }
    for metric in ("arc", "loop", "helix"):
        plane = tuple(selected_planes[f"{metric}_plane"])
        test = time_shuffle_test(
            coordinates3,
            metric,
            plane=plane,
            permutations=criteria["resampling"]["time_shuffles"],
            bootstraps=criteria["resampling"]["puzzle_bootstraps"],
            seed=stable_seed(criteria["seed"], *seed_parts, metric),
        )
        local_plane, local_score = select_plane(local_coordinates, metric)
        test["global_to_favorable_local_score_ratio"] = float(
            test["observed_median"] / max(local_score, 1e-12)
        )
        result["tests"][metric] = test
        result["favorable_local_scores"][metric] = {
            "score": local_score,
            "plane": list(local_plane),
        }
        scores = []
        planes = random_planes[metric]
        for index, coefficient in enumerate(random_coefficients):
            random_coordinates = coordinates32 @ coefficient
            scores.append(
                aggregate_metric(
                    random_coordinates, metric, tuple(int(v) for v in planes[index])
                )
            )
        test["matched_rank_random_projection_percentile"] = (
            percentile_against_controls(test["observed_median"], scores)
        )
        result["random_projection_controls"][metric] = {
            "count": len(scores),
            "score_median": float(np.median(scores)),
            "score_95_percent_interval": [
                float(np.quantile(scores, 0.025)),
                float(np.quantile(scores, 0.975)),
            ],
            "scores": scores,
        }
    return result


def add_adjusted_q_values(metrics):
    for metric in ("continuity", "arc", "loop", "helix"):
        references = []
        p_values = []
        for model_name in CHECKPOINT_NAMES:
            for representation_name in REPRESENTATION_NAMES:
                representation = metrics["models"][model_name]["representations"][
                    representation_name
                ]
                if metric == "continuity":
                    test = representation["continuity_test"]
                else:
                    test = representation["own_projection"]["tests"][metric]
                references.append(test)
                p_values.append(test["time_shuffle_p"])
                if model_name != "stable_plain" and metric != "continuity":
                    transfer_test = representation["stable_basis_transfer"]["tests"][
                        metric
                    ]
                    references.append(transfer_test)
                    p_values.append(transfer_test["time_shuffle_p"])
        adjusted = benjamini_hochberg(p_values)
        for reference, q_value in zip(references, adjusted):
            reference["time_shuffle_q"] = float(q_value)


def common_test_pass(test, criteria):
    common = criteria["criteria"]["single_cell_common"]
    return (
        test["time_shuffle_q"] <= common["maximum_adjusted_time_shuffle_q"]
        and test["ordered_minus_shuffle_effect"]
        >= common["minimum_ordered_minus_shuffle_effect"]
        and test["puzzle_bootstrap_95_percent_effect_interval"][0]
        > common["bootstrap_95_percent_lower_bound_must_exceed"]
    )


def projection_test_pass(metric, projection, criteria, *, transfer=False):
    test = projection["tests"][metric]
    common = criteria["criteria"]["single_cell_common"]
    if not common_test_pass(test, criteria):
        return False
    if (
        test["matched_rank_random_projection_percentile"]
        < common["minimum_random_projection_percentile"]
    ):
        return False
    claim = criteria["criteria"][f"shared_{metric}"]
    components = test["component_medians"]
    if metric == "arc":
        return (
            projection["rank3_dynamic_variance_fraction"]
            >= claim["minimum_heldout_rank3_dynamic_variance_fraction"]
            and (
                transfer
                or test["global_to_favorable_local_score_ratio"]
                >= claim["minimum_global_to_favorable_local_score_ratio"]
            )
        )
    if components["net_turns"] < claim["minimum_median_net_turns"]:
        return False
    if (
        components["angular_linearity_r2"]
        < claim["minimum_median_angular_linearity_r2"]
        or components["radius_cv"] > claim["maximum_median_radius_cv"]
    ):
        return False
    if metric == "helix":
        return (
            test["observed_median"] >= claim["minimum_median_helix_score"]
            and components["absolute_axial_spearman"]
            >= claim["minimum_median_absolute_axial_spearman"]
        )
    return True


def evaluate_verdicts(metrics, development, criteria):
    for model_name in CHECKPOINT_NAMES:
        for representation_name in REPRESENTATION_NAMES:
            representation = metrics["models"][model_name]["representations"][
                representation_name
            ]
            representation["continuity_test"]["passes_pre_registered_criteria"] = (
                common_test_pass(representation["continuity_test"], criteria)
            )
            for metric in ("arc", "loop", "helix"):
                representation["own_projection"]["tests"][metric][
                    "passes_pre_registered_criteria"
                ] = projection_test_pass(
                    metric, representation["own_projection"], criteria
                )
                if model_name != "stable_plain":
                    representation["stable_basis_transfer"]["tests"][metric][
                        "passes_pre_registered_criteria"
                    ] = projection_test_pass(
                        metric,
                        representation["stable_basis_transfer"],
                        criteria,
                        transfer=True,
                    )
    verdicts = {}
    for claim_name, metric in (
        ("temporal_continuity", "continuity"),
        ("shared_arc", "arc"),
        ("shared_loop", "loop"),
        ("shared_helix", "helix"),
    ):
        claim = criteria["criteria"][claim_name]
        checkpoint_counts = {}
        for model_name in CHECKPOINT_NAMES:
            count = 0
            for representation_name in REPRESENTATION_NAMES:
                representation = metrics["models"][model_name]["representations"][
                    representation_name
                ]
                if metric == "continuity":
                    passed = representation["continuity_test"][
                        "passes_pre_registered_criteria"
                    ]
                else:
                    passed = representation["own_projection"]["tests"][metric][
                        "passes_pre_registered_criteria"
                    ]
                count += int(passed)
            checkpoint_counts[model_name] = count
        passing_checkpoints = [
            name
            for name, count in checkpoint_counts.items()
            if count >= claim["minimum_passing_representations_per_checkpoint"]
        ]
        result = {
            "passes": len(passing_checkpoints) >= claim["minimum_passing_checkpoints"],
            "passing_representations_by_checkpoint": checkpoint_counts,
            "passing_checkpoints": passing_checkpoints,
        }
        if metric != "continuity":
            split_fractions = {
                model_name: development["models"][model_name]["split_robustness"][
                    "pass_fraction"
                ][metric]
                for model_name in CHECKPOINT_NAMES
            }
            robust_split_models = [
                name
                for name, fraction in split_fractions.items()
                if fraction >= claim["minimum_development_split_pass_fraction"]
            ]
            transfer_targets = []
            for target_name in CHECKPOINT_NAMES:
                if target_name == "stable_plain":
                    continue
                test = metrics["models"][target_name]["representations"][
                    PRIMARY_REPRESENTATION
                ]["stable_basis_transfer"]["tests"][metric]
                if test["passes_pre_registered_criteria"]:
                    transfer_targets.append(target_name)
            result.update(
                {
                    "development_split_pass_fraction": split_fractions,
                    "development_split_robust_models": robust_split_models,
                    "stable_basis_transfer_targets": transfer_targets,
                }
            )
            result["passes"] = (
                result["passes"]
                and len(robust_split_models) >= claim["minimum_passing_checkpoints"]
                and len(transfer_targets)
                >= claim["minimum_distinct_non_source_transfer_checkpoints"]
            )
        verdicts[claim_name] = result
    return verdicts


def run_final(output_dir, checkpoints, arrow_path, *, allow_rerun=False):
    final_path = output_dir / "robustness_metrics.json"
    if final_path.exists() and not allow_rerun:
        raise RuntimeError(
            "robustness_metrics.json already exists; refusing to inspect the holdout again"
        )
    criteria_path = SCRIPT_DIR / "acceptance_criteria.json"
    criteria = load_json(criteria_path)
    criteria_hash = file_sha256(criteria_path)
    frozen_choices = load_json(output_dir / "frozen_choices.json")
    development = load_json(output_dir / "development_metrics.json")
    if frozen_choices["criteria_sha256"] != criteria_hash:
        raise RuntimeError("acceptance criteria changed after development")
    projection_path = output_dir / "frozen_projection_choices.npz"
    if file_sha256(projection_path) != frozen_choices["frozen_projection_file_sha256"]:
        raise RuntimeError("frozen projection file changed after validation")
    checkpoint_info = checkpoint_metadata(checkpoints)
    for name in CHECKPOINT_NAMES:
        if checkpoint_info[name]["sha256"] != frozen_choices["checkpoint_metadata"][name][
            "sha256"
        ]:
            raise RuntimeError(f"checkpoint changed after development: {name}")
    sample, splits, manifest = prepare_sample(criteria, arrow_path)
    saved_manifest = load_json(output_dir / "split_manifest.json")["rows"]
    if manifest != saved_manifest:
        raise RuntimeError("puzzle split changed after development")
    holdout_indices = splits["holdout"]
    iterations = tuple(criteria["trajectory"]["iterations"])
    frozen = np.load(projection_path, allow_pickle=False)
    device = torch.device("cpu")
    metrics = {
        "phase": "final_holdout",
        "holdout_evaluations": 1,
        "criteria_sha256": criteria_hash,
        "config": {
            "device": str(device),
            "checkpoint_metadata": checkpoint_info,
            "sample_size": len(holdout_indices),
            "puzzles_per_rating_bucket": criteria["sample"][
                "puzzles_per_bucket_per_split"
            ],
            "iterations": list(iterations),
            "time_shuffles": criteria["resampling"]["time_shuffles"],
            "puzzle_bootstraps": criteria["resampling"]["puzzle_bootstraps"],
            "matched_rank_random_projections": criteria["projection"][
                "matched_rank_random_projections"
            ],
        },
        "models": {},
        "gallery_coordinates": {"real": {}},
        "synthetic_nulls": development["synthetic_nulls"],
    }
    started = time.time()
    stable_choices = frozen_choices["choices"]["stable_plain"]
    for model_name in CHECKPOINT_NAMES:
        log(f"final holdout: collecting {model_name}")
        model = load_model(model_name, checkpoints[model_name], device)
        collected = collect_trajectory(model, sample, holdout_indices, iterations, device)
        representations = build_representations(
            collected["states"], collected["updates"]
        )
        model_result = {
            "solved_fraction_by_iteration": collected["solved_fraction"],
            "representations": {},
        }
        metrics["gallery_coordinates"]["real"][model_name] = {}
        for representation_name in REPRESENTATION_NAMES:
            values = representations[representation_name]
            representation_result = {
                "continuity_test": time_shuffle_test(
                    values,
                    "continuity",
                    permutations=criteria["resampling"]["time_shuffles"],
                    bootstraps=criteria["resampling"]["puzzle_bootstraps"],
                    seed=stable_seed(
                        criteria["seed"], "holdout", model_name, representation_name,
                        "continuity"
                    ),
                )
            }
            own_prefix = f"{model_name}__{representation_name}"
            own_coordinates32 = project(
                values,
                frozen[f"{own_prefix}__mean"],
                frozen[f"{own_prefix}__basis"],
            )
            own_coordinates3 = own_coordinates32[:, :, :3]
            local_coordinates = local_pca_coordinates(values, rank=3)
            own_random_planes = {
                metric: frozen[f"{own_prefix}__random_{metric}_planes"]
                for metric in ("arc", "loop", "helix")
            }
            own_projection = evaluate_projection(
                own_coordinates3,
                own_coordinates32,
                values,
                local_coordinates,
                frozen[f"{own_prefix}__random_coefficients"],
                own_random_planes,
                frozen_choices["choices"][model_name][representation_name],
                criteria,
                ("holdout", model_name, representation_name, "own"),
            )
            representation_result["own_projection"] = own_projection
            if model_name != "stable_plain":
                stable_prefix = f"stable_plain__{representation_name}"
                stable_coordinates32 = project(
                    values,
                    frozen[f"{stable_prefix}__mean"],
                    frozen[f"{stable_prefix}__basis"],
                )
                stable_coordinates3 = stable_coordinates32[:, :, :3]
                stable_random_planes = {
                    metric: frozen[f"{stable_prefix}__random_{metric}_planes"]
                    for metric in ("arc", "loop", "helix")
                }
                representation_result["stable_basis_transfer"] = evaluate_projection(
                    stable_coordinates3,
                    stable_coordinates32,
                    values,
                    local_coordinates,
                    frozen[f"{stable_prefix}__random_coefficients"],
                    stable_random_planes,
                    stable_choices[representation_name],
                    criteria,
                    ("holdout", model_name, representation_name, "stable_transfer"),
                )
            if representation_name == PRIMARY_REPRESENTATION:
                metrics["gallery_coordinates"]["real"][model_name] = {
                    "global": own_coordinates3[0].tolist(),
                    "local": local_coordinates[0].tolist(),
                }
            model_result["representations"][representation_name] = representation_result
        metrics["models"][model_name] = model_result
        del model, collected, representations
    add_adjusted_q_values(metrics)
    metrics["verdicts"] = evaluate_verdicts(metrics, development, criteria)
    metrics["elapsed_seconds"] = time.time() - started
    metrics["limitations"] = {
        "checkpoint_evidence_missing": [],
        "all_four_canonical_checkpoints_loaded": True,
        "single_seed_per_checkpoint": True,
        "per_puzzle_pca_is_circular_control_only": True,
    }
    write_json(final_path, metrics)
    render_saved_metrics_in_fresh_process(output_dir)
    log(f"final holdout complete in {metrics['elapsed_seconds']:.1f}s")
    return metrics


def _plot_path(axis, coordinates, title, plane=(0, 1), color="#1f6f8b"):
    coordinates = np.asarray(coordinates)
    axis.plot(coordinates[:, plane[0]], coordinates[:, plane[1]], color=color, lw=1.5)
    colors = np.linspace(0.0, 1.0, len(coordinates))
    axis.scatter(
        coordinates[:, plane[0]], coordinates[:, plane[1]], c=colors,
        cmap="viridis", s=10, zorder=3
    )
    axis.scatter(
        coordinates[0, plane[0]], coordinates[0, plane[1]],
        marker="s", color="#d1495b", s=28, zorder=4
    )
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_aspect("equal", adjustable="datalim")


def render_gallery(output_dir, metrics):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 4, figsize=(13, 9), constrained_layout=True)
    for column, metric in enumerate(("arc", "loop", "helix")):
        gallery = metrics["synthetic_nulls"][metric]["gallery"]
        plane = gallery["plane"][:2]
        _plot_path(
            axes[0, column], gallery["coordinates"],
            f"Synthetic {metric}\nselected score {gallery['score']:.2f}",
            plane=plane,
            color="#7a5195",
        )
    axes[0, 3].axis("off")
    lines = ["Selection inflation (median)"]
    for metric in ("arc", "loop", "helix"):
        value = metrics["synthetic_nulls"][metric]["median_selection_inflation"]
        lines.append(f"{metric}: +{value:.2f}")
    lines.extend(["", "No Sudoku signal.", "Axes chosen after seeing", "the full null path."])
    axes[0, 3].text(0.05, 0.92, "\n".join(lines), va="top", fontsize=10)
    for column, model_name in enumerate(CHECKPOINT_NAMES):
        paths = metrics["gallery_coordinates"]["real"][model_name]
        _plot_path(
            axes[1, column], paths["global"],
            f"{model_name}\nglobal discovery PCA",
        )
        _plot_path(
            axes[2, column], paths["local"],
            f"{model_name}\nper-puzzle look-ahead PCA",
            color="#ef8354",
        )
    figure.suptitle(
        "Deceptive nulls and real normalized-state paths\n"
        "Color runs from early (purple) to late (yellow); red square is iteration 0",
        fontsize=13,
    )
    figure.savefig(output_dir / "adversarial_gallery.png", dpi=180)
    plt.close(figure)


def render_robustness_summary(output_dir, metrics):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for axis, metric in zip(axes.flat, ("continuity", "arc", "loop", "helix")):
        matrix = np.zeros((len(REPRESENTATION_NAMES), len(CHECKPOINT_NAMES)))
        passed = np.zeros_like(matrix, dtype=bool)
        for column, model_name in enumerate(CHECKPOINT_NAMES):
            for row, representation_name in enumerate(REPRESENTATION_NAMES):
                representation = metrics["models"][model_name]["representations"][
                    representation_name
                ]
                if metric == "continuity":
                    test = representation["continuity_test"]
                else:
                    test = representation["own_projection"]["tests"][metric]
                matrix[row, column] = test["ordered_minus_shuffle_effect"]
                passed[row, column] = test["passes_pre_registered_criteria"]
        scale = max(float(np.abs(matrix).max()), 0.1)
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-scale, vmax=scale)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                marker = "✓" if passed[row, column] else "×"
                axis.text(
                    column, row, f"{matrix[row, column]:.2f}\n{marker}",
                    ha="center", va="center", fontsize=8,
                )
        axis.set_xticks(range(len(CHECKPOINT_NAMES)), CHECKPOINT_NAMES, rotation=25, ha="right")
        axis.set_yticks(range(len(REPRESENTATION_NAMES)), REPRESENTATION_NAMES)
        axis.set_title(f"{metric}: ordered − shuffled")
        figure.colorbar(image, ax=axis, shrink=0.7)
    figure.savefig(output_dir / "robustness_summary.png", dpi=180)
    plt.close(figure)


def render_checkpoint_transfer(output_dir, metrics):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    targets = [name for name in CHECKPOINT_NAMES if name != "stable_plain"]
    for axis, metric in zip(axes, ("arc", "loop", "helix")):
        own = []
        transferred = []
        for target in targets:
            representation = metrics["models"][target]["representations"][
                PRIMARY_REPRESENTATION
            ]
            own.append(representation["own_projection"]["tests"][metric]["observed_median"])
            transferred.append(
                representation["stable_basis_transfer"]["tests"][metric][
                    "observed_median"
                ]
            )
        x = np.arange(len(targets))
        axis.bar(x - 0.18, own, width=0.36, label="own basis", color="#1f6f8b")
        axis.bar(
            x + 0.18, transferred, width=0.36,
            label="stable basis", color="#ef8354"
        )
        axis.set_xticks(x, targets, rotation=25, ha="right")
        axis.set_title(metric)
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Checkpoint transfer on the final normalized-state holdout")
    figure.savefig(output_dir / "checkpoint_transfer.png", dpi=180)
    plt.close(figure)


def render_html(output_dir, metrics):
    verdict_rows = []
    for claim, result in metrics["verdicts"].items():
        verdict_rows.append(
            f"<tr><td>{html.escape(claim)}</td><td>{'PASS' if result['passes'] else 'FAIL'}</td>"
            f"<td>{html.escape(', '.join(result['passing_checkpoints']) or 'none')}</td></tr>"
        )
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Arm 12 adversarial audit</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}}
img{{max-width:100%;border:1px solid #ddd;margin:1rem 0}}table{{border-collapse:collapse}}
th,td{{border:1px solid #bbb;padding:.45rem;text-align:left}}code{{background:#eee;padding:.1rem .25rem}}</style>
</head><body><h1>Arm 12 — adversarial audit</h1>
<p>The plots illustrate the preregistered held-out measurements. Visual appeal is not an acceptance criterion.</p>
<table><thead><tr><th>Claim</th><th>Verdict</th><th>Checkpoints meeting representation rule</th></tr></thead>
<tbody>{''.join(verdict_rows)}</tbody></table>
<h2>Null and real gallery</h2><img src="adversarial_gallery.png" alt="Adversarial gallery">
<h2>Robustness matrix</h2><img src="robustness_summary.png" alt="Robustness summary">
<h2>Checkpoint transfer</h2><img src="checkpoint_transfer.png" alt="Checkpoint transfer">
<p>Exact values: <a href="robustness_metrics.json">robustness_metrics.json</a>. Frozen criteria:
<a href="acceptance_criteria.json">acceptance_criteria.json</a>.</p></body></html>"""
    temporary = output_dir / "index.html.tmp"
    temporary.write_text(document)
    temporary.replace(output_dir / "index.html")


def render_artifacts(output_dir, metrics):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sotaku-arm12-matplotlib")
    render_gallery(output_dir, metrics)
    render_robustness_summary(output_dir, metrics)
    render_checkpoint_transfer(output_dir, metrics)
    render_html(output_dir, metrics)


def render_saved_metrics_in_fresh_process(output_dir):
    """Keep Matplotlib isolated from the long-lived PyTorch process on macOS."""

    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": "/tmp/sotaku-arm12-matplotlib",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "render",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        env=environment,
    )


def render_saved_metrics(output_dir):
    with open(output_dir / "robustness_metrics.json") as input_file:
        metrics = json.load(input_file)
    render_artifacts(output_dir, metrics)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("development", "final", "render", "all"),
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR)
    parser.add_argument("--test-arrow", type=Path, default=DEFAULT_TEST_ARROW)
    parser.add_argument("--stable-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["stable_plain"])
    parser.add_argument("--collapsed-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["collapsed_plain"])
    parser.add_argument("--late-state-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["late_state_ce"])
    parser.add_argument("--combined-checkpoint", type=Path, default=DEFAULT_CHECKPOINTS["combined_margin"])
    parser.add_argument("--allow-final-rerun", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoints = {
        "stable_plain": args.stable_checkpoint,
        "collapsed_plain": args.collapsed_checkpoint,
        "late_state_ce": args.late_state_checkpoint,
        "combined_margin": args.combined_checkpoint,
    }
    if args.phase in ("development", "all"):
        run_development(args.output_dir, checkpoints, args.test_arrow)
    if args.phase in ("final", "all"):
        run_final(
            args.output_dir,
            checkpoints,
            args.test_arrow,
            allow_rerun=args.allow_final_rerun,
        )
    if args.phase == "render":
        render_saved_metrics(args.output_dir)


if __name__ == "__main__":
    main()
