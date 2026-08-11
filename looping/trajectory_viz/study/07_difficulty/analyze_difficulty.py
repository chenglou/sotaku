"""Test whether recurrent geometry tracks Sudoku difficulty and solve latency."""

import json
import math
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import RATING_BUCKETS, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


DEFAULT_HORIZONS = (4, 8, 16, 32, 64, 128)
DEFAULT_COMPONENTS = (2, 4, 8, 12)
DEFAULT_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
SPLIT_NAMES = ("discovery", "validation", "final")


def rank_values(values):
    """Return average ranks, including ties."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman_correlation(left, right):
    left_rank = rank_values(left)
    right_rank = rank_values(right)
    if np.std(left_rank) < 1e-12 or np.std(right_rank) < 1e-12:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def make_stratified_splits(bucket_names, per_split):
    splits = {name: [] for name in SPLIT_NAMES}
    for bucket in [entry[2] for entry in RATING_BUCKETS]:
        indices = [index for index, value in enumerate(bucket_names) if value == bucket]
        required = per_split * len(SPLIT_NAMES)
        if len(indices) != required:
            raise ValueError(f"bucket {bucket} has {len(indices)} rows, expected {required}")
        for split_index, split_name in enumerate(SPLIT_NAMES):
            start = split_index * per_split
            splits[split_name].extend(indices[start : start + per_split])
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in splits.items()}


def fit_standardized_ridge(features, targets, alpha):
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    mean = features.mean(axis=0, keepdims=True)
    scale = features.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    standardized = (features - mean) / scale
    design = np.concatenate([np.ones((len(features), 1)), standardized], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {"mean": mean, "scale": scale, "weights": weights}


def predict_ridge(model, features):
    standardized = (np.asarray(features, dtype=np.float64) - model["mean"]) / model["scale"]
    design = np.concatenate([np.ones((len(standardized), 1)), standardized], axis=1)
    return design @ model["weights"]


def fit_pca(discovery, component_count):
    discovery = np.asarray(discovery, dtype=np.float64)
    mean = discovery.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(discovery - mean, full_matrices=False)
    basis = right[: min(component_count, len(right))].T
    return mean, basis


def project(values, mean, basis):
    return (np.asarray(values, dtype=np.float64) - mean) @ basis


def regression_metrics(targets, predictions):
    targets = np.asarray(targets, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    residual = targets - predictions
    denominator = np.square(targets - targets.mean()).sum()
    r_squared = 0.0 if denominator < 1e-12 else 1.0 - np.square(residual).sum() / denominator
    return {
        "spearman": spearman_correlation(targets, predictions),
        "mae": float(np.abs(residual).mean()),
        "r_squared": float(r_squared),
    }


def categorical_metrics(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(scores).argmax(axis=1)
    recalls = []
    for label in range(5):
        mask = labels == label
        recalls.append(float((predicted[mask] == label).mean()) if mask.any() else 0.0)
    return {
        "accuracy": float((predicted == labels).mean()),
        "macro_recall": float(np.mean(recalls)),
        "predictions": predicted.tolist(),
    }


def sample_balanced_puzzles(per_bucket, seed):
    dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    candidates = {name: [] for _, _, name in RATING_BUCKETS}
    for index, example in enumerate(dataset):
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= example["rating"] <= maximum:
                candidates[name].append(index)
                break
    generator = random.Random(seed)
    selected = []
    buckets = []
    for _, _, name in RATING_BUCKETS:
        chosen = generator.sample(candidates[name], per_bucket)
        selected.extend(chosen)
        buckets.extend([name] * per_bucket)
    examples = [dataset[index] for index in selected]
    inputs = model_module.encode_puzzles([example["question"] for example in examples])
    targets = model_module.encode_solutions([example["answer"] for example in examples]).long()
    def clue_count(question):
        if isinstance(question, str):
            return sum(character not in "0." for character in question if character.isdigit() or character == ".")
        return int((np.asarray(question) != 0).sum())

    return {
        "inputs": inputs,
        "targets": targets,
        "empty_mask": inputs[:, :, 0].bool(),
        "ratings": np.asarray([example["rating"] for example in examples], dtype=np.float64),
        "bucket_names": buckets,
        "puzzles": [example["question"] for example in examples],
        "clue_counts": np.asarray([clue_count(example["question"]) for example in examples], dtype=np.float64),
    }


def collect_model_geometry(model, inputs, targets, empty_mask, horizons, maximum_iteration=1024):
    requested = set(horizons)
    if maximum_iteration < max(horizons):
        raise ValueError("maximum_iteration must include every feature horizon")
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    hidden = model.initial_encoder(inputs)
    initial = hidden.detach().float()
    predictions = torch.zeros(inputs.size(0), 81, 9, device=inputs.device)
    solved_history = []
    representations = {}
    scalars = {}
    previous_update = None
    with torch.no_grad():
        for iteration in range(maximum_iteration + 1):
            logits = model.output_head(hidden)
            predicted = logits.argmax(dim=-1)
            solved = ((predicted == targets) | ~empty_mask).all(dim=1)
            solved_history.append(solved.cpu())
            if iteration == maximum_iteration:
                break
            next_hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
            update = next_hidden - hidden
            if iteration in requested:
                displacement = hidden - initial
                flat_displacement = displacement.flatten(1).float()
                flat_update = update.flatten(1).float()
                representations[str(iteration)] = {
                    "state_displacement_direction": F.normalize(flat_displacement, dim=1, eps=1e-12).cpu().numpy(),
                    "update_direction": F.normalize(flat_update, dim=1, eps=1e-12).cpu().numpy(),
                }
                cell_update_norm = update.float().norm(dim=-1)
                displacement_norm = flat_displacement.norm(dim=1)
                update_norm = flat_update.norm(dim=1)
                update_displacement_cosine = F.cosine_similarity(flat_update, flat_displacement, dim=1, eps=1e-12)
                if previous_update is None:
                    turn_cosine = torch.zeros_like(update_norm)
                else:
                    turn_cosine = F.cosine_similarity(flat_update, previous_update, dim=1, eps=1e-12)
                scalar_features = torch.stack(
                    [
                        torch.log1p(displacement_norm),
                        torch.log1p(update_norm),
                        update_displacement_cosine,
                        turn_cosine,
                        cell_update_norm.std(dim=1) / cell_update_norm.mean(dim=1).clamp_min(1e-12),
                    ],
                    dim=1,
                )
                scalars[str(iteration)] = scalar_features.cpu().numpy()
            previous_update = update.flatten(1).float()
            hidden = next_hidden
            predictions = F.softmax(model.output_head(hidden), dim=-1)
    solved_history = torch.stack(solved_history, dim=1).numpy()
    first_solve = np.full(inputs.size(0), maximum_iteration + 1, dtype=np.int64)
    stable_solve = np.full(inputs.size(0), maximum_iteration + 1, dtype=np.int64)
    for index, row in enumerate(solved_history):
        solved_indices = np.flatnonzero(row)
        if len(solved_indices):
            first_solve[index] = int(solved_indices[0])
        if row[-1]:
            unsolved_indices = np.flatnonzero(~row)
            stable_solve[index] = int(unsolved_indices[-1] + 1) if len(unsolved_indices) else 0
    return {
        "representations": representations,
        "scalars": scalars,
        "first_solve": first_solve,
        "stable_solve": stable_solve,
        "solved_at_maximum": solved_history[:, -1],
    }


def prepare_candidates(geometry, splits, components):
    candidates = []
    discovery = splits["discovery"]
    for horizon, representations in geometry["representations"].items():
        for representation_name, values in representations.items():
            mean, full_basis = fit_pca(values[discovery], max(components))
            for component_count in components:
                basis = full_basis[:, :component_count]
                candidates.append(
                    {
                        "name": f"{representation_name}@{horizon}/pca{component_count}",
                        "horizon": int(horizon),
                        "representation": representation_name,
                        "component_count": int(basis.shape[1]),
                        "features": project(values, mean, basis),
                        "raw_values": values,
                    }
                )
        candidates.append(
            {
                "name": f"scalar_geometry@{horizon}",
                "horizon": int(horizon),
                "representation": "scalar_geometry",
                "component_count": int(geometry["scalars"][horizon].shape[1]),
                "features": geometry["scalars"][horizon],
                "raw_values": None,
            }
        )
    return candidates


def select_regression(candidates, targets, splits, alphas):
    best = None
    for candidate in candidates:
        for alpha in alphas:
            model = fit_standardized_ridge(candidate["features"][splits["discovery"]], targets[splits["discovery"]], alpha)
            validation_predictions = predict_ridge(model, candidate["features"][splits["validation"]])
            score = spearman_correlation(targets[splits["validation"]], validation_predictions)
            record = (score, -candidate["component_count"], -math.log10(alpha), candidate, alpha)
            if best is None or record[:3] > best[:3]:
                best = record
    _, _, _, candidate, alpha = best
    model = fit_standardized_ridge(candidate["features"][splits["discovery"]], targets[splits["discovery"]], alpha)
    predictions = {name: predict_ridge(model, candidate["features"][indices]) for name, indices in splits.items()}
    return candidate, alpha, model, predictions


def evaluate_fixed_features(features, targets, splits, alphas):
    best = None
    for alpha in alphas:
        model = fit_standardized_ridge(features[splits["discovery"]], targets[splits["discovery"]], alpha)
        predictions = predict_ridge(model, features[splits["validation"]])
        score = spearman_correlation(targets[splits["validation"]], predictions)
        if best is None or score > best[0]:
            best = (score, alpha, model)
    _, alpha, model = best
    final_predictions = predict_ridge(model, features[splits["final"]])
    return {"alpha": alpha, **regression_metrics(targets[splits["final"]], final_predictions)}


def controls_for_selection(candidate, target, splits, alpha, clue_counts, seed, observed_spearman):
    final_indices = splits["final"]
    discovery_indices = splits["discovery"]
    selected_features = candidate["features"]
    clue_features = clue_counts[:, None]
    clue_result = evaluate_fixed_features(clue_features, target, splits, DEFAULT_ALPHAS)
    combined = np.concatenate([clue_features, selected_features], axis=1)
    combined_result = evaluate_fixed_features(combined, target, splits, DEFAULT_ALPHAS)

    generator = np.random.default_rng(seed)
    shuffled_scores = []
    for _ in range(100):
        shuffled = target[discovery_indices].copy()
        generator.shuffle(shuffled)
        model = fit_standardized_ridge(selected_features[discovery_indices], shuffled, alpha)
        predictions = predict_ridge(model, selected_features[final_indices])
        shuffled_scores.append(spearman_correlation(target[final_indices], predictions))

    random_projection_scores = []
    if candidate["raw_values"] is not None:
        raw = candidate["raw_values"].astype(np.float64)
        centered = raw - raw[discovery_indices].mean(axis=0, keepdims=True)
        for _ in range(20):
            gaussian = generator.normal(size=(raw.shape[1], candidate["component_count"]))
            basis, _ = np.linalg.qr(gaussian)
            random_features = centered @ basis
            result = evaluate_fixed_features(random_features, target, splits, (alpha,))
            random_projection_scores.append(result["spearman"])
    return {
        "clue_count_baseline": clue_result,
        "clue_plus_geometry": combined_result,
        "shuffled_labels": {
            "runs": 100,
            "median_spearman": float(np.median(shuffled_scores)),
            "p95_spearman": float(np.quantile(shuffled_scores, 0.95)),
            "empirical_p_one_sided": float(
                (1 + np.sum(np.asarray(shuffled_scores) >= observed_spearman))
                / (1 + len(shuffled_scores))
            ),
        },
        "matched_rank_random_projection": None
        if not random_projection_scores
        else {
            "runs": 20,
            "median_spearman": float(np.median(random_projection_scores)),
            "p95_spearman": float(np.quantile(random_projection_scores, 0.95)),
            "empirical_p_one_sided": float(
                (1 + np.sum(np.asarray(random_projection_scores) >= observed_spearman))
                / (1 + len(random_projection_scores))
            ),
        },
    }


def categorical_probe(features, bucket_labels, splits, alphas, seed):
    one_hot = np.eye(5)[bucket_labels]
    best = None
    for alpha in alphas:
        model = fit_standardized_ridge(features[splits["discovery"]], one_hot[splits["discovery"]], alpha)
        validation = categorical_metrics(bucket_labels[splits["validation"]], predict_ridge(model, features[splits["validation"]]))
        if best is None or validation["accuracy"] > best[0]:
            best = (validation["accuracy"], alpha, model)
    _, alpha, model = best
    final = categorical_metrics(bucket_labels[splits["final"]], predict_ridge(model, features[splits["final"]]))
    generator = np.random.default_rng(seed)
    shuffled_accuracies = []
    for _ in range(100):
        shuffled = bucket_labels[splits["discovery"]].copy()
        generator.shuffle(shuffled)
        shuffled_one_hot = np.eye(5)[shuffled]
        shuffled_model = fit_standardized_ridge(
            features[splits["discovery"]],
            shuffled_one_hot,
            alpha,
        )
        shuffled_result = categorical_metrics(
            bucket_labels[splits["final"]],
            predict_ridge(shuffled_model, features[splits["final"]]),
        )
        shuffled_accuracies.append(shuffled_result["accuracy"])
    final["alpha"] = alpha
    final["chance_accuracy"] = 0.2
    final["shuffled_labels"] = {
        "runs": 100,
        "median_accuracy": float(np.median(shuffled_accuracies)),
        "p95_accuracy": float(np.quantile(shuffled_accuracies, 0.95)),
        "empirical_p_one_sided": float(
            (1 + np.sum(np.asarray(shuffled_accuracies) >= final["accuracy"]))
            / (1 + len(shuffled_accuracies))
        ),
    }
    return final


def analyze_target(name, target, candidates, splits, clue_counts, bucket_labels, seed):
    candidate, alpha, _, predictions = select_regression(candidates, target, splits, DEFAULT_ALPHAS)
    split_metrics = {split: regression_metrics(target[splits[split]], predictions[split]) for split in SPLIT_NAMES}
    controls = controls_for_selection(
        candidate,
        target,
        splits,
        alpha,
        clue_counts,
        seed,
        split_metrics["final"]["spearman"],
    )
    result = {
        "target": name,
        "selected_without_final": {
            "candidate": candidate["name"],
            "horizon": candidate["horizon"],
            "representation": candidate["representation"],
            "component_count": candidate["component_count"],
            "alpha": alpha,
        },
        "metrics": split_metrics,
        "controls": controls,
        "final_targets": target[splits["final"]].tolist(),
        "final_predictions": predictions["final"].tolist(),
    }
    if name == "log_rating":
        result["unconstrained_bucket_probe"] = categorical_probe(
            candidate["features"],
            bucket_labels,
            splits,
            DEFAULT_ALPHAS,
            seed + 5000,
        )
    return result


def plot_results(results, output_dir):
    model_names = list(results["models"])
    figure, axes = plt.subplots(len(model_names), 2, figsize=(10, 3.2 * len(model_names)), constrained_layout=True)
    for row, model_name in enumerate(model_names):
        for column, target_name in enumerate(("log_rating", "log_stable_solve_latency")):
            result = results["models"][model_name]["targets"][target_name]
            actual = np.asarray(result["final_targets"])
            predicted = np.asarray(result["final_predictions"])
            axes[row, column].scatter(actual, predicted, c="#0072B2", alpha=0.8)
            lower = min(actual.min(), predicted.min())
            upper = max(actual.max(), predicted.max())
            axes[row, column].plot([lower, upper], [lower, upper], "--", color="#777777")
            score = result["metrics"]["final"]["spearman"]
            axes[row, column].set_title(f"{model_name}: {target_name}\nfinal Spearman={score:.2f}")
            axes[row, column].set_xlabel("actual")
            axes[row, column].set_ylabel("predicted")
    figure.savefig(os.path.join(output_dir, "heldout_predictions.png"), dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    x = np.arange(len(model_names))
    for target_name, color in (("log_rating", "#0072B2"), ("log_stable_solve_latency", "#D55E00")):
        values = [results["models"][name]["targets"][target_name]["metrics"]["final"]["spearman"] for name in model_names]
        axes[0].plot(x, values, marker="o", label=target_name, color=color)
    axes[0].axhline(0, color="#777777", linewidth=1)
    axes[0].set_title("Final held-out rank correlation")
    axes[0].set_xticks(x, model_names, rotation=25, ha="right")
    axes[0].legend(fontsize=8)
    rating_accuracy = [results["models"][name]["targets"]["log_rating"]["unconstrained_bucket_probe"]["accuracy"] for name in model_names]
    axes[1].bar(x, rating_accuracy, color="#009E73")
    axes[1].axhline(0.2, linestyle="--", color="#777777")
    axes[1].set_title("Final rating-bucket accuracy")
    axes[1].set_xticks(x, model_names, rotation=25, ha="right")
    clue = [results["models"][name]["targets"]["log_rating"]["controls"]["clue_count_baseline"]["spearman"] for name in model_names]
    geometry = [results["models"][name]["targets"]["log_rating"]["metrics"]["final"]["spearman"] for name in model_names]
    axes[2].bar(x - 0.18, clue, width=0.36, label="clues", color="#999999")
    axes[2].bar(x + 0.18, geometry, width=0.36, label="geometry", color="#CC79A7")
    axes[2].set_title("Rating: clue baseline vs geometry")
    axes[2].set_xticks(x, model_names, rotation=25, ha="right")
    axes[2].legend(fontsize=8)
    figure.savefig(os.path.join(output_dir, "summary_controls.png"), dpi=180)
    plt.close(figure)


def run(output_dir, per_split_bucket=4, seed=7027, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()
    sample = sample_balanced_puzzles(per_split_bucket * 3, seed)
    splits = make_stratified_splits(sample["bucket_names"], per_split_bucket)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs = sample["inputs"].to(resolved_device)
    targets = sample["targets"].to(resolved_device)
    empty_mask = sample["empty_mask"].to(resolved_device)
    bucket_order = [entry[2] for entry in RATING_BUCKETS]
    bucket_labels = np.asarray([bucket_order.index(name) for name in sample["bucket_names"]], dtype=np.int64)
    results = {
        "protocol": {
            "seed": seed,
            "per_split": int(len(splits["discovery"])),
            "per_rating_bucket_per_split": per_split_bucket,
            "splits": {name: indices.tolist() for name, indices in splits.items()},
            "horizons": list(DEFAULT_HORIZONS),
            "components": list(DEFAULT_COMPONENTS),
            "alphas": list(DEFAULT_ALPHAS),
            "final_holdout_policy": "selection uses discovery and validation only; final is evaluated once",
        },
        "sample": {
            "ratings": sample["ratings"].tolist(),
            "buckets": sample["bucket_names"],
            "clue_counts": sample["clue_counts"].tolist(),
        },
        "models": {},
    }
    for model_index, model_config in enumerate(DEFAULT_MODELS):
        model_name = model_config["name"]
        print(f"Collecting {model_name}", flush=True)
        model = _load_model(model_config, resolved_device)
        geometry = collect_model_geometry(
            model,
            inputs,
            targets,
            empty_mask,
            DEFAULT_HORIZONS,
            maximum_iteration=1024,
        )
        candidates = prepare_candidates(geometry, splits, DEFAULT_COMPONENTS)
        target_values = {
            "log_rating": np.log1p(sample["ratings"]),
            "log_first_solve_latency": np.log1p(geometry["first_solve"].astype(np.float64)),
            "log_stable_solve_latency": np.log1p(geometry["stable_solve"].astype(np.float64)),
        }
        model_results = {
            "checkpoint": model_config["path"],
            "latencies": {
                "first_solve": geometry["first_solve"].tolist(),
                "stable_solve": geometry["stable_solve"].tolist(),
                "solved_at_1024": int(geometry["solved_at_maximum"].sum()),
                "total": int(len(inputs)),
            },
            "targets": {},
        }
        for target_index, (target_name, target) in enumerate(target_values.items()):
            model_results["targets"][target_name] = analyze_target(
                target_name,
                target,
                candidates,
                splits,
                sample["clue_counts"],
                bucket_labels,
                seed + model_index * 100 + target_index,
            )
        results["models"][model_name] = model_results
        del model, geometry, candidates
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    results["elapsed_seconds"] = time.time() - started
    plot_results(results, output_dir)
    result_path = os.path.join(output_dir, "difficulty_metrics.json")
    temporary_path = result_path + ".tmp"
    with open(temporary_path, "w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, result_path)
    print(f"Wrote {result_path}", flush=True)
    return results


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu")
