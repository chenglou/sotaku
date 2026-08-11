"""Test whether recurrent cell states encode Sudoku digits on a helix-like geometry."""

import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


ITERATIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def ridge_fit(features, targets, ridge=1e-3):
    features = features.double()
    targets = targets.double()
    mean = features.mean(0, keepdim=True)
    scale = features.std(0, keepdim=True).clamp_min(1e-6)
    standardized = (features - mean) / scale
    design = torch.cat(
        [standardized, torch.ones(len(standardized), 1, dtype=torch.double)], 1
    )
    penalty = torch.eye(design.size(1), dtype=torch.double) * ridge
    penalty[-1, -1] = 0
    weights = torch.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return mean, scale, weights


def ridge_predict(features, fitted):
    mean, scale, weights = fitted
    standardized = (features.double() - mean) / scale
    design = torch.cat(
        [standardized, torch.ones(len(standardized), 1, dtype=torch.double)], 1
    )
    return design @ weights


def r_squared(target, prediction):
    residual = (target - prediction).square().sum()
    total = (target - target.mean(0, keepdim=True)).square().sum().clamp_min(1e-12)
    return float(1 - residual / total)


def digit_targets(digits):
    angle = 2 * math.pi * digits.double() / 9
    cyclic = torch.stack([torch.cos(angle), torch.sin(angle)], 1)
    categorical = F.one_hot(digits.long(), 9).double()
    return cyclic, categorical


def fit_digit_probes(train_features, train_digits, test_features, test_digits):
    train_cyclic, train_categorical = digit_targets(train_digits)
    test_cyclic, test_categorical = digit_targets(test_digits)
    cyclic_fit = ridge_fit(train_features, train_cyclic)
    categorical_fit = ridge_fit(train_features, train_categorical)
    cyclic_prediction = ridge_predict(test_features, cyclic_fit)
    categorical_prediction = ridge_predict(test_features, categorical_fit)
    predicted_angle = torch.atan2(cyclic_prediction[:, 1], cyclic_prediction[:, 0])
    predicted_digit = torch.remainder(
        torch.round(predicted_angle * 9 / (2 * math.pi)), 9
    ).long()
    return {
        "cyclic_r2": r_squared(test_cyclic, cyclic_prediction),
        "cyclic_nearest_digit_accuracy": float((predicted_digit == test_digits).double().mean()),
        "categorical_r2": r_squared(test_categorical, categorical_prediction),
        "categorical_accuracy": float((categorical_prediction.argmax(1) == test_digits).double().mean()),
    }, cyclic_fit


def shuffled_digit_control(
    train_features, train_digits, test_features, test_digits, permutations=20, seed=42
):
    generator = torch.Generator().manual_seed(seed)
    scores = []
    for _ in range(permutations):
        digit_permutation = torch.randperm(9, generator=generator)
        shuffled_train = digit_permutation[train_digits]
        shuffled_test = digit_permutation[test_digits]
        score, _ = fit_digit_probes(
            train_features, shuffled_train, test_features, shuffled_test
        )
        scores.append(score["cyclic_r2"])
    return {
        "cyclic_r2_mean": float(np.mean(scores)),
        "cyclic_r2_p95": float(np.quantile(scores, 0.95)),
        "cyclic_r2_values": scores,
    }


def centroid_fourier_fraction(features, digits):
    centroids = torch.stack([features[digits == digit].double().mean(0) for digit in range(9)])
    centroids = centroids - centroids.mean(0, keepdim=True)
    angle = 2 * math.pi * torch.arange(9, dtype=torch.double) / 9
    harmonics = torch.stack([torch.cos(angle), torch.sin(angle)], 1)
    reconstruction = harmonics @ torch.linalg.lstsq(harmonics, centroids).solution
    return float(reconstruction.square().sum() / centroids.square().sum().clamp_min(1e-12))


def collect(model, inputs, targets, empty_mask):
    selected = set(ITERATIONS)
    states = []
    logits = []
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    with torch.no_grad():
        hidden = model.initial_encoder(inputs)
        predictions = torch.zeros(len(inputs), 81, 9, device=inputs.device)
        for iteration in range(ITERATIONS[-1] + 1):
            if iteration in selected:
                states.append(hidden.float().cpu())
                logits.append(model.output_head(hidden).float().cpu())
            if iteration == ITERATIONS[-1]:
                break
            hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
            predictions = F.softmax(model.output_head(hidden), dim=-1)
    logits = torch.stack(logits, 1)
    probabilities = logits.softmax(-1)
    top_two = probabilities.topk(2, dim=-1).values
    return {
        "states": torch.stack(states, 1),
        "predicted": logits.argmax(-1),
        "confidence": probabilities.max(-1).values,
        "margin": top_two[..., 0] - top_two[..., 1],
        "targets": targets.cpu(),
        "empty_mask": empty_mask.cpu(),
    }


def _flatten_puzzles(values, puzzle_indices):
    return values[puzzle_indices].reshape(-1, values.size(-1))


def analyze_model(collected, seed):
    states = collected["states"]
    puzzle_count = states.size(0)
    split = puzzle_count // 2
    train_puzzles = torch.arange(split)
    test_puzzles = torch.arange(split, puzzle_count)
    target_digits = collected["targets"]
    predicted_digits = collected["predicted"]
    results = {"iterations": {}}
    projection = None
    for time_index, iteration in enumerate(ITERATIONS):
        train_features = states[train_puzzles, time_index].reshape(-1, states.size(-1))
        test_features = states[test_puzzles, time_index].reshape(-1, states.size(-1))
        train_true = target_digits[train_puzzles].reshape(-1)
        test_true = target_digits[test_puzzles].reshape(-1)
        train_predicted = predicted_digits[train_puzzles, time_index].reshape(-1)
        test_predicted = predicted_digits[test_puzzles, time_index].reshape(-1)
        train_positions = torch.arange(81).repeat(len(train_puzzles))
        test_positions = torch.arange(81).repeat(len(test_puzzles))
        true_scores, true_projection = fit_digit_probes(
            train_features, train_true, test_features, test_true
        )
        predicted_scores, _ = fit_digit_probes(
            train_features, train_predicted, test_features, test_predicted
        )
        margin_train = collected["margin"][train_puzzles, time_index].reshape(-1, 1)
        margin_test = collected["margin"][test_puzzles, time_index].reshape(-1, 1).double()
        margin_prediction = ridge_predict(
            test_features, ridge_fit(train_features, margin_train)
        )
        confidence_train = collected["confidence"][train_puzzles, time_index].reshape(-1, 1)
        confidence_test = collected["confidence"][test_puzzles, time_index].reshape(-1, 1).double()
        confidence_prediction = ridge_predict(
            test_features, ridge_fit(train_features, confidence_train)
        )
        position_fit = ridge_fit(
            train_features, F.one_hot(train_positions, 81).double()
        )
        position_prediction = ridge_predict(test_features, position_fit)
        train_position_means = torch.stack([
            train_features[train_positions == position].mean(0)
            for position in range(81)
        ])
        position_residual_scores, _ = fit_digit_probes(
            train_features - train_position_means[train_positions],
            train_true,
            test_features - train_position_means[test_positions],
            test_true,
        )
        results["iterations"][str(iteration)] = {
            "true_digit": true_scores,
            "predicted_digit": predicted_scores,
            "true_digit_shuffle": shuffled_digit_control(
                train_features, train_true, test_features, test_true, seed=seed + iteration
            ),
            "true_digit_centroid_first_harmonic_fraction": centroid_fourier_fraction(
                test_features, test_true
            ),
            "margin_r2": r_squared(margin_test, margin_prediction),
            "confidence_r2": r_squared(confidence_test, confidence_prediction),
            "cell_position_accuracy": float(
                (position_prediction.argmax(1) == test_positions).double().mean()
            ),
            "true_digit_after_position_centering": position_residual_scores,
            "test_prediction_accuracy": float((test_predicted == test_true).double().mean()),
        }
        if iteration == 128:
            projection = true_projection

    train_all = states[train_puzzles].reshape(-1, states.size(-1))
    test_all = states[test_puzzles].reshape(-1, states.size(-1))
    train_iteration = torch.arange(len(ITERATIONS)).view(1, -1, 1).expand(
        len(train_puzzles), -1, 81
    ).reshape(-1, 1).double()
    test_iteration = torch.arange(len(ITERATIONS)).view(1, -1, 1).expand(
        len(test_puzzles), -1, 81
    ).reshape(-1, 1).double()
    iteration_prediction = ridge_predict(
        test_all, ridge_fit(train_all, train_iteration)
    )
    results["iteration_index_r2"] = r_squared(test_iteration, iteration_prediction)
    return results, projection


def plot_model(name, collected, projection, output_dir):
    split = collected["states"].size(0) // 2
    test_states = collected["states"][split:]
    test_targets = collected["targets"][split:]
    time_index = ITERATIONS.index(128)
    features = test_states[:, time_index].reshape(-1, test_states.size(-1))
    points = ridge_predict(features, projection).numpy()
    digits = test_targets.reshape(-1).numpy()
    colors = plt.cm.hsv(np.arange(9) / 9)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for digit in range(9):
        chosen = digits == digit
        axes[0].scatter(points[chosen, 0], points[chosen, 1], s=5, alpha=0.18,
                        color=colors[digit], label=str(digit + 1))
    centroids = np.stack([points[digits == digit].mean(0) for digit in range(9)])
    closed = np.concatenate([centroids, centroids[:1]], 0)
    axes[0].plot(closed[:, 0], closed[:, 1], color="black", linewidth=1)
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c=colors, s=55,
                    edgecolor="black", linewidth=0.5)
    for digit, point in enumerate(centroids):
        axes[0].text(point[0], point[1], str(digit + 1), ha="center", va="center", fontsize=8)
    axes[0].set_title("Held-out cells at iteration 128\nprojection trained for numeric cyclic order")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(ncol=3, markerscale=2, fontsize=7)

    puzzle = 0
    cells = np.linspace(0, 80, 12, dtype=int)
    trajectory_features = test_states[puzzle, :, cells].permute(1, 0, 2).reshape(-1, test_states.size(-1))
    trajectories = ridge_predict(trajectory_features, projection).reshape(len(cells), -1, 2).numpy()
    for cell_index, trajectory in zip(cells, trajectories):
        digit = int(test_targets[puzzle, cell_index])
        axes[1].plot(trajectory[:, 0], trajectory[:, 1], color=colors[digit], alpha=0.75)
        axes[1].scatter(trajectory[-1, 0], trajectory[-1, 1], color=colors[digit], s=18)
    axes[1].set_title("Twelve held-out cell trajectories\niterations 0 to 1024")
    axes[1].set_aspect("equal", adjustable="datalim")
    figure.suptitle(name)
    figure.savefig(os.path.join(output_dir, f"{name}_cyclic_projection.png"), dpi=180)
    plt.close(figure)

    train_puzzles = torch.arange(split)
    test_puzzles = torch.arange(split, collected["states"].size(0))
    train_all = collected["states"][train_puzzles].reshape(-1, test_states.size(-1))
    train_iteration = torch.arange(len(ITERATIONS)).view(1, -1, 1).expand(
        len(train_puzzles), -1, 81
    ).reshape(-1, 1).double()
    iteration_projection = ridge_fit(train_all, train_iteration)
    chosen_cells = np.linspace(0, 80, 18, dtype=int)
    chosen_states = test_states[0, :, chosen_cells].permute(1, 0, 2)
    flat_states = chosen_states.reshape(-1, chosen_states.size(-1))
    xy = ridge_predict(flat_states, projection).reshape(len(chosen_cells), -1, 2).numpy()
    z = ridge_predict(flat_states, iteration_projection).reshape(len(chosen_cells), -1).numpy()
    target_digits = collected["targets"][test_puzzles[0], chosen_cells].numpy()
    figure = plt.figure(figsize=(8, 7), constrained_layout=True)
    axis = figure.add_subplot(111, projection="3d")
    for digit, x_y, height in zip(target_digits, xy, z):
        axis.plot(x_y[:, 0], x_y[:, 1], height, color=colors[digit], alpha=0.72)
        axis.scatter(x_y[-1, 0], x_y[-1, 1], height[-1], color=colors[digit], s=18)
    axis.set_xlabel("fitted digit cosine")
    axis.set_ylabel("fitted digit sine")
    axis.set_zlabel("fitted iteration coordinate")
    axis.set_title(
        f"{name}: held-out cell paths\n"
        "All three axes are supervised; shape alone is not evidence"
    )
    figure.savefig(os.path.join(output_dir, f"{name}_supervised_3d.png"), dpi=180)
    plt.close(figure)


def run(output_dir, examples_per_bucket=4, seed=42, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, solutions, buckets = _load_balanced_sample(
        examples_per_bucket, seed
    )
    inputs, targets, empty_mask = inputs.to(resolved_device), targets.to(resolved_device), empty_mask.to(resolved_device)
    summary = {"config": {"iterations": ITERATIONS, "examples_per_bucket": examples_per_bucket,
                           "sample_size": len(inputs), "seed": seed, "models": DEFAULT_MODELS},
               "sample": {"puzzles": puzzles, "solutions": solutions, "buckets": buckets}, "models": {}}
    started = time.time()
    for model_index, model_config in enumerate(DEFAULT_MODELS):
        print(f"Analyzing {model_config['name']}", flush=True)
        model = _load_model(model_config, resolved_device)
        collected = collect(model, inputs, targets, empty_mask)
        results, projection = analyze_model(collected, seed + 1000 * model_index)
        summary["models"][model_config["name"]] = results
        plot_model(model_config["name"], collected, projection, output_dir)
        del model, collected
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    summary["elapsed_seconds"] = time.time() - started
    with open(os.path.join(output_dir, "helix_results.json"), "w") as result_file:
        json.dump(summary, result_file, indent=2)
        result_file.write("\n")
    return summary


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cpu")
