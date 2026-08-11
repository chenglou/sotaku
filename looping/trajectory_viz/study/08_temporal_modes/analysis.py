"""Held-out tests for recurrent temporal modes in Sotaku hidden states."""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample, _load_model
from looping.eval_trajectory_geometry import DEFAULT_MODELS


SEED = 8082026
EXAMPLES_PER_BUCKET = 12
PROJECTION_DIMENSION = 16
WINDOWS = {"early": (0, 128), "late": (768, 1024)}
SPLIT_NAMES = ("discovery", "validation", "final")
REPRESENTATIONS = ("normalized_state", "normalized_update")
CONTROL_REPEATS = 32


def balanced_split_indices(bucket_names):
    """Split every rating bucket equally, without mixing puzzle identities."""
    groups = {}
    for index, bucket in enumerate(bucket_names):
        groups.setdefault(bucket, []).append(index)
    result = {name: [] for name in SPLIT_NAMES}
    for bucket, indices in groups.items():
        if len(indices) % 3:
            raise ValueError(f"bucket {bucket!r} cannot be split equally")
        width = len(indices) // 3
        for split_index, name in enumerate(SPLIT_NAMES):
            start = split_index * width
            result[name].extend(indices[start : start + width])
    return {name: sorted(indices) for name, indices in result.items()}


def random_orthonormal_projection(rows, columns, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(rows, columns, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix, mode="reduced")
    return basis.float()


def phase_randomize(sequences, rng):
    """Preserve each channel's Fourier magnitude while randomizing phase."""
    sequences = np.asarray(sequences, dtype=np.float64)
    spectrum = np.fft.rfft(sequences, axis=1)
    phases = rng.uniform(-np.pi, np.pi, size=spectrum.shape)
    phases[:, 0, :] = 0.0
    if sequences.shape[1] % 2 == 0:
        phases[:, -1, :] = 0.0
    randomized = np.abs(spectrum) * np.exp(1j * phases)
    randomized[:, 0, :] = spectrum[:, 0, :]
    if sequences.shape[1] % 2 == 0:
        randomized[:, -1, :] = spectrum[:, -1, :].real
    return np.fft.irfft(randomized, n=sequences.shape[1], axis=1)


def shuffle_time(sequences, rng):
    shuffled = np.empty_like(sequences)
    for puzzle_index in range(len(sequences)):
        shuffled[puzzle_index] = sequences[puzzle_index, rng.permutation(sequences.shape[1])]
    return shuffled


def _safe_cosine(left, right):
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.sum(left * right, axis=-1) / np.maximum(denominator, 1e-12)


def temporal_summary(sequences):
    """Projection-space curvature and spectral summaries."""
    sequences = np.asarray(sequences, dtype=np.float64)
    velocity = np.diff(sequences, axis=1)
    acceleration = np.diff(velocity, axis=1)
    turn_cosine = _safe_cosine(velocity[:, :-1], velocity[:, 1:])
    relative_acceleration = np.linalg.norm(acceleration, axis=-1) / np.maximum(
        np.linalg.norm(velocity[:, :-1], axis=-1), 1e-12
    )

    centered = sequences - sequences.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=1)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(sequences.shape[1])
    nonzero = frequencies > 0
    nonzero_power = power[:, nonzero].sum(axis=(1, 2))
    high_frequency = frequencies >= 1 / 32
    high_power = power[:, high_frequency].sum(axis=(1, 2))
    high_frequency_fraction = high_power / np.maximum(nonzero_power, 1e-12)

    pooled_power = power.sum(axis=(0, 2))
    pooled_power[0] = 0.0
    dominant_index = int(np.argmax(pooled_power))
    frequency_slice = spectrum[:, dominant_index, :]
    cross_spectrum = frequency_slice.conj().T @ frequency_slice
    eigenvalues = np.linalg.eigvalsh(cross_spectrum).real.clip(min=0)
    coherence = float(eigenvalues[-1] / max(eigenvalues.sum(), 1e-12))

    return {
        "turn_cosine_mean": float(np.mean(turn_cosine)),
        "turn_cosine_p10": float(np.quantile(turn_cosine, 0.1)),
        "relative_acceleration_mean": float(np.mean(relative_acceleration)),
        "high_frequency_power_fraction": float(np.mean(high_frequency_fraction)),
        "dominant_frequency": float(frequencies[dominant_index]),
        "dominant_period": (
            float(1 / frequencies[dominant_index])
            if frequencies[dominant_index] > 0
            else None
        ),
        "dominant_cross_feature_coherence": coherence,
    }


def fit_dmd(sequences, ridge=1e-4):
    """Fit an affine one-step linear map on several trajectories."""
    sequences = np.asarray(sequences, dtype=np.float64)
    left = sequences[:, :-1].reshape(-1, sequences.shape[-1])
    right = sequences[:, 1:].reshape(-1, sequences.shape[-1])
    design = np.concatenate([left, np.ones((len(left), 1))], axis=1)
    gram = design.T @ design
    penalty = ridge * np.eye(gram.shape[0])
    penalty[-1, -1] = 0.0
    coefficients = np.linalg.solve(gram + penalty, design.T @ right)
    return coefficients[:-1], coefficients[-1]


def evaluate_dmd(sequences, matrix, bias):
    sequences = np.asarray(sequences, dtype=np.float64)
    left = sequences[:, :-1]
    right = sequences[:, 1:]
    prediction = left @ matrix + bias
    residual = np.square(right - prediction).sum()
    centered = right - right.mean(axis=(0, 1), keepdims=True)
    total = np.square(centered).sum()
    persistence = np.square(right - left).sum()
    return {
        "r2": float(1 - residual / max(total, 1e-12)),
        "relative_to_persistence": float(1 - residual / max(persistence, 1e-12)),
        "mse": float(np.mean(np.square(right - prediction))),
    }


def dmd_spectrum(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    radius = np.abs(eigenvalues)
    angle = np.abs(np.angle(eigenvalues))
    active = radius > 0.5
    oscillatory = active & (angle > 0.02)
    return {
        "eigenvalues": [[float(value.real), float(value.imag)] for value in eigenvalues],
        "spectral_radius": float(radius.max()),
        "active_oscillatory_fraction": float(oscillatory.sum() / max(active.sum(), 1)),
        "largest_active_angle": float(angle[active].max()) if active.any() else 0.0,
    }


def control_distribution(sequences, statistic, control, repeats, seed):
    values = []
    for repeat in range(repeats):
        rng = np.random.default_rng(seed + repeat)
        controlled = control(sequences, rng)
        values.append(float(statistic(controlled)))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p05": float(np.quantile(values, 0.05)),
        "p95": float(np.quantile(values, 0.95)),
        "values": values,
    }


def _collect_model(model, inputs, projection_bases, device):
    final_iteration = max(end for _, end in WINDOWS.values())
    rope_cos = model_module.ROPE_COS.to(device)
    rope_sin = model_module.ROPE_SIN.to(device)
    projections = [basis.to(device) for basis in projection_bases]
    collected = {
        basis_name: {
            window_name: {"state": [], "state_norm": [], "update_norm": []}
            for window_name in WINDOWS
        }
        for basis_name in ("a", "b")
    }

    hidden = model.initial_encoder(inputs)
    predictions = torch.zeros(inputs.size(0), 81, 9, device=device)
    with torch.no_grad():
        for iteration in range(final_iteration + 1):
            active_windows = [
                name for name, (start, end) in WINDOWS.items()
                if start <= iteration <= end
            ]
            if active_windows:
                flat_hidden = hidden.flatten(1).float()
                hidden_norm = flat_hidden.norm(dim=1).cpu()
                for basis_name, basis in zip(("a", "b"), projections):
                    projected = (flat_hidden @ basis).cpu()
                    for window_name in active_windows:
                        collected[basis_name][window_name]["state"].append(projected)
                        collected[basis_name][window_name]["state_norm"].append(hidden_norm)

            if iteration == final_iteration:
                break
            next_hidden = model.recurrent_step(
                hidden,
                predictions,
                rope_cos,
                rope_sin,
            )
            update_norm = (next_hidden - hidden).flatten(1).float().norm(dim=1).cpu()
            for window_name, (start, end) in WINDOWS.items():
                if start <= iteration < end:
                    for basis_name in ("a", "b"):
                        collected[basis_name][window_name]["update_norm"].append(update_norm)
            hidden = next_hidden
            predictions = F.softmax(model.output_head(hidden), dim=-1)

    result = {}
    for basis_name in ("a", "b"):
        result[basis_name] = {}
        for window_name in WINDOWS:
            state = torch.stack(collected[basis_name][window_name]["state"], dim=1)
            state_norm = torch.stack(
                collected[basis_name][window_name]["state_norm"], dim=1
            )
            update_norm = torch.stack(
                collected[basis_name][window_name]["update_norm"], dim=1
            )
            normalized_state = state / state_norm.unsqueeze(-1).clamp_min(1e-12)
            projected_update = state[:, 1:] - state[:, :-1]
            normalized_update = projected_update / update_norm.unsqueeze(-1).clamp_min(1e-12)
            result[basis_name][window_name] = {
                "normalized_state": normalized_state.numpy(),
                "normalized_update": normalized_update.numpy(),
            }
    return result


def collect_trajectories(device="cuda"):
    resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, _, _, puzzles, _, buckets = _load_balanced_sample(EXAMPLES_PER_BUCKET, SEED)
    inputs = inputs.to(resolved_device)
    split_indices = balanced_split_indices(buckets)
    board_dimension = 81 * 128
    bases = (
        random_orthonormal_projection(board_dimension, PROJECTION_DIMENSION, SEED),
        random_orthonormal_projection(board_dimension, PROJECTION_DIMENSION, SEED + 1),
    )
    trajectories = {}
    for config in DEFAULT_MODELS:
        model = _load_model(config, resolved_device)
        trajectories[config["name"]] = _collect_model(
            model,
            inputs,
            bases,
            resolved_device,
        )
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "trajectories": trajectories,
        "splits": split_indices,
        "puzzles": puzzles,
        "buckets": buckets,
        "models": list(DEFAULT_MODELS),
    }


def _subset(array, indices):
    return np.asarray(array)[np.asarray(indices)]


def analyze_collected(collected):
    trajectories = collected["trajectories"]
    splits = collected["splits"]
    results = {"variants": {}, "selection": {}, "transfer": {}}
    selection_candidates = []

    for model_name, model_data in trajectories.items():
        results["variants"][model_name] = {}
        for basis_name in ("a", "b"):
            results["variants"][model_name][basis_name] = {}
            for window_name in WINDOWS:
                results["variants"][model_name][basis_name][window_name] = {}
                for representation in REPRESENTATIONS:
                    sequences = model_data[basis_name][window_name][representation]
                    discovery = _subset(sequences, splits["discovery"])
                    validation = _subset(sequences, splits["validation"])
                    final = _subset(sequences, splits["final"])
                    matrix, bias = fit_dmd(discovery)
                    shuffled_discovery = shuffle_time(
                        discovery,
                        np.random.default_rng(SEED + 100),
                    )
                    shuffled_matrix, shuffled_bias = fit_dmd(shuffled_discovery)
                    validation_ordered = evaluate_dmd(validation, matrix, bias)
                    validation_shuffled_fit = evaluate_dmd(
                        validation, shuffled_matrix, shuffled_bias
                    )
                    variant = {
                        "discovery_summary": temporal_summary(discovery),
                        "validation_summary": temporal_summary(validation),
                        "validation_dmd": validation_ordered,
                        "validation_shuffled_fit_dmd": validation_shuffled_fit,
                        "final_summary": temporal_summary(final),
                        "final_dmd": evaluate_dmd(final, matrix, bias),
                        "final_shuffled_fit_dmd": evaluate_dmd(
                            final, shuffled_matrix, shuffled_bias
                        ),
                        "dmd_spectrum": dmd_spectrum(matrix),
                    }
                    results["variants"][model_name][basis_name][window_name][representation] = variant
                    if basis_name == "a":
                        selection_candidates.append(
                            {
                                "model": model_name,
                                "window": window_name,
                                "representation": representation,
                                "validation_effect": (
                                    validation_ordered["r2"]
                                    - validation_shuffled_fit["r2"]
                                ),
                            }
                        )

    grouped = {}
    for candidate in selection_candidates:
        key = (candidate["window"], candidate["representation"])
        grouped.setdefault(key, []).append(candidate["validation_effect"])
    selected_key = max(grouped, key=lambda key: np.mean(grouped[key]))
    results["selection"] = {
        "window": selected_key[0],
        "representation": selected_key[1],
        "criterion": "mean validation DMD R2 advantage over shuffled-time fit across checkpoints",
        "candidate_mean_effects": {
            f"{key[0]}/{key[1]}": float(np.mean(values))
            for key, values in grouped.items()
        },
    }

    selected_window, selected_representation = selected_key
    for model_name, model_data in trajectories.items():
        results["variants"][model_name]["selected_final_controls"] = {}
        for basis_name in ("a", "b"):
            sequences = model_data[basis_name][selected_window][selected_representation]
            discovery = _subset(sequences, splits["discovery"])
            final = _subset(sequences, splits["final"])
            matrix, bias = fit_dmd(discovery)
            actual_r2 = evaluate_dmd(final, matrix, bias)["r2"]
            statistic = lambda controlled: evaluate_dmd(controlled, matrix, bias)["r2"]
            phase_control = control_distribution(
                final,
                statistic,
                phase_randomize,
                CONTROL_REPEATS,
                SEED + 200,
            )
            shuffle_control = control_distribution(
                final,
                statistic,
                shuffle_time,
                CONTROL_REPEATS,
                SEED + 300,
            )
            results["variants"][model_name]["selected_final_controls"][basis_name] = {
                "actual_dmd_r2": actual_r2,
                "phase_randomized_dmd_r2": phase_control,
                "shuffled_time_dmd_r2": shuffle_control,
                "actual_minus_phase_mean": actual_r2 - phase_control["mean"],
                "actual_minus_shuffle_mean": actual_r2 - shuffle_control["mean"],
            }

    for source_name, source_data in trajectories.items():
        source_discovery = _subset(
            source_data["a"][selected_window][selected_representation],
            splits["discovery"],
        )
        matrix, bias = fit_dmd(source_discovery)
        results["transfer"][source_name] = {}
        for target_name, target_data in trajectories.items():
            target_final = _subset(
                target_data["a"][selected_window][selected_representation],
                splits["final"],
            )
            results["transfer"][source_name][target_name] = evaluate_dmd(
                target_final,
                matrix,
                bias,
            )
    return results


def _plot_results(results, output_dir):
    model_names = list(results["variants"])
    selected_window = results["selection"]["window"]
    selected_representation = results["selection"]["representation"]

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, window_name in enumerate(WINDOWS):
        for column, representation in enumerate(REPRESENTATIONS):
            axis = axes[row, column]
            ordered = []
            shuffled = []
            for model_name in model_names:
                variant = results["variants"][model_name]["a"][window_name][representation]
                ordered.append(variant["final_dmd"]["r2"])
                shuffled.append(variant["final_shuffled_fit_dmd"]["r2"])
            positions = np.arange(len(model_names))
            axis.bar(positions - 0.18, ordered, 0.36, label="ordered")
            axis.bar(positions + 0.18, shuffled, 0.36, label="shuffled fit")
            axis.set_title(f"{window_name}: {representation}")
            axis.set_xticks(positions, model_names, rotation=20, ha="right")
            axis.set_ylabel("final DMD R2")
            axis.axhline(0, color="black", linewidth=0.8)
            axis.legend()
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "dmd_heldout_comparison.png"), dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for basis_name, axis in zip(("a", "b"), axes):
        actual = []
        phase = []
        shuffled = []
        for model_name in model_names:
            controls = results["variants"][model_name]["selected_final_controls"][basis_name]
            actual.append(controls["actual_dmd_r2"])
            phase.append(controls["phase_randomized_dmd_r2"]["mean"])
            shuffled.append(controls["shuffled_time_dmd_r2"]["mean"])
        positions = np.arange(len(model_names))
        axis.bar(positions - 0.25, actual, 0.25, label="actual")
        axis.bar(positions, phase, 0.25, label="phase randomized")
        axis.bar(positions + 0.25, shuffled, 0.25, label="time shuffled")
        axis.set_title(f"Random projection {basis_name}")
        axis.set_xticks(positions, model_names, rotation=20, ha="right")
        axis.set_ylabel("final DMD R2")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.legend()
    figure.suptitle(f"Selected on validation: {selected_window}/{selected_representation}")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "selected_controls.png"), dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(10, 9))
    for axis, model_name in zip(axes.flat, model_names):
        spectrum = results["variants"][model_name]["a"][selected_window][selected_representation]["dmd_spectrum"]
        eigenvalues = np.asarray(spectrum["eigenvalues"])
        axis.scatter(eigenvalues[:, 0], eigenvalues[:, 1], s=28)
        circle = plt.Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
        axis.add_patch(circle)
        axis.axhline(0, color="black", linewidth=0.7)
        axis.axvline(0, color="black", linewidth=0.7)
        axis.set_xlim(-1.1, 1.1)
        axis.set_ylim(-1.1, 1.1)
        axis.set_aspect("equal")
        axis.set_title(model_name)
        axis.set_xlabel("real")
        axis.set_ylabel("imaginary")
    figure.suptitle("Discovery-fit DMD eigenvalues")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "dmd_eigenvalues.png"), dpi=170)
    plt.close(figure)

    transfer = np.array([
        [results["transfer"][source][target]["r2"] for target in model_names]
        for source in model_names
    ])
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(transfer, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set_xticks(range(len(model_names)), model_names, rotation=25, ha="right")
    axis.set_yticks(range(len(model_names)), model_names)
    axis.set_xlabel("target checkpoint")
    axis.set_ylabel("source DMD fit")
    for row in range(len(model_names)):
        for column in range(len(model_names)):
            axis.text(column, row, f"{transfer[row, column]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="final R2")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "checkpoint_transfer.png"), dpi=170)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    metric_specs = (
        ("turn_cosine_mean", "Consecutive velocity cosine"),
        ("relative_acceleration_mean", "Relative acceleration"),
        ("high_frequency_power_fraction", "Power at period <= 32"),
        ("dominant_period", "Dominant Fourier period"),
    )
    positions = np.arange(len(model_names))
    for axis, (metric_name, title) in zip(axes.flat, metric_specs):
        offset = -0.3
        for window_name in WINDOWS:
            for basis_name in ("a", "b"):
                values = [
                    results["variants"][model_name][basis_name][window_name][
                        "normalized_update"
                    ]["final_summary"][metric_name]
                    for model_name in model_names
                ]
                label = f"{window_name}, projection {basis_name}"
                axis.bar(positions + offset, values, 0.2, label=label)
                offset += 0.2
        axis.set_title(title)
        axis.set_xticks(positions, model_names, rotation=20, ha="right")
        axis.axhline(0, color="black", linewidth=0.7)
        if metric_name == "dominant_period":
            axis.set_yscale("log")
        axis.legend(fontsize=8)
    figure.suptitle("Final-holdout normalized-update temporal statistics")
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, "curvature_fourier_summary.png"), dpi=170)
    plt.close(figure)


def run(output_dir, device="cuda"):
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()
    collected = collect_trajectories(device=device)
    results = analyze_collected(collected)
    payload = {
        "config": {
            "seed": SEED,
            "examples_per_bucket": EXAMPLES_PER_BUCKET,
            "sample_size": len(collected["puzzles"]),
            "split_sizes": {name: len(indices) for name, indices in collected["splits"].items()},
            "windows": WINDOWS,
            "projection_dimension": PROJECTION_DIMENSION,
            "control_repeats": CONTROL_REPEATS,
            "models": collected["models"],
        },
        "sample": {
            "puzzles": collected["puzzles"],
            "buckets": collected["buckets"],
            "splits": collected["splits"],
        },
        "results": results,
        "elapsed_seconds": time.time() - started,
    }
    temporary = os.path.join(output_dir, "metrics.json.tmp")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, os.path.join(output_dir, "metrics.json"))
    _plot_results(results, output_dir)
    return payload


if __name__ == "__main__":
    run(os.path.dirname(__file__), device="cuda")
