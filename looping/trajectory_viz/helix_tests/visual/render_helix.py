"""Static figures for held-out per-cell trajectory geometry."""

from __future__ import annotations

import colorsys
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from looping.trajectory_viz.helix_tests.visual.helix_geometry import (
    NATURAL_ORDER,
    class_centroids,
    cycle_lengths,
    cycle_tensor,
    downsample_indices,
    helix_code,
    pairwise_distances,
    periodic_code,
)


FIGURE_DPI = 190
DIGIT_PALETTE = np.asarray([
    colorsys.hsv_to_rgb(index / 9.0, 0.74, 0.72)
    for index in range(9)
])
MODEL_PALETTE = (
    "#3264a8",
    "#d0533d",
    "#2f8f68",
    "#8a5a9f",
    "#c58b24",
)


def _save(figure, path):
    figure.savefig(
        path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def _digit_scatter(axis, points, digits, *, size=8, alpha=0.62):
    for digit in range(9):
        mask = digits == digit
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=size,
            alpha=alpha,
            color=DIGIT_PALETTE[digit],
            edgecolors="none",
            rasterized=True,
            label=str(digit + 1),
        )


def _position_colors(rows, columns):
    colors = []
    for row, column in zip(rows, columns):
        hue = (float(column) + 0.5 * (float(row) % 2)) / 9.0
        saturation = 0.72
        value = 0.40 + 0.52 * float(row) / 8.0
        colors.append(colorsys.hsv_to_rgb(hue % 1.0, saturation, value))
    return np.asarray(colors)


def _position_key(axis):
    key = np.zeros((9, 9, 3))
    for row in range(9):
        for column in range(9):
            key[row, column] = _position_colors([row], [column])[0]
    inset = axis.inset_axes([1.04, 0.12, 0.28, 0.72])
    inset.imshow(key, origin="upper", interpolation="nearest")
    inset.set_xticks((0, 8), (1, 9), fontsize=7)
    inset.set_yticks((0, 8), (1, 9), fontsize=7)
    inset.set_xlabel("column", fontsize=7, labelpad=-2)
    inset.set_ylabel("row", fontsize=7, labelpad=-3)
    for spine in inset.spines.values():
        spine.set_linewidth(0.5)


def _target_ring(axis):
    angles = np.linspace(0, 2 * np.pi, 256)
    axis.plot(np.cos(angles), np.sin(angles), color="#9aa0a6", lw=0.8, alpha=0.6)
    code = periodic_code().numpy()
    for digit, point in enumerate(code):
        axis.text(
            point[0] * 1.12,
            point[1] * 1.12,
            str(digit + 1),
            color=DIGIT_PALETTE[digit],
            fontsize=8,
            fontweight="bold",
            ha="center",
            va="center",
        )


def render_attribute_atlas(
    coordinates: torch.Tensor,
    metadata: dict[str, torch.Tensor],
    path: str,
    *,
    title: str,
    axis_labels: tuple[str, str],
    seed: int,
    target_ring: bool,
):
    """Plot identical held-out coordinates with six requested color encodings."""

    selected = downsample_indices(
        metadata["true_digit"],
        metadata["iteration"],
        5200,
        seed=seed,
    )
    points = coordinates[selected].float().numpy()
    shown = {
        key: value[selected].cpu().numpy()
        for key, value in metadata.items()
    }
    lower = np.quantile(points, 0.005, axis=0)
    upper = np.quantile(points, 0.995, axis=0)
    span = np.maximum(upper - lower, 1e-5)
    lower -= 0.08 * span
    upper += 0.08 * span
    if target_ring:
        lower = np.minimum(lower, -1.28)
        upper = np.maximum(upper, 1.28)

    figure, axes = plt.subplots(2, 3, figsize=(14.2, 9.1), constrained_layout=True)
    figure.suptitle(title, fontsize=15, y=1.02)
    panels = axes.flat

    _digit_scatter(panels[0], points, shown["true_digit"])
    panels[0].set_title("true digit · cyclic hue 1→…→9")
    _digit_scatter(panels[1], points, shown["predicted_digit"])
    panels[1].set_title("predicted digit · same cyclic hue")

    confidence = panels[2].scatter(
        points[:, 0],
        points[:, 1],
        c=shown["confidence"],
        cmap="cividis",
        vmin=0,
        vmax=1,
        s=7,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    panels[2].set_title("prediction confidence")
    figure.colorbar(confidence, ax=panels[2], shrink=0.78, label="max probability")

    margin_limit = max(1e-5, float(np.quantile(np.abs(shown["target_margin"]), 0.98)))
    margin = panels[3].scatter(
        points[:, 0],
        points[:, 1],
        c=shown["target_margin"],
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=-margin_limit, vcenter=0, vmax=margin_limit),
        s=7,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    panels[3].set_title("true-digit margin")
    figure.colorbar(margin, ax=panels[3], shrink=0.78, label="true logit − best wrong")

    log_iteration = np.log2(shown["iteration"] + 1)
    iteration = panels[4].scatter(
        points[:, 0],
        points[:, 1],
        c=log_iteration,
        cmap="viridis",
        vmin=0,
        vmax=np.log2(max(1, shown["iteration"].max()) + 1),
        s=7,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    panels[4].set_title("iteration")
    colorbar = figure.colorbar(iteration, ax=panels[4], shrink=0.78)
    available = np.unique(shown["iteration"])
    desired = np.asarray((0, 1, 4, 16, 64, 256, 1024))
    ticks = desired[np.isin(desired, available)]
    colorbar.set_ticks(
        np.log2(ticks + 1),
        labels=[str(value) for value in ticks],
    )
    colorbar.set_label("recurrent iteration")

    position = _position_colors(shown["row"], shown["column"])
    panels[5].scatter(
        points[:, 0],
        points[:, 1],
        c=position,
        s=7,
        alpha=0.55,
        edgecolors="none",
        rasterized=True,
    )
    panels[5].set_title("cell position · row × column")
    _position_key(panels[5])

    for axis in panels:
        if target_ring:
            _target_ring(axis)
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
        axis.set_xlabel(axis_labels[0])
        axis.set_ylabel(axis_labels[1])
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#d5d8dc", linewidth=0.45, alpha=0.45)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        title="digit",
        ncol=9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.045),
        frameon=False,
        markerscale=1.7,
        columnspacing=1.1,
        handletextpad=0.2,
    )
    _save(figure, path)


def _centroids_in_projection(metrics, coordinates, split):
    feature_centroids = torch.tensor(
        metrics["intrinsic_digit_geometry"][f"{split}_centroids"]
    )
    return coordinates["pca_projection"].project(feature_centroids)[:, :2]


def _draw_centroid_cycle(
    axis,
    points,
    *,
    marker,
    label,
    alpha=1.0,
    linestyle="-",
    line_color="#747b83",
):
    closed = torch.cat((points, points[:1]), dim=0).numpy()
    axis.plot(
        closed[:, 0],
        closed[:, 1],
        color=line_color,
        lw=1.15,
        ls=linestyle,
        alpha=0.72,
    )
    for digit, point in enumerate(points.numpy()):
        axis.scatter(
            point[0],
            point[1],
            marker=marker,
            s=66,
            facecolor=DIGIT_PALETTE[digit] if marker != "o" else "none",
            edgecolor=DIGIT_PALETTE[digit],
            linewidth=1.5,
            alpha=alpha,
            label=label if digit == 0 else None,
            zorder=3,
        )
        axis.text(point[0], point[1], str(digit + 1), fontsize=7, ha="center", va="center")


def render_centroid_diagnostics(
    model_name: str,
    model_metrics: dict,
    coordinate_sets: dict[str, dict],
    path: str,
):
    figure, axes = plt.subplots(2, 3, figsize=(14.5, 8.2), constrained_layout=True)
    figure.suptitle(
        f"{model_name} · intrinsic digit-centroid checks on held-out blanks",
        fontsize=15,
    )
    ideal_spectrum = np.linalg.svd(
        helix_code().numpy() - helix_code().numpy().mean(0, keepdims=True),
        compute_uv=False,
    ) ** 2
    ideal_spectrum = ideal_spectrum / ideal_spectrum.sum()
    for row, representation in enumerate(("unit_state", "unit_update")):
        metrics = model_metrics["representations"][representation]
        coordinates = coordinate_sets[representation]
        train_centroids = _centroids_in_projection(metrics, coordinates, "train")
        test_centroids = _centroids_in_projection(metrics, coordinates, "test")
        centroid_axis = axes[row, 0]
        _draw_centroid_cycle(
            centroid_axis,
            train_centroids,
            marker="o",
            label="train centroids",
            alpha=0.75,
            linestyle="--",
            line_color="#9ba2aa",
        )
        _draw_centroid_cycle(
            centroid_axis,
            test_centroids,
            marker="s",
            label="test centroids",
            linestyle="-",
            line_color="#4f5862",
        )
        centroid_axis.set_aspect("equal", adjustable="datalim")
        centroid_axis.set_xlabel("train PC1")
        centroid_axis.set_ylabel("train PC2")
        centroid_axis.set_title(
            f"{representation.replace('_', ' ')} · natural order connected"
        )
        centroid_axis.legend(frameon=False, fontsize=8)
        centroid_axis.grid(alpha=0.25)

        spectrum = np.asarray(
            metrics["intrinsic_digit_geometry"]["test_centroid_spectrum"][
                "eigenvalue_fractions"
            ]
        )
        spectrum_axis = axes[row, 1]
        spectrum_axis.bar(
            np.arange(1, 9) - 0.17,
            spectrum[:8],
            width=0.34,
            color="#5279a8",
            label="held-out centroids",
        )
        spectrum_axis.bar(
            np.arange(1, 4) + 0.17,
            ideal_spectrum,
            width=0.34,
            color="#d4864b",
            label="ideal 3D helix",
        )
        spectrum_axis.set_xticks(range(1, 9))
        spectrum_axis.set_xlabel("centered centroid component")
        spectrum_axis.set_ylabel("variance fraction")
        spectrum_axis.set_title(
            "rank test · top-3 = "
            f"{metrics['intrinsic_digit_geometry']['test_centroid_spectrum']['top3_fraction']:.2f}"
        )
        spectrum_axis.legend(frameon=False, fontsize=8)
        spectrum_axis.grid(axis="y", alpha=0.25)

        geometry = metrics["intrinsic_digit_geometry"]
        test_feature_centroids = torch.tensor(geometry["test_centroids"])
        lengths = cycle_lengths(pairwise_distances(test_feature_centroids), cycle_tensor())
        natural_length = geometry["natural_cycle_length"]
        selected_length = geometry["train_shortest_cycle_test_length"]
        null_axis = axes[row, 2]
        null_axis.hist(
            lengths.numpy(),
            bins=45,
            color="#aab2bb",
            edgecolor="white",
            linewidth=0.3,
        )
        null_axis.axvline(
            natural_length,
            color="#b12f3f",
            lw=2,
            label=(
                "natural 1→…→9 · "
                f"p={geometry['natural_cycle_shorter_percentile']:.2f}"
            ),
        )
        null_axis.axvline(
            selected_length,
            color="#287a62",
            lw=2,
            ls="--",
            label=(
                "train-shortest on test · "
                f"p={geometry['train_shortest_cycle_test_shorter_percentile']:.2f}"
            ),
        )
        null_axis.set_xlabel("closed-cycle path length in original 128D space")
        null_axis.set_ylabel("cycle count")
        null_axis.set_title("all 20,160 digit cycles")
        null_axis.legend(frameon=False, fontsize=8)
        null_axis.grid(axis="x", alpha=0.2)
    _save(figure, path)


def _style_3d(axis, labels, points):
    axis.set_xlabel(labels[0], labelpad=2)
    axis.set_ylabel(labels[1], labelpad=2)
    axis.set_zlabel(labels[2], labelpad=2)
    axis.tick_params(labelsize=7, pad=0)
    axis.view_init(elev=24, azim=-58)
    axis.set_proj_type("ortho")
    points = np.asarray(points)
    lower = np.quantile(points, 0.005, axis=0)
    upper = np.quantile(points, 0.995, axis=0)
    span = np.maximum(upper - lower, 1e-5)
    lower -= 0.06 * span
    upper += 0.06 * span
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(span)
    axis.grid(True, alpha=0.2)


def _digit_scatter_3d(axis, points, digits, *, size=7, alpha=0.42):
    for digit in range(9):
        mask = digits == digit
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            points[mask, 2],
            s=size,
            alpha=alpha,
            color=DIGIT_PALETTE[digit],
            edgecolors="none",
            rasterized=True,
        )


def _draw_ideal_helix(axis):
    code = helix_code().numpy()
    axis.plot(code[:, 0], code[:, 1], code[:, 2], color="#555d66", lw=1.1, alpha=0.65)
    for digit, point in enumerate(code):
        axis.text(
            point[0],
            point[1],
            point[2],
            str(digit + 1),
            color=DIGIT_PALETTE[digit],
            fontsize=8,
            fontweight="bold",
        )


def render_geometry_3d(
    model_name: str,
    metadata: dict[str, torch.Tensor],
    coordinate_sets: dict[str, dict],
    path: str,
    *,
    seed: int,
):
    selected = downsample_indices(
        metadata["true_digit"], metadata["iteration"], 3300, seed=seed
    )
    shown = {key: value[selected].numpy() for key, value in metadata.items()}
    state_pca = coordinate_sets["unit_state"]["pca_test"][selected].numpy()
    update_pca = coordinate_sets["unit_update"]["pca_test"][selected].numpy()
    state_helix = coordinate_sets["unit_state"]["helix_test"][selected].numpy()
    update_helix = coordinate_sets["unit_update"]["helix_test"][selected].numpy()
    figure = plt.figure(figsize=(13.5, 10.5), constrained_layout=True)
    figure.suptitle(
        f"{model_name} · 3D held-out blank-cell geometry",
        fontsize=15,
    )
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]
    _digit_scatter_3d(axes[0], state_pca, shown["true_digit"])
    axes[0].set_title(
        "unit states · train-fit PCA · true digit\n"
        "(cyclic hues match labeled target below)"
    )
    _style_3d(axes[0], ("PC1", "PC2", "PC3"), state_pca)

    iteration_colors = np.log2(shown["iteration"] + 1)
    iteration_scatter = axes[1].scatter(
        update_pca[:, 0],
        update_pca[:, 1],
        update_pca[:, 2],
        c=iteration_colors,
        cmap="viridis",
        s=7,
        alpha=0.45,
        edgecolors="none",
        rasterized=True,
    )
    axes[1].set_title("unit updates · train-fit PCA · iteration")
    _style_3d(axes[1], ("PC1", "PC2", "PC3"), update_pca)
    iteration_bar = figure.colorbar(
        iteration_scatter,
        ax=axes[1],
        shrink=0.55,
        pad=0.13,
    )
    available = np.unique(shown["iteration"])
    desired = np.asarray((0, 1, 4, 16, 64, 256, 1024))
    ticks = desired[np.isin(desired, available)]
    iteration_bar.set_ticks(
        np.log2(ticks + 1),
        labels=[str(value) for value in ticks],
    )
    iteration_bar.set_label("iteration")

    confidence = axes[2].scatter(
        state_helix[:, 0],
        state_helix[:, 1],
        state_helix[:, 2],
        c=shown["confidence"],
        cmap="cividis",
        vmin=0,
        vmax=1,
        s=7,
        alpha=0.5,
        edgecolors="none",
        rasterized=True,
    )
    _draw_ideal_helix(axes[2])
    axes[2].set_title(
        "unit states · target-imposed helix readout · confidence"
    )
    _style_3d(
        axes[2],
        ("cos readout", "sin readout", "digit-axis readout"),
        np.concatenate((state_helix, helix_code().numpy()), axis=0),
    )
    figure.colorbar(confidence, ax=axes[2], shrink=0.55, pad=0.08)

    limit = max(1e-5, float(np.quantile(np.abs(shown["target_margin"]), 0.98)))
    margin = axes[3].scatter(
        update_helix[:, 0],
        update_helix[:, 1],
        update_helix[:, 2],
        c=shown["target_margin"],
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        s=7,
        alpha=0.5,
        edgecolors="none",
        rasterized=True,
    )
    _draw_ideal_helix(axes[3])
    axes[3].set_title(
        "unit updates · target-imposed helix readout · target margin"
    )
    _style_3d(
        axes[3],
        ("cos readout", "sin readout", "digit-axis readout"),
        np.concatenate((update_helix, helix_code().numpy()), axis=0),
    )
    figure.colorbar(margin, ax=axes[3], shrink=0.55, pad=0.08)
    _save(figure, path)


def _trajectory_groups(metadata):
    groups = {}
    puzzles = metadata["puzzle"].tolist()
    cells = metadata["cell"].tolist()
    for index, key in enumerate(zip(puzzles, cells)):
        groups.setdefault(key, []).append(index)
    ranked = []
    for key, indices in groups.items():
        order = sorted(indices, key=lambda index: int(metadata["iteration"][index]))
        predictions = metadata["predicted_digit"][order]
        changes = int((predictions[1:] != predictions[:-1]).sum().item())
        final_correct = bool(metadata["correct"][order[-1]].item())
        minimum_margin = float(metadata["target_margin"][order].min().item())
        ranked.append((not final_correct, changes, -minimum_margin, key, order))
    ranked.sort(reverse=True)
    selected = []
    used_puzzles = set()
    for item in ranked:
        puzzle = item[3][0]
        if puzzle not in used_puzzles or len(selected) >= 3:
            selected.append(item)
            used_puzzles.add(puzzle)
        if len(selected) == 6:
            break
    return selected


def render_selected_trajectories(
    model_name: str,
    metadata: dict[str, torch.Tensor],
    state_coordinates: dict,
    path: str,
):
    selected = _trajectory_groups(metadata)
    figure = plt.figure(figsize=(14, 11.0), constrained_layout=True)
    figure.get_layout_engine().set(
        h_pad=0.18,
        hspace=0.16,
        rect=(0.0, 0.055, 1.0, 0.93),
    )
    figure.suptitle(
        f"{model_name} · selected held-out paths in train-fit, target-imposed helix coordinates",
        fontsize=14,
    )
    helix_points = state_coordinates["helix_test"].numpy()
    for panel, item in enumerate(selected):
        _, changes, _, (puzzle, cell), indices = item
        axis = figure.add_subplot(2, 3, panel + 1, projection="3d")
        points = helix_points[indices]
        predictions = metadata["predicted_digit"][indices].numpy()
        confidence = metadata["confidence"][indices].numpy()
        iterations = metadata["iteration"][indices].numpy()
        true_digit = int(metadata["true_digit"][indices[0]].item()) + 1
        axis.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="#6f7780",
            lw=1.25,
            alpha=0.75,
        )
        for digit in range(9):
            mask = predictions == digit
            axis.scatter(
                points[mask, 0],
                points[mask, 1],
                points[mask, 2],
                color=DIGIT_PALETTE[digit],
                s=18 + 34 * confidence[mask],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.92,
            )
        axis.scatter(
            points[0, 0],
            points[0, 1],
            points[0, 2],
            marker="s",
            s=48,
            facecolor="none",
            edgecolor="#16191c",
            linewidth=1.1,
            zorder=5,
        )
        axis.scatter(
            points[-1, 0],
            points[-1, 1],
            points[-1, 2],
            marker="X",
            s=52,
            color="#16191c",
            edgecolor="white",
            linewidth=0.5,
            zorder=5,
        )
        annotate = {0, 16, 128, int(iterations[-1])}
        for point, iteration in zip(points, iterations):
            if int(iteration) in annotate:
                axis.text(
                    point[0],
                    point[1],
                    point[2],
                    f"t{iteration}",
                    fontsize=6,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "none",
                        "alpha": 0.68,
                        "pad": 0.4,
                    },
                )
        _draw_ideal_helix(axis)
        row, column = divmod(cell, 9)
        axis.set_title(
            f"puzzle {puzzle} · cell ({row + 1},{column + 1}) · "
            f"true {true_digit} · {changes} prediction changes",
            fontsize=9,
        )
        _style_3d(
            axis,
            ("cos", "sin", "digit axis"),
            np.concatenate((points, helix_code().numpy()), axis=0),
        )
    figure.text(
        0.5,
        -0.025,
        "point hue = predicted digit · point area = confidence · □ first snapshot · ✕ last snapshot · gray curve = supervised target template",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    _save(figure, path)


def render_model_comparison(model_metrics: dict, path: str):
    names = list(model_metrics)
    x = np.arange(len(names))
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.8), constrained_layout=True)
    figure.suptitle(
        "Held-out geometry comparison · blank cells only",
        fontsize=15,
    )
    for row, representation in enumerate(("unit_state", "unit_update")):
        pca_capture = []
        centroid_top3 = []
        natural_cycle = []
        periodic_accuracy = []
        periodic_without_decoder = []
        no_decoder_name = f"{representation}_no_decoder"
        for name in names:
            metrics = model_metrics[name]["representations"][representation]
            removed = model_metrics[name]["representations"][no_decoder_name]
            pca_capture.append(metrics["pca"]["test_top3_captured"])
            centroid_top3.append(
                metrics["intrinsic_digit_geometry"]["test_centroid_spectrum"][
                    "top3_fraction"
                ]
            )
            natural_cycle.append(
                metrics["intrinsic_digit_geometry"][
                    "natural_cycle_shorter_percentile"
                ]
            )
            periodic_accuracy.append(
                metrics["periodic_readout"]["test_natural_cycle"][
                    "sector_accuracy"
                ]
            )
            periodic_without_decoder.append(
                removed["periodic_readout"]["test_natural_cycle"][
                    "sector_accuracy"
                ]
            )
        values_and_titles = (
            (
                pca_capture,
                "held-out energy in train PC1–3",
                "fraction about train mean",
                None,
            ),
            (centroid_top3, "digit-centroid top-3 variance", "fraction", None),
            (
                natural_cycle,
                "natural-cycle path percentile",
                "lower is more natural",
                0.05,
            ),
        )
        for column, (values, title, ylabel, reference) in enumerate(values_and_titles):
            axis = axes[row, column]
            axis.bar(x, values, color=MODEL_PALETTE[: len(names)])
            axis.set_ylim(0, 1.02)
            axis.set_title(title)
            axis.set_ylabel(ylabel)
            if reference is not None:
                axis.axhline(reference, color="#a52b3a", ls="--", lw=1)
            axis.grid(axis="y", alpha=0.25)
        axis = axes[row, 3]
        width = 0.28
        axis.bar(
            x - width / 2,
            periodic_accuracy,
            width=width,
            color="#3d79a8",
            label="full features",
        )
        axis.bar(
            x + width / 2,
            periodic_without_decoder,
            width=width,
            color="#a9b4bf",
            label="output-head span removed",
        )
        axis.set_ylim(0, 1.02)
        axis.set_title("natural-cycle periodic readout")
        axis.set_ylabel("held-out sector accuracy")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False, fontsize=7, loc="lower right")
        axes[row, 0].text(
            -0.28,
            0.5,
            representation.replace("_", " "),
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )
        for axis in axes[row]:
            axis.set_xticks(x, [name.replace("_", "\n") for name in names], fontsize=8)
    _save(figure, path)
