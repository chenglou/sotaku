"""Render the compact figures for the number-geometry statistics."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from looping.trajectory_viz.helix_tests.statistics.analyze_number_helix import (
    DIGIT_LABELS,
    MODEL_DISPLAY_NAMES,
    OUTPUT_NULL_REPRESENTATION,
    PRIMARY_REPRESENTATION,
    REPRESENTATIONS,
    SNAPSHOT_ITERATIONS,
    UPDATE_REPRESENTATION,
)


MODEL_COLORS = {
    "stable_plain": "#0072B2",
    "collapsed_plain": "#D55E00",
    "late_state_ce": "#009E73",
    "combined_margin": "#CC79A7",
}
FIT_COLORS = {
    "linear": "#0072B2",
    "cyclic": "#E69F00",
    "helix": "#009E73",
    "categorical": "#5B5B5B",
}
REPRESENTATION_STYLES = {
    PRIMARY_REPRESENTATION: ("cell-centered state", "o", "-"),
    OUTPUT_NULL_REPRESENTATION: ("outside output head", "s", "--"),
    UPDATE_REPRESENTATION: ("cell-centered update", "^", ":"),
}


def _axes_grid(row_count, column_count, *, width=5.0, height=2.8):
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(width * column_count, height * row_count),
        squeeze=False,
        constrained_layout=True,
    )
    return figure, axes


def _save(figure, output_dir, filename):
    path = os.path.join(output_dir, filename)
    figure.savefig(path, dpi=180, facecolor="white")
    plt.close(figure)
    return filename


def _iteration_x(iterations):
    return np.log2(np.asarray(iterations, dtype=np.float64) + 1.0)


def _set_iteration_axis(axis):
    ticks = np.asarray((0, 1, 4, 16, 64, 256, 1024))
    axis.set_xticks(_iteration_x(ticks), [str(value) for value in ticks])
    axis.set_xlabel("recurrent iteration (0 = encoder state)")


def plot_heldout_fit_summary(summary, output_dir):
    model_names = list(summary["models"])
    fit_names = ("linear", "cyclic", "helix", "categorical")
    figure, axes = _axes_grid(len(model_names), len(DIGIT_LABELS), width=5.5)
    x = np.arange(len(fit_names), dtype=np.float64)
    offsets = np.linspace(-0.18, 0.18, len(REPRESENTATIONS))
    domain_values = [0.0]
    for model in summary["models"].values():
        for representation in REPRESENTATIONS:
            for label_name in DIGIT_LABELS:
                fit = model["pooled"][representation][label_name]
                for fit_name in fit_names:
                    value = fit["partial_r2"][f"{fit_name}_over_base"]
                    domain_values.append(value)
                    interval = fit.get("bootstrap", {}).get("intervals", {}).get(
                        f"partial_r2.{fit_name}_over_base"
                    )
                    if interval:
                        domain_values.extend((interval["lower"], interval["upper"]))
    finite_domain = np.asarray(domain_values, dtype=np.float64)
    finite_domain = finite_domain[np.isfinite(finite_domain)]
    domain_span = max(0.05, float(finite_domain.max() - finite_domain.min()))
    y_limits = (
        float(finite_domain.min() - 0.06 * domain_span),
        float(finite_domain.max() + 0.06 * domain_span),
    )
    for row, model_name in enumerate(model_names):
        model = summary["models"][model_name]
        for column, label_name in enumerate(DIGIT_LABELS):
            axis = axes[row, column]
            for representation_index, representation in enumerate(REPRESENTATIONS):
                fit = model["pooled"][representation][label_name]
                values = [
                    fit["partial_r2"][f"{name}_over_base"]
                    for name in fit_names
                ]
                display, marker, _ = REPRESENTATION_STYLES[representation]
                lower = []
                upper = []
                for fit_name, value in zip(fit_names, values):
                    interval = fit.get("bootstrap", {}).get("intervals", {}).get(
                        f"partial_r2.{fit_name}_over_base"
                    )
                    if interval and np.isfinite(interval["lower"]):
                        lower.append(max(0.0, value - interval["lower"]))
                        upper.append(max(0.0, interval["upper"] - value))
                    else:
                        lower.append(0.0)
                        upper.append(0.0)
                axis.errorbar(
                    x + offsets[representation_index],
                    values,
                    yerr=np.asarray((lower, upper)),
                    marker=marker,
                    linestyle="none",
                    capsize=2,
                    linewidth=1.0,
                    markersize=4.5,
                    label=display,
                )
            axis.axhline(0, color="#777777", linewidth=0.7)
            axis.set_xticks(x, fit_names, rotation=20)
            axis.set_ylabel("held-out partial $R^2$")
            axis.set_title(
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} · "
                f"{label_name.replace('_', ' ')}"
            )
            axis.grid(axis="y", alpha=0.2)
            axis.set_ylim(*y_limits)
            if row == 0 and column == 0:
                axis.legend(
                    fontsize=8,
                    frameon=False,
                    title="conditional 95% intervals",
                    title_fontsize=7,
                )
    figure.suptitle(
        "Held-out digit fits after iteration and exact-position controls"
    )
    return _save(figure, output_dir, "heldout_fit_summary.png")


def plot_iteration_profiles(summary, output_dir):
    model_names = list(summary["models"])
    fit_names = ("linear", "cyclic", "helix", "categorical")
    figure, axes = _axes_grid(
        len(model_names), len(DIGIT_LABELS), width=5.5, height=3.1
    )
    iterations = np.asarray(SNAPSHOT_ITERATIONS)
    iteration_x = _iteration_x(iterations)
    for row, model_name in enumerate(model_names):
        by_iteration = summary["models"][model_name]["by_iteration"]
        for column, label_name in enumerate(DIGIT_LABELS):
            axis = axes[row, column]
            for fit_name in fit_names:
                values = [
                    by_iteration[str(iteration)][label_name]["partial_r2"]
                    [f"{fit_name}_over_base"]
                    for iteration in iterations
                ]
                axis.plot(
                    iteration_x,
                    values,
                    marker="o",
                    markersize=3,
                    linewidth=1.2,
                    color=FIT_COLORS[fit_name],
                    label=fit_name,
                )
            axis.axhline(0, color="#777777", linewidth=0.7)
            _set_iteration_axis(axis)
            if row != len(model_names) - 1:
                axis.set_xlabel("")
            axis.set_ylabel("held-out partial $R^2$")
            axis.grid(alpha=0.2)
            axis.set_title(
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} · "
                f"{label_name.replace('_', ' ')}"
            )
            if row == 0 and column == 0:
                axis.legend(ncol=2, fontsize=8, frameon=False)
    figure.suptitle("Per-iteration digit fits on cell-centered state directions")
    return _save(figure, output_dir, "iteration_profiles.png")


def plot_permutation_nulls(summary, null_store, output_dir, representation):
    model_names = list(summary["models"])
    figure, axes = _axes_grid(len(model_names), len(DIGIT_LABELS), width=5.4)
    for row, model_name in enumerate(model_names):
        for column, label_name in enumerate(DIGIT_LABELS):
            axis = axes[row, column]
            fit = summary["models"][model_name]["pooled"][representation][label_name]
            test = fit["natural_order_permutation"]["helix_partial_r2"]
            key = (
                f"order__{model_name}__{representation}__{label_name}__"
                "helix_partial_r2"
            )
            null = np.asarray(null_store[key])
            axis.hist(null, bins=35, color="#B8BDC6", edgecolor="none")
            axis.axvline(
                test["observed"], color="#D55E00", linewidth=2, label="natural order"
            )
            adjusted_p = test.get("p_value_holm")
            adjusted_text = (
                "" if adjusted_p is None else f", Holm p={adjusted_p:.3g}"
            )
            axis.text(
                0.02,
                0.96,
                f"p={test['p_value']:.3g}{adjusted_text}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
            )
            axis.set_xlabel("held-out helix partial $R^2$")
            axis.set_ylabel("count")
            axis.set_title(
                f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} · "
                f"{label_name.replace('_', ' ')}"
            )
            if row == 0 and column == 0:
                axis.legend(fontsize=8, frameon=False)
    display = REPRESENTATION_STYLES[representation][0]
    figure.suptitle(f"Is natural digit order exceptional? · {display}")
    filename = (
        "natural_order_permutations.png"
        if representation == PRIMARY_REPRESENTATION
        else "output_null_order_permutations.png"
    )
    return _save(figure, output_dir, filename)


def plot_nuisance_controls(summary, output_dir):
    model_names = list(summary["models"])
    contributions = (
        ("iteration_over_context", "iteration"),
        ("cell_position_over_iteration", "cell position"),
        (
            "confidence_decision_margin_over_position",
            "confidence + decision-margin bins",
        ),
        (
            "target_probability_margin_over_decision",
            "true-probability + target-margin bins",
        ),
    )
    representations = (PRIMARY_REPRESENTATION, OUTPUT_NULL_REPRESENTATION)
    figure, axes = _axes_grid(1, 2, width=6.2, height=4.0)
    group_x = np.arange(len(model_names))
    width = 0.19
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    for column, representation in enumerate(representations):
        axis = axes[0, column]
        for index, ((key, label), color) in enumerate(zip(contributions, colors)):
            values = [
                summary["models"][model]["control_contributions"][representation]
                ["partial_r2"][key]
                for model in model_names
            ]
            axis.bar(
                group_x + (index - 1.5) * width,
                values,
                width=width,
                color=color,
                label=label,
            )
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_xticks(
            group_x,
            [MODEL_DISPLAY_NAMES.get(name, name) for name in model_names],
            rotation=20,
            ha="right",
        )
        axis.set_ylabel("incremental held-out partial $R^2$")
        axis.set_title(REPRESENTATION_STYLES[representation][0])
        axis.grid(axis="y", alpha=0.2)
        if column == 0:
            axis.legend(fontsize=8, frameon=False)
    figure.suptitle(
        "Sequential held-out contributions: context → iteration → position → "
        "fixed-bin certainty"
    )
    return _save(figure, output_dir, "nuisance_controls.png")


def plot_confidence_margin_trajectories(summary, output_dir):
    model_names = list(summary["models"])
    figure, axes = _axes_grid(len(model_names), 2, width=5.8, height=2.8)
    iterations = np.asarray(SNAPSHOT_ITERATIONS)
    iteration_x = _iteration_x(iterations)
    for row, model_name in enumerate(model_names):
        statistics = summary["models"][model_name]["sample_statistics"]
        correct = np.asarray(
            [statistics[str(value)]["correct_fraction"] for value in iterations]
        )
        confidence = np.asarray(
            [statistics[str(value)]["mean_max_confidence"] for value in iterations]
        )
        true_probability = np.asarray(
            [statistics[str(value)]["mean_true_probability"] for value in iterations]
        )
        target_margin_p10 = np.asarray(
            [statistics[str(value)]["target_margin_p10"] for value in iterations]
        )
        decision_margin_p10 = np.asarray(
            [statistics[str(value)]["decision_margin_p10"] for value in iterations]
        )
        axis = axes[row, 0]
        axis.plot(
            iteration_x, correct, color="#0072B2", marker="o", markersize=3,
            label="correct originally blank cells",
        )
        axis.plot(
            iteration_x, confidence, color="#009E73", marker="s", markersize=3,
            label="mean max confidence",
        )
        axis.plot(
            iteration_x,
            true_probability,
            color="#CC79A7",
            marker="d",
            markersize=3,
            label="mean true probability",
        )
        axis.set_ylim(-0.03, 1.03)
        axis.set_ylabel("fraction / probability")
        _set_iteration_axis(axis)
        axis.grid(alpha=0.2)

        margin_axis = axes[row, 1]
        margin_axis.plot(
            iteration_x,
            target_margin_p10,
            color="#D55E00",
            marker="^",
            markersize=3,
            label="target margin: mean puzzle p10",
        )
        margin_axis.plot(
            iteration_x,
            decision_margin_p10,
            color="#E69F00",
            marker="v",
            markersize=3,
            label="decision margin: mean puzzle p10",
        )
        margin_axis.axhline(0, color="#777777", linewidth=0.7)
        margin_axis.set_yscale("symlog", linthresh=1.0)
        margin_axis.set_ylabel("logit margin")
        _set_iteration_axis(margin_axis)
        margin_axis.grid(alpha=0.2)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        axis.set_title(f"{display_name} · accuracy and confidence")
        margin_axis.set_title(f"{display_name} · weakest-tail margins")
        if row == 0:
            axis.legend(fontsize=7, frameon=False)
            margin_axis.legend(fontsize=7, frameon=False)
    figure.suptitle("Prediction accuracy, confidence, and logit margins")
    return _save(figure, output_dir, "confidence_margin_trajectories.png")


def plot_parameter_geometry(summary, null_store, output_dir):
    model_names = list(summary["models"])
    sources = (
        "input_encoder_digit_columns",
        "prediction_feedback_columns",
        "output_head_rows",
    )
    source_titles = ("input digit columns", "prediction-feedback columns", "output-head rows")
    fit_names = ("linear", "cyclic", "helix")
    figure, axes = _axes_grid(len(sources), 2, width=6.0, height=3.1)
    x = np.arange(len(model_names), dtype=np.float64)
    offsets = np.linspace(-0.18, 0.18, len(fit_names))
    for row, (source, source_title) in enumerate(zip(sources, source_titles)):
        raw_axis, centered_axis = axes[row]
        for fit_index, fit_name in enumerate(fit_names):
            observed = []
            null_mean = []
            centered = []
            for model_name in model_names:
                fit = (
                    summary["models"][model_name]["parameter_geometry"]
                    ["sources"][source]["fits"][fit_name]
                )
                observed.append(fit["r2"])
                key = f"parameter__{model_name}__{source}__{fit_name}"
                permutation_mean = float(np.mean(null_store[key]))
                null_mean.append(permutation_mean)
                centered.append(fit["r2"] - permutation_mean)
            point_x = x + offsets[fit_index]
            raw_axis.scatter(
                point_x,
                observed,
                marker="o",
                color=FIT_COLORS[fit_name],
                label=fit_name,
            )
            raw_axis.scatter(
                point_x,
                null_mean,
                marker="_",
                s=90,
                linewidths=1.5,
                color=FIT_COLORS[fit_name],
            )
            centered_axis.scatter(
                point_x,
                centered,
                marker="o",
                color=FIT_COLORS[fit_name],
                label=fit_name,
            )
        labels = [MODEL_DISPLAY_NAMES.get(name, name) for name in model_names]
        for axis in (raw_axis, centered_axis):
            axis.set_xticks(x, labels, rotation=20, ha="right")
            axis.grid(axis="y", alpha=0.2)
        raw_axis.set_ylabel("between-digit $R^2$")
        raw_axis.set_ylim(bottom=0)
        centered_axis.set_ylabel("observed $R^2$ − permutation mean")
        centered_axis.axhline(0, color="#777777", linewidth=0.7)
        raw_axis.set_title(f"{source_title}: raw fit")
        centered_axis.set_title(f"{source_title}: order-specific excess")
        if row == 0:
            raw_axis.legend(ncol=3, fontsize=8, frameon=False)
    figure.suptitle(
        "Learned parameter geometry (round markers observed; short ticks permutation mean)"
    )
    return _save(figure, output_dir, "parameter_geometry.png")


def plot_geometric_adequacy(summary, output_dir):
    figure, axes = _axes_grid(1, 2, width=6.0, height=4.4)
    for column, label_name in enumerate(DIGIT_LABELS):
        axis = axes[0, column]
        for model_name, model in summary["models"].items():
            for representation in (
                PRIMARY_REPRESENTATION,
                OUTPUT_NULL_REPRESENTATION,
            ):
                fit = model["pooled"][representation][label_name]
                descriptors = fit["coefficient_geometry"]["fold_descriptors"]
                circularity = float(
                    np.mean([item["cyclic_singular_value_ratio"] for item in descriptors])
                )
                marker = REPRESENTATION_STYLES[representation][1]
                axis.scatter(
                    fit["helix_categorical_gain_fraction"],
                    circularity,
                    marker=marker,
                    s=55,
                    color=MODEL_COLORS.get(model_name, "#444444"),
                    label=(
                        f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} · "
                        f"{REPRESENTATION_STYLES[representation][0]}"
                    ),
                )
        axis.axhline(1, color="#777777", linewidth=0.7, linestyle="--")
        axis.axvline(
            3.0 / 8.0,
            color="#777777",
            linewidth=0.7,
            linestyle=":",
            label="rank-only 3/8 reference" if column == 0 else None,
        )
        axis.scatter([1.0], [1.0], marker="*", s=70, color="#777777")
        axis.set_xlabel("helix gain / categorical digit gain")
        axis.set_ylabel("cyclic coefficient $s_2/s_1$ (1 = circle)")
        axis.set_xlim(-0.03, 1.03)
        axis.set_ylim(-0.03, 1.03)
        axis.set_title(label_name.replace("_", " "))
        axis.grid(alpha=0.2)
        if column == 0:
            axis.legend(fontsize=7, frameon=False, ncol=2)
    figure.suptitle(
        "Descriptive adequacy; digit-order permutations determine evidence"
    )
    return _save(figure, output_dir, "geometric_adequacy.png")


def plot_natural_order_test_matrix(summary, output_dir):
    statistics = (
        "linear_partial_r2",
        "cyclic_partial_r2",
        "helix_partial_r2",
        "cyclic_given_linear_partial_r2",
        "linear_given_cyclic_partial_r2",
    )
    column_labels = (
        "linear",
        "cyclic",
        "helix",
        "cyclic | linear",
        "linear | cyclic",
    )
    rows = []
    row_labels = []
    for representation in (PRIMARY_REPRESENTATION, OUTPUT_NULL_REPRESENTATION):
        representation_label = REPRESENTATION_STYLES[representation][0]
        for model_name, model in summary["models"].items():
            for label_name in DIGIT_LABELS:
                tests = model["pooled"][representation][label_name]
                tests = tests["natural_order_permutation"]
                rows.append([
                    tests[name].get("p_value_holm", tests[name]["p_value"])
                    for name in statistics
                ])
                row_labels.append(
                    f"{representation_label} · "
                    f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} · "
                    f"{label_name.replace('_', ' ')}"
                )
    q_values = np.asarray(rows, dtype=np.float64)
    color_values = -np.log10(np.clip(q_values, 1e-4, 1.0))
    figure, axes = _axes_grid(1, 1, width=8.5, height=8.0)
    axis = axes[0, 0]
    image = axis.imshow(color_values, aspect="auto", cmap="viridis", vmin=0, vmax=4)
    axis.set_xticks(np.arange(len(column_labels)), column_labels, rotation=20, ha="right")
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    for row in range(len(row_labels)):
        for column in range(len(column_labels)):
            value = q_values[row, column]
            text_color = "white" if color_values[row, column] < 1.7 else "black"
            axis.text(
                column,
                row,
                f"{value:.3g}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
            )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.75)
    colorbar.set_label(r"$-\log_{10}$(Holm-adjusted p)")
    axis.set_title("Natural digit-order tests (cell values are Holm-adjusted p)")
    return _save(figure, output_dir, "natural_order_test_matrix.png")


def plot_certainty_sensitivity(summary, output_dir):
    model_names = list(summary["models"])
    design_names = ("position", "decision_adjusted", "target_adjusted")
    design_labels = (
        "position-controlled primary",
        "+ confidence/decision-margin bins",
        "+ target-informed bins",
    )
    colors = ("#0072B2", "#009E73", "#CC79A7")
    x = np.arange(len(model_names), dtype=np.float64)
    offsets = np.linspace(-0.18, 0.18, len(design_names))
    figure, axes = _axes_grid(1, len(DIGIT_LABELS), width=6.0, height=4.0)
    for column, label_name in enumerate(DIGIT_LABELS):
        axis = axes[0, column]
        for design_index, (design_name, design_label, color) in enumerate(
            zip(design_names, design_labels, colors)
        ):
            values = []
            for model_name in model_names:
                model = summary["models"][model_name]
                if design_name == "position":
                    fit = model["pooled"][PRIMARY_REPRESENTATION][label_name]
                else:
                    fit = model["certainty_sensitivity"][PRIMARY_REPRESENTATION]
                    fit = fit[label_name][design_name]
                values.append(fit["partial_r2"]["helix_over_base"])
            axis.scatter(
                x + offsets[design_index],
                values,
                color=color,
                marker="o",
                label=design_label,
            )
        axis.axhline(0, color="#777777", linewidth=0.7)
        axis.set_xticks(
            x,
            [MODEL_DISPLAY_NAMES.get(name, name) for name in model_names],
            rotation=20,
            ha="right",
        )
        axis.set_ylabel("held-out helix partial $R^2$")
        axis.set_title(label_name.replace("_", " "))
        axis.grid(axis="y", alpha=0.2)
        if column == 0:
            axis.legend(fontsize=7, frameon=False, loc="center left")
    figure.suptitle("Sensitivity to fixed-bin confidence and margin adjustment")
    return _save(figure, output_dir, "certainty_sensitivity.png")


def plot_response_scales(summary, output_dir):
    model_names = list(summary["models"])
    iterations = np.asarray(SNAPSHOT_ITERATIONS)
    iteration_x = _iteration_x(iterations)
    figure, axes = _axes_grid(len(model_names), 2, width=5.8, height=2.8)
    for row, model_name in enumerate(model_names):
        scale = summary["models"][model_name]["response_scale_statistics"]

        def series(name):
            return np.asarray([scale[str(value)][name] for value in iterations])

        norm_axis = axes[row, 0]
        for name, label, color, marker in (
            ("cell_centered_hidden_rms_norm", "cell-centered state", "#0072B2", "o"),
            ("output_null_hidden_rms_norm", "outside output head", "#E69F00", "s"),
            ("cell_centered_update_rms_norm", "one-step update", "#009E73", "^"),
        ):
            values = series(name)
            norm_axis.plot(
                iteration_x,
                np.maximum(values, np.finfo(float).tiny),
                color=color,
                marker=marker,
                markersize=3,
                label=label,
            )
        norm_axis.set_yscale("log")
        norm_axis.set_ylabel("RMS vector norm")
        _set_iteration_axis(norm_axis)
        norm_axis.grid(alpha=0.2)

        fraction_axis = axes[row, 1]
        fraction_axis.plot(
            iteration_x,
            series("output_null_energy_fraction"),
            color="#D55E00",
            marker="s",
            markersize=3,
            label="state energy outside output head",
        )
        fraction_axis.plot(
            iteration_x,
            series("update_near_zero_fraction_at_1e-8"),
            color="#CC79A7",
            marker="x",
            markersize=4,
            label="updates with norm ≤ 1e-8",
        )
        fraction_axis.set_ylim(-0.03, 1.03)
        fraction_axis.set_ylabel("energy / cell fraction")
        _set_iteration_axis(fraction_axis)
        fraction_axis.grid(alpha=0.2)
        display_name = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        norm_axis.set_title(f"{display_name} · pre-normalization norms")
        fraction_axis.set_title(f"{display_name} · normalization checks")
        if row == 0:
            norm_axis.legend(fontsize=7, frameon=False)
            fraction_axis.legend(fontsize=7, frameon=False)
    figure.suptitle("Scale checks for direction-normalized responses")
    return _save(figure, output_dir, "response_scales.png")


def render_all_plots(summary, null_store, output_dir):
    """Render every requested figure and return their filenames."""

    return [
        plot_heldout_fit_summary(summary, output_dir),
        plot_iteration_profiles(summary, output_dir),
        plot_permutation_nulls(
            summary, null_store, output_dir, PRIMARY_REPRESENTATION
        ),
        plot_permutation_nulls(
            summary, null_store, output_dir, OUTPUT_NULL_REPRESENTATION
        ),
        plot_nuisance_controls(summary, output_dir),
        plot_confidence_margin_trajectories(summary, output_dir),
        plot_parameter_geometry(summary, null_store, output_dir),
        plot_geometric_adequacy(summary, output_dir),
        plot_natural_order_test_matrix(summary, output_dir),
        plot_certainty_sensitivity(summary, output_dir),
        plot_response_scales(summary, output_dir),
    ]
