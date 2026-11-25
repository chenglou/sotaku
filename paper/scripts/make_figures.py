"""Build the paper's figures from downloaded results.

    python -m paper.scripts.make_figures

Inputs are read from `runs/paper/` (downloaded from the `cheng-sotaku-outputs` Modal volume,
see `paper/README.md`) and from result files already checked into the repository. A figure
whose inputs are missing is skipped with a message.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs" / "paper"
FIGURES = ROOT / "paper" / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
VIOLET, MAGENTA, RED = "#6250d6", "#e87ba4", "#e34948"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#ffffff"
NEUTRAL = "#b9b8b2"
# One light-to-dark blue ramp for the five rating buckets, easiest to hardest.
BUCKET_RAMP = ["#a9c9f0", "#7fb0ea", "#5596e1", "#2a78d6", "#134f9c"]
BUCKETS = ["0", "1-2", "3-10", "11-50", "51+"]

CHECKPOINTS = {
    "v2": ("Later-iteration training (released)", BLUE),
    "v1": ("Original recipe, accurate run", AQUA),
    "lr25e4": ("Original recipe, LR $2.5\\times10^{-3}$", ORANGE),
    "lr3e3": ("Original recipe, LR $3\\times10^{-3}$", VIOLET),
}

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "font.family": "DejaVu Sans",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 1.6, "pdf.fonttype": 42, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "legend.frameon": False,
})


def save(figure, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / f"{name}.pdf")
    figure.savefig(FIGURES / f"{name}.png", dpi=200)
    plt.close(figure)
    print(f"wrote {name}")


def load_trajectory(name):
    path = DATA / "analysis" / name / "trajectory.npz"
    return np.load(path) if path.exists() else None


def log2_axis(axis, largest=4096, smallest=1, sparse=False):
    axis.set_xscale("log", base=2)
    candidates = (1, 16, 256, 4096) if sparse else (1, 4, 16, 64, 256, 1024, 4096)
    ticks = [value for value in candidates if smallest <= value <= largest]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(value) for value in ticks])
    axis.set_xlim(smallest, largest)
    axis.minorticks_off()


def figure_iteration_scaling():
    """Accuracy against the number of inference iterations."""
    trajectories = {name: load_trajectory(name) for name in CHECKPOINTS}
    result_path = DATA / "evaluations" / "v2_fp32" / "result.json"
    if any(value is None for value in trajectories.values()) or not result_path.exists():
        return print("skip iteration_scaling: missing analysis or evaluation")
    figure, (left, right) = plt.subplots(1, 2, figsize=(6.6, 2.5), gridspec_kw={"wspace": 0.24})
    iterations = np.arange(1, 4097)
    for name, (label, color) in CHECKPOINTS.items():
        left.plot(iterations, 100 * trajectories[name]["solved"].mean(1), color=color, label=label)
    left.axvline(16, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)))
    left.text(19, 9, "loss covers\niterations 1-16", color=MUTED, fontsize=7, va="center", ha="left")
    log2_axis(left)
    left.set_ylim(0, 102)
    left.set_yticks([0, 25, 50, 75, 100])
    left.set_xlabel("Inference iterations")
    left.set_ylabel("Puzzles solved (%)")
    left.set_title("Four checkpoints, same 1,000 puzzles", loc="left")
    left.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1, handlelength=1.6)

    scores = json.loads(result_path.read_text())["scores"]
    horizons = sorted(int(key) for key in scores)
    for bucket, color in zip(BUCKETS, BUCKET_RAMP):
        values = [100 * scores[str(h)]["buckets"][bucket]["solved"] / scores[str(h)]["buckets"][bucket]["total"]
                  for h in horizons]
        right.plot(horizons, values, color=color, marker="o", markersize=2.6, label=f"rating {bucket}")
    right.plot(horizons, [scores[str(h)]["accuracy_percent"] for h in horizons], color=INK,
               linewidth=1.2, linestyle=(0, (4, 2)), label="all 25,000")
    log2_axis(right, smallest=4)
    right.set_ylim(40, 101)
    right.set_xlabel("Inference iterations")
    right.set_title("Released checkpoint by difficulty rating", loc="left")
    right.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.6,
                 columnspacing=1.4)
    save(figure, "iteration_scaling")


def figure_state_dynamics():
    trajectories = {name: load_trajectory(name) for name in CHECKPOINTS}
    if any(value is None for value in trajectories.values()):
        return print("skip state_dynamics: missing analysis")
    figure, axes = plt.subplots(1, 3, figsize=(6.6, 2.25), gridspec_kw={"wspace": 0.42})
    iterations = np.arange(1, 4097)
    for name, (label, color) in CHECKPOINTS.items():
        data = trajectories[name]
        axes[0].plot(iterations, np.median(data["state_rms"], axis=1), color=color, label=label)
        axes[1].plot(iterations, np.median(data["update_rms"], axis=1), color=color)
        axes[2].plot(iterations, np.percentile(data["min_margin"], 10, axis=1), color=color)
    for axis in axes:
        log2_axis(axis, sparse=True)
        axis.set_xlabel("Iteration")
    axes[0].set_yscale("log")
    axes[0].set_title("Hidden-state RMS", loc="left")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.2, 6)
    axes[1].set_yticks([0.25, 0.5, 1, 2, 4])
    axes[1].set_yticklabels(["0.25", "0.5", "1", "2", "4"])
    axes[1].minorticks_off()
    axes[1].set_title("RMS of one iteration's update", loc="left")
    axes[2].set_yscale("symlog", linthresh=10)
    axes[2].axhline(0, color=MUTED, linewidth=0.8)
    axes[2].set_yticks([-1000, -100, -10, 0, 10, 100, 1000])
    axes[2].set_title("Weakest correct-answer margin\n(10th percentile of puzzles)", loc="left")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
                  bbox_to_anchor=(0.5, -0.3), handlelength=1.6, columnspacing=1.6)
    save(figure, "state_dynamics")


def draw_board(axis, puzzle, answer, predictions, title):
    axis.set_xlim(0, 9)
    axis.set_ylim(9, 0)
    axis.set_aspect("equal")
    axis.axis("off")
    wrong_count = 0
    for cell in range(81):
        row, column = divmod(cell, 9)
        given = puzzle[cell] != "."
        digit = puzzle[cell] if given else str(int(predictions[cell]) + 1)
        wrong = (not given) and digit != answer[cell]
        wrong_count += wrong
        face = "#ecebe6" if given else (ORANGE if wrong else SURFACE)
        axis.add_patch(Rectangle((column, row), 1, 1, facecolor=face, edgecolor="none"))
        axis.text(column + 0.5, row + 0.54, digit, ha="center", va="center", fontsize=7,
                  color=MUTED if given else (SURFACE if wrong else INK),
                  fontweight="bold" if wrong else "normal")
    for line in range(10):
        width = 1.1 if line % 3 == 0 else 0.25
        axis.plot([0, 9], [line, line], color=INK, linewidth=width)
        axis.plot([line, line], [0, 9], color=INK, linewidth=width)
    axis.set_title(f"{title}: {wrong_count} wrong", fontsize=7.5)


def figure_board_evolution(example=None):
    data = load_trajectory("v2")
    if data is None:
        return print("skip board_evolution: missing analysis")
    solved = data["solved"][:1024, data["example_positions"]]
    first = np.where(solved.any(0), solved.argmax(0) + 1, 0)
    if example is None:
        example = int(np.argmax(np.where(data["buckets"][data["example_positions"]] == "51+", first, -1)))
    puzzle = str(data["example_questions"][example])
    answer = str(data["example_answers"][example])
    shown = [1, 2, 4, 8, 16, 32, 64, int(first[example])]
    figure, axes = plt.subplots(2, 4, figsize=(6.4, 3.75), gridspec_kw={"wspace": 0.08, "hspace": 0.28})
    for axis, iteration in zip(axes.flat, shown):
        draw_board(axis, puzzle, answer, data["example_predictions"][iteration - 1, example],
                   f"iteration {iteration}")
    save(figure, "board_evolution")
    return {"example": example, "first_solved": int(first[example]),
            "rating": int(data["ratings"][data["example_positions"][example]])}


def attention_masses(attention):
    rows, columns = np.divmod(np.arange(81), 9)
    boxes = (rows // 3) * 3 + columns // 3
    same_row = rows[:, None] == rows[None, :]
    same_column = columns[:, None] == columns[None, :]
    same_box = boxes[:, None] == boxes[None, :]
    identity = np.eye(81, dtype=bool)
    masks = {
        "same row": same_row & ~identity,
        "same column": same_column & ~identity,
        "same box only": same_box & ~same_row & ~same_column,
        "itself": identity,
        "elsewhere": ~(same_row | same_column | same_box),
    }
    return {name: attention[..., mask].sum(-1) / 81 for name, mask in masks.items()}


def figure_attention(iteration=16):
    data = load_trajectory("v2")
    if data is None:
        return print("skip attention: missing analysis")
    index = list(data["attention_iterations"]).index(iteration)
    attention = data["attention"][index]
    masses = attention_masses(attention)
    colors = {"same row": BLUE, "same column": ORANGE, "same box only": AQUA,
              "itself": "#3b3b39", "elsewhere": "#d9d8d2"}
    figure = plt.figure(figsize=(6.8, 2.9))
    grid = figure.add_gridspec(2, 4, width_ratios=[2.5, 0.25, 1, 1], wspace=0.12, hspace=0.42)
    bars = figure.add_subplot(grid[:, 0])
    labels, positions = [], []
    for layer in range(4):
        for head in range(4):
            position = layer * 4.6 + head
            positions.append(position)
            labels.append(f"layer {layer + 1}, head {head + 1}")
            left = 0.0
            for name, color in colors.items():
                value = float(masses[name][layer, head])
                bars.barh(position, value, left=left, height=0.8, color=color, edgecolor=SURFACE,
                          linewidth=0.7, label=name if layer == 0 and head == 0 else None)
                left += value
    chance_position = positions[-1] + 1.8
    left = 0.0
    for name, chance in zip(colors, (8 / 81, 8 / 81, 4 / 81, 1 / 81, 60 / 81)):
        bars.barh(chance_position, chance, left=left, height=0.8, color=colors[name],
                  edgecolor=SURFACE, linewidth=0.7)
        left += chance
    bars.set_yticks(positions + [chance_position])
    bars.set_yticklabels(labels + ["uniform attention"], fontsize=6.5)
    bars.invert_yaxis()
    bars.set_xlim(0, 1)
    bars.set_xlabel("Share of attention")
    bars.grid(False)
    bars.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=5, handlelength=0.9,
                columnspacing=0.8, fontsize=6.8, handletextpad=0.4)

    query_row, query_column = 1, 6
    query = query_row * 9 + query_column
    colormap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#a9c9f0", "#2a78d6", "#0d2f63"])
    examples = [((3, 2), "a row head"), ((3, 0), "a column head"),
                ((2, 0), "a box head"), ((1, 2), "a head that looks elsewhere")]
    for (layer, head), description in examples:
        axis = figure.add_subplot(grid[examples.index(((layer, head), description)) // 2,
                                       2 + examples.index(((layer, head), description)) % 2])
        weights = attention[layer, head, query].reshape(9, 9)
        axis.imshow(weights, cmap=colormap, vmin=0, vmax=max(0.12, float(weights.max())),
                    extent=(0, 9, 9, 0), interpolation="nearest")
        for line in range(0, 10, 3):
            axis.plot([0, 9], [line, line], color=INK, linewidth=0.8)
            axis.plot([line, line], [0, 9], color=INK, linewidth=0.8)
        axis.add_patch(Rectangle((query_column, query_row), 1, 1, fill=False, edgecolor=ORANGE,
                                 linewidth=1.4))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(f"{description}\n(layer {layer + 1}, head {head + 1})", fontsize=6.8)
    save(figure, "attention_heads")
    return {name: np.round(value, 3).tolist() for name, value in masses.items()}


def figure_training_curves():
    """1024-iteration monitoring accuracy during training, original recipe vs later-iteration training."""
    curves_path = ROOT / "looping" / "weight_tying" / "results" / "learning_curves.json"
    curves = json.loads(curves_path.read_text())
    cohort = {}
    for path in sorted((DATA / "curves").glob("paper_*.json")):
        # FP32 re-scoring of every saved checkpoint (paper/scripts/checkpoint_curve.py).
        rows = json.loads(path.read_text())["rows"]
        arm = "late" if "late_state_ce" in path.name else "early"
        cohort.setdefault(arm, []).append((
            [row["step"] for row in rows],
            [100 * row["solved_1024"] / row["total"] for row in rows],
        ))
    panels = 2 if cohort else 1
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"wspace": 0.2}, squeeze=False)
    axes = axes[0]
    if not cohort:
        axes[1].axis("off")
        axes[1].text(0.5, 0.5, "50K-step runs in progress", transform=axes[1].transAxes,
                     ha="center", va="center", color=MUTED)
    styles = {"early": ("Loss on iterations 1-16 only", ORANGE),
              "late": ("Later-iteration training", BLUE)}
    for arm, (label, color) in styles.items():
        for index, seed in enumerate((20260902, 20260903, 20260904)):
            rows = curves[f"tied_{arm}_seed{seed}"]
            steps = [row["updates"] for row in rows]
            values = [100 * row["scores"]["1024"]["accuracy"] for row in rows]
            axes[0].plot(np.asarray(steps) / 1000, values, color=color, linewidth=1.2,
                         label=label if index == 0 else None)
    axes[0].set_title("20K-step schedule, three paired seeds", loc="left")
    if cohort:
        for arm, (label, color) in styles.items():
            for steps, values in cohort.get(arm, []):
                axes[1].plot(np.asarray(steps) / 1000, values, color=color, linewidth=1.2)
        axes[1].set_title("50K-step schedule, three new paired seeds", loc="left")
    for axis in axes[:panels]:
        axis.set_ylim(-2, 102)
        axis.set_xlabel("Training step (thousands)")
    axes[0].set_ylabel("Solved at 1024 iterations (%)")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                  bbox_to_anchor=(0.5, -0.14), ncol=2, handlelength=1.6)
    save(figure, "training_curves")


def figure_precision():
    paths = {name: DATA / "precision" / f"{name}.json" for name in ("v2", "v1")}
    if not all(path.exists() for path in paths.values()):
        return print("skip precision: missing results")
    variants = {"fp32": ("FP32", BLUE), "bf16_fp32_state": ("BF16 compute, FP32 state", AQUA),
                "bf16": ("BF16 compute, BF16 state", ORANGE)}
    titles = {"v2": "Released checkpoint", "v1": "Original recipe, accurate run"}
    figure, axes = plt.subplots(1, 3, figsize=(6.6, 2.1), gridspec_kw={"wspace": 0.34})
    for axis, name in zip(axes[:2], ("v2", "v1")):
        result = json.loads(paths[name].read_text())["variants"]
        for variant, (label, color) in variants.items():
            horizons = sorted(int(key) for key in result[variant])
            axis.plot(horizons, [100 * result[variant][str(h)]["solved"] / result[variant][str(h)]["total"]
                                 for h in horizons], color=color, marker="o", markersize=2.8, label=label)
        log2_axis(axis, smallest=16, sparse=True)
        axis.set_xticks([16, 64, 256, 1024, 4096])
        axis.set_xticklabels(["16", "64", "256", "1024", "4096"])
        axis.set_xlim(14, 4700)
        axis.set_ylim(0, 103)
        axis.set_xlabel("Inference iterations")
        axis.set_title(titles[name], loc="left")
    axes[0].set_ylabel("Puzzles solved (%)")
    for name, style in (("v2", "-"), ("v1", (0, (4, 2)))):
        result = json.loads(paths[name].read_text())["variants"]["bf16"]
        horizons = sorted(int(key) for key in result)
        axes[2].plot(horizons, [100 * result[str(h)]["unchanged_state_fraction"] for h in horizons],
                     color=ORANGE, linestyle=style, marker="o", markersize=2.8)
    axes[2].text(18, 98, "solid: released checkpoint\ndashed: original recipe", fontsize=6.8,
                 color=MUTED, va="top")
    log2_axis(axes[2], smallest=16, sparse=True)
    axes[2].set_xticks([16, 64, 256, 1024, 4096])
    axes[2].set_xticklabels(["16", "64", "256", "1024", "4096"])
    axes[2].set_xlim(14, 4700)
    axes[2].set_ylim(0, 103)
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("BF16 state: elements an\niteration leaves unchanged (%)", loc="left")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3,
                  bbox_to_anchor=(0.5, -0.2), handlelength=1.6)
    save(figure, "precision")


def figure_where_accuracy_came_from():
    """Accuracy on Sudoku-Extreme after each change that was kept, in the order the changes were made.

    Values are single runs copied from STALE_EXPERIMENTS_DOC.md, cosine/EXPERIMENTS_COSINE.md,
    pos_embedding/EXPERIMENTS_POS.md and iters/EXPERIMENTS_ITERS.md; the last three rows come
    from this paper's FP32 evaluations of the two published checkpoints.
    """
    v1 = DATA / "evaluations" / "v1_fp32" / "result.json"
    v2 = DATA / "evaluations" / "v2_fp32" / "result.json"
    if not (v1.exists() and v2.exists()):
        return print("skip where_accuracy_came_from: missing evaluations")
    v1 = json.loads(v1.read_text())["scores"]
    v2 = json.loads(v2.read_text())["scores"]
    basics, simpler, iterations = BLUE, "#a9c9f0", AQUA
    rows = [
        ("Loop model trained on easier Kaggle puzzles", 31.7, basics),
        ("Train on Sudoku-Extreme (400K puzzles)", 63.4, basics),
        ("2.7M training puzzles", 71.4, basics),
        ("Batch size 512 to 4096", 76.3, basics),
        ("Learning-rate warmup", 78.5, basics),
        ("Cosine learning-rate decay", 84.0, basics),
        ("Remove sharpness-aware minimization", 83.6, simpler),
        ("Grid coordinates only (2D rotary), 50K steps", 82.5, simpler),
        ("Batch size 2048, learning rate 2e-3", v1["16"]["accuracy_percent"], simpler),
        ("Same weights, 128 iterations", v1["128"]["accuracy_percent"], iterations),
        ("Same weights, 1024 iterations", v1["1024"]["accuracy_percent"], iterations),
        ("Later-iteration training, 1024 iterations", v2["1024"]["accuracy_percent"], iterations),
    ]
    figure, axis = plt.subplots(figsize=(6.2, 2.9))
    positions = np.arange(len(rows))
    for position, (label, value, color) in zip(positions, rows):
        axis.barh(position, value, height=0.72, color=color)
        axis.text(value + 1, position, f"{value:.1f}", va="center", ha="left", fontsize=7.5, color=INK)
    axis.set_yticks(positions)
    axis.set_yticklabels([label for label, _, _ in rows])
    axis.invert_yaxis()
    axis.set_xlim(0, 108)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("Puzzles solved (%)")
    axis.grid(axis="y", visible=False)
    axis.tick_params(axis="y", length=0)
    handles = [Rectangle((0, 0), 1, 1, color=color) for color in (basics, simpler, iterations)]
    axis.legend(handles, ["data and optimization basics, 16 iterations",
                          "changes kept for other reasons, 16 iterations",
                          "more inference iterations"],
                loc="lower center", bbox_to_anchor=(0.3, -0.42), ncol=1, handlelength=1.0)
    save(figure, "where_accuracy_came_from")


def figure_attention_all_heads(iterations=(1, 16, 1024), query_row=1, query_column=6):
    """Appendix: average attention from one cell for all 16 heads at three iterations."""
    data = load_trajectory("v2")
    if data is None:
        return print("skip attention_all_heads: missing analysis")
    available = list(data["attention_iterations"])
    query = query_row * 9 + query_column
    colormap = LinearSegmentedColormap.from_list("blues", ["#ffffff", "#a9c9f0", "#2a78d6", "#0d2f63"])
    figure = plt.figure(figsize=(6.8, 6.0))
    outer = figure.add_gridspec(len(iterations), 1, hspace=0.3)
    for block, iteration in enumerate(iterations):
        attention = data["attention"][available.index(iteration)]
        inner = outer[block].subgridspec(2, 8, wspace=0.08, hspace=0.22)
        for layer in range(4):
            for head in range(4):
                index = layer * 4 + head
                axis = figure.add_subplot(inner[index // 8, index % 8])
                weights = attention[layer, head, query].reshape(9, 9)
                axis.imshow(weights, cmap=colormap, vmin=0, vmax=max(0.12, float(weights.max())),
                            extent=(0, 9, 9, 0), interpolation="nearest")
                for line in range(0, 10, 3):
                    axis.plot([0, 9], [line, line], color=INK, linewidth=0.6)
                    axis.plot([line, line], [0, 9], color=INK, linewidth=0.6)
                axis.add_patch(Rectangle((query_column, query_row), 1, 1, fill=False,
                                         edgecolor=ORANGE, linewidth=1.1))
                axis.set_xticks([])
                axis.set_yticks([])
                axis.grid(False)
                for spine in axis.spines.values():
                    spine.set_visible(False)
                axis.set_title(f"layer {layer + 1}, head {head + 1}", fontsize=6, pad=2)
                if index == 0:
                    axis.text(0, 1.3, f"Iteration {iteration}", transform=axis.transAxes,
                              fontsize=8.5, fontweight="bold", ha="left", va="bottom")
    save(figure, "attention_all_heads")


def cell_timeline(data, example, horizon=1024):
    puzzle = str(data["example_questions"][example])
    answer = np.array([int(character) - 1 for character in str(data["example_answers"][example])])
    empty = np.array([character == "." for character in puzzle])
    predictions = data["example_predictions"][:horizon, example].astype(int)
    return predictions[:, empty] == answer[empty][None, :]


def figure_cell_timeline(example=None):
    """Appendix: which empty cells are right at each iteration, for one puzzle in two models."""
    released, failing = load_trajectory("v2"), load_trajectory("lr25e4")
    if released is None or failing is None:
        return print("skip cell_timeline: missing analysis")
    if example is None:
        solved = released["solved"][:1024, released["example_positions"]]
        first = np.where(solved.any(0), solved.argmax(0) + 1, 0)
        example = int(np.argmax(np.where(released["buckets"][released["example_positions"]] == "51+", first, -1)))
    panels = [("Later-iteration training (released)", cell_timeline(released, example)),
              ("Original recipe, LR $2.5\\times10^{-3}$ (loses its answers)", cell_timeline(failing, example))]
    # One row order for both panels: cells sorted by when the released model last got them wrong.
    reference = panels[0][1]
    last_wrong = np.where((~reference).any(0), reference.shape[0] - np.argmax(~reference[::-1], axis=0), 0)
    order = np.argsort(last_wrong, kind="stable")
    colormap = LinearSegmentedColormap.from_list("wrong", ["#f3f2ee", ORANGE])
    edges = np.arange(0.5, reference.shape[0] + 1)
    figure, axes = plt.subplots(2, 1, figsize=(6.8, 4.4), sharex=True, gridspec_kw={"hspace": 0.25})
    summary = {}
    for axis, (title, correct) in zip(axes, panels):
        wrong = (~correct[:, order]).T.astype(float)
        axis.pcolormesh(edges, np.arange(wrong.shape[0] + 1), wrong, cmap=colormap, vmin=0, vmax=1,
                        rasterized=True)
        axis.set_xscale("log", base=2)
        axis.set_xlim(0.5, reference.shape[0] + 0.5)
        ticks = [1, 4, 16, 64, 256, 1024]
        axis.set_xticks(ticks)
        axis.set_xticklabels([str(tick) for tick in ticks])
        axis.minorticks_off()
        axis.set_ylabel("Empty cells")
        axis.set_yticks([])
        axis.grid(False)
        axis.set_title(title, loc="left")
        solved_at = np.flatnonzero(correct.all(1)) + 1
        summary[title] = {"first_solved": int(solved_at[0]) if solved_at.size else None,
                          "wrong_at_end": int((~correct[-1]).sum()), "empty_cells": int(correct.shape[1])}
    axes[-1].set_xlabel("Inference iteration (orange: the cell's predicted digit is wrong)")
    save(figure, "cell_timeline")
    return summary


def figure_perturbation_maps(model="20k_20260907"):
    """Appendix: outcome of 2,048 FP64 iterations over a 2D slice of starting states, zoomed six times."""
    folder = ROOT / "looping" / "basin_diagnostics" / "slow_puzzles" / "results" / model / "slow"
    if not (folder / "level_0.npz").exists():
        return print("skip perturbation_maps: missing arrays")
    colormap = plt.get_cmap("viridis").copy()
    colormap.set_bad("#d9d8d2")
    figure, axes = plt.subplots(2, 3, figsize=(6.8, 4.9), gridspec_kw={"wspace": 0.1, "hspace": 0.2})
    summary = []
    for level, axis in enumerate(axes.flat):
        arrays = np.load(folder / f"level_{level}.npz")
        details = json.loads((folder / f"level_{level}.json").read_text())
        last_change = arrays["last_change"].astype(float)
        last_change[~arrays["successful"]] = np.nan
        half = details["width"]  # the saved "width" is the distance from the center to an edge
        center_x, center_y = details["center"]
        extent = (center_x - half, center_x + half, center_y - half, center_y + half)
        image = axis.imshow(np.log2(np.maximum(last_change, 1)), cmap=colormap, origin="lower",
                            extent=extent, vmin=np.log2(40), vmax=np.log2(400), interpolation="nearest")
        if level < 5:
            zoom = details["next_zoom"]
            zoom_half = zoom["width"]
            axis.add_patch(Rectangle((zoom["center"][0] - zoom_half, zoom["center"][1] - zoom_half),
                                     2 * zoom_half, 2 * zoom_half, fill=False, edgecolor="#ffffff",
                                     linewidth=1.0))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_title(f"magnified {details['magnification']}$\\times$", fontsize=8)
        summary.append({"level": level, "magnification": details["magnification"], "width": details["width"],
                        "successful": round(float(arrays["successful"].mean()), 4),
                        "min": float(np.nanmin(last_change)), "max": float(np.nanmax(last_change))})
    bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    ticks = [50, 100, 200, 400]
    bar.set_ticks([np.log2(tick) for tick in ticks])
    bar.set_ticklabels([str(tick) for tick in ticks])
    bar.set_label("Iteration at which the predicted board last changed")
    save(figure, "perturbation_maps")
    return summary


if __name__ == "__main__":
    figure_where_accuracy_came_from()
    figure_iteration_scaling()
    figure_state_dynamics()
    print(figure_board_evolution())
    figure_attention()
    figure_training_curves()
    figure_precision()
    figure_attention_all_heads()
    print(figure_cell_timeline())
    print(figure_perturbation_maps())
