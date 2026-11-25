"""Write the LaTeX fragments in `paper/generated/` from result files.

    python -m paper.scripts.make_tables

Numbers from the new Modal runs are read from `runs/paper/` (see `paper/README.md`). Numbers
from earlier studies are copied from the result files named beside them, so every table row can
be traced to a file. Fragments whose inputs are missing are written with a "pending" note.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs" / "paper"
OUT = ROOT / "paper" / "generated"
BUCKETS = ["0", "1-2", "3-10", "11-50", "51+"]
SEEDS = (20261119, 20261120, 20261121)
HORIZONS = (16, 128, 1024, 2048, 4096)

# Final checkpoints, FP32, frozen 25K benchmark, accuracy at 16/128/1024/2048/4096.
# looping/width_50k/RESULTS.md (width-128 controls)
CONTROLS_50K = {20260910: (81.076, 96.060, 98.244, 97.720, 91.688),
                20260911: (81.180, 96.276, 97.848, 97.432, 97.236),
                20260912: (80.764, 96.432, 99.228, 99.368, 99.436)}
# release/PRECISION_RESULTS.md, full 25K benchmark, accuracy at 128/1024/2048/4096.
PRECISION_25K = [
    ("Released checkpoint", "FP32", (96.288, 99.116, 99.048, 98.632)),
    ("Released checkpoint", "BF16", (96.472, 98.796, 92.976, 43.704)),
    ("Released checkpoint", "BF16, compiled in 16-iteration chunks", (96.372, 98.952, 97.600, 92.512)),
    ("Original recipe, accurate run", "FP32", (95.704, 98.888, 99.028, 99.020)),
    ("Original recipe, accurate run", "BF16", (95.256, 98.876, 98.844, 84.888)),
    ("Original recipe, accurate run", "BF16, compiled in 16-iteration chunks", (95.572, 98.744, 98.784, 98.748)),
]


def write(name, text):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.tex").write_text(text.rstrip() + "%\n")
    print(f"wrote {name}")


def load_scores(name):
    path = DATA / "evaluations" / name / "result.json"
    return json.loads(path.read_text())["scores"] if path.exists() else None


def fmt(value, digits=1):
    return f"{value:.{digits}f}"


def table_scaling():
    v2, v1 = load_scores("v2_fp32"), load_scores("v1_fp32")
    if v2 is None:
        write("headline_1024", "99.1\\%")
        write("table_scaling", "\\emph{(Iteration-scaling table pending.)}")
        return
    write("headline_1024", f"{v2['1024']['accuracy_percent']:.1f}\\%")
    horizons = sorted(int(key) for key in v2)
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Accuracy (\\%) by number of inference iterations, FP32. The first block uses the "
        "25,000-puzzle benchmark; both checkpoints were trained with at most 16 differentiable "
        "iterations. The second block uses all 422,786 puzzles of the Sudoku-Extreme test split.}",
        "\\label{tab:scaling}",
        "\\setlength{\\tabcolsep}{3.3pt}",
        "\\begin{tabular}{@{}l" + "r" * len(horizons) + "@{}}", "\\toprule",
        "Inference iterations & " + " & ".join(str(h) for h in horizons) + " \\\\", "\\midrule",
    ]
    rows = [("Released (later-iteration training)", v2), ("Version 1 (original recipe)", v1)]
    for label, scores in rows:
        if scores is None:
            lines.append(f"{label} & \\multicolumn{{{len(horizons)}}}{{c}}{{pending}} \\\\")
            continue
        lines.append(label + " & " + " & ".join(fmt(scores[str(h)]["accuracy_percent"]) for h in horizons) + " \\\\")
    lines.append("\\midrule")
    lines.append("\\multicolumn{" + str(len(horizons) + 1) + "}{@{}l}{\\emph{Released checkpoint by rating bucket}} \\\\")
    for bucket in BUCKETS:
        values = [100 * v2[str(h)]["buckets"][bucket]["solved"] / v2[str(h)]["buckets"][bucket]["total"]
                  for h in horizons]
        lines.append(f"\\quad rating {bucket} & " + " & ".join(fmt(value) for value in values) + " \\\\")
    own_full = load_scores("v2_fp32_full_test")
    write("full_test_1024", f"{own_full['1024']['accuracy_percent']:.2f}" if own_full else "pending")
    full = [(label, load_scores(name)) for label, name in (
        ("Released (later-iteration training)", "v2_fp32_full_test"),
        ("Version 1 (original recipe)", "v1_fp32_full_test"))]
    if any(scores is not None for _, scores in full):
        lines.append("\\midrule")
        lines.append("\\multicolumn{" + str(len(horizons) + 1) + "}{@{}l}{\\emph{Full test split}} \\\\")
        for label, scores in full:
            if scores is None:
                continue
            cells = [fmt(scores[str(h)]["accuracy_percent"], 2) if str(h) in scores else "" for h in horizons]
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_scaling", "\n".join(lines))


def cohort_scores():
    cohort = {}
    for arm in ("vanilla", "late_state_ce"):
        for seed in SEEDS:
            scores = load_scores(f"model_paper_{arm}_seed{seed}")
            result_path = DATA / "cohort" / f"result_paper_{arm}_seed{seed}.json"
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            cohort[(arm, seed)] = (scores, result)
    return cohort


def row(label, seed, values, digits=1):
    cells = " & ".join("--" if value is None else fmt(value, digits) for value in values)
    return f"{label} & {seed} & {cells} \\\\"


def table_late():
    cohort = cohort_scores()
    v2 = load_scores("v2_fp32")
    released = tuple(v2[str(h)]["accuracy_percent"] for h in HORIZONS) if v2 else (None, 96.288, 99.116, 99.048, 98.632)
    lines = [
        "\\begin{table}[t]", "\\centering", "\\small",
        "\\caption{Final checkpoints of 50K-step runs on the 25,000-puzzle benchmark, FP32 inference, "
        "accuracy (\\%) by inference iterations. No checkpoint selection. The runs marked ``new'' were "
        "trained for this paper in a fresh environment, in pairs that share a seed.}",
        "\\label{tab:late}",
        "\\begin{tabular}{@{}llrrrrr@{}}", "\\toprule",
        "Recipe & Seed & 16 & 128 & 1024 & 2048 & 4096 \\\\", "\\midrule",
        row("Later-iteration training (released)", 20260730, released),
    ]
    for seed, values in CONTROLS_50K.items():
        lines.append(row("Later-iteration training", seed, values))
    pending = False
    for arm, label in (("late_state_ce", "Later-iteration training (new)"), ("vanilla", "Original recipe (new)")):
        lines.append("\\midrule")
        for seed in SEEDS:
            scores, _ = cohort[(arm, seed)]
            if scores is None:
                pending = True
                lines.append(f"{label} & {seed} & \\multicolumn{{5}}{{c}}{{run in progress}} \\\\")
            else:
                lines.append(row(label, seed, [scores[str(h)]["accuracy_percent"] for h in HORIZONS]))
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_late", "\n".join(lines))
    return pending


def table_precision():
    lines = [
        "\\begin{table}[h]", "\\centering", "\\small",
        "\\caption{The effect of inference precision on the same weights, accuracy (\\%) by inference "
        "iterations. \\emph{Top:} the 25,000-puzzle benchmark. \\emph{Bottom:} 1,000 of those puzzles, "
        "separating the precision of the arithmetic from the precision in which the carried state is stored.}",
        "\\label{tab:precision}",
        "\\begin{tabular}{@{}llrrrr@{}}", "\\toprule",
        "Checkpoint & Inference & 128 & 1024 & 2048 & 4096 \\\\", "\\midrule",
    ]
    previous = None
    for checkpoint, setting, values in PRECISION_25K:
        if previous is not None and checkpoint != previous:
            lines.append("\\addlinespace")
        lines.append(f"{checkpoint if checkpoint != previous else ''} & {setting} & "
                     + " & ".join(fmt(value) for value in values) + " \\\\")
        previous = checkpoint
    labels = {"fp32": "FP32", "bf16": "BF16 arithmetic, BF16 state",
              "bf16_fp32_state": "BF16 arithmetic, FP32 state"}
    names = {"v2": "Released checkpoint", "v1": "Original recipe, accurate run"}
    if all((DATA / "precision" / f"{name}.json").exists() for name in names):
        lines.append("\\midrule")
        for name, checkpoint in names.items():
            variants = json.loads((DATA / "precision" / f"{name}.json").read_text())["variants"]
            for index, (variant, label) in enumerate(labels.items()):
                values = [100 * variants[variant][str(h)]["solved"] / variants[variant][str(h)]["total"]
                          for h in (128, 1024, 2048, 4096)]
                lines.append(f"{checkpoint if index == 0 else ''} & {label} & "
                             + " & ".join(fmt(value) for value in values) + " \\\\")
            lines.append("\\addlinespace")
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write("table_precision", "\n".join(lines))


if __name__ == "__main__":
    table_scaling()
    table_late()
    table_precision()
