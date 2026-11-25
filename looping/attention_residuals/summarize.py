"""Print the result tables from downloaded evaluation records.

Usage: python -m looping.attention_residuals.summarize [runs/attention_residuals/eval]

Each run's record is `<directory>/attnres20k_<arm>_seed<seed>/result.json`, as written by
`modal_run.py::evaluate`.
"""

import json
import sys
from pathlib import Path

ARMS = ("baseline", "attnres", "message", "attnres_message")
SEEDS = (20261123, 20261124, 20261125)
HORIZONS = ("16", "128", "1024", "4096")


def load_scores(directory):
    scores = {}
    for arm in ARMS:
        for seed in SEEDS:
            path = Path(directory) / f"attnres20k_{arm}_seed{seed}" / "result.json"
            if path.exists():
                scores[arm, seed] = json.loads(path.read_text())["scores"]
    return scores


def mean(values):
    return sum(values) / len(values)


def promising(baseline, variant):
    """The rule in README.md. Both arguments map horizon -> list of per-seed accuracies."""
    def seeds_improved(horizon):
        return sum(new > old for new, old in zip(variant[horizon], baseline[horizon]))

    def difference(horizon):
        return mean(variant[horizon]) - mean(baseline[horizon])

    by_1024 = difference("1024") >= 0.5 and difference("4096") >= 0 and seeds_improved("1024") >= 2
    by_4096 = difference("4096") >= 5 and difference("1024") >= -0.5 and seeds_improved("4096") >= 2
    return by_1024 or by_4096


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "runs/attention_residuals/eval"
    scores = load_scores(directory)

    print("| Arm | Seed | " + " | ".join(f"{h} iterations" for h in HORIZONS) + " |")
    print("|---|---:|" + "---:|" * len(HORIZONS))
    for (arm, seed), record in scores.items():
        cells = " | ".join(f"{record[h]['accuracy_percent']:.3f}%" for h in HORIZONS)
        print(f"| `{arm}` | {seed} | {cells} |")

    print()
    print("| Arm | " + " | ".join(f"Mean at {h}" for h in HORIZONS) + " | Solved at 1024, then lost by 4096 |")
    print("|---|" + "---:|" * (len(HORIZONS) + 1))
    by_arm = {}
    for arm in ARMS:
        records = [scores[arm, seed] for seed in SEEDS if (arm, seed) in scores]
        if not records:
            continue
        by_arm[arm] = {h: [record[h]["accuracy_percent"] for record in records] for h in HORIZONS}
        lost = " / ".join(str(record["4096"]["lost_since_previous_recorded_horizon"]) for record in records)
        cells = " | ".join(f"{mean(by_arm[arm][h]):.3f}%" for h in HORIZONS)
        print(f"| `{arm}` | {cells} | {lost} |")

    print()
    for arm in ARMS[1:]:
        if arm in by_arm and len(by_arm[arm]["1024"]) == len(by_arm["baseline"]["1024"]):
            verdict = "promising" if promising(by_arm["baseline"], by_arm[arm]) else "not promising"
            print(f"{arm}: {verdict} under the predeclared rule")


if __name__ == "__main__":
    main()
