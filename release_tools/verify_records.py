"""Verify downloaded evaluation records against the pinned dataset, without a GPU."""

import argparse
import json
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save, validate_config
from dataset_utils import DATASET_NAME, DATASET_REVISION, validate_benchmark
from runtime_utils import file_sha256
from stabilize.exp_testbed_20k import encode_puzzles, encode_solutions


def verify_evaluation(directory, dataset):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    identity = json.loads((directory / "identity.json").read_text())
    validate_config(result["identity"], identity)
    benchmark = json.loads((directory / "benchmark.json").read_text())
    rows = validate_benchmark(dataset, benchmark)
    if identity["benchmark_rows_sha256"] != benchmark["rows_sha256"]:
        raise ValueError("Evaluation and benchmark identities differ")
    array_path = directory / "per_puzzle.npz"
    if file_sha256(array_path) != result["per_puzzle_sha256"]:
        raise ValueError("Downloaded prediction checksum does not match")
    empty = encode_puzzles(rows["question"])[:, :, 0].bool().numpy()
    targets = encode_solutions(rows["answer"]).numpy()
    labels = np.asarray(benchmark["bucket_names"])
    with np.load(array_path, allow_pickle=False) as arrays:
        if not np.array_equal(arrays["indices"], benchmark["indices"]):
            raise ValueError("Prediction row indices differ from the benchmark")
        previous = None
        for horizon in identity["iterations"]:
            predictions = arrays[f"predictions_{horizon}"]
            if predictions.shape != targets.shape or not np.issubdtype(predictions.dtype, np.integer):
                raise ValueError("Prediction dimensions or dtype are invalid")
            if (predictions < 0).any() or (predictions > 8).any():
                raise ValueError("Predicted digit is outside 0-8")
            solved = ((predictions == targets) | ~empty).all(-1)
            if not np.array_equal(solved, arrays[f"solved_{horizon}"]):
                raise ValueError("Saved correctness flags disagree with the puzzle answers")
            score = result["scores"][str(horizon)]
            if score["solved"] != int(solved.sum()) or score["total"] != len(rows):
                raise ValueError("Saved score disagrees with the predictions")
            if score["accuracy_percent"] != 100 * int(solved.sum()) / len(rows):
                raise ValueError("Saved percentage disagrees with the count")
            for label, bucket in score["buckets"].items():
                selected = labels == label
                if bucket != {"solved": int(solved[selected].sum()), "total": int(selected.sum())}:
                    raise ValueError("Saved bucket score disagrees with the predictions")
            if previous is not None:
                if score["lost_since_previous_recorded_horizon"] != int((previous & ~solved).sum()):
                    raise ValueError("Saved regression count disagrees with the predictions")
                if score["gained_since_previous_recorded_horizon"] != int((~previous & solved).sum()):
                    raise ValueError("Saved gain count disagrees with the predictions")
            previous = solved
    return {
        "result_sha256": file_sha256(directory / "result.json"),
        "weights_sha256": identity["weights_sha256"],
        "rows_sha256": benchmark["rows_sha256"],
        "solved": {key: value["solved"] for key, value in result["scores"].items()},
        "total": len(rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--cached-arrow", help="Already downloaded pinned test split")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import Dataset, load_dataset
    dataset = Dataset.from_file(args.cached_arrow) if args.cached_arrow else load_dataset(
        DATASET_NAME, revision=DATASET_REVISION, split="test",
    )
    records = sorted(args.records.rglob("result.json"))
    if not records:
        raise ValueError("No completed evaluation records found")
    verified = {}
    for result_path in records:
        name = str(result_path.parent.relative_to(args.records))
        verified[name] = verify_evaluation(result_path.parent, dataset)
        print(f"Verified {name}: {verified[name]['solved']}", flush=True)
    atomic_json_save({"dataset_revision": DATASET_REVISION, "verified": verified}, args.output)


if __name__ == "__main__":
    main()
