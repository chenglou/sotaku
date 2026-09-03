"""Prepare paired training data and a newly generated, unopened test set."""

import csv
import io
import json
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from checkpoint_utils import atomic_json_save
from dataset_utils import DATASET_NAME, DATASET_REVISION, validate_benchmark
from looping.weight_tying.common import atomic_npz, protocol, protocol_sha256
from runtime_utils import file_sha256


def encode_rows(questions, answers):
    questions = [question.replace("0", ".") for question in questions]
    digits = np.frombuffer("".join(questions).encode("ascii"), dtype=np.uint8).reshape(-1, 81)
    targets = np.frombuffer("".join(answers).encode("ascii"), dtype=np.uint8).reshape(-1, 81)
    if not np.all((digits == ord(".")) | ((digits >= 49) & (digits <= 57))):
        raise ValueError("Invalid puzzle character")
    if not np.all((targets >= 49) & (targets <= 57)):
        raise ValueError("Invalid answer character")
    return np.where(digits == ord("."), 0, digits - 48).astype(np.uint8), (targets - 49).astype(np.uint8)


def canonical_digits(board):
    if isinstance(board, str):
        board = board.replace("0", ".").encode("ascii")
    order = bytes(dict.fromkeys(board.replace(b".", b"")))
    return board.translate(bytes.maketrans(order, b"123456789"[:len(order)]))


def symmetry_keys(board):
    grid = np.frombuffer(board.encode("ascii"), dtype=np.uint8).reshape(9, 9)
    return {canonical_digits(view.tobytes()) for turns in range(4)
            for view in (np.rot90(grid, turns), np.fliplr(np.rot90(grid, turns)))}


def validate_solution(question, answer):
    if len(question) != 81 or len(answer) != 81:
        raise ValueError("A Sudoku board must contain 81 cells")
    if not set(question) <= set(".123456789") or set(answer) != set("123456789"):
        raise ValueError("Invalid Sudoku characters")
    if any(given != "." and given != solved for given, solved in zip(question, answer)):
        raise ValueError("Solution changes a given digit")
    grid = np.asarray(list(answer)).reshape(9, 9)
    units = [*grid, *grid.T, *(grid[row:row + 3, col:col + 3].flat
                              for row in (0, 3, 6) for col in (0, 3, 6))]
    if any(set(unit) != set("123456789") for unit in units):
        raise ValueError("Invalid completed Sudoku grid")


def parse_qqwing(text, expected_difficulty):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or rows[0][:2] != ["Puzzle", "Solution"]:
        raise ValueError("Unexpected QQWing CSV header")
    # QQWing prints solution count immediately after puzzle and solution.
    if "solution" not in rows[0][2].lower():
        raise ValueError("QQWing uniqueness count is missing")
    difficulty_column = rows[0].index("Difficulty")
    accepted = []
    for row in rows[1:]:
        if not row:
            continue
        if int(row[2]) != 1:
            raise ValueError("Generated puzzle does not have exactly one solution")
        question, answer = row[:2]
        validate_solution(question, answer)
        if row[difficulty_column].lower() != expected_difficulty:
            raise ValueError("Unexpected QQWing difficulty")
        accepted.append({"question": question, "answer": answer, "difficulty": expected_difficulty})
    return accepted


def load_manifest(directory, *, verify=()):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["protocol_sha256"] != protocol_sha256():
        raise ValueError("Prepared data belongs to a different study protocol")
    if manifest["dataset_revision"] != DATASET_REVISION:
        raise ValueError("Prepared data uses a different dataset revision")
    for filename in verify:
        if file_sha256(directory / filename) != manifest["files"][filename]:
            raise ValueError(f"Prepared data checksum mismatch: {filename}")
    return manifest


def prepare_data(output_dir):
    from datasets import load_dataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        return load_manifest(output_dir, verify=("train.npz", "validation.npz", "holdout.npz", "holdout.json"))
    settings = protocol()
    evaluation = settings["evaluation"]
    executable = shutil.which("qqwing")
    if executable is None:
        raise RuntimeError("QQWing must be installed for data construction")
    version = subprocess.check_output([executable, "--version"], text=True).strip()
    if version.split()[-1] != evaluation["holdout_generator_version"]:
        raise ValueError(f"Unexpected QQWing version: {version}")
    started = time.perf_counter()
    candidates = []
    raw_files = []
    for difficulty in evaluation["holdout_difficulties"]:
        raw_path = output_dir / f"qqwing_{difficulty}.csv"
        if not raw_path.exists():
            print(f"Generating new {difficulty} puzzles", flush=True)
            generated = subprocess.run([
                executable, "--generate", str(evaluation["holdout_per_difficulty"] + 64),
                "--difficulty", difficulty, "--puzzle", "--solution", "--count-solutions",
                "--stats", "--csv",
            ], check=True, capture_output=True, text=True, timeout=7200)
            temporary = raw_path.with_suffix(".tmp")
            temporary.write_text(generated.stdout)
            temporary.replace(raw_path)
        candidates.extend(parse_qqwing(raw_path.read_text(), difficulty))
        raw_files.append(raw_path.name)

    question_keys, answer_keys = defaultdict(set), defaultdict(set)
    rejected = set()
    for index, row in enumerate(candidates):
        questions, answers = symmetry_keys(row["question"]), symmetry_keys(row["answer"])
        if any(key in question_keys for key in questions) or any(key in answer_keys for key in answers):
            rejected.add(index)
        for key in questions:
            question_keys[key].add(index)
        for key in answers:
            answer_keys[key].add(index)
    within_duplicates = len(rejected)
    scanned = {}
    datasets = {}
    for split in ("train", "test"):
        dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=split)
        datasets[split] = dataset
        scanned[split] = len(dataset)
        print(f"Checking all {len(dataset):,} {split} questions and answers for overlap", flush=True)
        for batch in dataset.select_columns(["question", "answer"]).iter(batch_size=50000):
            for question, answer in zip(batch["question"], batch["answer"]):
                rejected.update(question_keys.get(canonical_digits(question), ()))
                rejected.update(answer_keys.get(canonical_digits(answer), ()))
    selected = []
    counts = {}
    for difficulty in evaluation["holdout_difficulties"]:
        rows = [row for index, row in enumerate(candidates)
                if index not in rejected and row["difficulty"] == difficulty]
        if len(rows) < evaluation["holdout_per_difficulty"]:
            raise ValueError("Too few fresh unique puzzles; do not silently shrink the test set")
        selected.extend(rows[:evaluation["holdout_per_difficulty"]])
        counts[difficulty] = evaluation["holdout_per_difficulty"]
    atomic_json_save(selected, output_dir / "holdout.json")
    digits, targets = encode_rows([row["question"] for row in selected], [row["answer"] for row in selected])
    atomic_npz(output_dir / "holdout.npz", digits=digits, targets=targets,
               labels=np.asarray([row["difficulty"] for row in selected]))

    training = datasets["train"].select(range(settings["training"]["train_rows"]))
    digit_parts, target_parts, rating_parts = [], [], []
    for batch in training.iter(batch_size=50000):
        digits, targets = encode_rows(batch["question"], batch["answer"])
        digit_parts.append(digits)
        target_parts.append(targets)
        rating_parts.append(np.asarray(batch["rating"], dtype=np.int16))
    atomic_npz(output_dir / "train.npz", digits=np.concatenate(digit_parts),
               targets=np.concatenate(target_parts), ratings=np.concatenate(rating_parts))
    benchmark_path = Path(__file__).parents[2] / "release" / "benchmark_25k.json"
    benchmark = json.loads(benchmark_path.read_text())
    rows = validate_benchmark(datasets["test"], benchmark)
    digits, targets = encode_rows(rows["question"], rows["answer"])
    labels = np.asarray(benchmark["bucket_names"])
    per_bucket = settings["training"]["probe_per_bucket"]
    probe_indices = np.concatenate([np.flatnonzero(labels == name)[:per_bucket]
                                    for name in ("0", "1-2", "3-10", "11-50", "51+")])
    atomic_npz(output_dir / "validation.npz", digits=digits[probe_indices], targets=targets[probe_indices],
               labels=labels[probe_indices], indices=np.asarray(benchmark["indices"])[probe_indices])
    atomic_npz(output_dir / "development.npz", digits=digits, targets=targets, labels=labels,
               indices=np.asarray(benchmark["indices"]))
    files = ["train.npz", "validation.npz", "development.npz", "holdout.npz", "holdout.json", *raw_files]
    manifest = {
        "schema_version": 1, "study_id": settings["study_id"], "protocol_sha256": protocol_sha256(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": DATASET_NAME, "dataset_revision": DATASET_REVISION,
        "training_rows": len(training), "scanned_rows": scanned, "holdout_counts": counts,
        "generator_version": version, "generator_binary_sha256": file_sha256(executable),
        "duplicate_candidates": within_duplicates, "rejected_candidates": len(rejected),
        "duplicate_screen": "questions and answers, digit relabeling and D4 rotations/reflections",
        "development_benchmark_sha256": file_sha256(benchmark_path),
        "generation_and_preparation_seconds": time.perf_counter() - started,
        "files": {filename: file_sha256(output_dir / filename) for filename in files},
        "holdout_status": "No model evaluation; open only after all preregistered runs are fixed",
    }
    atomic_json_save(manifest, output_dir / "manifest.json")
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return manifest
