"""Dataset identity and the historical balanced development benchmark."""

import hashlib
import json
import random

DATASET_NAME = "sapientinc/sudoku-extreme"
DATASET_REVISION = "58942f96baeb572ca3127e2a9e9c70f330783d6b"
RATING_BUCKETS = ((0, 0, "0"), (1, 2, "1-2"), (3, 10, "3-10"),
                  (11, 50, "11-50"), (51, 1000, "51+"))


def phase_bucket_keys(keys, minimum_rating):
    # Historical curricula select whole buckets, e.g. cutoff 21 selects 51+.
    return [key for key in keys if key[0] >= minimum_rating]


def balanced_indices(ratings, per_bucket=5000, seed=42):
    if not isinstance(per_bucket, int) or per_bucket <= 0:
        raise ValueError("per_bucket must be a positive integer")
    buckets = {}
    for index, rating in enumerate(ratings):
        for lower, upper, name in RATING_BUCKETS:
            if lower <= rating <= upper:
                buckets.setdefault(name, []).append(index)
                break
        else:
            raise ValueError(f"No rating bucket for row {index}: {rating}")
    rng = random.Random(seed)
    # Sampling in first-encounter order is part of the original benchmark.
    for name, indices in buckets.items():
        if len(indices) > per_bucket:
            buckets[name] = rng.sample(indices, per_bucket)
    ordered_indices, names = [], []
    for _, _, name in RATING_BUCKETS:
        indices = buckets.get(name, [])
        ordered_indices.extend(indices)
        names.extend([name] * len(indices))
    return ordered_indices, names


def benchmark_manifest(dataset, per_bucket=5000, seed=42):
    indices, names = balanced_indices(dataset["rating"], per_bucket, seed)
    return manifest_for_indices(dataset, indices, names, per_bucket=per_bucket, seed=seed)


def manifest_for_indices(dataset, indices, names, *, per_bucket=None, seed=None):
    if len(indices) != len(names) or not indices or len(set(indices)) != len(indices):
        raise ValueError("Benchmark rows and bucket names must be aligned and unique")
    if any(type(index) is not int or not 0 <= index < len(dataset) for index in indices):
        raise ValueError("Invalid benchmark row index")
    rows = dataset.select(indices)
    digest = hashlib.sha256()
    for index, name, row in zip(indices, names, rows):
        expected_name = next(label for low, high, label in RATING_BUCKETS if low <= row["rating"] <= high)
        if name != expected_name:
            raise ValueError(f"Wrong rating bucket for row {index}")
        digest.update(json.dumps(
            [index, row["question"], row["answer"], int(row["rating"])],
            separators=(",", ":"),
        ).encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "split": "test",
        "purpose": "reused development benchmark, not an untouched holdout",
        "seed": seed,
        "per_bucket": per_bucket,
        "sampling_order": ("first encounter, then numeric bucket concatenation"
                           if seed is not None else "explicit recorded indices"),
        "indices": indices,
        "bucket_names": names,
        "rows_sha256": digest.hexdigest(),
    }


def validate_benchmark(dataset, manifest):
    if (manifest.get("schema_version") != 1 or manifest.get("dataset") != DATASET_NAME
            or manifest.get("revision") != DATASET_REVISION or manifest.get("split") != "test"):
        raise ValueError("Unsupported dataset or benchmark revision")
    actual = manifest_for_indices(dataset, manifest["indices"], manifest["bucket_names"])
    if actual["rows_sha256"] != manifest["rows_sha256"]:
        raise ValueError("Benchmark row content does not match the recorded checksum")
    return dataset.select(manifest["indices"])
