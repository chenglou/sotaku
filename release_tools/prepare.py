"""Prepare private release assets without publishing or modifying their weight bytes."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import torch

from checkpoint_utils import atomic_json_save
from dataset_utils import DATASET_NAME, DATASET_REVISION, benchmark_manifest
from model_io import build_model, load_model, settings_from_training_config, write_model_manifest
from runtime_utils import file_sha256


def export_weights(weights_path, checkpoint_path, output_path, *, trust_checkpoint=False):
    if not trust_checkpoint:
        raise ValueError("Export requires explicit trust in the training checkpoint's pickled state")
    weights_path, checkpoint_path, output_path = map(Path, (weights_path, checkpoint_path, output_path))
    source = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if set(weights) != set(source["model_state_dict"]):
        raise ValueError("Weight keys do not match the training checkpoint")
    if any(not torch.equal(value, source["model_state_dict"][key]) for key, value in weights.items()):
        raise ValueError("The selected weight file is not the supplied checkpoint's model")
    settings = source.get("model_settings") or settings_from_training_config(source["config"])
    model = build_model(settings)
    model.load_state_dict(weights)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace release asset {output_path}")
    shutil.copyfile(weights_path, output_path)
    manifest = write_model_manifest(
        output_path, model,
        training={"config": source["config"], "last_step": source["step"], "optimizer_updates": source["step"] + 1},
        provenance={
            "source_weights": weights_path.name, "source_weights_sha256": file_sha256(weights_path),
            "source_checkpoint": checkpoint_path.name, "source_checkpoint_sha256": file_sha256(checkpoint_path),
            "dataset": DATASET_NAME, "dataset_revision": DATASET_REVISION,
            "training_code_revision": None,
            "metadata_origin": "recovered training config and verified historical implementation defaults",
        },
    )
    if file_sha256(output_path) != file_sha256(weights_path):
        raise AssertionError("Release export changed the original weight bytes")
    return manifest


def package_records(weights_path, records_dir, benchmark_path, archive_path):
    weights_path, records_dir, benchmark_path, archive_path = map(
        Path, (weights_path, records_dir, benchmark_path, archive_path),
    )
    load_model(weights_path)
    verified = json.loads((records_dir / "verified_records.json").read_text())["verified"]
    results = sorted(records_dir.rglob("result.json"))
    if not results:
        raise ValueError("No evaluated records to package")
    for result_path in results:
        name = str(result_path.parent.relative_to(records_dir))
        if verified.get(name, {}).get("result_sha256") != file_sha256(result_path):
            raise ValueError(f"Run verify_records before packaging {name}")
        result = json.loads(result_path.read_text())
        if file_sha256(result_path.parent / "per_puzzle.npz") != result["per_puzzle_sha256"]:
            raise ValueError(f"Prediction records changed for {name}")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(benchmark_path, "benchmark_25k.json")
        for path in sorted(records_dir.rglob("*")):
            is_source_archive = path.name.startswith("sources_") and path.name.endswith(".tar.gz")
            if path.is_file() and (path.suffix in (".json", ".npz", ".md") or is_source_archive):
                archive.write(path, Path("validation") / path.relative_to(records_dir))
    files = (weights_path, Path(str(weights_path) + ".json"), archive_path)
    checksums = {path.name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in files}
    atomic_json_save(checksums, archive_path.with_suffix(".checksums.json"))
    return checksums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser("export")
    export.add_argument("--weights", required=True)
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--trust-training-checkpoint", action="store_true")
    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--output", required=True)
    benchmark.add_argument("--cached-arrow", help="Use an already downloaded copy of the pinned test split")
    package = subparsers.add_parser("package")
    package.add_argument("--weights", required=True)
    package.add_argument("--records", required=True)
    package.add_argument("--benchmark", required=True)
    package.add_argument("--output", required=True, help="New zip file; existing archives are never replaced")
    args = parser.parse_args()
    if args.command == "export":
        print(json.dumps(export_weights(args.weights, args.checkpoint, args.output,
                                       trust_checkpoint=args.trust_training_checkpoint), indent=2))
    elif args.command == "package":
        print(json.dumps(package_records(args.weights, args.records, args.benchmark, args.output), indent=2))
    else:
        from datasets import Dataset, load_dataset
        if Path(args.output).exists():
            raise FileExistsError(args.output)
        dataset = Dataset.from_file(args.cached_arrow) if args.cached_arrow else load_dataset(
            DATASET_NAME, revision=DATASET_REVISION, split="test",
        )
        manifest = benchmark_manifest(dataset)
        atomic_json_save(manifest, args.output)
        print(f"Frozen {len(manifest['indices'])} rows; SHA-256 {manifest['rows_sha256']}")


if __name__ == "__main__":
    main()
