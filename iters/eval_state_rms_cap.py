"""Evaluate recurrent-state RMS caps on stable, collapsed, and rescued models."""

import argparse
import importlib
import json
import os
import random
import re
import time
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from datasets import load_dataset

from iters.state_norm import cap_token_rms, per_token_rms


RATING_BUCKETS = (
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
)
DEFAULT_MODELS = (
    ("stable_bp", "/outputs/model_baseline_lr2e3.pt"),
    ("collapsed_pre_es", "/outputs/model_baseline_lr2e3_clean_a.pt"),
    ("rescued_post_es", "/outputs/model_es_ft_collapsed.pt"),
)
DEFAULT_CAPS = (None, 32.0, 64.0, 128.0)
DEFAULT_CHECKPOINTS = (16, 128, 1024, 2048)


def _load_model(module, model_path, device, model_kwargs):
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    model = module.SudokuTransformer(**model_kwargs).to(device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def _load_balanced_sample(module, examples_per_bucket, seed):
    dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, example in enumerate(dataset):
        rating = example["rating"]
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= rating <= maximum:
                bucket_indices[name].append(index)
                break

    generator = random.Random(seed)
    selected_indices = []
    bucket_names = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        if len(candidates) < examples_per_bucket:
            raise ValueError(
                f"bucket {name!r} has {len(candidates)} examples, "
                f"fewer than requested {examples_per_bucket}"
            )
        selected_indices.extend(generator.sample(candidates, examples_per_bucket))
        bucket_names.extend([name] * examples_per_bucket)

    puzzles = [dataset[index]["question"] for index in selected_indices]
    solutions = [dataset[index]["answer"] for index in selected_indices]
    inputs = module.encode_puzzles(puzzles)
    targets = module.encode_solutions(solutions).long()
    empty_mask = torch.tensor(
        [[character == "." for character in puzzle] for puzzle in puzzles],
        dtype=torch.bool,
    )
    return inputs, targets, empty_mask, bucket_names


def _cap_label(maximum_rms):
    return "uncapped" if maximum_rms is None else f"cap_{maximum_rms:g}"


def _constraint_label(maximum_rms, model_kwargs):
    if maximum_rms is not None:
        return _cap_label(maximum_rms)
    if model_kwargs.get("outer_state_norm"):
        return "rmsnorm"
    return "uncapped"


def _summarize_rms(chunks):
    values = torch.cat(chunks).float()
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "p90": torch.quantile(values, 0.9).item(),
        "max": values.max().item(),
    }


def _summarize_checkpoint(
    predictions,
    targets,
    empty_mask,
    bucket_names,
    pre_cap_rms,
    post_cap_rms,
    maximum_rms,
):
    predictions = torch.cat(predictions)
    solved = ((predictions == targets) | ~empty_mask).all(dim=1)
    per_bucket = {}
    for _, _, name in RATING_BUCKETS:
        bucket_mask = torch.tensor([bucket_name == name for bucket_name in bucket_names])
        per_bucket[name] = {
            "solved": int((solved & bucket_mask).sum().item()),
            "total": int(bucket_mask.sum().item()),
        }

    pre_cap_values = torch.cat(pre_cap_rms).float()
    result = {
        "solved": int(solved.sum().item()),
        "total": len(solved),
        "accuracy": solved.float().mean().item(),
        "per_bucket": per_bucket,
        "pre_cap_token_rms": _summarize_rms(pre_cap_rms),
        "post_cap_token_rms": _summarize_rms(post_cap_rms),
    }
    if maximum_rms is not None:
        result["capped_token_fraction"] = (pre_cap_values > maximum_rms).float().mean().item()
    return result


def evaluate_trajectory(
    model,
    module,
    inputs,
    targets,
    empty_mask,
    bucket_names,
    checkpoints,
    maximum_rms,
    batch_size,
    device,
):
    saved = {
        iteration: {"predictions": [], "pre_cap_rms": [], "post_cap_rms": []}
        for iteration in checkpoints
    }
    rope_cos = module.ROPE_COS.to(device)
    rope_sin = module.ROPE_SIN.to(device)
    max_iterations = max(checkpoints)

    for start in range(0, len(inputs), batch_size):
        batch_inputs = inputs[start:start + batch_size].to(device)
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if device.type == "cuda"
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            hidden_state = model.initial_encoder(batch_inputs)
            predictions = torch.zeros(
                batch_inputs.size(0), 81, 9, device=device
            )
            for iteration in range(1, max_iterations + 1):
                next_hidden = hidden_state + model.pred_proj(predictions)
                for layer in model.layers:
                    next_hidden = layer(next_hidden, rope_cos, rope_sin)

                pre_cap = per_token_rms(next_hidden)
                normalize_outer_state = getattr(model, "normalize_outer_state", None)
                if normalize_outer_state is not None:
                    next_hidden = normalize_outer_state(next_hidden)
                if maximum_rms is not None:
                    next_hidden = cap_token_rms(next_hidden, maximum_rms)
                post_cap = per_token_rms(next_hidden)
                hidden_state = next_hidden
                logits = model.output_head(hidden_state)
                predictions = F.softmax(logits, dim=-1)

                if iteration in saved:
                    saved[iteration]["predictions"].append(
                        logits.argmax(dim=-1).cpu()
                    )
                    saved[iteration]["pre_cap_rms"].append(pre_cap.cpu().flatten())
                    saved[iteration]["post_cap_rms"].append(post_cap.cpu().flatten())

    return {
        str(iteration): _summarize_checkpoint(
            values["predictions"],
            targets,
            empty_mask,
            bucket_names,
            values["pre_cap_rms"],
            values["post_cap_rms"],
            maximum_rms,
        )
        for iteration, values in saved.items()
    }


def evaluate(
    model_configs=DEFAULT_MODELS,
    experiment_module="iters.exp_baseline_lr2e3",
    caps=DEFAULT_CAPS,
    checkpoints=DEFAULT_CHECKPOINTS,
    examples_per_bucket=200,
    batch_size=250,
    seed=42,
    device="cuda",
    output_dir=None,
    output_prefix="state_rms_cap_probe",
    model_kwargs=None,
):
    if not checkpoints or any(iteration <= 0 for iteration in checkpoints):
        raise ValueError("checkpoints must contain positive iteration counts")
    if tuple(sorted(set(checkpoints))) != tuple(checkpoints):
        raise ValueError("checkpoints must be strictly increasing")
    if any(cap is not None and cap <= 0 for cap in caps):
        raise ValueError("caps must be positive or None")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")

    model_kwargs = dict(model_kwargs or {})

    module = importlib.import_module(experiment_module)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    inputs, targets, empty_mask, bucket_names = _load_balanced_sample(
        module, examples_per_bucket, seed
    )

    log_file = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = open(os.path.join(output_dir, f"{output_prefix}.log"), "w")

    def log(message=""):
        print(message, flush=True)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    evaluation_config = {
        "experiment_module": experiment_module,
        "models": dict(model_configs),
        "caps": list(caps),
        "checkpoints": list(checkpoints),
        "examples_per_bucket": examples_per_bucket,
        "sample_size": len(inputs),
        "batch_size": batch_size,
        "seed": seed,
        "device": str(device),
        "model_kwargs": model_kwargs,
        "normalization_axis": "per token over d_model",
    }
    if model_kwargs.get("outer_state_norm"):
        evaluation_config["model_constraint_behavior"] = (
            "divide every token by its feature RMS after each recurrent loop"
        )
    if any(cap is not None for cap in caps):
        evaluation_config["additional_cap_behavior"] = (
            "rescale only above threshold; no mean subtraction"
        )
    summary = {"config": evaluation_config, "models": {}}

    started_at = time.time()
    log(
        f"Recurrent-state probe: {len(inputs)} puzzles, models={len(model_configs)}, "
        f"constraints={[_constraint_label(cap, model_kwargs) for cap in caps]}, "
        f"checkpoints={list(checkpoints)}"
    )
    for model_name, model_path in model_configs:
        log(f"\nMODEL {model_name}: {model_path}")
        model = _load_model(module, model_path, device, model_kwargs)
        summary["models"][model_name] = {}
        for maximum_rms in caps:
            label = _constraint_label(maximum_rms, model_kwargs)
            trajectory_started_at = time.time()
            metrics = evaluate_trajectory(
                model,
                module,
                inputs,
                targets,
                empty_mask,
                bucket_names,
                checkpoints,
                maximum_rms,
                batch_size,
                device,
            )
            summary["models"][model_name][label] = metrics
            result_text = ", ".join(
                f"@{iteration} {metrics[str(iteration)]['solved']}/{len(inputs)}"
                for iteration in checkpoints
            )
            log(
                f"  {label:>10}: {result_text} "
                f"({time.time() - trajectory_started_at:.1f}s)"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary["elapsed_seconds"] = time.time() - started_at
    if output_dir:
        result_path = os.path.join(output_dir, f"{output_prefix}.json")
        with open(result_path, "w") as result_file:
            json.dump(summary, result_file, indent=2)
            result_file.write("\n")
        log(f"\nStructured results: {result_path}")
    log(f"Total time: {summary['elapsed_seconds']:.1f}s")
    if log_file:
        log_file.close()
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-per-bucket", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--caps", type=float, nargs="+", default=(32.0, 64.0, 128.0))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir")
    parser.add_argument("--output-prefix", default="state_rms_cap_probe")
    arguments = parser.parse_args()
    evaluate(
        caps=(None, *arguments.caps),
        checkpoints=tuple(arguments.checkpoints),
        examples_per_bucket=arguments.examples_per_bucket,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        device=arguments.device,
        output_dir=arguments.output_dir,
        output_prefix=arguments.output_prefix,
    )
