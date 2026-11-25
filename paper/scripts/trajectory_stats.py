"""Per-iteration measurements of one checkpoint's recurrent trajectory, for the paper's figures.

Runs ordinary FP32 inference on the first `puzzles_per_bucket` puzzles of each rating bucket of
the frozen 25K benchmark and records, at every iteration and for every puzzle:

- whether the puzzle is solved, and how many empty cells are wrong
- hidden-state RMS and the RMS of the one-iteration update
- how much the per-cell state direction changed (1 - cosine with the previous iteration)
- the smallest correct-answer margin over the empty cells
  (correct digit's logit minus the largest other logit)

It also saves the predictions of a few example puzzles at every iteration, and the mean
attention pattern of every head at selected iterations.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from dataset_utils import DATASET_NAME, DATASET_REVISION
from model_io import load_model
from stabilize.exp_testbed_20k import (
    ROPE_COS, ROPE_SIN, apply_rope, encode_puzzles, encode_solutions,
)

ATTENTION_ITERATIONS = (1, 2, 4, 16, 64, 256, 1024)
EXAMPLES_PER_BUCKET = 4
EXAMPLE_ITERATIONS = 1024


def benchmark_sample(puzzles_per_bucket, benchmark_path="release/benchmark_25k.json"):
    benchmark = json.loads(Path(benchmark_path).read_text())
    taken, positions = {}, []
    for position, bucket in enumerate(benchmark["bucket_names"]):
        if taken.get(bucket, 0) < puzzles_per_bucket:
            taken[bucket] = taken.get(bucket, 0) + 1
            positions.append(position)
    indices = [benchmark["indices"][position] for position in positions]
    buckets = [benchmark["bucket_names"][position] for position in positions]
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split="test")
    rows = dataset.select(indices)
    return rows["question"], rows["answer"], [int(value) for value in rows["rating"]], buckets, indices


class AttentionRecorder:
    """Averages every head's 81x81 attention matrix over puzzles when enabled."""

    def __init__(self, model):
        self.model = model
        self.enabled = False
        self.current = []
        self.hooks = [
            layer.norm1.register_forward_hook(self._hook(layer)) for layer in model.layers
        ]

    def _hook(self, layer):
        def record(module, inputs, normalized):
            if not self.enabled:
                return
            batch, length, _ = normalized.shape
            shape = (batch, length, layer.n_heads, layer.head_dim)
            queries = layer.q_proj(normalized).view(shape).transpose(1, 2)
            keys = layer.k_proj(normalized).view(shape).transpose(1, 2)
            cos, sin = ROPE_COS.to(normalized.device), ROPE_SIN.to(normalized.device)
            queries, keys = apply_rope(queries, cos, sin), apply_rope(keys, cos, sin)
            weights = F.softmax((queries @ keys.transpose(-2, -1)) * layer.head_dim ** -0.5, dim=-1)
            self.current.append(weights.mean(0).double().cpu())
        return record

    def close(self):
        for hook in self.hooks:
            hook.remove()


@torch.inference_mode()
def collect(model_path, output_dir, *, legacy_exp=None, max_iterations=4096,
            puzzles_per_bucket=200, device="cuda"):
    torch.set_float32_matmul_precision("highest")
    device = torch.device(device)
    model, manifest = load_model(
        model_path, legacy_exp=legacy_exp, legacy_defaults=bool(legacy_exp), device=device,
    )
    model = model.float().eval()
    questions, answers, ratings, buckets, indices = benchmark_sample(puzzles_per_bucket)
    inputs = encode_puzzles(questions).to(device)
    targets = encode_solutions(answers).long().to(device)
    empty = inputs[:, :, 0].bool()
    target_one_hot = F.one_hot(targets, 9).bool()
    count = inputs.shape[0]
    rope_cos, rope_sin = ROPE_COS.to(device), ROPE_SIN.to(device)

    example_positions = []
    seen = {}
    for position, bucket in enumerate(buckets):
        if seen.get(bucket, 0) < EXAMPLES_PER_BUCKET:
            seen[bucket] = seen.get(bucket, 0) + 1
            example_positions.append(position)
    example_index = torch.tensor(example_positions, device=device)

    float_names = ("state_rms", "update_rms", "direction_change", "update_cosine", "min_margin")
    series = {name: np.zeros((max_iterations, count), dtype=np.float32) for name in float_names}
    solved = np.zeros((max_iterations, count), dtype=bool)
    wrong_cells = np.zeros((max_iterations, count), dtype=np.int16)
    changed_cells = np.zeros((max_iterations, count), dtype=np.int16)
    example_steps = min(EXAMPLE_ITERATIONS, max_iterations)
    example_predictions = np.zeros((example_steps, len(example_positions), 81), dtype=np.uint8)
    example_confidence = np.zeros((example_steps, len(example_positions), 81), dtype=np.float16)
    example_margin = np.zeros((example_steps, len(example_positions), 81), dtype=np.float32)

    recorder = AttentionRecorder(model)
    attention = {}
    hidden = model.initial_encoder(inputs)
    predictions = torch.zeros(count, 81, 9, device=device)
    previous_update = torch.zeros_like(hidden)
    previous_digits = torch.full((count, 81), -1, device=device, dtype=torch.long)
    nonfinite_at = None
    for iteration in range(1, max_iterations + 1):
        recorder.enabled = iteration in ATTENTION_ITERATIONS
        recorder.current = []
        new_hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
        if recorder.enabled:
            attention[iteration] = torch.stack(recorder.current).numpy().astype(np.float32)
        logits = model.output_head(new_hidden)
        predictions = F.softmax(logits, dim=-1)
        if nonfinite_at is None and not (torch.isfinite(new_hidden).all() and torch.isfinite(logits).all()):
            nonfinite_at = iteration
        update = new_hidden - hidden
        row = iteration - 1
        series["state_rms"][row] = new_hidden.pow(2).mean((1, 2)).sqrt().cpu().numpy()
        series["update_rms"][row] = update.pow(2).mean((1, 2)).sqrt().cpu().numpy()
        series["direction_change"][row] = (
            1 - F.cosine_similarity(new_hidden, hidden, dim=-1).mean(1)
        ).cpu().numpy()
        series["update_cosine"][row] = F.cosine_similarity(
            update.flatten(1), previous_update.flatten(1), dim=-1,
        ).cpu().numpy()
        correct_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        strongest_other = logits.masked_fill(target_one_hot, float("-inf")).amax(-1)
        margin = correct_logit - strongest_other
        series["min_margin"][row] = margin.masked_fill(~empty, float("inf")).amin(1).cpu().numpy()
        digits = logits.argmax(-1)
        wrong = (digits != targets) & empty
        wrong_cells[row] = wrong.sum(1).cpu().numpy()
        solved[row] = (~wrong.any(1)).cpu().numpy()
        changed_cells[row] = ((digits != previous_digits) & empty).sum(1).cpu().numpy()
        if iteration <= example_steps:
            example_predictions[row] = digits[example_index].cpu().numpy()
            example_confidence[row] = predictions[example_index].amax(-1).cpu().numpy()
            example_margin[row] = margin[example_index].cpu().numpy()
        hidden, previous_update, previous_digits = new_hidden, update, digits
    recorder.close()

    os.makedirs(output_dir, exist_ok=True)
    arrays = {
        **series, "solved": solved, "wrong_cells": wrong_cells, "changed_cells": changed_cells,
        "ratings": np.asarray(ratings), "buckets": np.asarray(buckets),
        "dataset_indices": np.asarray(indices),
        "empty_cells": empty.sum(1).cpu().numpy(),
        "example_positions": np.asarray(example_positions),
        "example_predictions": example_predictions,
        "example_confidence": example_confidence, "example_margin": example_margin,
        "example_questions": np.asarray([questions[position] for position in example_positions]),
        "example_answers": np.asarray([answers[position] for position in example_positions]),
        "attention_iterations": np.asarray(sorted(attention)),
        "attention": np.stack([attention[key] for key in sorted(attention)]),
    }
    temporary = os.path.join(output_dir, "trajectory.npz.tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, os.path.join(output_dir, "trajectory.npz"))

    checkpoints = [value for value in (16, 128, 512, 1024, 2048, 4096) if value <= max_iterations]
    ever = solved.any(0)
    first_solved = np.where(ever, solved.argmax(0) + 1, 0)
    lost_later = ever & (solved[:-1] & ~solved[1:]).any(0)
    summary = {
        "model": str(model_path), "weights_sha256": manifest["weights"]["sha256"],
        "puzzles": int(count), "max_iterations": max_iterations, "precision": "fp32",
        "matmul_precision": "highest", "nonfinite_at": nonfinite_at,
        "solved_percent": {str(value): float(100 * solved[value - 1].mean()) for value in checkpoints},
        "mean_state_rms": {str(value): float(series["state_rms"][value - 1].mean()) for value in checkpoints},
        "ever_solved": int(ever.sum()), "solved_then_lost_at_least_once": int(lost_later.sum()),
        "median_first_solved_iteration": float(np.median(first_solved[ever])) if ever.any() else None,
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    parser.add_argument("output_dir")
    parser.add_argument("--legacy-exp")
    parser.add_argument("--max-iterations", type=int, default=4096)
    parser.add_argument("--puzzles-per-bucket", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    print(json.dumps(collect(
        arguments.model_path, arguments.output_dir, legacy_exp=arguments.legacy_exp,
        max_iterations=arguments.max_iterations,
        puzzles_per_bucket=arguments.puzzles_per_bucket, device=arguments.device,
    ), indent=2))
