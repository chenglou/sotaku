"""Score every saved training checkpoint of one run with ordinary FP32 inference.

The trainer's own monitoring uses BF16 autocast, which can lower scores at large iteration
counts. This script re-scores the same checkpoints in FP32 on the first `puzzles_per_bucket`
puzzles of each rating bucket of the frozen benchmark, so training curves in the paper do not
depend on inference precision.
"""

import glob
import json
import os
import re

import torch
import torch.nn.functional as F

from model_io import build_model, settings_from_training_config
from paper.scripts.trajectory_stats import benchmark_sample
from stabilize.exp_testbed_20k import ROPE_COS, ROPE_SIN, encode_puzzles, encode_solutions


@torch.inference_mode()
def solved_counts(model, inputs, targets, iterations):
    empty = inputs[:, :, 0].bool()
    rope_cos, rope_sin = ROPE_COS.to(inputs.device), ROPE_SIN.to(inputs.device)
    hidden = model.initial_encoder(inputs)
    predictions = torch.zeros(inputs.shape[0], 81, 9, device=inputs.device)
    counts = {}
    for iteration in range(1, max(iterations) + 1):
        hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
        logits = model.output_head(hidden)
        predictions = F.softmax(logits, dim=-1)
        if iteration in iterations:
            counts[iteration] = int((((logits.argmax(-1) == targets) | ~empty).all(-1)).sum())
    return counts


def score_checkpoints(checkpoint_dir, run_name, output_path, *, iterations=(128, 1024),
                      puzzles_per_bucket=200, device="cuda"):
    torch.set_float32_matmul_precision("highest")
    device = torch.device(device)
    questions, answers, _, _, _ = benchmark_sample(puzzles_per_bucket)
    inputs = encode_puzzles(questions).to(device)
    targets = encode_solutions(answers).long().to(device)
    paths = {}
    for path in glob.glob(os.path.join(checkpoint_dir, f"{run_name}_checkpoint_step*.pt")):
        paths[int(re.search(r"step(\d+)\.pt$", path).group(1))] = path
    rows = []
    for step in sorted(paths):
        # These are our own training checkpoints; they include pickled RNG state.
        checkpoint = torch.load(paths[step], map_location="cpu", weights_only=False)
        settings = checkpoint.get("model_settings") or settings_from_training_config(checkpoint["config"])
        model = build_model(settings).float()
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        counts = solved_counts(model, inputs, targets, tuple(iterations))
        rows.append({"step": step, "total": int(inputs.shape[0]),
                     **{f"solved_{iteration}": count for iteration, count in counts.items()}})
        print(rows[-1], flush=True)
    summary = {
        "run": run_name, "precision": "fp32", "matmul_precision": "highest",
        "iterations": list(iterations), "checkpoints": len(rows), "rows": rows,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path + ".tmp", "w") as handle:
        json.dump(summary, handle, indent=2)
    os.replace(output_path + ".tmp", output_path)
    return summary
