"""Test whether BF16 inference fails at large iteration counts because of how the state is stored.

The recurrent state's magnitude grows roughly linearly with the iteration count while each
iteration's update stays about the same size. A BF16 number has 8 significant bits, so once a
state element is larger than about 256 times the update, adding the update can round to no
change at all. This script compares three ways of running the same weights on the same puzzles:

- `fp32`: everything in FP32.
- `bf16`: BF16 autocast, with the carried state also stored in BF16 (what the trainer's
  monitoring and the historical evaluations did).
- `bf16_fp32_state`: the same BF16 autocast for every matrix multiply and attention call, but
  the carried state is stored in FP32. The only change is casting the initial state to FP32;
  PyTorch's type promotion then keeps every residual addition in FP32.

It records accuracy at several iteration counts, and for each variant the fraction of state
elements that an iteration leaves bit-for-bit unchanged.
"""

import json
import os

import torch
import torch.nn.functional as F

from model_io import load_model
from paper.scripts.trajectory_stats import benchmark_sample
from stabilize.exp_testbed_20k import ROPE_COS, ROPE_SIN, encode_puzzles, encode_solutions

RECORD_AT = (16, 64, 128, 256, 512, 1024, 2048, 4096)


@torch.inference_mode()
def run_variant(model, inputs, targets, variant, max_iterations):
    empty = inputs[:, :, 0].bool()
    rope_cos, rope_sin = ROPE_COS.to(inputs.device), ROPE_SIN.to(inputs.device)
    use_autocast = variant != "fp32"
    torch.set_float32_matmul_precision("highest" if variant == "fp32" else "high")
    rows = {}
    with torch.autocast(inputs.device.type, dtype=torch.bfloat16, enabled=use_autocast):
        hidden = model.initial_encoder(inputs)
        if variant == "bf16_fp32_state":
            hidden = hidden.float()
        predictions = torch.zeros(inputs.shape[0], 81, 9, device=inputs.device)
        for iteration in range(1, max_iterations + 1):
            new_hidden = model.recurrent_step(hidden, predictions, rope_cos, rope_sin)
            logits = model.output_head(new_hidden)
            predictions = F.softmax(logits.float(), dim=-1)
            if iteration in RECORD_AT:
                solved = ((logits.argmax(-1) == targets) | ~empty).all(-1)
                rows[str(iteration)] = {
                    "solved": int(solved.sum()), "total": int(solved.numel()),
                    "state_dtype": str(new_hidden.dtype),
                    "state_rms": float(new_hidden.float().pow(2).mean().sqrt()),
                    "update_rms": float((new_hidden.float() - hidden.float()).pow(2).mean().sqrt()),
                    "unchanged_state_fraction": float((new_hidden == hidden).float().mean()),
                }
            hidden = new_hidden
    return rows


def compare(model_path, output_path, *, legacy_exp=None, max_iterations=4096,
            puzzles_per_bucket=200, device="cuda"):
    device = torch.device(device)
    model, manifest = load_model(
        model_path, legacy_exp=legacy_exp, legacy_defaults=bool(legacy_exp), device=device,
    )
    model = model.float().eval()
    questions, answers, _, _, _ = benchmark_sample(puzzles_per_bucket)
    inputs = encode_puzzles(questions).to(device)
    targets = encode_solutions(answers).long().to(device)
    result = {
        "model": str(model_path), "weights_sha256": manifest["weights"]["sha256"],
        "puzzles": int(inputs.shape[0]),
        "variants": {
            variant: run_variant(model, inputs, targets, variant, max_iterations)
            for variant in ("fp32", "bf16", "bf16_fp32_state")
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path + ".tmp", "w") as handle:
        json.dump(result, handle, indent=2)
    os.replace(output_path + ".tmp", output_path)
    return result
