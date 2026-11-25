"""Check how far the message gates open and how large the carried state gets.

Usage: python -m looping.attention_residuals.gate_check [runs/attention_residuals/train]

Runs each arm's final weights for 64 iterations on 200 benchmark puzzles (CPU, FP32) and prints,
at a few iteration counts, how many puzzles are solved, how many answers changed in that
iteration, and the RMS of the carried state. For the message arms it also prints, per gated
layer input at iteration 64, the mean of |1 - keep|, the mean of |gain|, and the message's share
of the mixed input's norm.
"""

import json
import sys

import torch
from datasets import load_dataset

import iters.eval_more_iters as evaluator
from inference import ROPE_COS, ROPE_SIN
from looping.attention_residuals.evaluate import variant_loader
from looping.attention_residuals.model import MessageGate, unit_rms
from looping.attention_residuals.summarize import ARMS, SEEDS
from model_io import load_model

REPORTED_ITERATIONS = (1, 4, 16, 32, 64)


def check(model, inputs, targets):
    gate_records = {"closed": [], "gain": [], "message_share": []}

    def record_gate(gate, arguments, mixed):
        _, current_unit, message = arguments
        reduced = gate.from_input(current_unit) + gate.from_message(unit_rms(message))
        gain = torch.tanh(gate.gain(reduced))
        gate_records["closed"].append(torch.tanh(gate.keep(reduced)).abs().mean().item())
        gate_records["gain"].append(gain.abs().mean().item())
        gate_records["message_share"].append(((gain * message).norm() / mixed.norm()).item())

    gates = [module for module in model.modules() if isinstance(module, MessageGate)]
    handles = [gate.register_forward_hook(record_gate) for gate in gates]
    rows = {}
    with torch.no_grad():
        state = model.initial_encoder(inputs)
        predictions = torch.zeros(inputs.shape[0], 81, 9)
        previous_digits = None
        for iteration in range(1, max(REPORTED_ITERATIONS) + 1):
            state = model.recurrent_step(state, predictions, ROPE_COS, ROPE_SIN)
            logits = model.output_head(state)
            predictions = logits.softmax(-1)
            digits = logits.argmax(-1)
            if iteration in REPORTED_ITERATIONS:
                rows[iteration] = {
                    "solved": int((digits == targets).all(-1).sum()),
                    "changed": None if previous_digits is None
                    else int((digits != previous_digits).any(-1).sum()),
                    # The message arms carry four width-128 parts; the last is the second block's sum.
                    "state_rms": round(state[..., -128:].square().mean().sqrt().item(), 1),
                }
            previous_digits = digits
    for handle in handles:
        handle.remove()
    last_iteration = {name: [round(value, 2) for value in values[-len(gates):]]
                      for name, values in gate_records.items()} if gates else None
    return rows, last_iteration


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else "runs/attention_residuals/train"
    manifest = json.load(open("release/benchmark_25k.json"))
    dataset = load_dataset(evaluator.DATASET_NAME, revision=evaluator.DATASET_REVISION, split="test")
    puzzles = dataset[manifest["indices"][::125]]  # 200 puzzles, 40 from each difficulty bucket
    inputs = evaluator.encode_puzzles(puzzles["question"])
    targets = evaluator.encode_solutions(puzzles["answer"]).long()
    for arm in ARMS:
        for seed in SEEDS:
            with variant_loader(arm):
                model, _ = load_model(f"{directory}/model_attnres20k_{arm}_seed{seed}.pt")
            rows, gates = check(model.eval(), inputs, targets)
            print(f"{arm} {seed}")
            for iteration, row in rows.items():
                print(f"  iteration {iteration}: {row}")
            if gates:
                print(f"  gates at iteration {max(REPORTED_ITERATIONS)}: {gates}")


if __name__ == "__main__":
    main()
