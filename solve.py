"""Run one puzzle with ordinary recurrent inference and preserve the given digits."""

import argparse

import torch

from inference import RecurrentRunner
from model_io import load_model
from stabilize.exp_testbed_20k import encode_puzzles


def normalize_puzzle(text):
    puzzle = "".join(text.split()).replace("0", ".")
    if len(puzzle) != 81 or any(character not in ".123456789" for character in puzzle):
        raise ValueError("A puzzle must contain 81 digits, dots, or zeros, optionally separated by whitespace")
    return puzzle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights")
    parser.add_argument("puzzle", help="81 characters; use . or 0 for blanks")
    parser.add_argument("--iters", type=int, default=1024)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--manifest")
    args = parser.parse_args()
    puzzle = normalize_puzzle(args.puzzle)
    device = torch.device(args.device)
    precision = args.precision
    if precision == "bf16" and device.type != "cuda":
        parser.error("BF16 inference requires CUDA; use FP32 on CPU")
    model, _ = load_model(args.weights, manifest_path=args.manifest, device=device)
    torch.set_float32_matmul_precision("highest" if precision == "fp32" else "high")
    autocast = torch.autocast(device.type, dtype=torch.bfloat16, enabled=precision == "bf16")
    with autocast:
        outputs, _ = RecurrentRunner(model, compiled=args.compiled).run_batch(encode_puzzles([puzzle]).to(device), [args.iters])
    predictions = outputs[args.iters].argmax(-1)[0].tolist()
    answer = "".join(str(predictions[index] + 1) if digit == "." else digit for index, digit in enumerate(puzzle))
    for row in range(9):
        print(answer[row * 9:(row + 1) * 9])


if __name__ == "__main__":
    main()
