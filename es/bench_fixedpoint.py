# Measure the early-exit ceiling for ES fitness evaluation: for each puzzle, find
# the last iteration at which the model's answer (argmax per cell) changes, over the
# full 1024-iteration horizon. If most puzzles settle early, a fixed-point early exit
# would cut fitness cost proportionally; if answers keep drifting, it wouldn't.
#
# Run on Modal:  modal run --detach modal_run.py --exp es.bench_fixedpoint

import os

import torch
import torch.nn.functional as F
from datasets import load_dataset

from iters.exp_baseline_lr2e3 import ROPE_COS, ROPE_SIN, SudokuTransformer, encode_puzzles

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "bench_fixedpoint_checkpoint_step"
total_steps = 1
eval_every = 1
log_name = "bench_fixedpoint.log"

SEED_MODEL = "model_viridian_canonical_final50k.pt"
PUZZLES = 384
ITERS = 1024


def train(output_dir="."):
    device = torch.device("cuda")
    log_file = open(os.path.join(output_dir, log_name), "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    rows = dataset[2_700_000:2_700_000 + PUZZLES]
    x = encode_puzzles(rows["question"]).to(device)

    model = SudokuTransformer().to(device)
    state = torch.load(os.path.join(output_dir, SEED_MODEL), map_location=device, weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    model.load_state_dict(state)
    model.eval()
    rope_cos = ROPE_COS.to(device)
    rope_sin = ROPE_SIN.to(device)

    last_change = torch.zeros(PUZZLES, dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        h_prev = model.initial_encoder(x)
        preds = torch.zeros(x.size(0), 81, 9, device=device)
        prev_answer = None
        for it in range(ITERS):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            logits = model.output_head(h)
            preds = F.softmax(logits, dim=-1)
            answer = logits.argmax(dim=-1)
            if prev_answer is not None:
                changed = (answer != prev_answer).any(dim=-1)
                last_change[changed] = it
            prev_answer = answer

    lc = last_change.cpu()
    qs = torch.quantile(lc.float(), torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
    log(f"last-change iteration over {PUZZLES} puzzles at {ITERS} iters:")
    log(f"  p50 {qs[0]:.0f} | p75 {qs[1]:.0f} | p90 {qs[2]:.0f} | p95 {qs[3]:.0f} | p99 {qs[4]:.0f} | max {qs[5]:.0f}")
    # Chunk-level exit (whole batch stops when every puzzle has settled, checked
    # every 32 iterations) saves proportionally to max; per-puzzle exit saves vs mean.
    log(f"  mean {lc.float().mean():.0f} | still changing at final iteration: {(lc >= ITERS - 2).sum().item()}/{PUZZLES}")
    log_file.close()
    return {}


if __name__ == "__main__":
    train()
