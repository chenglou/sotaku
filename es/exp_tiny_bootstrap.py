# Backprop bootstrap for the growth program: train the 52K-parameter tiny model
# (es/exp_es_tiny.py architecture) with ordinary supervised training at 16 iterations,
# just long enough to build the feature structure that gives ES a fitness slope —
# exp_es_tiny.py showed pure ES from random weights has none, at any tested scale or
# horizon. 5,000 steps, mixed sampling (no curriculum), cosine decay. The product,
# model_tiny_bootstrap.pt, seeds the ES-plus-growth stage. Long-horizon stability is
# NOT a goal here; only short-horizon competence is.

import os
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from es.exp_es_tiny import TINY_ROPE_COS, TINY_ROPE_SIN, TinySudokuTransformer
from iters.exp_baseline_lr2e3 import RATING_BUCKETS, encode_puzzles

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "tiny_bootstrap_checkpoint_step"

CONFIG = {
    'experiment': 'exp_tiny_bootstrap',
    'd_model': 32,
    'n_heads': 4,
    'd_ff': 128,
    'n_layers': 4,
    'total_steps': 5000,
    'batch_size': 2048,
    'peak_lr': 2e-3,
    'train_iterations': 16,
    'sampling': 'mixed',
}

total_steps = 5000
eval_every = 1000
batch_size = 2048
peak_lr = 2e-3
warmup_steps = 200
final_lr_frac = 0.05
train_iterations = 16
train_rows = 1_000_000
log_name = "exp_tiny_bootstrap.log"


def train(output_dir="."):
    device = torch.device("cuda")

    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("Loading training rows...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    rows = dataset[:train_rows]
    x_all = encode_puzzles(rows["question"]).to(device)
    targets_all = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in rows["answer"]], device=device)
    empty_all = torch.tensor([[p[j] == '.' for j in range(81)] for p in rows["question"]], device=device)
    log(f"Training pool: {x_all.size(0)} puzzles")

    log("Loading validation probe (test split, 200 per rating bucket)...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    probe_idx = []
    per_bucket = {name: 0 for _, _, name in RATING_BUCKETS}
    for i in range(len(test_dataset)):
        r = test_dataset[i]['rating']
        for min_r, max_r, name in RATING_BUCKETS:
            if min_r <= r <= max_r and per_bucket[name] < 200:
                probe_idx.append(i)
                per_bucket[name] += 1
                break
        if len(probe_idx) >= 1000:
            break
    probe_x = encode_puzzles([test_dataset[i]['question'] for i in probe_idx]).to(device)
    probe_targets = torch.tensor([[int(test_dataset[i]['answer'][j]) - 1 for j in range(81)] for i in probe_idx], device=device)
    probe_empty = torch.tensor([[test_dataset[i]['question'][j] == '.' for j in range(81)] for i in probe_idx], device=device)
    log(f"Validation probe: {probe_x.size(0)} puzzles")

    model = TinySudokuTransformer().to(device)
    rope_cos = TINY_ROPE_COS.to(device)
    rope_sin = TINY_ROPE_SIN.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=0.01)

    start_step = 0
    checkpoint_path, _ = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step'] + 1
        log(f"Resumed from step {ckpt['step']}")

    def lr_at(step):
        if step < warmup_steps:
            return peak_lr * (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return peak_lr * (final_lr_frac + (1 - final_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))

    def probe(n_iters):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            solved = 0
            for start in range(0, probe_x.size(0), 500):
                xb = probe_x[start:start + 500]
                h_prev = model.initial_encoder(xb)
                preds = torch.zeros(xb.size(0), 81, 9, device=device)
                for _ in range(n_iters):
                    h = h_prev + model.pred_proj(preds)
                    for layer in model.layers:
                        h = layer(h, rope_cos, rope_sin)
                    h_prev = h
                    preds = F.softmax(model.output_head(h), dim=-1)
                final = model.output_head(h_prev).argmax(dim=-1)
                tb = probe_targets[start:start + 500]
                eb = probe_empty[start:start + 500]
                solved += int((((final == tb) & eb) | ~eb).all(dim=1).sum().item())
        return solved

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step)

        idx = torch.randint(0, x_all.size(0), (batch_size,), device=device)
        xb, tb, eb = x_all[idx], targets_all[idx], empty_all[idx]

        with torch.autocast('cuda', dtype=torch.bfloat16):
            h_prev = model.initial_encoder(xb)
            preds = torch.zeros(batch_size, 81, 9, device=device)
            loss = 0.0
            for _ in range(train_iterations):
                h = h_prev + model.pred_proj(preds)
                for layer in model.layers:
                    h = layer(h, rope_cos, rope_sin)
                h_prev = h
                logits = model.output_head(h)
                loss = loss + F.cross_entropy(logits.float()[eb], tb[eb])
                preds = F.softmax(logits, dim=-1)
            loss = loss / train_iterations

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            log(f"STEP {step:5d} | loss {loss.item():.4f} | lr {lr_at(step):.2e} | {time.time() - t0:.0f}s")

        if step % eval_every == 0 or step == total_steps - 1:
            model.eval()
            p16 = probe(16)
            p128 = probe(128)
            model.train()
            log(f"PROBE {step:5d} | 16-iter: {p16}/1000 | 128-iter: {p128}/1000")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
            }, os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt"))

    final_path = os.path.join(output_dir, "model_tiny_bootstrap.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_probe_16': p16}


if __name__ == "__main__":
    train()
