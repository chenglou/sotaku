# Pixel-level sudoku — the drop-one-level probe. The current unit of composition is
# the CELL: a 10-way one-hot in, a 9-way softmax out. This drops below the symbol:
# each cell is an 8x8 BITMAP (a rendered digit glyph or blank), the model reads raw
# pixels, iterates the usual weight-shared refinement, and writes its answer back as
# pixels. Digits, as a concept, exist only in the training targets' glyph shapes.
# The question: does iteration scaling survive below the symbol level (probe at 128
# vs 16)? If yes, everything 2D-visual becomes one task family — a maze or ARC port
# needs a renderer, not an encoding layer.
#
# First rung honesty: tokens are still cell-aligned 8x8 patches (81 tokens, same
# attention geometry as the digit model), so this drops the symbol, not yet the grid
# alignment. Pure-pixel tokens are the rung below, if this one holds.

import os
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from iters.exp_baseline_lr2e3 import COL_IDX, RATING_BUCKETS, ROW_IDX, RoPETransformerLayer

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "pixel_bootstrap_checkpoint_step"

CONFIG = {
    'experiment': 'exp_pixel_bootstrap',
    'd_model': 64,
    'n_heads': 8,
    'd_ff': 256,
    'n_layers': 4,
    'cell_pixels': 64,
    'total_steps': 8000,
    'batch_size': 1024,
    'peak_lr': 1e-3,
    'train_iterations': 16,
}

total_steps = 8000
eval_every = 1000
batch_size = 1024
peak_lr = 1e-3
warmup_steps = 200
final_lr_frac = 0.05
train_iterations = 16
train_rows = 1_000_000
log_name = "exp_pixel_bootstrap.log"

d_model = 64
n_heads = 8
d_ff = 256
n_layers = 4
CELL_PIXELS = 64   # 8x8 bitmap per cell

# An 8x8 bitmap glyph per digit ('#' = on). Index 0 is the blank cell.
GLYPH_ART = {
    0: ["........", "........", "........", "........", "........", "........", "........", "........"],
    1: ["...#....", "..##....", ".#.#....", "...#....", "...#....", "...#....", ".#####..", "........"],
    2: ["..###...", ".#...#..", ".....#..", "....#...", "...#....", "..#.....", ".#####..", "........"],
    3: ["#####...", "....#...", "...#....", "..###...", ".....#..", "#....#..", ".####...", "........"],
    4: ["....#...", "...##...", "..#.#...", ".#..#...", "#####...", "....#...", "....#...", "........"],
    5: [".#####..", ".#......", ".####...", ".....#..", ".....#..", ".#...#..", "..###...", "........"],
    6: ["...##...", "..#.....", ".#......", ".####...", ".#...#..", ".#...#..", "..###...", "........"],
    7: ["#######.", ".....#..", "....#...", "...#....", "..#.....", ".#......", "#.......", "........"],
    8: [".#####..", ".#...#..", ".#...#..", ".#####..", ".#...#..", ".#...#..", ".#####..", "........"],
    9: ["..####..", ".#....#.", ".#....#.", "..#####.", "......#.", ".....#..", "..###...", "........"],
}
GLYPHS = torch.tensor(
    [[[1.0 if ch == '#' else 0.0 for ch in row] for row in GLYPH_ART[d]] for d in range(10)]
).reshape(10, CELL_PIXELS)


# 2D RoPE for head_dim 8, same construction as the tiny model
head_dim = d_model // n_heads
rope_half = head_dim // 2
rope_pairs = rope_half // 2
_freqs = 1.0 / (10.0 ** (torch.arange(rope_pairs).float() * 2 / rope_half))
_angles = torch.cat([ROW_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0),
                     COL_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)], dim=-1)
PIXEL_ROPE_COS = _angles.cos()
PIXEL_ROPE_SIN = _angles.sin()


class PixelSudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Linear(CELL_PIXELS, d_model)
        self.pred_proj = nn.Linear(CELL_PIXELS, d_model)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, CELL_PIXELS)


def digits_tensor(strings, blank_char, device):
    """Puzzle/solution strings to per-cell glyph indices (0 = blank)."""
    out = torch.zeros(len(strings), 81, dtype=torch.long, device=device)
    for i, s in enumerate(strings):
        for j in range(81):
            out[i, j] = 0 if s[j] == blank_char else int(s[j])
    return out


def run_iterations(model, x_pixels, n_iters, rope_cos, rope_sin):
    """x_pixels: (B, 81, 64) raw bitmaps. Returns final pixel logits (B, 81, 64)."""
    h_prev = model.initial_encoder(x_pixels)
    preds = torch.zeros_like(x_pixels)
    for _ in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        preds = torch.sigmoid(model.output_head(h))
    return model.output_head(h_prev)


def train(output_dir="."):
    device = torch.device("cuda")
    log_file = open(os.path.join(output_dir, log_name), "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    glyphs = GLYPHS.to(device)
    rope_cos = PIXEL_ROPE_COS.to(device)
    rope_sin = PIXEL_ROPE_SIN.to(device)

    log("Loading training rows...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    rows = dataset[:train_rows]
    puzzle_digits = digits_tensor(rows["question"], '.', device)   # (N, 81), 0 = blank
    answer_digits = digits_tensor(rows["answer"], '_', device)     # answers have no blanks
    log(f"Training pool: {puzzle_digits.size(0)} puzzles (stored as glyph indices; rendered per batch)")

    log("Loading validation probe...")
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
    probe_puzzle_digits = digits_tensor([test_dataset[i]['question'] for i in probe_idx], '.', device)
    probe_answer_digits = digits_tensor([test_dataset[i]['answer'] for i in probe_idx], '_', device)
    log(f"Validation probe: {probe_puzzle_digits.size(0)} puzzles")

    model = PixelSudokuTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Pixel model: {n_params} parameters")
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
        """Decode by nearest glyph on empty cells; count fully-solved puzzles and
        per-cell decode accuracy."""
        solved = cells_right = cells_total = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, probe_puzzle_digits.size(0), 250):
                pd = probe_puzzle_digits[start:start + 250]
                ad = probe_answer_digits[start:start + 250]
                x = glyphs[pd]
                probs = torch.sigmoid(run_iterations(model, x, n_iters, rope_cos, rope_sin).float())
                # distance to each digit glyph (blank excluded): (B, 81, 9)
                dists = (probs.unsqueeze(2) - glyphs[1:].view(1, 1, 9, CELL_PIXELS)).abs().sum(dim=-1)
                decoded = dists.argmin(dim=-1) + 1
                empty = pd == 0
                hits = (decoded == ad) & empty
                cells_right += int(hits.sum().item())
                cells_total += int(empty.sum().item())
                solved += int((hits | ~empty).all(dim=1).sum().item())
        return solved, cells_right, cells_total

    if start_step >= total_steps:
        final_path = os.path.join(output_dir, "model_pixel_bootstrap.pt")
        torch.save(model.state_dict(), final_path)
        log(f"Run already complete at step {start_step - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True}

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step)

        idx = torch.randint(0, puzzle_digits.size(0), (batch_size,), device=device)
        pd, ad = puzzle_digits[idx], answer_digits[idx]
        x = glyphs[pd]                      # (B, 81, 64) rendered on the fly
        target_pixels = glyphs[ad]          # (B, 81, 64)
        empty = (pd == 0).unsqueeze(-1).expand_as(target_pixels)

        with torch.autocast('cuda', dtype=torch.bfloat16):
            h_prev = model.initial_encoder(x)
            preds = torch.zeros_like(x)
            loss = 0.0
            for _ in range(train_iterations):
                h = h_prev + model.pred_proj(preds)
                for layer in model.layers:
                    h = layer(h, rope_cos, rope_sin)
                h_prev = h
                logits = model.output_head(h)
                loss = loss + F.binary_cross_entropy_with_logits(logits.float()[empty], target_pixels[empty])
                preds = torch.sigmoid(logits)
            loss = loss / train_iterations

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 200 == 0:
            log(f"STEP {step:5d} | pixel loss {loss.item():.4f} | lr {lr_at(step):.2e} | {time.time() - t0:.0f}s")

        if step % eval_every == 0 or step == total_steps - 1:
            model.eval()
            s16, c16, ct = probe(16)
            s128, c128, _ = probe(128)
            model.train()
            log(f"PROBE {step:5d} | 16-iter: {s16}/1000 solved, {c16}/{ct} cells | "
                f"128-iter: {s128}/1000 solved, {c128}/{ct} cells")
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': CONFIG,
            }, os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt"))

    final_path = os.path.join(output_dir, "model_pixel_bootstrap.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_solved_128': s128}


if __name__ == "__main__":
    train()
