# Mipmap a trained network DOWN one scale: compress the d_model=128 canonical model
# (96.0% at 1024 iterations) to d_model=64 by literal 2x2 box-filtering of adjacent
# channel pairs, the way an image mipmap averages adjacent pixels. Theory says this
# should mostly destroy the function — image mipmaps work because adjacent pixels are
# correlated, and a network's channels have no adjacency structure — so the residual
# accuracy measures how redundantly the trained network organized its channels.
#
# The compression is the best a pair-averaging basis can do: with A the (64, 128)
# pair-averaging matrix, every stream-to-stream weight W becomes 2 A W A^T (the 2x2
# box filter times two, from the pseudo-inverse of A), stream vectors become A v, and
# the feed-forward hidden width stays 512 so only the residual stream is compressed.
# Attention is the uncontrolled part: head_dim drops 32 -> 16, so the RoPE tables and
# the 1/sqrt(head_dim) scale change beyond the linear map.
#
# Probes both models at 16 / 128 / 1024 iterations on the standard 1,000-puzzle set.

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from iters.exp_baseline_lr2e3 import (
    COL_IDX,
    RATING_BUCKETS,
    ROW_IDX,
    RoPETransformerLayer,
    SudokuTransformer,
    encode_puzzles,
)

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "mipdown_checkpoint_step"
total_steps = 1
eval_every = 1
log_name = "exp_mipdown.log"

SEED_MODEL = "model_viridian_canonical_final50k.pt"
D_SMALL = 64
N_HEADS = 4
D_FF = 512
N_LAYERS = 4


def rope_tables(d_model, n_heads):
    head_dim = d_model // n_heads
    rope_half = head_dim // 2
    rope_pairs = rope_half // 2
    freqs = 1.0 / (10.0 ** (torch.arange(rope_pairs).float() * 2 / rope_half))
    row_angles = ROW_IDX.float().unsqueeze(1) * freqs.unsqueeze(0)
    col_angles = COL_IDX.float().unsqueeze(1) * freqs.unsqueeze(0)
    angles = torch.cat([row_angles, col_angles], dim=-1)
    return angles.cos(), angles.sin()


def make_model(d_model):
    model = nn.Module()
    model.initial_encoder = nn.Linear(10, d_model)
    model.pred_proj = nn.Linear(9, d_model)
    model.layers = nn.ModuleList([
        RoPETransformerLayer(d_model, N_HEADS, D_FF)
        for _ in range(N_LAYERS)
    ])
    model.output_head = nn.Linear(d_model, 9)
    return model


def mip_state_dict(state):
    """Box-filter every stream dimension of the d=128 state dict down to d=64."""
    d_big = 128
    A = torch.zeros(D_SMALL, d_big)
    for i in range(D_SMALL):
        A[i, 2 * i] = 0.5
        A[i, 2 * i + 1] = 0.5
    up = 2 * A.t()   # pseudo-inverse of A: duplicate each compressed value to its pair

    def rows(w):      # (d_big, in) -> (D_SMALL, in): average output pairs
        return A @ w

    def cols(w):      # (out, d_big) -> (out, D_SMALL): best read from averaged inputs
        return w @ up

    def both(w):      # stream -> stream
        return A @ w @ up

    def vec(v):       # stream vector
        return A @ v

    new = {}
    for key, w in state.items():
        w = w.float().cpu()
        if key.startswith('initial_encoder.') or key.startswith('pred_proj.'):
            new[key] = rows(w) if w.dim() == 2 else vec(w)
        elif key.startswith('output_head.'):
            new[key] = cols(w) if w.dim() == 2 else w.clone()
        elif '.norm' in key:
            new[key] = vec(w)
        elif any(p in key for p in ('.q_proj.', '.k_proj.', '.v_proj.', '.out_proj.')):
            new[key] = both(w) if w.dim() == 2 else vec(w)
        elif '.linear1.' in key:   # (d_ff, d_model) — compress input side only
            new[key] = cols(w) if w.dim() == 2 else w.clone()
        elif '.linear2.' in key:   # (d_model, d_ff) — compress output side only
            new[key] = rows(w) if w.dim() == 2 else vec(w)
        else:
            raise ValueError(f"unmapped key: {key}")
    return new


def run_iterations(model, x, n_iters, rope_cos, rope_sin):
    h_prev = model.initial_encoder(x)
    preds = torch.zeros(x.size(0), 81, 9, device=x.device)
    for _ in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        preds = F.softmax(model.output_head(h), dim=-1)
    return model.output_head(h_prev)


def train(output_dir="."):
    device = torch.device("cuda")
    log_file = open(os.path.join(output_dir, log_name), "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

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

    state = torch.load(os.path.join(output_dir, SEED_MODEL), map_location='cpu', weights_only=True)
    if 'model_state_dict' in state:
        state = state['model_state_dict']

    big = SudokuTransformer().to(device)
    big.load_state_dict(state)
    big.eval()

    small = make_model(D_SMALL).to(device)
    small.load_state_dict(mip_state_dict(state))
    small.eval()

    def probe(model, d_model, n_iters):
        rope_cos, rope_sin = rope_tables(d_model, N_HEADS)
        rope_cos, rope_sin = rope_cos.to(device), rope_sin.to(device)
        solved = cells = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, probe_x.size(0), 500):
                final = run_iterations(model, probe_x[start:start + 500], n_iters, rope_cos, rope_sin).argmax(dim=-1)
                tb = probe_targets[start:start + 500]
                eb = probe_empty[start:start + 500]
                hits = (final == tb) & eb
                cells += int(hits.sum().item())
                solved += int((hits | ~eb).all(dim=1).sum().item())
        return solved, cells

    total_empty = int(probe_empty.sum().item())
    for n_iters in (16, 128, 1024):
        s_big, c_big = probe(big, 128, n_iters)
        s_small, c_small = probe(small, D_SMALL, n_iters)
        log(f"ITER {n_iters:4d} | d=128: {s_big}/1000 solved, {c_big}/{total_empty} cells | "
            f"mipdown d=64: {s_small}/1000 solved, {c_small}/{total_empty} cells")

    log_file.close()
    return {}


if __name__ == "__main__":
    train()
