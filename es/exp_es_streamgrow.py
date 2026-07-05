# Stream growth: the axis exp_es_grow left untested. That run widened d_ff (128->512)
# function-preservingly and the learning slope did not change — the 32-dimensional
# residual stream is the suspected bottleneck (the mipmap-down probe showed the full
# model saturates all 128 of its stream dimensions). This run widens the STREAM:
# d_model 32 -> 64 by duplicate-and-halve — every stream vector becomes [v, v], every
# writer duplicates its output rows, every reader halves-and-duplicates its input
# columns, and heads double (4 -> 8) at constant head_dim 8, so the per-head attention
# geometry and the RoPE tables are untouched. LayerNorm survives duplication exactly
# (duplicating values changes neither mean nor variance). Seed: model_es_streamgrow.pt is
# NOT used — the seed is exp_es_grow's final model (d=32, d_ff=512, probe 260/1000 at
# 128 iterations), so the pre-growth slope directly continues that run's and the
# post-growth slope answers whether the stream was the constraint.

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from es.exp_es_tiny import TINY_ROPE_COS, TINY_ROPE_SIN
from iters.exp_baseline_lr2e3 import RATING_BUCKETS, RoPETransformerLayer, encode_puzzles

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "es_streamgrow_checkpoint_step"

CONFIG = {
    'experiment': 'exp_es_streamgrow',
    'd_model': 32,
    'n_heads': 4,
    'd_model_small': 32,
    'd_model_big': 64,
    'd_ff': 512,
    'grow_at_gen': 200,
    'n_layers': 4,
    'es_generations': 1000,
    'population_pairs': 16,
    'sigma': 'calibrated',
    'lr': 3e-4,
    'anchor_lambda': 0.0,
    'fitness': 'dense_cells',
    'fitness_puzzles': 384,
    'fitness_iters': 128,
}

total_steps = 1000
eval_every = 100
population_pairs = 16
sigma_ladder = [3e-3, 1e-3, 3e-4, 1e-4]
lr = 3e-4
fitness_puzzles = 384
fitness_iters = 128
fitness_pool_offset = 2_700_000
fitness_pool_size = 20_000
log_name = "exp_es_streamgrow.log"

D_MODEL_SMALL = 32
D_MODEL_BIG = 64
HEAD_DIM = 8
D_FF = 512
n_layers = 4
GROW_AT_GEN = 200

COMPILE_CHUNK = 32


def make_model(d_model):
    n_heads = d_model // HEAD_DIM   # heads scale with width; head_dim stays 8
    model = nn.Module()
    model.initial_encoder = nn.Linear(10, d_model)
    model.pred_proj = nn.Linear(9, d_model)
    model.layers = nn.ModuleList([
        RoPETransformerLayer(d_model, n_heads, D_FF)
        for _ in range(n_layers)
    ])
    model.output_head = nn.Linear(d_model, 9)
    return model


def grown_state_dict(state):
    """Duplicate-and-halve the residual stream, exactly function-preserving: stream
    vectors become [v, v] (writers duplicate output rows; LayerNorm params duplicate —
    duplication changes neither mean nor variance), and every reader becomes
    [W/2, W/2] so it computes the same dot products from the duplicated input.
    Heads double at constant head_dim, so per-head attention and RoPE are untouched."""
    def dup_rows(w):    # stream writer: (d, in) -> (2d, in)
        return torch.cat([w, w], dim=0)

    def half_cols(w):   # stream reader: (out, d) -> (out, 2d)
        return torch.cat([w / 2, w / 2], dim=1)

    def dup_vec(v):
        return torch.cat([v, v], dim=0)

    new_state = {}
    for key, w in state.items():
        w = w.float().cpu()
        if key.startswith('initial_encoder.') or key.startswith('pred_proj.'):
            new_state[key] = dup_rows(w) if w.dim() == 2 else dup_vec(w)
        elif key.startswith('output_head.'):
            new_state[key] = half_cols(w) if w.dim() == 2 else w.clone()
        elif '.norm' in key:
            new_state[key] = dup_vec(w)
        elif any(p in key for p in ('.q_proj.', '.k_proj.', '.v_proj.', '.out_proj.')):
            new_state[key] = dup_rows(half_cols(w)) if w.dim() == 2 else dup_vec(w)
        elif '.linear1.' in key:   # (d_ff, d): reader only
            new_state[key] = half_cols(w) if w.dim() == 2 else w.clone()
        elif '.linear2.' in key:   # (d, d_ff): writer only
            new_state[key] = dup_rows(w) if w.dim() == 2 else dup_vec(w)
        else:
            raise ValueError(f"unmapped key: {key}")
    return new_state


def run_iterations(model, x, n_iters):
    """Eager probe path."""
    device = x.device
    rope_cos = TINY_ROPE_COS.to(device)
    rope_sin = TINY_ROPE_SIN.to(device)
    h_prev = model.initial_encoder(x)
    preds = torch.zeros(x.size(0), 81, 9, device=device)
    for _ in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        preds = F.softmax(model.output_head(h), dim=-1)
    return model.output_head(h_prev)


def make_compiled_runner(model, device):
    rope_cos = TINY_ROPE_COS.to(device)
    rope_sin = TINY_ROPE_SIN.to(device)

    def iter_block(h_prev, preds):
        for _ in range(COMPILE_CHUNK):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            preds = F.softmax(model.output_head(h), dim=-1)
        return h_prev, preds

    compiled_block = torch.compile(iter_block, dynamic=False)

    def run(x, n_iters):
        assert n_iters % COMPILE_CHUNK == 0
        h_prev = model.initial_encoder(x)
        preds = torch.zeros(x.size(0), 81, 9, device=device)
        for _ in range(n_iters // COMPILE_CHUNK):
            h_prev, preds = compiled_block(h_prev, preds)
        return model.output_head(h_prev)

    return run


def count_solved(model, x, puzzles, solutions, n_iters):
    solved = 0
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for start in range(0, x.size(0), 256):
            batch_x = x[start:start + 256]
            final = run_iterations(model, batch_x, n_iters).argmax(dim=-1).cpu()
            for b, (puzzle, solution) in enumerate(zip(puzzles[start:start + 256], solutions[start:start + 256])):
                pred = list(puzzle)
                for i in range(81):
                    if puzzle[i] == '.':
                        pred[i] = str(final[b, i].item() + 1)
                if ''.join(pred) == solution:
                    solved += 1
    return solved


def perturb(params, seed, scale):
    gen = torch.Generator(device=params[0].device)
    gen.manual_seed(seed)
    with torch.no_grad():
        for p in params:
            z = torch.randn(p.shape, generator=gen, device=p.device, dtype=torch.float32)
            p.add_(z, alpha=scale)


def train(output_dir="."):
    device = torch.device("cuda")

    print("Loading fitness pool...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[fitness_pool_offset:fitness_pool_offset + fitness_pool_size]
    pool_puzzles = pool_rows["question"]
    pool_solutions = pool_rows["answer"]
    pool_x = encode_puzzles(pool_puzzles).to(device)
    pool_targets = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in pool_solutions], device=device)
    pool_empty = torch.tensor([[p[j] == '.' for j in range(81)] for p in pool_puzzles], device=device)
    print(f"Fitness pool: {len(pool_puzzles)} puzzles")

    print("Loading validation probe...")
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
    probe_puzzles = [test_dataset[i]['question'] for i in probe_idx]
    probe_solutions = [test_dataset[i]['answer'] for i in probe_idx]
    probe_x = encode_puzzles(probe_puzzles).to(device)
    print(f"Validation probe: {len(probe_puzzles)} puzzles")

    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    checkpoint_path, _ = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        current_d_model = int(ckpt.get('d_model', D_MODEL_SMALL))
        model = make_model(current_d_model).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        start_gen = ckpt['step'] + 1
        log(f"Resumed from generation {ckpt['step']} at d_model={current_d_model}")
    else:
        seed_path = os.path.join(os.getcwd(), "seed_model.pt")
        state = torch.load(seed_path, map_location='cpu', weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        current_d_model = D_MODEL_SMALL
        model = make_model(current_d_model).to(device)
        model.load_state_dict(state)
        start_gen = 0
        log(f"Loaded bootstrap seed from {seed_path}")

    model.eval()
    params = list(model.parameters())
    run_compiled = make_compiled_runner(model, device)
    n_params = sum(p.numel() for p in params)
    log(f"Model: {n_params} parameters (d_model={current_d_model})")

    def save_checkpoint(gen):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{gen}.pt")
        torch.save({
            'step': gen,
            'd_model': current_d_model,
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
        }, path)
        log(f"Checkpoint saved: {path}")

    def score_slice(lo):
        fx = pool_x[lo:lo + fitness_puzzles]
        ft = pool_targets[lo:lo + fitness_puzzles]
        fm = pool_empty[lo:lo + fitness_puzzles]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            final = run_compiled(fx, fitness_iters).argmax(dim=-1)
        return int(((final == ft) & fm).sum().item())

    def calibrate():
        unperturbed = score_slice(0)
        snapshot = [p.detach().clone() for p in model.parameters()]
        chosen = sigma_ladder[-1]
        for candidate in sigma_ladder:
            perturb(list(model.parameters()), 777, candidate)
            score = score_slice(0)
            with torch.no_grad():
                for p, s in zip(model.parameters(), snapshot):
                    p.copy_(s)
            log(f"CALIBRATE sigma={candidate:.0e}: {score} (unperturbed {unperturbed})")
            if score >= unperturbed // 2:
                chosen = candidate
                break
        log(f"CALIBRATE chose sigma={chosen:.0e}")
        return chosen

    baseline_probe = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
    log(f"GEN {start_gen - 1:4d} | validation {fitness_iters}-iter: {baseline_probe}/{len(probe_puzzles)} (starting point)")
    sigma = calibrate()

    if start_gen >= total_steps:
        # Resuming from the final checkpoint: the run is already complete. Save the
        # final model and exit cleanly — without this, the loop below never runs and
        # the return would crash on loop-local variables, failing every retry.
        final_path = os.path.join(output_dir, "model_es_streamgrow.pt")
        torch.save(model.state_dict(), final_path)
        log(f"Run already complete at generation {start_gen - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True}

    rng = random.Random(1234 + start_gen)
    for gen in range(start_gen, total_steps):
        if gen == GROW_AT_GEN and current_d_model == D_MODEL_SMALL:
            pre = score_slice(0)
            grown = grown_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            current_d_model = D_MODEL_BIG
            model = make_model(current_d_model).to(device)
            model.load_state_dict(grown)
            model.eval()
            params = list(model.parameters())
            run_compiled = make_compiled_runner(model, device)
            post = score_slice(0)
            n_params = sum(p.numel() for p in params)
            log(f"GROW at gen {gen}: d_model {D_MODEL_SMALL} -> {D_MODEL_BIG} ({n_params} params) | "
                f"fitness before {pre}, after {post} (function-preserving: should match)")
            sigma = calibrate()

        t0 = time.time()
        lo = (gen * fitness_puzzles) % (fitness_pool_size - fitness_puzzles)

        snapshot = [p.detach().clone() for p in params]
        seeds = [rng.randrange(2**62) for _ in range(population_pairs)]
        scores_plus, scores_minus = [], []
        for seed in seeds:
            perturb(params, seed, sigma)
            scores_plus.append(score_slice(lo))
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)
            perturb(params, seed, -sigma)
            scores_minus.append(score_slice(lo))
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)

        all_scores = np.array(scores_plus + scores_minus, dtype=np.float64)
        spread = all_scores.std()
        if spread > 1e-9:
            utilities = (all_scores - all_scores.mean()) / spread
            u_plus, u_minus = utilities[:population_pairs], utilities[population_pairs:]
            with torch.no_grad():
                for i, seed in enumerate(seeds):
                    w = float(u_plus[i] - u_minus[i]) / (2 * population_pairs)
                    gen_t = torch.Generator(device=device)
                    gen_t.manual_seed(seed)
                    for p in params:
                        z = torch.randn(p.shape, generator=gen_t, device=device, dtype=torch.float32)
                        p.add_(z, alpha=lr * w)

        if gen % 10 == 0 or gen == total_steps - 1:
            probe_solved = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
            log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.0f} "
                f"best {max(all_scores):.0f} | validation {fitness_iters}-iter: "
                f"{probe_solved}/{len(probe_puzzles)} | {time.time() - t0:.1f}s")

        if gen % eval_every == 0 or gen == total_steps - 1:
            save_checkpoint(gen)

    final_path = os.path.join(output_dir, "model_es_streamgrow.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_probe': probe_solved}


if __name__ == "__main__":
    train()
