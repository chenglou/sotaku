# CE fitness from scratch at HORIZON 1: the iteration loop reduced to a single pass.
# Both from-scratch failures judged the model through 16+ scrambling iterations; at
# horizon 1 the network is just a feedforward map and the fitness landscape is as
# unscrambled as it gets. If ES can learn single-pass cell prediction here, a horizon
# ladder (1 -> 4 -> 16 -> ...) continues; if even this is floor-bound, from-scratch
# scalar-fitness ES is out of excuses.

import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from iters.exp_baseline_lr2e3 import (
    COL_IDX,
    RATING_BUCKETS,
    ROW_IDX,
    RoPETransformerLayer,
    encode_puzzles,
)

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "es_ce_h1_checkpoint_step"

CONFIG = {
    'experiment': 'exp_es_ce_h1',
    'd_model': 32,
    'n_heads': 4,
    'd_ff': 128,
    'n_layers': 4,
    'es_generations': 5000,
    'population_pairs': 16,
    'sigma': 'calibrated',
    'lr': 3e-4,
    'anchor_lambda': 0.0,
    'fitness': 'neg_ce_empty_cells',
    'fitness_puzzles': 384,
    'fitness_iters': 1,
}

total_steps = 5000        # generations; horizon-1 rollouts are nearly free
eval_every = 500          # checkpoint every N generations
population_pairs = 16     # antithetic pairs per generation (32 evaluations)
sigma_ladder = [3e-3, 1e-3, 3e-4, 1e-4]
lr = 3e-4                 # update step size
fitness_dense = True
fitness_puzzles = 384
fitness_iters = 1         # a single pass: the most unscrambled landscape possible
fitness_pool_offset = 2_700_000
fitness_pool_size = 20_000
log_name = "exp_es_ce_h1.log"

d_model = 32
n_heads = 4
d_ff = 128
n_layers = 4

COMPILE_CHUNK = 1    # a single-iteration block: the whole horizon

# 2D RoPE tables for the tiny head size (head_dim 8: 2 frequency pairs per axis)
head_dim = d_model // n_heads
rope_half = head_dim // 2
rope_pairs = rope_half // 2
rope_base = 10.0
_freqs = 1.0 / (rope_base ** (torch.arange(rope_pairs).float() * 2 / rope_half))
_row_angles = ROW_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)
_col_angles = COL_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)
_angles = torch.cat([_row_angles, _col_angles], dim=-1)
TINY_ROPE_COS = _angles.cos()
TINY_ROPE_SIN = _angles.sin()


class TinySudokuTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, 9)


def run_iterations(model, x, n_iters):
    """Forward-only iterative refinement, eager mode (probe path)."""
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
        assert n_iters % COMPILE_CHUNK == 0, f"{n_iters} not a multiple of {COMPILE_CHUNK}"
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
    """Add scale * z to every parameter, z regenerated from the seed (never stored)."""
    gen = torch.Generator(device=params[0].device)
    gen.manual_seed(seed)
    with torch.no_grad():
        for p in params:
            z = torch.randn(p.shape, generator=gen, device=p.device, dtype=torch.float32)
            p.add_(z, alpha=scale)


def train(output_dir="."):
    device = torch.device("cuda")

    print("Loading fitness pool (train rows beyond the first-order training cut)...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[fitness_pool_offset:fitness_pool_offset + fitness_pool_size]
    pool_puzzles = pool_rows["question"]
    pool_solutions = pool_rows["answer"]
    pool_x = encode_puzzles(pool_puzzles).to(device)
    pool_targets = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in pool_solutions], device=device)
    pool_empty = torch.tensor([[p[j] == '.' for j in range(81)] for p in pool_puzzles], device=device)
    print(f"Fitness pool: {len(pool_puzzles)} puzzles")

    print("Loading validation probe (test split, 200 per rating bucket)...")
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

    model = TinySudokuTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Tiny model: {n_params} parameters")

    checkpoint_path, start_gen = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_gen = ckpt['step'] + 1
        print(f"Resumed ES state from generation {ckpt['step']}")
    else:
        seed_path = os.path.join(os.getcwd(), "seed_model.pt")
        state = torch.load(seed_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        start_gen = 0
        print(f"Loaded seed model from {seed_path}")

    model.eval()
    params = list(model.parameters())
    run_compiled = make_compiled_runner(model, device)

    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    def save_checkpoint(gen):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{gen}.pt")
        torch.save({
            'step': gen,
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
        }, path)
        log(f"Checkpoint saved: {path}")

    def score_slice(lo):
        fx = pool_x[lo:lo + fitness_puzzles]
        ft = pool_targets[lo:lo + fitness_puzzles]
        fm = pool_empty[lo:lo + fitness_puzzles]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            logits = run_compiled(fx, fitness_iters)
        ce = F.cross_entropy(logits.float()[fm], ft[fm])
        return -float(ce.item())

    baseline_probe = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
    log(f"GEN {start_gen - 1:4d} | {n_params} params | validation {fitness_iters}-iter: "
        f"{baseline_probe}/{len(probe_puzzles)} (starting point)")

    unperturbed = score_slice(0)
    snapshot = [p.detach().clone() for p in model.parameters()]
    sigma = sigma_ladder[-1]
    for candidate in sigma_ladder:
        perturb(list(model.parameters()), 777, candidate)
        score = score_slice(0)
        with torch.no_grad():
            for p, s in zip(model.parameters(), snapshot):
                p.copy_(s)
        log(f"CALIBRATE sigma={candidate:.0e}: {score} (unperturbed {unperturbed})")
        if score >= unperturbed - 0.3:   # allow ~0.3 nats degradation (chance is -ln 9 = -2.20)
            sigma = candidate
            break
    log(f"CALIBRATE chose sigma={sigma:.0e}")

    if start_gen >= total_steps:
        # Resuming from the final checkpoint: the run is already complete. Save the
        # final model and exit cleanly — without this, the loop below never runs and
        # the return would crash on loop-local variables, failing every retry.
        final_path = os.path.join(output_dir, "model_es_ce_h1.pt")
        torch.save(model.state_dict(), final_path)
        log(f"Run already complete at generation {start_gen - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True}

    rng = random.Random(1234 + start_gen)
    for gen in range(start_gen, total_steps):
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
            log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.4f} "
                f"best {max(all_scores):.4f} | validation {fitness_iters}-iter: "
                f"{probe_solved}/{len(probe_puzzles)} | {time.time() - t0:.1f}s")

        if gen % eval_every == 0 or gen == total_steps - 1:
            save_checkpoint(gen)

    final_path = os.path.join(output_dir, "model_es_ce_h1.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_fitness': float(np.mean(all_scores))}


if __name__ == "__main__":
    train()
