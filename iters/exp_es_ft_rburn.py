# Evolution-strategies fine-tuning of a trained checkpoint, directly on the
# 1024-iteration solve rate — the metric first-order training cannot see (backprop
# through 1024 iterations is memory-impossible; ES needs only forward passes).
# Plain antithetic ES: perturb weights with seeded Gaussian noise, score each
# perturbation by solved puzzles, update along the rank-weighted average direction.
# Starts from ./seed_model.pt (packaged into the job by submit.py --seed-model).
# Generations chain across jobs: each ~2h job runs what fits, checkpoints every 20
# generations, and a resubmit with --resume-checkpoint-key continues from there.

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from iters.exp_baseline_lr2e3 import (
    RATING_BUCKETS,
    ROPE_COS,
    ROPE_SIN,
    SudokuTransformer,
    encode_puzzles,
)

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "es_ft_rburn_checkpoint_step"

CONFIG = {
    'experiment': 'exp_es_ft_rburn',
    'es_generations': 60,
    'population_pairs': 16,
    'sigma': 'calibrated',
    'lr': 3e-4,
    'anchor_lambda': 1e-3,
    'fitness_puzzles': 384,
    'fitness_iters': 1024,
}

total_steps = 60          # generations; submit.py reads this for checkpoint names
eval_every = 20           # checkpoint every N generations
population_pairs = 16     # antithetic pairs per generation (32 evaluations)
sigma_ladder = [3e-4, 1e-4, 3e-5, 1e-5]   # calibrated at startup: largest scale that
                                          # only mildly degrades fitness. At 1024
                                          # iterations the model is extremely sensitive
                                          # to weight noise (1e-3 zeroes a 96% model).
lr = 3e-4                 # update step size
anchor_lambda = 1e-3      # pull toward the seed weights each generation
fitness_puzzles = 384     # puzzles per fitness evaluation (rotated per generation)
fitness_iters = 1024      # the deployment horizon — the point of all this
fitness_pool_offset = 2_700_000   # train rows beyond the first-order training cut
fitness_pool_size = 20_000
log_name = "exp_es_ft_rburn.log"


def run_iterations(model, x, n_iters):
    """Forward-only iterative refinement, eager mode. Returns final logits."""
    device = x.device
    rope_cos = ROPE_COS.to(device)
    rope_sin = ROPE_SIN.to(device)
    h_prev = model.initial_encoder(x)
    preds = torch.zeros(x.size(0), 81, 9, device=device)
    for _ in range(n_iters):
        h = h_prev + model.pred_proj(preds)
        for layer in model.layers:
            h = layer(h, rope_cos, rope_sin)
        h_prev = h
        preds = F.softmax(model.output_head(h), dim=-1)
    return model.output_head(h_prev)


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

    model = SudokuTransformer().to(device)

    checkpoint_path, start_gen = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        anchor = [t.to(device) for t in ckpt['anchor']]
        start_gen = ckpt['step'] + 1
        print(f"Resumed ES state from generation {ckpt['step']}")
    else:
        seed_path = os.path.join(os.getcwd(), "seed_model.pt")
        state = torch.load(seed_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        anchor = [p.detach().clone() for p in model.parameters()]
        start_gen = 0
        print(f"Loaded seed model from {seed_path}")

    model.eval()  # no dropout anywhere in ES — fitness must reflect deployment
    params = list(model.parameters())

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
            'anchor': [t.cpu() for t in anchor],
            'config': CONFIG,
        }, path)
        log(f"Checkpoint saved: {path}")

    baseline_probe = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
    log(f"GEN {start_gen - 1:4d} | validation 1024-iter: {baseline_probe}/{len(probe_puzzles)} (starting point)")

    # Calibrate the perturbation scale: pick the largest sigma whose perturbed model
    # keeps at least half the unperturbed fitness. Too large and every member scores
    # zero (no signal); too small and the finite-difference signal drowns in noise.
    calib_x = pool_x[:fitness_puzzles]
    calib_p = pool_puzzles[:fitness_puzzles]
    calib_s = pool_solutions[:fitness_puzzles]
    unperturbed = count_solved(model, calib_x, calib_p, calib_s, fitness_iters)
    snapshot = [p.detach().clone() for p in model.parameters()]
    sigma = sigma_ladder[-1]
    for candidate in sigma_ladder:
        perturb(list(model.parameters()), 777, candidate)
        score = count_solved(model, calib_x, calib_p, calib_s, fitness_iters)
        with torch.no_grad():
            for p, s in zip(model.parameters(), snapshot):
                p.copy_(s)
        log(f"CALIBRATE sigma={candidate:.0e}: {score}/{fitness_puzzles} (unperturbed {unperturbed})")
        if score >= unperturbed // 2:
            sigma = candidate
            break
    log(f"CALIBRATE chose sigma={sigma:.0e}")

    rng = random.Random(1234 + start_gen)
    for gen in range(start_gen, total_steps):
        t0 = time.time()
        # Same probe slice for every population member (common random numbers),
        # rotated each generation so we don't overfit one subset.
        lo = (gen * fitness_puzzles) % (fitness_pool_size - fitness_puzzles)
        fx = pool_x[lo:lo + fitness_puzzles]
        fp = pool_puzzles[lo:lo + fitness_puzzles]
        fs = pool_solutions[lo:lo + fitness_puzzles]

        snapshot = [p.detach().clone() for p in params]
        seeds = [rng.randrange(2**62) for _ in range(population_pairs)]
        scores_plus, scores_minus = [], []
        for seed in seeds:
            perturb(params, seed, sigma)
            scores_plus.append(count_solved(model, fx, fp, fs, fitness_iters))
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)
            perturb(params, seed, -sigma)
            scores_minus.append(count_solved(model, fx, fp, fs, fitness_iters))
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)

        # Score-normalized weights, tie-safe: a generation where every member scores
        # the same (no signal) takes no step at all. The update direction is the
        # fitness-weighted average of the unit-variance perturbation directions; sigma
        # deliberately does not appear (it is absorbed into the effective step size).
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
                for p, a in zip(params, anchor):
                    p.add_(p - a, alpha=-anchor_lambda)

        probe_solved = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
        log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.1f}/{fitness_puzzles} "
            f"best {max(all_scores)}/{fitness_puzzles} | validation 1024-iter: {probe_solved}/{len(probe_puzzles)} "
            f"| {time.time() - t0:.0f}s")

        if gen % eval_every == 0 or gen == total_steps - 1:
            save_checkpoint(gen)

    final_path = os.path.join(output_dir, "model_es_ft_rburn.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_validation': probe_solved}


if __name__ == "__main__":
    train()
