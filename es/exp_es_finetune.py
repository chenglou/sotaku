# Evolution-strategies fine-tuning of a trained checkpoint, directly on the
# 1024-iteration solve rate — the metric first-order training cannot see (backprop
# through 1024 iterations is memory-impossible; ES needs only forward passes).
# By default, evaluate 32 independent Gaussian perturbations and update along their
# score-weighted average direction. Antithetic sampling remains available as a control.
# Starts from ./seed_model.pt (packaged into the job by submit.py --seed-model).
# Generations chain across jobs: each ~2h job runs what fits, checkpoints every 20
# generations, and a resubmit with --resume-checkpoint-key continues from there.

import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import atomic_torch_save, find_latest_checkpoint
from es.es_sampling import (
    DEFAULT_SAMPLING_MODE,
    POPULATION_PAIRS,
    POPULATION_SIZE,
    SAMPLING_MODES,
    direction_weights,
    generation_direction_seeds,
)
from iters.exp_baseline_lr2e3 import (
    RATING_BUCKETS,
    ROPE_COS,
    ROPE_SIN,
    SudokuTransformer,
    encode_puzzles,
)

torch.set_float32_matmul_precision('high')

sampling_mode = DEFAULT_SAMPLING_MODE
CHECKPOINT_PREFIX = f"es_finetune_{sampling_mode}_checkpoint_step"

CONFIG = {
    'experiment': 'exp_es_finetune',
    'es_generations': 120,
    'sampling_mode': sampling_mode,
    'population_evaluations': POPULATION_SIZE,
    'sigma': 'calibrated',
    'lr': 3e-4,
    'anchor_lambda': 1e-3,
    'fitness': 'dense_cells',
    'fitness_puzzles': 384,
    'fitness_iters': 1024,
}

total_steps = 120         # generations; submit.py reads this for checkpoint names
eval_every = 20           # checkpoint every N generations
sigma_ladder = [3e-4, 1e-4, 3e-5, 1e-5]   # calibrated at startup: largest scale that
                                          # only mildly degrades fitness. At 1024
                                          # iterations the model is extremely sensitive
                                          # to weight noise (1e-3 zeroes a 96% model).
lr = 3e-4                 # update step size
anchor_lambda = 1e-3      # pull toward the seed weights each generation
fitness_dense = True      # dense cell-level fitness gives near-collapsed seeds a usable
                          # slope; solved-puzzle fitness matches the deployment metric
                          # exactly and is fine for any seed that already solves some
fitness_puzzles = 384     # puzzles per fitness evaluation (rotated per generation)
fitness_iters = 1024      # the deployment horizon — the point of all this
fitness_pool_offset = 2_700_000   # train rows beyond the first-order training cut
fitness_pool_size = 20_000
log_name = f"exp_es_finetune_{sampling_mode}.log"


COMPILE_CHUNK = 32   # iterations per compiled block; fitness_iters must divide evenly


def run_iterations(model, x, n_iters):
    """Forward-only iterative refinement, eager mode. Returns final logits.

    The validation probe stays on this exact path so probe values remain comparable
    across every run in the project's history; only fitness evaluation uses the
    compiled path below (3.4x faster, numerically a different-but-equally-valid
    realization — reduction-order changes compound over 1024 iterations)."""
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


def make_compiled_runner(model, device):
    """Compiled fitness path: one 32-iteration block, compiled once, called in a loop.
    Compiling the block (not the whole horizon) keeps compile time to a few minutes
    while removing eager per-op launch overhead, the dominant cost for this tiny model."""
    rope_cos = ROPE_COS.to(device)
    rope_sin = ROPE_SIN.to(device)

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
    if sampling_mode not in SAMPLING_MODES:
        raise ValueError("sampling_mode must be 'paired' or 'independent'")

    device = torch.device("cuda")

    print("Loading fitness pool (train rows beyond the first-order training cut)...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[fitness_pool_offset:fitness_pool_offset + fitness_pool_size]
    pool_puzzles = pool_rows["question"]
    pool_solutions = pool_rows["answer"]
    pool_x = encode_puzzles(pool_puzzles).to(device)
    # Dense fitness needs per-cell targets and the empty-cell mask: counting only
    # fully-solved puzzles gives near-collapsed seeds a flat fitness landscape (every
    # perturbation scores ~0), while correct-cell counts differ everywhere.
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

    model = SudokuTransformer().to(device)

    checkpoint_path, start_gen = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        for key, expected in CONFIG.items():
            actual = ckpt.get('config', {}).get(key)
            if actual != expected:
                raise ValueError(f"resume config mismatch for {key}: {actual!r} != {expected!r}")
        model.load_state_dict(ckpt['model_state_dict'])
        anchor = [t.to(device) for t in ckpt['anchor']]
        sigma = float(ckpt['sigma'])
        start_gen = ckpt['step'] + 1
        print(f"Resumed ES state from generation {ckpt['step']}")
    else:
        seed_path = os.path.join(os.getcwd(), "seed_model.pt")
        state = torch.load(seed_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        anchor = [p.detach().clone() for p in model.parameters()]
        sigma = None
        start_gen = 0
        print(f"Loaded seed model from {seed_path}")

    model.eval()  # no dropout anywhere in ES — fitness must reflect deployment
    params = list(model.parameters())
    run_compiled = make_compiled_runner(model, device)  # fitness path; first call compiles (~5 min)

    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    def save_checkpoint(gen):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{gen}.pt")
        atomic_torch_save({
            'step': gen,
            'model_state_dict': model.state_dict(),
            'anchor': [t.cpu() for t in anchor],
            'sigma': sigma,
            'config': CONFIG,
        }, path)
        log(f"Checkpoint saved: {path}")

    def score_slice(lo):
        # One full-batch evaluation on the compiled path (chunking removed: it was
        # 1.8x slower and per-puzzle results don't depend on batch composition).
        fx = pool_x[lo:lo + fitness_puzzles]
        ft = pool_targets[lo:lo + fitness_puzzles]
        fm = pool_empty[lo:lo + fitness_puzzles]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            final = run_compiled(fx, fitness_iters).argmax(dim=-1)
        hits = (final == ft) & fm
        if fitness_dense:
            return int(hits.sum().item())
        # A puzzle is solved when every empty cell is correct (givens always match).
        return int((hits | ~fm).all(dim=1).sum().item())

    baseline_probe = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
    log(f"GEN {start_gen - 1:4d} | validation 1024-iter: {baseline_probe}/{len(probe_puzzles)} (starting point)")

    # Calibrate the perturbation scale: pick the largest sigma whose perturbed model
    # keeps at least half the unperturbed fitness. Too large and every member scores
    # zero (no signal); too small and the finite-difference signal drowns in noise.
    if sigma is None:
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
            if score >= unperturbed // 2:
                sigma = candidate
                break
        log(f"CALIBRATE chose sigma={sigma:.0e}")

    if start_gen >= total_steps:
        # Resuming from the final checkpoint: the run is already complete. Save the
        # final model and exit cleanly — without this, the loop below never runs and
        # the return would crash on loop-local variables, failing every retry.
        final_path = os.path.join(output_dir, f"model_es_finetune_{sampling_mode}.pt")
        atomic_torch_save(model.state_dict(), final_path)
        log(f"Run already complete at generation {start_gen - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True, 'sampling_mode': sampling_mode}

    for gen in range(start_gen, total_steps):
        t0 = time.time()
        # Same probe slice for every population member (common random numbers),
        # rotated each generation so we don't overfit one subset.
        lo = (gen * fitness_puzzles) % (fitness_pool_size - fitness_puzzles)

        snapshot = [p.detach().clone() for p in params]
        all_direction_seeds = generation_direction_seeds(1234, gen)
        if sampling_mode == 'paired':
            direction_seeds = all_direction_seeds[:POPULATION_PAIRS]
            scores_plus, scores_minus = [], []
            for direction_seed in direction_seeds:
                perturb(params, direction_seed, sigma)
                scores_plus.append(score_slice(lo))
                with torch.no_grad():
                    for p, saved in zip(params, snapshot):
                        p.copy_(saved)
                perturb(params, direction_seed, -sigma)
                scores_minus.append(score_slice(lo))
                with torch.no_grad():
                    for p, saved in zip(params, snapshot):
                        p.copy_(saved)
            all_scores = np.asarray(scores_plus + scores_minus, dtype=np.float64)
        else:
            direction_seeds = all_direction_seeds
            scores = []
            for direction_seed in direction_seeds:
                perturb(params, direction_seed, sigma)
                scores.append(score_slice(lo))
                with torch.no_grad():
                    for p, saved in zip(params, snapshot):
                        p.copy_(saved)
            all_scores = np.asarray(scores, dtype=np.float64)

        # Centered, score-normalized weights are tie-safe: a generation where every
        # member scores the same takes no step. Sigma is absorbed into the learning rate.
        weights, spread = direction_weights(all_scores, sampling_mode)
        if spread > 1e-9:
            with torch.no_grad():
                for direction_seed, weight in zip(direction_seeds, weights):
                    gen_t = torch.Generator(device=device)
                    gen_t.manual_seed(direction_seed)
                    for p in params:
                        z = torch.randn(p.shape, generator=gen_t, device=device, dtype=torch.float32)
                        p.add_(z, alpha=lr * float(weight))
                for p, a in zip(params, anchor):
                    p.add_(p - a, alpha=-anchor_lambda)

        probe_solved = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
        log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.0f} "
            f"best {max(all_scores):.0f} | validation 1024-iter: {probe_solved}/{len(probe_puzzles)} "
            f"| {time.time() - t0:.0f}s")

        if (gen + 1) % eval_every == 0 or gen == total_steps - 1:
            save_checkpoint(gen)

    final_path = os.path.join(output_dir, f"model_es_finetune_{sampling_mode}.pt")
    atomic_torch_save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_validation': probe_solved, 'sampling_mode': sampling_mode}


if __name__ == "__main__":
    train()
