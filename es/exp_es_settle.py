# ES on SETTLEDNESS at 2048 iterations: fitness counts puzzles that are solved AND
# whose answer stopped changing over the last 128 iterations. Every model fine-tuned
# at 1024 softens or collapses at 2048 (95.2% -> 59.0%, 94.6% -> 89.5%) — they were
# only ever graded on a snapshot. This grades the settling behavior itself, at a
# horizon beyond deployment. Seed: the 96.5% record model (es_ft_stable final), the
# only one that currently holds 2048; the question is whether rewarding stopping
# deepens its basin further and generalizes past the graded horizon (probe logs
# solved and settled separately at 2048). Per the train/deploy-consistency rule,
# a model trained this way would also legitimize run-until-settled deployment.

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

CHECKPOINT_PREFIX = "es_settle_checkpoint_step"

CONFIG = {
    'experiment': 'exp_es_settle',
    'es_generations': 60,
    'population_pairs': 16,
    'sigma': 'calibrated',
    'lr': 3e-4,
    'anchor_lambda': 1e-3,
    'fitness': 'solved_and_settled',
    'fitness_puzzles': 384,
    'fitness_iters': 2048,
    'settle_window': 128,
}

total_steps = 60
eval_every = 20
population_pairs = 16
sigma_ladder = [3e-4, 1e-4, 3e-5, 1e-5]
lr = 3e-4
anchor_lambda = 1e-3
fitness_puzzles = 384
fitness_iters = 2048
SETTLE_WINDOW = 128       # answer must be unchanged from iteration 1920 to 2048
fitness_pool_offset = 2_700_000
fitness_pool_size = 20_000
log_name = "exp_es_settle.log"

COMPILE_CHUNK = 32


def make_two_phase_runner(model, device):
    """Compiled runner that returns the answer at (n_total - window) and at n_total,
    so settledness is one extra argmax instead of a second full run."""
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

    def run(x, n_total, window):
        assert n_total % COMPILE_CHUNK == 0 and window % COMPILE_CHUNK == 0
        h_prev = model.initial_encoder(x)
        preds = torch.zeros(x.size(0), 81, 9, device=device)
        for _ in range((n_total - window) // COMPILE_CHUNK):
            h_prev, preds = compiled_block(h_prev, preds)
        answer_early = model.output_head(h_prev).argmax(dim=-1)
        for _ in range(window // COMPILE_CHUNK):
            h_prev, preds = compiled_block(h_prev, preds)
        answer_final = model.output_head(h_prev).argmax(dim=-1)
        return answer_early, answer_final

    return run


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
    probe_x = encode_puzzles([test_dataset[i]['question'] for i in probe_idx]).to(device)
    probe_targets = torch.tensor([[int(test_dataset[i]['answer'][j]) - 1 for j in range(81)] for i in probe_idx], device=device)
    probe_empty = torch.tensor([[test_dataset[i]['question'][j] == '.' for j in range(81)] for i in probe_idx], device=device)
    print(f"Validation probe: {probe_x.size(0)} puzzles")

    model = SudokuTransformer().to(device)

    checkpoint_path, _ = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
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

    model.eval()
    params = list(model.parameters())
    run_two = make_two_phase_runner(model, device)

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

    def solved_settled(x, targets, empty, chunk):
        """Counts over one batch set: (solved, settled, both). Settledness is judged
        on empty cells only — given cells' predictions are irrelevant to the answer."""
        solved = settled = both = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, x.size(0), chunk):
                a_early, a_final = run_two(x[start:start + chunk], fitness_iters, SETTLE_WINDOW)
                tb = targets[start:start + chunk]
                eb = empty[start:start + chunk]
                ok = (((a_final == tb) & eb) | ~eb).all(dim=1)
                still = (((a_final == a_early) & eb) | ~eb).all(dim=1)
                solved += int(ok.sum().item())
                settled += int(still.sum().item())
                both += int((ok & still).sum().item())
        return solved, settled, both

    def score_slice(lo):
        _, _, both = solved_settled(
            pool_x[lo:lo + fitness_puzzles],
            pool_targets[lo:lo + fitness_puzzles],
            pool_empty[lo:lo + fitness_puzzles],
            fitness_puzzles,
        )
        return both

    s0, t0_, b0 = solved_settled(probe_x, probe_targets, probe_empty, 500)
    log(f"GEN {start_gen - 1:4d} | validation 2048-iter: solved {s0}/1000, settled {t0_}/1000, both {b0}/1000 (starting point)")

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
        final_path = os.path.join(output_dir, "model_es_settle.pt")
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
                for p, a in zip(params, anchor):
                    p.add_(p - a, alpha=-anchor_lambda)

        if gen % 5 == 0 or gen == total_steps - 1:
            s, t, b = solved_settled(probe_x, probe_targets, probe_empty, 500)
            log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.0f} best {max(all_scores):.0f} | "
                f"validation 2048-iter: solved {s}/1000, settled {t}/1000, both {b}/1000 | {time.time() - t0:.0f}s")
        else:
            log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.0f} best {max(all_scores):.0f} | {time.time() - t0:.0f}s")

        if gen % eval_every == 0 or gen == total_steps - 1:
            save_checkpoint(gen)

    final_path = os.path.join(output_dir, "model_es_settle.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_both': b}


if __name__ == "__main__":
    train()
