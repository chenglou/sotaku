# The ES/settledness stack, ported below the symbol level. Seed: the pixel bootstrap
# model (d_model=64, ~345/1000 solved at 16 and 128 after 8K supervised steps).
# Fitness: puzzles whose glyph-decoded answer is correct AND unchanged over the last
# 64 of 256 iterations — the settledness recipe that produced the digit domain's best
# model, now grading drawn pixels. Probes log solved/settled/both at 256 every 5
# generations and a 1024-iteration transfer reading every 25. The question: does the
# repair stack work on a representation it was not developed on?

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from es.exp_pixel_bootstrap import (
    CELL_PIXELS,
    GLYPHS,
    PIXEL_ROPE_COS,
    PIXEL_ROPE_SIN,
    PixelSudokuTransformer,
    digits_tensor,
)
from iters.exp_baseline_lr2e3 import RATING_BUCKETS

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "es_pixel_checkpoint_step"

CONFIG = {
    'experiment': 'exp_es_pixel',
    'es_generations': 300,
    'population_pairs': 16,
    'sigma': 'calibrated',
    'lr': 3e-4,
    'anchor_lambda': 1e-3,
    'fitness': 'solved_and_settled_pixels',
    'fitness_puzzles': 384,
    'fitness_iters': 256,
    'settle_window': 64,
}

total_steps = 300
eval_every = 50
population_pairs = 16
sigma_ladder = [3e-4, 1e-4, 3e-5, 1e-5]
lr = 3e-4
anchor_lambda = 1e-3
fitness_puzzles = 384
fitness_iters = 256
SETTLE_WINDOW = 64
fitness_pool_offset = 2_700_000
fitness_pool_size = 20_000
log_name = "exp_es_pixel.log"

COMPILE_CHUNK = 32


def make_two_phase_runner(model, device):
    rope_cos = PIXEL_ROPE_COS.to(device)
    rope_sin = PIXEL_ROPE_SIN.to(device)

    def iter_block(h_prev, preds):
        for _ in range(COMPILE_CHUNK):
            h = h_prev + model.pred_proj(preds)
            for layer in model.layers:
                h = layer(h, rope_cos, rope_sin)
            h_prev = h
            preds = torch.sigmoid(model.output_head(h))
        return h_prev, preds

    compiled_block = torch.compile(iter_block, dynamic=False)

    def run(x, n_total, window):
        assert n_total % COMPILE_CHUNK == 0 and window % COMPILE_CHUNK == 0
        h_prev = model.initial_encoder(x)
        preds = torch.zeros_like(x)
        for _ in range((n_total - window) // COMPILE_CHUNK):
            h_prev, preds = compiled_block(h_prev, preds)
        probs_early = torch.sigmoid(model.output_head(h_prev).float())
        for _ in range(window // COMPILE_CHUNK):
            h_prev, preds = compiled_block(h_prev, preds)
        probs_final = torch.sigmoid(model.output_head(h_prev).float())
        return probs_early, probs_final

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
    glyphs = GLYPHS.to(device)

    def decode(probs):
        """Nearest digit glyph per cell: (B, 81) decoded digits 1-9."""
        dists = (probs.unsqueeze(2) - glyphs[1:].view(1, 1, 9, CELL_PIXELS)).abs().sum(dim=-1)
        return dists.argmin(dim=-1) + 1

    print("Loading fitness pool...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[fitness_pool_offset:fitness_pool_offset + fitness_pool_size]
    pool_puzzle_digits = digits_tensor(pool_rows["question"], '.', device)
    pool_answer_digits = digits_tensor(pool_rows["answer"], '_', device)
    print(f"Fitness pool: {pool_puzzle_digits.size(0)} puzzles")

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
    probe_puzzle_digits = digits_tensor([test_dataset[i]['question'] for i in probe_idx], '.', device)
    probe_answer_digits = digits_tensor([test_dataset[i]['answer'] for i in probe_idx], '_', device)
    print(f"Validation probe: {probe_puzzle_digits.size(0)} puzzles")

    model = PixelSudokuTransformer().to(device)

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

    def solved_settled(puzzle_digits, answer_digits, n_total, window, chunk):
        solved = settled = both = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, puzzle_digits.size(0), chunk):
                pd = puzzle_digits[start:start + chunk]
                ad = answer_digits[start:start + chunk]
                probs_early, probs_final = run_two(glyphs[pd], n_total, window)
                d_early = decode(probs_early)
                d_final = decode(probs_final)
                empty = pd == 0
                ok = (((d_final == ad) & empty) | ~empty).all(dim=1)
                still = (((d_final == d_early) & empty) | ~empty).all(dim=1)
                solved += int(ok.sum().item())
                settled += int(still.sum().item())
                both += int((ok & still).sum().item())
        return solved, settled, both

    def score_slice(lo):
        _, _, both = solved_settled(
            pool_puzzle_digits[lo:lo + fitness_puzzles],
            pool_answer_digits[lo:lo + fitness_puzzles],
            fitness_iters, SETTLE_WINDOW, fitness_puzzles,
        )
        return both

    s0, t0_, b0 = solved_settled(probe_puzzle_digits, probe_answer_digits, fitness_iters, SETTLE_WINDOW, 250)
    log(f"GEN {start_gen - 1:4d} | validation {fitness_iters}-iter: solved {s0}/1000, settled {t0_}/1000, both {b0}/1000 (starting point)")

    if start_gen >= total_steps:
        final_path = os.path.join(output_dir, "model_es_pixel.pt")
        torch.save(model.state_dict(), final_path)
        log(f"Run already complete at generation {start_gen - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True}

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
            s, t, b = solved_settled(probe_puzzle_digits, probe_answer_digits, fitness_iters, SETTLE_WINDOW, 250)
            transfer = ""
            if gen % 25 == 0 or gen == total_steps - 1:
                s1024, t1024, b1024 = solved_settled(probe_puzzle_digits, probe_answer_digits, 1024, SETTLE_WINDOW, 250)
                transfer = f" | 1024-iter both: {b1024}/1000"
            log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.0f} best {max(all_scores):.0f} | "
                f"validation {fitness_iters}-iter: solved {s}/1000, settled {t}/1000, both {b}/1000{transfer} | {time.time() - t0:.0f}s")
        else:
            log(f"GEN {gen:4d} | fitness mean {np.mean(all_scores):.0f} best {max(all_scores):.0f} | {time.time() - t0:.0f}s")

        if gen % eval_every == 0 or gen == total_steps - 1:
            save_checkpoint(gen)

    final_path = os.path.join(output_dir, "model_es_pixel.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_both': b}


if __name__ == "__main__":
    train()
