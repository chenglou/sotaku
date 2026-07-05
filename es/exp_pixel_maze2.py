# Maze, harder. The first maze run saturated instantly (996/1000 at the first probe,
# perfect from step 5,000, both horizons) — the port works, but at that difficulty 16
# iterations already suffices, so the iteration-scaling question never gets asked.
# Same task, two dials turned: minimum path length 6 -> 14 and wall density
# 0.28 -> 0.30. Longer corridors mean information must propagate farther; if more
# thinking loops help anywhere on mazes, it is here.

import os
import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from checkpoint_utils import find_latest_checkpoint
from es.exp_pixel_bootstrap import (
    CELL_PIXELS,
    PIXEL_ROPE_COS,
    PIXEL_ROPE_SIN,
    PixelSudokuTransformer,
)

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "pixel_maze2_checkpoint_step"

CONFIG = {
    'experiment': 'exp_pixel_maze2',
    'd_model': 64,
    'n_heads': 8,
    'd_ff': 256,
    'n_layers': 4,
    'total_steps': 10000,
    'batch_size': 1024,
    'peak_lr': 1e-3,
    'train_iterations': 16,
    'wall_density': 0.30,
    'min_path_len': 14,
}

total_steps = 10000
eval_every = 1000
batch_size = 1024
peak_lr = 1e-3
warmup_steps = 200
final_lr_frac = 0.05
train_iterations = 16
n_mazes = 101_000     # last 1,000 held out as the probe set
log_name = "exp_pixel_maze2.log"

FLOOR, WALL, START, GOAL, PATH = 0, 1, 2, 3, 4
MAZE_GLYPH_ART = {
    FLOOR: ["........", "........", "........", "........", "........", "........", "........", "........"],
    WALL:  ["########", "########", "########", "########", "########", "########", "########", "########"],
    START: ["........", "..####..", ".#......", "..###...", ".....#..", ".####...", "........", "........"],
    GOAL:  ["........", "..####..", ".#......", ".#..##..", ".#...#..", "..####..", "........", "........"],
    PATH:  ["........", "........", "...##...", "..####..", "..####..", "...##...", "........", "........"],
}
MAZE_GLYPHS = torch.tensor(
    [[[1.0 if ch == '#' else 0.0 for ch in row] for row in MAZE_GLYPH_ART[k]] for k in range(5)]
).reshape(5, CELL_PIXELS)

NEIGHBORS = [[] for _ in range(81)]
for r in range(9):
    for c in range(9):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < 9 and 0 <= cc < 9:
                NEIGHBORS[r * 9 + c].append(rr * 9 + cc)


def generate_mazes(n, seed):
    """Rejection-sample mazes with a unique shortest path. Returns (input_idx,
    target_idx) uint8 arrays of shape (n, 81): input marks walls/start/goal, target
    additionally marks the path cells."""
    rng = np.random.RandomState(seed)
    inputs = np.zeros((n, 81), dtype=np.uint8)
    targets = np.zeros((n, 81), dtype=np.uint8)
    made = attempts = 0
    while made < n:
        attempts += 1
        walls = rng.rand(81) < CONFIG['wall_density']
        free = np.flatnonzero(~walls)
        if len(free) < 12:
            continue
        s, g = rng.choice(free, 2, replace=False)
        dist = [-1] * 81
        count = [0] * 81
        dist[s], count[s] = 0, 1
        queue = deque([s])
        while queue:
            v = queue.popleft()
            for u in NEIGHBORS[v]:
                if walls[u]:
                    continue
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    count[u] = count[v]
                    queue.append(u)
                elif dist[u] == dist[v] + 1:
                    count[u] += count[v]
        if dist[g] < CONFIG['min_path_len'] or count[g] != 1:
            continue
        path = []
        v = g
        while v != s:
            v = next(u for u in NEIGHBORS[v] if not walls[u] and dist[u] == dist[v] - 1)
            if v != s:
                path.append(v)
        cell = np.zeros(81, dtype=np.uint8)
        cell[walls] = WALL
        cell[s], cell[g] = START, GOAL
        inputs[made] = cell
        target = cell.copy()
        target[path] = PATH
        targets[made] = target
        made += 1
    return inputs, targets, attempts


def run_iterations(model, x_pixels, n_iters, rope_cos, rope_sin):
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

    glyphs = MAZE_GLYPHS.to(device)
    rope_cos = PIXEL_ROPE_COS.to(device)
    rope_sin = PIXEL_ROPE_SIN.to(device)

    log(f"Generating {n_mazes} unique-shortest-path mazes...")
    t_gen = time.time()
    inputs_np, targets_np, attempts = generate_mazes(n_mazes, seed=42)
    log(f"Generated in {time.time() - t_gen:.0f}s ({attempts} attempts, "
        f"{n_mazes / attempts:.0%} acceptance)")
    inputs = torch.from_numpy(inputs_np).long().to(device)
    targets = torch.from_numpy(targets_np).long().to(device)
    train_inputs, train_targets = inputs[:-1000], targets[:-1000]
    probe_inputs, probe_targets = inputs[-1000:], targets[-1000:]

    model = PixelSudokuTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Maze model: {n_params} parameters (unchanged pixel sudoku architecture)")
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
        """A maze is solved when every floor cell is correctly classified path or
        not-path (nearest glyph among FLOOR and PATH)."""
        solved = cells_right = cells_total = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, probe_inputs.size(0), 250):
                pi = probe_inputs[start:start + 250]
                pt = probe_targets[start:start + 250]
                probs = torch.sigmoid(run_iterations(model, glyphs[pi], n_iters, rope_cos, rope_sin).float())
                d_floor = (probs - glyphs[FLOOR].view(1, 1, CELL_PIXELS)).abs().sum(-1)
                d_path = (probs - glyphs[PATH].view(1, 1, CELL_PIXELS)).abs().sum(-1)
                decoded_path = d_path < d_floor
                floor = pi == FLOOR
                hits = (decoded_path == (pt == PATH)) & floor
                cells_right += int(hits.sum().item())
                cells_total += int(floor.sum().item())
                solved += int((hits | ~floor).all(dim=1).sum().item())
        return solved, cells_right, cells_total

    if start_step >= total_steps:
        final_path = os.path.join(output_dir, "model_pixel_maze2.pt")
        torch.save(model.state_dict(), final_path)
        log(f"Run already complete at step {start_step - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True}

    model.train()
    t0 = time.time()
    for step in range(start_step, total_steps):
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step)

        idx = torch.randint(0, train_inputs.size(0), (batch_size,), device=device)
        pi, pt = train_inputs[idx], train_targets[idx]
        x = glyphs[pi]
        target_pixels = glyphs[pt]
        floor = (pi == FLOOR).unsqueeze(-1).expand_as(target_pixels)

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
                loss = loss + F.binary_cross_entropy_with_logits(logits.float()[floor], target_pixels[floor])
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

    final_path = os.path.join(output_dir, "model_pixel_maze2.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_solved_128': s128}


if __name__ == "__main__":
    train()
