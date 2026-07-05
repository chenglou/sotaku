# Reward-modulated Hebbian learning, from scratch — the dumbest version on purpose.
# The from-scratch wall: backprop tells every weight individually how it affected every
# cell; a scalar fitness tells the whole model one number (and ES from scratch sat at
# chance in every configuration tested). This rule sits between the two: every Linear
# accumulates a local eligibility trace (the input-output correlation, summed over all
# iterations and cells of one rollout), and the GLOBAL fitness advantage gates whether
# those correlations get strengthened or weakened — local credit granularity,
# forward-only, one rollout per update instead of ES's 32.
#
# Deliberately minimal per the exploration guideline: no Oja stabilization, no
# per-layer learning rates, no trace decay — normalized traces, an EMA baseline, and
# one step size. The question is binary: does fitness move off chance where scalar-
# fitness ES was flat? Judge the trend, then add machinery only if the trend is real.

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import find_latest_checkpoint
from es.exp_es_tiny import TINY_ROPE_COS, TINY_ROPE_SIN, TinySudokuTransformer
from iters.exp_baseline_lr2e3 import RATING_BUCKETS, encode_puzzles

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "hebb_checkpoint_step"

CONFIG = {
    'experiment': 'exp_hebb',
    'rule': 'reward_modulated_hebbian',
    'updates': 3000,
    'lr': 1e-3,
    'baseline_ema': 0.9,
    'fitness': 'dense_cells',
    'fitness_puzzles': 384,
    'fitness_iters': 16,
}

total_steps = 3000        # updates; one rollout each
eval_every = 300
lr = 1e-3
baseline_ema = 0.9
fitness_puzzles = 384
fitness_iters = 16
fitness_pool_offset = 2_700_000
fitness_pool_size = 20_000
log_name = "exp_hebb.log"


def run_iterations(model, x, n_iters):
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


class TraceRecorder:
    """Forward hooks on every Linear: accumulate the input-output outer product
    (summed over batch, cells, and iterations) — the eligibility trace, shaped like
    the weight it belongs to."""

    def __init__(self, model):
        self.linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.traces = None
        self.bias_traces = None
        self.handles = []

    def _hook(self, idx):
        def hook(module, args, output):
            x = args[0].detach().float().reshape(-1, module.in_features)
            y = output.detach().float().reshape(-1, module.out_features)
            self.traces[idx] += y.t() @ x
            self.bias_traces[idx] += y.sum(dim=0)
        return hook

    def start(self):
        device = self.linears[0].weight.device
        self.traces = [torch.zeros_like(m.weight, dtype=torch.float32, device=device) for m in self.linears]
        self.bias_traces = [torch.zeros(m.out_features, dtype=torch.float32, device=device) for m in self.linears]
        self.handles = [m.register_forward_hook(self._hook(i)) for i, m in enumerate(self.linears)]

    def stop(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def train(output_dir="."):
    device = torch.device("cuda")

    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log("Loading fitness pool...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[fitness_pool_offset:fitness_pool_offset + fitness_pool_size]
    pool_puzzles = pool_rows["question"]
    pool_solutions = pool_rows["answer"]
    pool_x = encode_puzzles(pool_puzzles).to(device)
    pool_targets = torch.tensor([[int(s[j]) - 1 for j in range(81)] for s in pool_solutions], device=device)
    pool_empty = torch.tensor([[p[j] == '.' for j in range(81)] for p in pool_puzzles], device=device)
    log(f"Fitness pool: {len(pool_puzzles)} puzzles")

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
    probe_puzzles = [test_dataset[i]['question'] for i in probe_idx]
    probe_solutions = [test_dataset[i]['answer'] for i in probe_idx]
    probe_x = encode_puzzles(probe_puzzles).to(device)
    log(f"Validation probe: {len(probe_puzzles)} puzzles")

    model = TinySudokuTransformer().to(device)

    checkpoint_path, _ = find_latest_checkpoint(output_dir, CHECKPOINT_PREFIX)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state_dict'])
        start_step = ckpt['step'] + 1
        baseline = float(ckpt['baseline'])
        log(f"Resumed from update {ckpt['step']}")
    else:
        seed_path = os.path.join(os.getcwd(), "seed_model.pt")
        state = torch.load(seed_path, map_location=device, weights_only=True)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        model.load_state_dict(state)
        start_step = 0
        baseline = None
        log(f"Loaded seed model from {seed_path}")

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    recorder = TraceRecorder(model)

    def save_checkpoint(step, baseline):
        path = os.path.join(output_dir, f"{CHECKPOINT_PREFIX}{step}.pt")
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'baseline': baseline if baseline is not None else 0.0,
            'config': CONFIG,
        }, path)
        log(f"Checkpoint saved: {path}")

    baseline_probe = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
    log(f"STEP {start_step - 1:5d} | {n_params} params | validation {fitness_iters}-iter: "
        f"{baseline_probe}/{len(probe_puzzles)} (starting point)")

    if start_step >= total_steps:
        final_path = os.path.join(output_dir, "model_hebb.pt")
        torch.save(model.state_dict(), final_path)
        log(f"Run already complete at update {start_step - 1}; final model saved: {final_path}")
        log_file.close()
        return {'already_complete': True}

    t_start = time.time()
    for step in range(start_step, total_steps):
        lo = (step * fitness_puzzles) % (fitness_pool_size - fitness_puzzles)
        fx = pool_x[lo:lo + fitness_puzzles]
        ft = pool_targets[lo:lo + fitness_puzzles]
        fm = pool_empty[lo:lo + fitness_puzzles]

        recorder.start()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            final = run_iterations(model, fx, fitness_iters).argmax(dim=-1)
        recorder.stop()
        fitness = float(((final == ft) & fm).sum().item())

        if baseline is None:
            baseline = fitness
        advantage = fitness - baseline
        baseline = baseline_ema * baseline + (1 - baseline_ema) * fitness

        with torch.no_grad():
            for module, trace, bias_trace in zip(recorder.linears, recorder.traces, recorder.bias_traces):
                norm = trace.norm()
                if norm > 1e-12:
                    module.weight.add_(trace / norm, alpha=lr * advantage / fitness_puzzles)
                bias_norm = bias_trace.norm()
                if module.bias is not None and bias_norm > 1e-12:
                    module.bias.add_(bias_trace / bias_norm, alpha=lr * advantage / fitness_puzzles)

        if step % 25 == 0 or step == total_steps - 1:
            probe_solved = count_solved(model, probe_x, probe_puzzles, probe_solutions, fitness_iters)
            log(f"STEP {step:5d} | fitness {fitness:.0f} baseline {baseline:.0f} adv {advantage:+.0f} | "
                f"validation {fitness_iters}-iter: {probe_solved}/{len(probe_puzzles)} | "
                f"{time.time() - t_start:.0f}s")

        if step % eval_every == 0 or step == total_steps - 1:
            save_checkpoint(step, baseline)

    final_path = os.path.join(output_dir, "model_hebb.pt")
    torch.save(model.state_dict(), final_path)
    log(f"Final model saved: {final_path}")
    log_file.close()
    return {'final_fitness': fitness}


if __name__ == "__main__":
    train()
