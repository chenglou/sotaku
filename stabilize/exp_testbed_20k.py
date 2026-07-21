# 20K-step testbed for stabilization experiments: the exp_baseline_lr2e3 recipe with
# the schedule compressed 2.5x, plus an in-training probe that measures 128- and
# 1024-iteration accuracy on 1,000 fixed test puzzles every 1,000 steps. Runs in ~1h
# on H200 and shows the full stability trajectory, not just the final outcome.
# Baseline for A/B testing stabilization ideas; variants copy this file.

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
import json
import random
import os
import math
import numpy as np
import re
from checkpoint_utils import atomic_torch_save, find_latest_checkpoint, load_checkpoint
from iters.state_norm import DEFAULT_EPSILON, cap_token_rms, rms_normalize

torch.set_float32_matmul_precision('high')

CHECKPOINT_PREFIX = "testbed_20k_checkpoint_step"

CONFIG = {
    'experiment': 'exp_testbed_20k',
    'd_model': 128,
    'd_ff': 512,
    'n_layers': 4,
    'batch_size': 2048,
    'lr': 2e-3,
    'warmup_steps': 560,
    'lr_min_ratio': 0.01,
    'total_steps': 20000,
}

d_model = 128
n_heads = 4
d_ff = 512
n_layers = 4
n_iterations = 16
lr = 2e-3
warmup_steps = 560
lr_min_ratio = 0.01
total_steps = 20000
batch_size = 2048
train_size = 2700000
eval_every = 2000
checkpoint_prefix = CHECKPOINT_PREFIX
log_name = "exp_testbed_20k.log"

PHASES = [
    (0, 4000, 21, "Phase 1: Hard only (rating 21+)"),
    (4000, 8000, 6, "Phase 2: Medium+ (rating 6+)"),
    (8000, 12000, 1, "Phase 3: Easy+ (rating 1+)"),
    (12000, 20000, 0, "Phase 4: All (rating 0+)"),
]

RATING_BUCKETS = [
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
]

ROW_IDX = torch.tensor([i // 9 for i in range(81)])
COL_IDX = torch.tensor([i % 9 for i in range(81)])

CHAR_TO_DIGIT = np.zeros(256, dtype=np.uint8)
for i in range(1, 10):
    CHAR_TO_DIGIT[ord(str(i))] = i

CHAR_TO_TARGET = np.zeros(256, dtype=np.uint8)
for i in range(1, 10):
    CHAR_TO_TARGET[ord(str(i))] = i - 1

CHAR_TO_DIGIT[ord('.')] = 0

ENCODE_CHUNK_SIZE = 50000
ONE_HOT = np.eye(10, dtype=np.float32)

# Precompute 2D RoPE cos/sin for all 81 positions
head_dim = d_model // n_heads  # 32
rope_half = head_dim // 2  # 16 dims per spatial axis
rope_pairs = rope_half // 2  # 8 frequency pairs per axis
rope_base = 10.0

_freqs = 1.0 / (rope_base ** (torch.arange(rope_pairs).float() * 2 / rope_half))
_row_angles = ROW_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)
_col_angles = COL_IDX.float().unsqueeze(1) * _freqs.unsqueeze(0)
_angles = torch.cat([_row_angles, _col_angles], dim=-1)
ROPE_COS = _angles.cos()
ROPE_SIN = _angles.sin()


def apply_rope(x, cos, sin):
    B, H, L, D = x.shape
    pairs = x.reshape(B, H, L, D // 2, 2)
    x0, x1 = pairs[..., 0], pairs[..., 1]
    c = cos.unsqueeze(0).unsqueeze(0)
    s = sin.unsqueeze(0).unsqueeze(0)
    out0 = x0 * c - x1 * s
    out1 = x0 * s + x1 * c
    return torch.stack([out0, out1], dim=-1).reshape(B, H, L, D)


class RoPETransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.attn_dropout_p = dropout

    def forward(self, x, rope_cos, rope_sin):
        h = self.norm1(x)
        B, L, D = h.shape

        q = self.q_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout_p if self.training else 0.0)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.dropout1(self.out_proj(attn_out))

        h2 = self.norm2(x)
        h2 = self.linear2(self.dropout(F.relu(self.linear1(h2))))
        x = x + self.dropout2(h2)
        return x


class SudokuTransformer(nn.Module):
    def __init__(
        self,
        outer_state_norm=False,
        outer_state_norm_epsilon=DEFAULT_EPSILON,
        outer_state_rms_cap=None,
    ):
        super().__init__()
        if outer_state_norm and outer_state_rms_cap is not None:
            raise ValueError("outer state normalization and capping are mutually exclusive")
        if outer_state_rms_cap is not None and outer_state_rms_cap <= 0:
            raise ValueError("outer_state_rms_cap must be positive")
        self.outer_state_norm = outer_state_norm
        self.outer_state_norm_epsilon = outer_state_norm_epsilon
        self.outer_state_rms_cap = outer_state_rms_cap
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        self.output_head = nn.Linear(d_model, 9)

    def normalize_outer_state(self, hidden_state):
        if self.outer_state_norm:
            return rms_normalize(hidden_state, self.outer_state_norm_epsilon)
        if self.outer_state_rms_cap is not None:
            return cap_token_rms(
                hidden_state,
                self.outer_state_rms_cap,
                self.outer_state_norm_epsilon,
            )
        return hidden_state

    def forward(self, x, return_all=False):
        batch_size = x.size(0)
        device = x.device
        rope_cos = ROPE_COS.to(device)
        rope_sin = ROPE_SIN.to(device)

        h_prev = self.initial_encoder(x)
        preds = torch.zeros(batch_size, 81, 9, device=device)

        all_logits = []
        for _ in range(n_iterations):
            h = h_prev + self.pred_proj(preds)
            for layer in self.layers:
                h = layer(h, rope_cos, rope_sin)
            h = self.normalize_outer_state(h)
            h_prev = h
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
            if return_all:
                all_logits.append(logits)
        return all_logits if return_all else logits


def encode_puzzles(puzzles):
    if not puzzles:
        return torch.empty((0, 81, 10), dtype=torch.float32)
    chunks = []
    for start in range(0, len(puzzles), ENCODE_CHUNK_SIZE):
        chunk = puzzles[start:start + ENCODE_CHUNK_SIZE]
        buf = ''.join(chunk).encode('ascii')
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(chunk), 81)
        digits = CHAR_TO_DIGIT[arr]
        chunks.append(torch.from_numpy(ONE_HOT[digits]))
    return torch.cat(chunks, dim=0)


def encode_solutions(solutions):
    if not solutions:
        return torch.empty((0, 81), dtype=torch.uint8)
    chunks = []
    for start in range(0, len(solutions), ENCODE_CHUNK_SIZE):
        chunk = solutions[start:start + ENCODE_CHUNK_SIZE]
        buf = ''.join(chunk).encode('ascii')
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(len(chunk), 81)
        digits = CHAR_TO_TARGET[arr]
        chunks.append(torch.from_numpy(digits))
    return torch.cat(chunks, dim=0)


def get_lr(step, schedule_warmup_steps=warmup_steps, schedule_total_steps=total_steps):
    if step < schedule_warmup_steps:
        return lr * (step + 1) / schedule_warmup_steps
    progress = (
        (step - schedule_warmup_steps)
        / (schedule_total_steps - schedule_warmup_steps)
    )
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (lr_min_ratio + (1 - lr_min_ratio) * cosine_decay)


def resolve_schedule(overrides=None):
    schedule = {
        'warmup_steps': warmup_steps,
        'total_steps': total_steps,
        'eval_every': eval_every,
        'probe_every': 1000,
        'phases': tuple(PHASES),
    }
    if overrides:
        unknown_keys = set(overrides) - set(schedule)
        if unknown_keys:
            raise ValueError(f"unknown schedule settings: {sorted(unknown_keys)}")
        schedule.update(overrides)

    for key in ('warmup_steps', 'total_steps', 'eval_every', 'probe_every'):
        if not isinstance(schedule[key], int) or schedule[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if schedule['warmup_steps'] >= schedule['total_steps']:
        raise ValueError("warmup_steps must be smaller than total_steps")

    phases = tuple(tuple(phase) for phase in schedule['phases'])
    expected_start = 0
    for phase in phases:
        if len(phase) != 4:
            raise ValueError("each training phase must have four fields")
        start, end, _, _ = phase
        if start != expected_start or end <= start:
            raise ValueError("training phases must be contiguous and increasing")
        expected_start = end
    if expected_start != schedule['total_steps']:
        raise ValueError("training phases must cover exactly total_steps")
    schedule['phases'] = phases
    return schedule


def train(
    output_dir=".",
    *,
    experiment_name=CONFIG['experiment'],
    run_name=None,
    outer_state_norm=False,
    outer_state_norm_epsilon=DEFAULT_EPSILON,
    outer_state_rms_cap=None,
    random_seed=None,
    checkpoint_on_probe=False,
    schedule=None,
):
    if run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")
    if outer_state_norm_epsilon <= 0:
        raise ValueError("outer_state_norm_epsilon must be positive")
    if outer_state_norm and outer_state_rms_cap is not None:
        raise ValueError("outer state normalization and capping are mutually exclusive")
    if outer_state_rms_cap is not None and outer_state_rms_cap <= 0:
        raise ValueError("outer_state_rms_cap must be positive")

    run_schedule = resolve_schedule(schedule)
    run_warmup_steps = run_schedule['warmup_steps']
    run_total_steps = run_schedule['total_steps']
    run_eval_every = run_schedule['eval_every']
    run_probe_every = run_schedule['probe_every']
    run_phases = run_schedule['phases']

    run_config = dict(CONFIG)
    run_config['experiment'] = experiment_name
    if schedule is not None:
        run_config.update(run_schedule)
    if run_name is not None:
        run_config.update({
            'run_name': run_name,
            'outer_state_norm': 'rmsnorm_no_affine' if outer_state_norm else 'none',
            'outer_state_norm_epsilon': outer_state_norm_epsilon,
            'random_seed': random_seed,
            'checkpoint_on_probe': checkpoint_on_probe,
        })
        if outer_state_rms_cap is not None:
            run_config['outer_state_rms_cap'] = outer_state_rms_cap
        run_checkpoint_prefix = f"{run_name}_checkpoint_step"
        run_log_name = f"{run_name}.log"
        final_model_name = f"model_{run_name}.pt"
        best_model_name = f"model_{run_name}_best_probe.pt"
        result_name = f"result_{run_name}.json"
    else:
        run_checkpoint_prefix = checkpoint_prefix
        run_log_name = log_name
        final_model_name = "model_testbed_20k.pt"
        best_model_name = "model_testbed_20k_best_probe.pt"
        result_name = "result_testbed_20k.json"

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)

    device = torch.device("cuda")
    print(
        "SDPA backends enabled: "
        f"flash={torch.backends.cuda.flash_sdp_enabled()}, "
        f"mem_efficient={torch.backends.cuda.mem_efficient_sdp_enabled()}, "
        f"math={torch.backends.cuda.math_sdp_enabled()}"
    )

    print("Loading sudoku-extreme train split...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    print(f"Total available: {len(dataset)}")
    train_size_local = min(train_size, len(dataset))
    print(f"Using first {train_size_local} for training")

    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    print(f"Test set: {len(test_dataset)}")

    print("\nEncoding training data by rating...")
    train_rows = dataset[:train_size_local]
    ratings = np.asarray(train_rows["rating"], dtype=np.int16)
    puzzles_all = train_rows["question"]
    solutions_all = train_rows["answer"]
    x_all = encode_puzzles(puzzles_all)
    targets_all = encode_solutions(solutions_all)
    del puzzles_all, solutions_all, train_rows
    train_data = {}
    for min_r, max_r, name in RATING_BUCKETS:
        idx = np.where((ratings >= min_r) & (ratings <= max_r))[0]
        if idx.size == 0:
            continue
        print(f"  Rating {name}: {len(idx)} puzzles...", end=" ", flush=True)
        train_data[(min_r, max_r)] = {
            'idx': torch.from_numpy(idx),
            'size': int(idx.size),
        }
        print("done")

    phase_buckets = {}
    for start, end, min_rating, name in run_phases:
        buckets_for_phase = [k for k in train_data.keys() if k[0] >= min_rating]
        total = sum(train_data[k]['size'] for k in buckets_for_phase)
        phase_buckets[min_rating] = buckets_for_phase
        print(f"  {name}: {total} puzzles")

    print("\nPreparing test data...")
    test_data = {}
    for min_r, max_r, name in RATING_BUCKETS:
        indices = [i for i in range(len(test_dataset)) if min_r <= test_dataset[i]['rating'] <= max_r]
        if len(indices) == 0:
            continue
        if len(indices) > 5000:
            indices = random.sample(indices, 5000)
        puzzles = [test_dataset[i]['question'] for i in indices]
        solutions = [test_dataset[i]['answer'] for i in indices]
        x_test = encode_puzzles(puzzles).to(device)
        test_data[name] = {
            'x': x_test,
            'puzzles': puzzles,
            'solutions': solutions,
        }
        print(f"  Test {name}: {len(puzzles)} puzzles")

    # Fixed probe set for the in-training long-horizon probe: 200 puzzles per bucket.
    probe_x_parts, probe_puzzles, probe_solutions = [], [], []
    for name, data in test_data.items():
        probe_x_parts.append(data['x'][:200])
        probe_puzzles.extend(data['puzzles'][:200])
        probe_solutions.extend(data['solutions'][:200])
    probe_x = torch.cat(probe_x_parts, dim=0)
    print(f"Long-horizon probe set: {len(probe_puzzles)} puzzles")

    model = SudokuTransformer(
        outer_state_norm=outer_state_norm,
        outer_state_norm_epsilon=outer_state_norm_epsilon,
        outer_state_rms_cap=outer_state_rms_cap,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {param_count:,}")

    checkpoint_path, start_step = None, 0
    checkpoint_data = None
    checkpoint_path, start_step = find_latest_checkpoint(output_dir, run_checkpoint_prefix)
    if checkpoint_path:
        print(f"Found checkpoint: {checkpoint_path}")
        checkpoint_data = load_checkpoint(checkpoint_path, model, run_config)
        start_step = int(checkpoint_data['step']) + 1
        print(f"Loaded model weights through step {start_step - 1}")

    model = torch.compile(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    if checkpoint_data:
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        rng_state = checkpoint_data.get('rng_state')
        if rng_state:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            torch.cuda.set_rng_state_all(rng_state['cuda'])
        print(f"Resuming at step {start_step}")

    print(f"\nExperiment: {experiment_name}")
    print(f"Architecture: d_model={d_model}, d_ff={d_ff}, n_layers={n_layers}")
    print(f"Iterations: {n_iterations}")
    print(f"Batch size: {batch_size}, lr: {lr}, warmup_steps: {run_warmup_steps}")
    print(f"Total steps: {run_total_steps}")
    if outer_state_norm:
        outer_state_constraint = "RMSNorm without affine"
    elif outer_state_rms_cap is not None:
        outer_state_constraint = f"per-token RMS cap at {outer_state_rms_cap:g}"
    else:
        outer_state_constraint = "none"
    print(f"Outer state constraint: {outer_state_constraint}")
    print(f"Random seed: {random_seed}")
    print(f"Output directory: {output_dir}")

    log_path = os.path.join(output_dir, run_log_name)
    log_file = open(log_path, "a")

    probe_history = list(checkpoint_data.get('probe_history', [])) if checkpoint_data else []
    best_probe = dict(checkpoint_data.get('best_probe', {'step': -1, 'solved_1024': -1})) if checkpoint_data else {'step': -1, 'solved_1024': -1}

    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    def get_phase(step):
        for start, end, min_rating, name in run_phases:
            if start <= step < end:
                return phase_buckets[min_rating], name
        return None, None

    def sample_batch(active_buckets, bs):
        sizes = np.array([train_data[b]['size'] for b in active_buckets], dtype=np.int64)
        total = sizes.sum()
        probs = sizes / total
        counts = np.random.multinomial(bs, probs)
        x_parts = []
        t_parts = []
        for b, count in zip(active_buckets, counts):
            if count == 0:
                continue
            bucket_idx = train_data[b]['idx']
            sel = bucket_idx[torch.randint(0, train_data[b]['size'], (count,))]
            x_parts.append(x_all[sel])
            t_parts.append(targets_all[sel])
        x_batch = torch.cat(x_parts, dim=0)
        t_batch = torch.cat(t_parts, dim=0)
        perm = torch.randperm(bs)
        x_batch = x_batch[perm]
        t_batch = t_batch[perm]
        return x_batch, t_batch

    def compute_loss(x_batch, t_batch):
        all_logits = model(x_batch, return_all=True)
        mask = x_batch[:, :, 0]
        mask = mask.to(dtype=torch.float32)
        t_batch = t_batch.to(dtype=torch.long)
        loss = 0
        for logits in all_logits:
            per_cell = F.cross_entropy(logits.reshape(-1, 9), t_batch.reshape(-1), reduction='none')
            per_cell = per_cell.view(t_batch.size(0), 81)
            loss = loss + (per_cell * mask).sum() / mask.sum()
        loss = loss / len(all_logits)
        return loss, all_logits, mask, t_batch

    def evaluate_all():
        model.eval()
        results = {}
        total_solved = 0
        total_puzzles = 0
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for name, data in test_data.items():
                x_test = data['x']
                puzzles = data['puzzles']
                solutions = data['solutions']
                puzzles_solved = 0
                for start in range(0, len(puzzles), 256):
                    end = min(start + 256, len(puzzles))
                    batch_x = x_test[start:end]
                    logits = model(batch_x)
                    preds_full = logits.argmax(dim=-1).cpu()
                    for b, (puzzle, solution) in enumerate(zip(puzzles[start:end], solutions[start:end])):
                        pred_solution = list(puzzle)
                        for i in range(81):
                            if puzzle[i] == '.':
                                pred_solution[i] = str(preds_full[b, i].item() + 1)
                        if ''.join(pred_solution) == solution:
                            puzzles_solved += 1
                results[name] = {'solved': puzzles_solved, 'total': len(puzzles)}
                total_solved += puzzles_solved
                total_puzzles += len(puzzles)
        results['_total'] = {'solved': total_solved, 'total': total_puzzles}
        return results

    def probe_long_horizon(n_iters):
        # Runs the uncompiled module (shared weights) in eager mode so the iteration
        # count is a plain Python loop, not something torch.compile specialized on.
        m = getattr(model, '_orig_mod', model)
        m.eval()
        solved = 0
        rope_cos = ROPE_COS.to(device)
        rope_sin = ROPE_SIN.to(device)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for start in range(0, probe_x.size(0), 256):
                batch_x = probe_x[start:start + 256]
                bs = batch_x.size(0)
                h_prev = m.initial_encoder(batch_x)
                preds = torch.zeros(bs, 81, 9, device=device)
                for _ in range(n_iters):
                    h = h_prev + m.pred_proj(preds)
                    for layer in m.layers:
                        h = layer(h, rope_cos, rope_sin)
                    h = m.normalize_outer_state(h)
                    h_prev = h
                    preds = F.softmax(m.output_head(h), dim=-1)
                final_preds = m.output_head(h_prev).argmax(dim=-1).cpu()
                for b, (puzzle, solution) in enumerate(zip(probe_puzzles[start:start + 256], probe_solutions[start:start + 256])):
                    pred_solution = list(puzzle)
                    for i in range(81):
                        if puzzle[i] == '.':
                            pred_solution[i] = str(final_preds[b, i].item() + 1)
                    if ''.join(pred_solution) == solution:
                        solved += 1
        return solved

    def do_save_checkpoint(step):
        path = os.path.join(output_dir, f"{run_checkpoint_prefix}{step}.pt")
        atomic_torch_save({
            'step': step,
            'model_state_dict': {k.replace('_orig_mod.', ''): v for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'config': run_config,
            'probe_history': probe_history,
            'best_probe': best_probe,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
            },
        }, path)
        print(f"Checkpoint saved: {path}")

    def save_model(path):
        state_dict = {
            key.replace('_orig_mod.', ''): value
            for key, value in model.state_dict().items()
        }
        atomic_torch_save(state_dict, path)

    current_phase_name = None
    current_buckets = None

    for step in range(start_step, run_total_steps):
        current_lr = get_lr(step, run_warmup_steps, run_total_steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        buckets, phase_name = get_phase(step)
        if phase_name != current_phase_name:
            current_phase_name = phase_name
            current_buckets = buckets
            total_puzzles = sum(train_data[b]['size'] for b in buckets)
            log(f"\n{'='*60}")
            log(f"Step {step}: Entering {phase_name}")
            log(f"Training pool: {total_puzzles} puzzles")
            log(f"{'='*60}\n")

        model.train()
        x_batch, t_batch = sample_batch(current_buckets, batch_size)
        x_batch = x_batch.to(device)
        t_batch = t_batch.to(device)

        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, all_logits, mask, t_batch = compute_loss(x_batch, t_batch)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == run_total_steps - 1:
            with torch.no_grad():
                final_logits = all_logits[-1]
                preds = final_logits.argmax(dim=-1)
                correct = (preds == t_batch) & (mask > 0)
                train_acc = correct.sum().item() / mask.sum().item()

            do_eval = step % run_eval_every == 0 or step == run_total_steps - 1
            if do_eval:
                results = evaluate_all()
                total_r = results.pop('_total')
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
            else:
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f} Acc: {train_acc:.2%}")

            do_probe = step % run_probe_every == 0 and step > 0
            if do_probe:
                solved_128 = probe_long_horizon(128)
                solved_1024 = probe_long_horizon(1024)
                probe_result = {
                    'step': step,
                    'solved_128': solved_128,
                    'solved_1024': solved_1024,
                    'total': len(probe_puzzles),
                }
                probe_history.append(probe_result)
                log(f"PROBE {step:5d} | 128-iter: {solved_128}/{len(probe_puzzles)} | 1024-iter: {solved_1024}/{len(probe_puzzles)}")
                if solved_1024 > best_probe['solved_1024']:
                    best_probe.clear()
                    best_probe.update(probe_result)
                    save_model(os.path.join(output_dir, best_model_name))
                    log(f"Best 1024-iteration probe so far; saved {best_model_name}")

            if do_eval or (checkpoint_on_probe and do_probe):
                do_save_checkpoint(step)

    log("\n" + "="*60)
    log(f"FINAL RESULTS - {experiment_name}")
    log("="*60)
    results = evaluate_all()
    total_r = results.pop('_total')
    for name, r in results.items():
        log(f"Rating {name:6s}: {r['solved']:5d}/{r['total']:5d} solved ({100*r['solved']/r['total']:5.1f}%)")
    log(f"\nTotal: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
    final_probe_128 = probe_long_horizon(128)
    final_probe_1024 = probe_long_horizon(1024)
    log(f"Final probe | 128-iter: {final_probe_128}/{len(probe_puzzles)} | 1024-iter: {final_probe_1024}/{len(probe_puzzles)}")
    final_probe_result = {
        'step': run_total_steps - 1,
        'solved_128': final_probe_128,
        'solved_1024': final_probe_1024,
        'total': len(probe_puzzles),
        'final': True,
    }
    probe_history.append(final_probe_result)
    if final_probe_1024 > best_probe['solved_1024']:
        best_probe.clear()
        best_probe.update(final_probe_result)
        save_model(os.path.join(output_dir, best_model_name))
        log(f"Final probe is the best 1024-iteration probe; saved {best_model_name}")

    final_path = os.path.join(output_dir, final_model_name)
    save_model(final_path)
    log(f"Final model saved: {final_path}")
    result = {
        'experiment': experiment_name,
        'run_name': run_name,
        'config': run_config,
        'final_16_iteration_solved': total_r['solved'],
        'final_16_iteration_total': total_r['total'],
        'final_probe_128': final_probe_128,
        'final_probe_1024': final_probe_1024,
        'probe_total': len(probe_puzzles),
        'probe_history': probe_history,
        'best_probe': best_probe,
        'final_16_iteration_per_bucket': results,
        'final_model_path': final_path,
    }
    result_path = os.path.join(output_dir, result_name)
    temporary_result_path = result_path + ".tmp"
    with open(temporary_result_path, "w") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)
        result_file.write("\n")
    os.replace(temporary_result_path, result_path)
    log(f"Structured results saved: {result_path}")
    log_file.close()
    return result


if __name__ == "__main__":
    train()
