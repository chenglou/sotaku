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
from checkpoint_utils import (
    atomic_torch_save,
    find_latest_checkpoint,
    load_branch_checkpoint,
    load_checkpoint,
)
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
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, residual_scale=1.0):
        super().__init__()
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.residual_scale = float(residual_scale)
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
        x = x + self.residual_scale * self.dropout1(self.out_proj(attn_out))

        h2 = self.norm2(x)
        h2 = self.linear2(self.dropout(F.relu(self.linear1(h2))))
        x = x + self.residual_scale * self.dropout2(h2)
        return x


class SudokuTransformer(nn.Module):
    def __init__(
        self,
        outer_state_norm=False,
        outer_state_norm_epsilon=DEFAULT_EPSILON,
        outer_state_rms_cap=None,
        unique_layers=n_layers,
        layer_schedule=None,
        residual_scale=1.0,
        feedback_scale=1.0,
        training_iterations=n_iterations,
    ):
        super().__init__()
        if outer_state_norm and outer_state_rms_cap is not None:
            raise ValueError("outer state normalization and capping are mutually exclusive")
        if outer_state_rms_cap is not None and outer_state_rms_cap <= 0:
            raise ValueError("outer_state_rms_cap must be positive")
        if not isinstance(unique_layers, int) or unique_layers <= 0:
            raise ValueError("unique_layers must be a positive integer")
        if residual_scale <= 0:
            raise ValueError("residual_scale must be positive")
        if feedback_scale < 0:
            raise ValueError("feedback_scale must be non-negative")
        if not isinstance(training_iterations, int) or training_iterations <= 0:
            raise ValueError("training_iterations must be a positive integer")
        if layer_schedule is None:
            layer_schedule = tuple(range(unique_layers))
        else:
            layer_schedule = tuple(layer_schedule)
        if not layer_schedule:
            raise ValueError("layer_schedule must not be empty")
        if any(
            not isinstance(layer_index, int)
            or layer_index < 0
            or layer_index >= unique_layers
            for layer_index in layer_schedule
        ):
            raise ValueError(
                "layer_schedule indices must refer to stored transformer layers"
            )
        self.outer_state_norm = outer_state_norm
        self.outer_state_norm_epsilon = outer_state_norm_epsilon
        self.outer_state_rms_cap = outer_state_rms_cap
        self.unique_layers = unique_layers
        self.layer_schedule = layer_schedule
        self.residual_scale = float(residual_scale)
        self.feedback_scale = float(feedback_scale)
        self.training_iterations = training_iterations
        self.initial_encoder = nn.Linear(10, d_model)
        self.pred_proj = nn.Linear(9, d_model)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(
                d_model,
                n_heads,
                d_ff,
                residual_scale=self.residual_scale,
            )
            for _ in range(unique_layers)
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

    def apply_recurrent_updates(
        self,
        hidden_state,
        predictions,
        rope_cos,
        rope_sin,
    ):
        hidden_state = (
            hidden_state + self.feedback_scale * self.pred_proj(predictions)
        )
        for layer_index in self.layer_schedule:
            hidden_state = self.layers[layer_index](
                hidden_state,
                rope_cos,
                rope_sin,
            )
        return hidden_state

    def recurrent_step(self, hidden_state, predictions, rope_cos, rope_sin):
        hidden_state = self.apply_recurrent_updates(
            hidden_state,
            predictions,
            rope_cos,
            rope_sin,
        )
        return self.normalize_outer_state(hidden_state)

    def forward(
        self,
        x,
        return_all=False,
        initial_state=None,
        return_state=False,
    ):
        batch_size = x.size(0)
        device = x.device
        rope_cos = ROPE_COS.to(device)
        rope_sin = ROPE_SIN.to(device)

        if initial_state is None:
            h_prev = self.initial_encoder(x)
            preds = torch.zeros(batch_size, 81, 9, device=device)
        else:
            h_prev, preds = initial_state

        all_logits = []
        for _ in range(self.training_iterations):
            h = self.recurrent_step(h_prev, preds, rope_cos, rope_sin)
            h_prev = h
            logits = self.output_head(h)
            preds = F.softmax(logits, dim=-1)
            if return_all:
                all_logits.append(logits)
        outputs = all_logits if return_all else logits
        if return_state:
            return outputs, (h_prev, preds)
        return outputs


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


def resolve_late_supervision(horizons, probability, mix):
    horizons = tuple(horizons)
    if any(
        not isinstance(horizon, int) or horizon <= 0 or horizon % n_iterations != 0
        for horizon in horizons
    ):
        raise ValueError(
            f"late-supervision horizons must be positive multiples of {n_iterations}"
        )
    if len(set(horizons)) != len(horizons):
        raise ValueError("late-supervision horizons must be unique")
    if not 0 <= probability <= 1:
        raise ValueError("late-supervision probability must be between 0 and 1")
    if not 0 <= mix <= 1:
        raise ValueError("late-supervision mix must be between 0 and 1")
    enabled_settings = (bool(horizons), probability > 0, mix > 0)
    if any(enabled_settings) and not all(enabled_settings):
        raise ValueError(
            "late-supervision horizons, probability, and mix must be enabled together"
        )
    return horizons


def resolve_late_supervision_timing(
    horizons,
    start_step,
    probability_ramp_steps,
    horizon_start_steps,
):
    if not isinstance(start_step, int) or start_step < 0:
        raise ValueError("late-supervision start step must be a non-negative integer")
    if (
        not isinstance(probability_ramp_steps, int)
        or probability_ramp_steps < 0
    ):
        raise ValueError(
            "late-supervision probability ramp must be a non-negative integer"
        )

    horizon_start_steps = tuple(horizon_start_steps)
    if not horizons:
        if start_step != 0:
            raise ValueError("late-supervision start step requires late supervision")
        if probability_ramp_steps != 0:
            raise ValueError(
                "late-supervision probability ramp requires late supervision"
            )
        if horizon_start_steps:
            raise ValueError(
                "late-supervision horizon start steps require late supervision"
            )
        return ()

    if not horizon_start_steps:
        horizon_start_steps = (start_step,) * len(horizons)
    if len(horizon_start_steps) != len(horizons):
        raise ValueError(
            "late-supervision horizons and horizon start steps must have equal length"
        )
    if any(
        not isinstance(horizon_step, int) or horizon_step < start_step
        for horizon_step in horizon_start_steps
    ):
        raise ValueError(
            "late-supervision horizon start steps must be integers at or after "
            "the late-supervision start step"
        )
    if tuple(sorted(horizon_start_steps)) != horizon_start_steps:
        raise ValueError(
            "late-supervision horizon start steps must be non-decreasing"
        )
    return horizon_start_steps


def late_supervision_plan_for_step(
    step,
    horizons,
    horizon_start_steps,
    target_probability,
    start_step,
    probability_ramp_steps,
):
    available_horizons = tuple(
        horizon
        for horizon, horizon_step in zip(horizons, horizon_start_steps)
        if step >= horizon_step
    )
    if not available_horizons:
        return (), 0.0
    if probability_ramp_steps == 0:
        return available_horizons, target_probability
    ramp_progress = min(
        1.0,
        max(0.0, (step - start_step) / probability_ramp_steps),
    )
    return available_horizons, target_probability * ramp_progress


def correct_prediction_consistency_loss(
    anchor_logits,
    future_logits,
    targets,
    mask,
):
    anchor_probabilities = F.softmax(anchor_logits.detach(), dim=-1)
    anchor_correct = anchor_logits.detach().argmax(dim=-1).eq(targets)
    eligible = anchor_correct & mask.bool()
    per_cell_kl = F.kl_div(
        F.log_softmax(future_logits, dim=-1),
        anchor_probabilities,
        reduction="none",
    ).sum(dim=-1)
    return (per_cell_kl * eligible).sum() / eligible.sum().clamp_min(1)


def solved_puzzle_margin_floor_loss(
    anchor_logits,
    future_logits,
    targets,
    mask,
    margin_floor,
):
    empty_mask = mask.bool()
    anchor_predictions = anchor_logits.detach().argmax(dim=-1)
    anchor_solved = (
        (anchor_predictions == targets) | ~empty_mask
    ).all(dim=1)

    target_logits = future_logits.gather(
        -1,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    wrong_logits = future_logits.masked_fill(
        F.one_hot(targets, num_classes=9).bool(),
        -torch.inf,
    )
    margins = target_logits - wrong_logits.max(dim=-1).values
    minimum_margin = margins.masked_fill(
        ~empty_mask,
        torch.inf,
    ).min(dim=1).values
    per_puzzle_loss = F.relu(margin_floor - minimum_margin)
    return (
        per_puzzle_loss * anchor_solved
    ).sum() / anchor_solved.sum().clamp_min(1)


def resolve_late_recheck(
    gaps,
    loss_weight,
    consistency_weight,
    late_supervision_enabled,
    margin_floor_weight=0.0,
    margin_floor=1.0,
):
    gaps = tuple(gaps)
    if any(
        not isinstance(gap, int) or gap <= 0 or gap % n_iterations != 0
        for gap in gaps
    ):
        raise ValueError(
            f"late recheck gaps must be positive multiples of {n_iterations}"
        )
    if not 0 <= loss_weight <= 1:
        raise ValueError("late recheck loss weight must be between 0 and 1")
    if consistency_weight < 0:
        raise ValueError("late consistency weight must be non-negative")
    if margin_floor_weight < 0:
        raise ValueError("late margin-floor weight must be non-negative")
    if margin_floor <= 0:
        raise ValueError("late margin floor must be positive")
    has_recheck_objective = (
        loss_weight > 0
        or consistency_weight > 0
        or margin_floor_weight > 0
    )
    if bool(gaps) != has_recheck_objective:
        raise ValueError(
            "late recheck gaps and a positive recheck objective must be "
            "enabled together"
        )
    if gaps and not late_supervision_enabled:
        raise ValueError("late rechecks require late supervision")
    if consistency_weight > 0 and not gaps:
        raise ValueError("late consistency requires late rechecks")
    if margin_floor_weight > 0 and not gaps:
        raise ValueError("late margin floor requires late rechecks")
    return gaps


def train(
    output_dir=".",
    *,
    experiment_name=CONFIG['experiment'],
    run_name=None,
    outer_state_norm=False,
    outer_state_norm_epsilon=DEFAULT_EPSILON,
    outer_state_rms_cap=None,
    unique_layers=n_layers,
    layer_schedule=None,
    residual_scale=1.0,
    feedback_scale=1.0,
    training_iterations=n_iterations,
    run_batch_size=batch_size,
    microbatch_size=None,
    random_seed=None,
    checkpoint_on_probe=False,
    schedule=None,
    late_supervision_horizons=(),
    late_supervision_probability=0.0,
    late_supervision_mix=0.0,
    late_supervision_start_step=0,
    late_supervision_probability_ramp_steps=0,
    late_supervision_horizon_start_steps=(),
    late_recheck_gaps=(),
    late_recheck_loss_weight=0.0,
    late_consistency_weight=0.0,
    late_margin_floor_weight=0.0,
    late_margin_floor=1.0,
    late_auxiliary_only=False,
    branch_checkpoint_path=None,
):
    if run_name is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")
    if branch_checkpoint_path is not None and run_name is None:
        raise ValueError("branch_checkpoint_path requires a run_name")
    if outer_state_norm_epsilon <= 0:
        raise ValueError("outer_state_norm_epsilon must be positive")
    if outer_state_norm and outer_state_rms_cap is not None:
        raise ValueError("outer state normalization and capping are mutually exclusive")
    if outer_state_rms_cap is not None and outer_state_rms_cap <= 0:
        raise ValueError("outer_state_rms_cap must be positive")
    if not isinstance(run_batch_size, int) or run_batch_size <= 0:
        raise ValueError("run_batch_size must be a positive integer")
    if microbatch_size is None:
        microbatch_size = run_batch_size
    if not isinstance(microbatch_size, int) or not 0 < microbatch_size <= run_batch_size:
        raise ValueError(
            "microbatch_size must be a positive integer no larger than run_batch_size"
        )

    late_supervision_horizons = resolve_late_supervision(
        late_supervision_horizons,
        late_supervision_probability,
        late_supervision_mix,
    )
    if microbatch_size < run_batch_size and late_supervision_horizons:
        raise ValueError(
            "gradient accumulation is not implemented for late supervision"
        )
    late_supervision_horizon_start_steps = resolve_late_supervision_timing(
        late_supervision_horizons,
        late_supervision_start_step,
        late_supervision_probability_ramp_steps,
        late_supervision_horizon_start_steps,
    )
    late_recheck_gaps = resolve_late_recheck(
        late_recheck_gaps,
        late_recheck_loss_weight,
        late_consistency_weight,
        bool(late_supervision_horizons),
        late_margin_floor_weight,
        late_margin_floor,
    )
    if late_auxiliary_only:
        if not late_recheck_gaps:
            raise ValueError(
                "late auxiliary-only training requires a recheck objective"
            )
        if late_recheck_loss_weight > 0:
            raise ValueError(
                "late auxiliary-only training cannot use recheck "
                "cross-entropy"
            )

    resolved_layer_schedule = (
        tuple(range(unique_layers))
        if layer_schedule is None
        else tuple(layer_schedule)
    )
    model_settings = {
        'unique_layers': unique_layers,
        'layer_schedule': resolved_layer_schedule,
        'residual_scale': residual_scale,
        'feedback_scale': feedback_scale,
        'training_iterations': training_iterations,
    }
    nondefault_model_settings = (
        unique_layers != n_layers
        or resolved_layer_schedule != tuple(range(n_layers))
        or residual_scale != 1.0
        or feedback_scale != 1.0
        or training_iterations != n_iterations
    )

    run_schedule = resolve_schedule(schedule)
    run_warmup_steps = run_schedule['warmup_steps']
    run_total_steps = run_schedule['total_steps']
    run_eval_every = run_schedule['eval_every']
    run_probe_every = run_schedule['probe_every']
    run_phases = run_schedule['phases']
    if (
        late_supervision_horizons
        and late_supervision_start_step >= run_total_steps
    ):
        raise ValueError("late-supervision start step must precede total_steps")
    if (
        late_supervision_horizon_start_steps
        and late_supervision_horizon_start_steps[-1] >= run_total_steps
    ):
        raise ValueError(
            "every late-supervision horizon must become active before total_steps"
        )

    run_config = dict(CONFIG)
    run_config['experiment'] = experiment_name
    run_config['batch_size'] = run_batch_size
    if microbatch_size < run_batch_size:
        run_config['microbatch_size'] = microbatch_size
    if nondefault_model_settings:
        run_config.update(model_settings)
    if schedule is not None:
        run_config.update(run_schedule)
    if late_supervision_horizons:
        run_config.update({
            'late_supervision_horizons': late_supervision_horizons,
            'late_supervision_probability': late_supervision_probability,
            'late_supervision_mix': late_supervision_mix,
            'late_supervision_start_step': late_supervision_start_step,
        })
        if late_supervision_probability_ramp_steps:
            run_config['late_supervision_probability_ramp_steps'] = (
                late_supervision_probability_ramp_steps
            )
        if any(
            horizon_step != late_supervision_start_step
            for horizon_step in late_supervision_horizon_start_steps
        ):
            run_config['late_supervision_horizon_start_steps'] = (
                late_supervision_horizon_start_steps
            )
        if late_recheck_gaps:
            run_config.update({
                'late_recheck_gaps': late_recheck_gaps,
                'late_recheck_loss_weight': late_recheck_loss_weight,
                'late_consistency_weight': late_consistency_weight,
            })
            if late_auxiliary_only:
                run_config['late_auxiliary_only'] = True
            if late_margin_floor_weight > 0:
                run_config.update({
                    'late_margin_floor_weight': late_margin_floor_weight,
                    'late_margin_floor': late_margin_floor,
                })
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
    if branch_checkpoint_path is not None:
        run_config['branch_source_checkpoint'] = os.path.basename(
            os.path.abspath(branch_checkpoint_path)
        )

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
        **model_settings,
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
    elif branch_checkpoint_path is not None:
        branch_source = os.path.abspath(branch_checkpoint_path)
        if not os.path.isfile(branch_source):
            raise FileNotFoundError(
                f"branch checkpoint does not exist: {branch_source}"
            )
        allowed_config_changes = {
            'branch_source_checkpoint',
            'experiment',
            'late_consistency_weight',
            'late_recheck_gaps',
            'late_recheck_loss_weight',
            'late_margin_floor',
            'late_margin_floor_weight',
            'late_auxiliary_only',
            'run_name',
        }
        print(f"Branching from checkpoint: {branch_source}")
        checkpoint_data = load_branch_checkpoint(
            branch_source,
            model,
            run_config,
            allowed_config_changes,
        )
        start_step = int(checkpoint_data['step']) + 1
        checkpoint_data = dict(checkpoint_data)
        checkpoint_data.update({
            'probe_history': [],
            'best_probe': {'step': -1, 'solved_1024': -1},
            'late_horizon_counts': {},
            'recheck_gap_counts': {},
        })
        print(f"Loaded branch source through step {start_step - 1}")

    model = torch.compile(model)

    original_model = getattr(model, '_orig_mod', model)
    if late_supervision_horizons:
        def _burnin_chunk(hidden_state, predictions):
            rope_cos = ROPE_COS.to(hidden_state.device)
            rope_sin = ROPE_SIN.to(hidden_state.device)
            for _ in range(n_iterations):
                hidden_state = original_model.recurrent_step(
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                )
                predictions = F.softmax(
                    original_model.output_head(hidden_state),
                    dim=-1,
                )
            return hidden_state, predictions

        burnin_chunk = torch.compile(_burnin_chunk)

        def detached_advance(initial_state, horizon):
            hidden_state, predictions = initial_state
            for _ in range(horizon // n_iterations):
                hidden_state, predictions = burnin_chunk(
                    hidden_state,
                    predictions,
                )
            return hidden_state.detach(), predictions.detach()

        def detached_burnin(x_batch, horizon):
            initial_state = (
                original_model.initial_encoder(x_batch),
                torch.zeros(
                    x_batch.size(0),
                    81,
                    9,
                    device=x_batch.device,
                ),
            )
            return detached_advance(initial_state, horizon)
    else:
        detached_burnin = None
        detached_advance = None

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
    schedule_text = "".join(chr(ord('A') + index) for index in resolved_layer_schedule)
    print(
        f"Architecture: d_model={d_model}, d_ff={d_ff}, "
        f"unique_layers={unique_layers}, layer_schedule={schedule_text}"
    )
    print(f"Training iterations: {training_iterations}")
    print(
        f"Residual scale: {residual_scale:g}, "
        f"prediction-feedback scale: {feedback_scale:g}"
    )
    print(
        f"Batch size: {run_batch_size}, lr: {lr}, "
        f"warmup_steps: {run_warmup_steps}"
    )
    if microbatch_size < run_batch_size:
        print(f"Gradient accumulation microbatch size: {microbatch_size}")
    print(f"Total steps: {run_total_steps}")
    if outer_state_norm:
        outer_state_constraint = "RMSNorm without affine"
    elif outer_state_rms_cap is not None:
        outer_state_constraint = f"per-token RMS cap at {outer_state_rms_cap:g}"
    else:
        outer_state_constraint = "none"
    print(f"Outer state constraint: {outer_state_constraint}")
    print(f"Random seed: {random_seed}")
    if late_supervision_horizons:
        print(
            "Detached late-state supervision: "
            f"horizons={late_supervision_horizons}, "
            f"probability={late_supervision_probability:g}, "
            f"late-loss mix={late_supervision_mix:g}, "
            f"start step={late_supervision_start_step}"
        )
        if late_supervision_probability_ramp_steps:
            print(
                "Late-state probability ramp: "
                f"{late_supervision_probability_ramp_steps} steps"
            )
        if any(
            horizon_step != late_supervision_start_step
            for horizon_step in late_supervision_horizon_start_steps
        ):
            print(
                "Late-state horizon start steps: "
                f"{late_supervision_horizon_start_steps}"
            )
        if late_recheck_gaps:
            print(
                "Late-state recheck: "
                f"gaps={late_recheck_gaps}, "
                f"loss weight={late_recheck_loss_weight:g}, "
                f"consistency weight={late_consistency_weight:g}, "
                f"margin-floor weight={late_margin_floor_weight:g}, "
                f"margin floor={late_margin_floor:g}"
            )
            if late_auxiliary_only:
                print(
                    "Late-state objective mode: auxiliary only; ordinary "
                    "cross-entropy remains active on every batch"
                )
    else:
        print("Detached late-state supervision: none")
    print(f"Output directory: {output_dir}")

    log_path = os.path.join(output_dir, run_log_name)
    log_file = open(log_path, "a")

    probe_history = list(checkpoint_data.get('probe_history', [])) if checkpoint_data else []
    best_probe = dict(checkpoint_data.get('best_probe', {'step': -1, 'solved_1024': -1})) if checkpoint_data else {'step': -1, 'solved_1024': -1}
    late_horizon_counts = dict(
        checkpoint_data.get('late_horizon_counts', {})
        if checkpoint_data
        else {}
    )
    recheck_gap_counts = dict(
        checkpoint_data.get('recheck_gap_counts', {})
        if checkpoint_data
        else {}
    )

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

    def compute_loss(
        x_batch,
        t_batch,
        initial_state=None,
        return_state=False,
    ):
        model_outputs = model(
            x_batch,
            return_all=True,
            initial_state=initial_state,
            return_state=return_state,
        )
        if return_state:
            all_logits, final_state = model_outputs
        else:
            all_logits = model_outputs
        mask = x_batch[:, :, 0]
        mask = mask.to(dtype=torch.float32)
        t_batch = t_batch.to(dtype=torch.long)
        loss = 0
        for logits in all_logits:
            per_cell = F.cross_entropy(logits.reshape(-1, 9), t_batch.reshape(-1), reduction='none')
            per_cell = per_cell.view(t_batch.size(0), 81)
            loss = loss + (per_cell * mask).sum() / mask.sum()
        loss = loss / len(all_logits)
        outputs = (loss, all_logits, mask, t_batch)
        if return_state:
            return outputs + (final_state,)
        return outputs

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
                    h = m.recurrent_step(h_prev, preds, rope_cos, rope_sin)
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
            'late_horizon_counts': late_horizon_counts,
            'recheck_gap_counts': recheck_gap_counts,
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
        x_batch, t_batch = sample_batch(current_buckets, run_batch_size)
        x_batch = x_batch.to(device)
        t_batch = t_batch.to(device)

        (
            available_late_horizons,
            current_late_probability,
        ) = late_supervision_plan_for_step(
            step,
            late_supervision_horizons,
            late_supervision_horizon_start_steps,
            late_supervision_probability,
            late_supervision_start_step,
            late_supervision_probability_ramp_steps,
        )
        use_late_supervision = (
            bool(available_late_horizons)
            and random.random() < current_late_probability
        )
        late_horizon = (
            random.choice(available_late_horizons)
            if use_late_supervision
            else None
        )
        if late_horizon is not None:
            horizon_key = str(late_horizon)
            late_horizon_counts[horizon_key] = (
                late_horizon_counts.get(horizon_key, 0) + 1
            )
        late_loss = None
        recheck_gap = None
        recheck_loss = None
        consistency_loss = None
        margin_floor_loss = None

        optimizer.zero_grad()
        if microbatch_size < run_batch_size:
            total_mask = x_batch[:, :, 0].float().sum()
            base_loss = torch.zeros((), device=device)
            train_correct = 0
            train_mask_count = 0
            for microbatch_start in range(0, run_batch_size, microbatch_size):
                microbatch_end = min(
                    microbatch_start + microbatch_size,
                    run_batch_size,
                )
                microbatch_x = x_batch[microbatch_start:microbatch_end]
                microbatch_targets = t_batch[microbatch_start:microbatch_end]
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    (
                        microbatch_loss,
                        microbatch_logits,
                        microbatch_mask,
                        microbatch_targets,
                    ) = compute_loss(microbatch_x, microbatch_targets)
                microbatch_weight = microbatch_mask.sum() / total_mask
                (microbatch_weight * microbatch_loss).backward()
                base_loss = (
                    base_loss
                    + microbatch_weight.detach() * microbatch_loss.detach()
                )
                with torch.no_grad():
                    microbatch_predictions = microbatch_logits[-1].argmax(dim=-1)
                    train_correct += (
                        (microbatch_predictions == microbatch_targets)
                        & (microbatch_mask > 0)
                    ).sum().item()
                    train_mask_count += microbatch_mask.sum().item()
            loss = base_loss
            train_acc = train_correct / train_mask_count
        else:
            base_weight = (
                1
                if late_auxiliary_only
                else (
                    1 - late_supervision_mix
                    if use_late_supervision
                    else 1
                )
            )
            if base_weight > 0:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    base_loss, all_logits, mask, t_batch = compute_loss(
                        x_batch,
                        t_batch,
                    )
                (base_weight * base_loss).backward()
            else:
                base_loss = None
                all_logits = None
                mask = None

            if use_late_supervision:
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    initial_state = detached_burnin(x_batch, late_horizon)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    if late_recheck_gaps:
                        (
                            late_loss,
                            late_logits,
                            late_mask,
                            t_batch,
                            late_final_state,
                        ) = compute_loss(
                            x_batch,
                            t_batch,
                            initial_state=initial_state,
                            return_state=True,
                        )
                    else:
                        (
                            late_loss,
                            late_logits,
                            late_mask,
                            t_batch,
                        ) = compute_loss(
                            x_batch,
                            t_batch,
                            initial_state=initial_state,
                        )

                late_primary_weight = (
                    0
                    if late_auxiliary_only
                    else (
                        1 - late_recheck_loss_weight
                        if late_recheck_gaps
                        else 1
                    )
                )
                if late_primary_weight > 0:
                    (
                        late_supervision_mix
                        * late_primary_weight
                        * late_loss
                    ).backward()
                late_objective = late_primary_weight * late_loss.detach()

                if late_recheck_gaps:
                    recheck_gap = random.choice(late_recheck_gaps)
                    gap_key = str(recheck_gap)
                    recheck_gap_counts[gap_key] = (
                        recheck_gap_counts.get(gap_key, 0) + 1
                    )
                    with torch.no_grad():
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            recheck_initial_state = detached_advance(
                                late_final_state,
                                recheck_gap,
                            )
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        (
                            recheck_loss,
                            recheck_logits,
                            recheck_mask,
                            t_batch,
                        ) = compute_loss(
                            x_batch,
                            t_batch,
                            initial_state=recheck_initial_state,
                        )
                        if late_consistency_weight > 0:
                            consistency_loss = sum(
                                correct_prediction_consistency_loss(
                                    late_logits[-1],
                                    future_logits,
                                    t_batch,
                                    recheck_mask,
                                )
                                for future_logits in recheck_logits
                            ) / len(recheck_logits)
                        else:
                            consistency_loss = recheck_loss.new_zeros(())
                        if late_margin_floor_weight > 0:
                            margin_floor_loss = sum(
                                solved_puzzle_margin_floor_loss(
                                    late_logits[-1],
                                    future_logits,
                                    t_batch,
                                    recheck_mask,
                                    late_margin_floor,
                                )
                                for future_logits in recheck_logits
                            ) / len(recheck_logits)
                        else:
                            margin_floor_loss = recheck_loss.new_zeros(())
                        recheck_objective = (
                            late_recheck_loss_weight * recheck_loss
                            + late_consistency_weight * consistency_loss
                            + late_margin_floor_weight * margin_floor_loss
                        )
                    (
                        late_supervision_mix * recheck_objective
                    ).backward()
                    late_objective = (
                        late_objective + recheck_objective.detach()
                    )

                loss = late_supervision_mix * late_objective
                if base_loss is not None:
                    loss = loss + base_weight * base_loss.detach()
                else:
                    if recheck_loss is not None:
                        all_logits = recheck_logits
                        mask = recheck_mask
                    else:
                        all_logits = late_logits
                        mask = late_mask
            else:
                loss = base_loss.detach()
            with torch.no_grad():
                final_logits = all_logits[-1]
                preds = final_logits.argmax(dim=-1)
                correct = (preds == t_batch) & (mask > 0)
                train_acc = correct.sum().item() / mask.sum().item()
        optimizer.step()

        if step % 100 == 0 or step == run_total_steps - 1:
            if late_loss is None:
                late_text = ""
            elif late_auxiliary_only:
                late_text = (
                    f" Anchor@{late_horizon}: {late_loss.item():.4f}"
                )
            else:
                late_text = (
                    f" Late@{late_horizon}: {late_loss.item():.4f}"
                )
            if recheck_loss is not None:
                late_text += (
                    f" Recheck+{recheck_gap}: {recheck_loss.item():.4f}"
                    f" Stable: {consistency_loss.item():.4f}"
                )
                if late_margin_floor_weight > 0:
                    late_text += (
                        f" Margin: {margin_floor_loss.item():.4f}"
                    )
            do_eval = step % run_eval_every == 0 or step == run_total_steps - 1
            if do_eval:
                results = evaluate_all()
                total_r = results.pop('_total')
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f}{late_text} Acc: {train_acc:.2%} | " +
                    " | ".join([f"{name}: {r['solved']}/{r['total']}" for name, r in results.items()]) +
                    f" | Total: {total_r['solved']}/{total_r['total']} ({100*total_r['solved']/total_r['total']:.1f}%)")
            else:
                log(f"Step {step:5d} | LR: {current_lr:.2e} | Loss: {loss.item():.4f}{late_text} Acc: {train_acc:.2%}")

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
    if late_horizon_counts:
        log(f"Late-state batch counts by horizon: {late_horizon_counts}")
    if recheck_gap_counts:
        log(f"Recheck batch counts by gap: {recheck_gap_counts}")
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
        'final_training_iteration_count': training_iterations,
        'final_training_iteration_solved': total_r['solved'],
        'final_training_iteration_total': total_r['total'],
        'final_probe_128': final_probe_128,
        'final_probe_1024': final_probe_1024,
        'probe_total': len(probe_puzzles),
        'probe_history': probe_history,
        'best_probe': best_probe,
        'late_horizon_counts': late_horizon_counts,
        'recheck_gap_counts': recheck_gap_counts,
        'final_training_iteration_per_bucket': results,
        'final_model_path': final_path,
    }
    if training_iterations == n_iterations:
        result.update({
            'final_16_iteration_solved': total_r['solved'],
            'final_16_iteration_total': total_r['total'],
            'final_16_iteration_per_bucket': results,
        })
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
