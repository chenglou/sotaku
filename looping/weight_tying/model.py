"""The v2 recurrence, with independent transformer blocks at selected depths."""

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from stabilize.exp_testbed_20k import RoPETransformerLayer


def rope_tables(width, heads=4):
    if width % heads or (width // heads) % 4:
        raise ValueError("Each attention head must contain a multiple of four features")
    half = width // heads // 2
    frequencies = 1.0 / (10.0 ** (torch.arange(half // 2).float() * 2 / half))
    positions = torch.arange(81)
    angles = torch.cat(((positions // 9)[:, None] * frequencies,
                        (positions % 9)[:, None] * frequencies), dim=-1)
    return angles.cos(), angles.sin()


class StudyTransformer(nn.Module):
    def __init__(self, width=128, feedforward_width=512, period=1, dropout=0.1):
        super().__init__()
        if period not in (1, 16):
            raise ValueError("The preregistered periods are 1 and 16")
        self.width = width
        self.feedforward_width = feedforward_width
        self.period = period
        self.initial_encoder = nn.Linear(10, width)
        self.pred_proj = nn.Linear(9, width)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(width, 4, feedforward_width, dropout=dropout)
            for _ in range(4)
        ])
        self.output_head = nn.Linear(width, 9)
        # Clones start equal but own distinct parameters and optimizer states.
        base_layers = list(self.layers)
        for _ in range(1, period):
            self.layers.extend(deepcopy(layer) for layer in base_layers)
        cosine, sine = rope_tables(width)
        self.register_buffer("rope_cos", cosine, persistent=False)
        self.register_buffer("rope_sin", sine, persistent=False)

    def initial_state(self, inputs):
        return self.initial_encoder(inputs), inputs.new_zeros(inputs.shape[0], 81, 9)

    def step(self, hidden, predictions, iteration, *, repeat=False):
        if self.period > 1 and iteration >= self.period and not repeat:
            raise ValueError("An untied 16-stage stack has no stage 17; repetition must be explicit")
        offset = (iteration % self.period) * 4
        hidden = hidden + self.pred_proj(predictions)
        for index in range(offset, offset + 4):
            hidden = self.layers[index](hidden, self.rope_cos, self.rope_sin)
        logits = self.output_head(hidden)
        return hidden, F.softmax(logits, dim=-1), logits

    def advance(self, hidden, predictions):
        """One complete 16-iteration cycle, also used for gradient-free burn-in."""
        for iteration in range(16):
            hidden, predictions, _ = self.step(hidden, predictions, iteration)
        return hidden, predictions

    def forward(self, inputs, targets, initial_state=None):
        hidden, predictions = self.initial_state(inputs) if initial_state is None else initial_state
        mask = inputs[:, :, 0]
        denominator = mask.sum().clamp_min(1)
        loss = inputs.new_zeros(())
        for iteration in range(16):
            hidden, predictions, logits = self.step(hidden, predictions, iteration)
            cell_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
            loss = loss + (cell_loss.reshape_as(mask) * mask).sum() / denominator
        return loss / 16, logits

    def settings(self):
        return {"width": self.width, "feedforward_width": self.feedforward_width,
                "period": self.period, "dropout": self.layers[0].attn_dropout_p}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def forward_flops_per_iteration(width, feedforward_width, batch_size=1):
    """Linear layers and attention matmuls; excludes nonlinearities and optimizer."""
    tokens = 81
    blocks = 4
    transformer = blocks * (2 * tokens * (4 * width**2 + 2 * width * feedforward_width)
                            + 4 * tokens**2 * width)
    feedback_and_head = 4 * tokens * 9 * width
    return batch_size * (transformer + feedback_and_head)
