"""Score candidate windows without labels and replay one with gradients."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from looping.weight_tying.train import rng_state
from stabilize.exp_testbed_20k import SudokuTransformer


class WindowTransformer(SudokuTransformer):
    def initial_state(self, inputs):
        return self.initial_encoder(inputs), inputs.new_zeros(inputs.shape[0], 81, 9)

    def scan_window(self, hidden, predictions):
        placeholder = hidden.new_empty(hidden.shape[0], 81, 10)
        logits, state = super().forward(placeholder, return_all=True,
                                         initial_state=(hidden, predictions), return_state=True)
        return state, torch.stack(logits)

    def forward(self, inputs, targets, initial_state=None):
        logits, state = super().forward(inputs, return_all=True,
                                       initial_state=initial_state, return_state=True)
        mask = inputs[:, :, 0]
        denominator = mask.sum().clamp_min(1)
        losses = [((F.cross_entropy(output.flatten(0, 1), targets.flatten(), reduction="none")
                    .reshape_as(mask) * mask).sum() / denominator) for output in logits]
        return torch.stack(losses).mean(), state, torch.stack(logits)


def confidence_score(logits, empty_mask):
    probabilities = logits.float().softmax(-1).amax(-1)
    weights = empty_mask.to(probabilities.dtype)
    return (probabilities * weights).sum() / (weights.sum().clamp_min(1) * logits.shape[0])


def select_index(scores, selector, random_index):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not len(scores):
        raise ValueError("Window scores must be a nonempty vector")
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite window confidence")
    if not 0 <= random_index < len(scores):
        raise ValueError("Random candidate index is out of range")
    if selector == "random":
        return int(random_index)
    if selector == "confidence":
        return int(scores.argmax())
    if selector == "latest":
        return len(scores) - 1
    raise ValueError("Unknown window selector")


@dataclass
class Scan:
    starts: tuple
    states: list
    rng_states: list
    logits: list
    scores: list
    end_rng: dict


@torch.no_grad()
def scan_candidates(model, inputs, advance, starts):
    starts = tuple(starts)
    length = model.training_iterations
    if not starts or starts != tuple(sorted(set(starts))) or any(start <= 0 or start % length for start in starts):
        raise ValueError("Candidate starts must be increasing positive multiples of window length")
    state = model.initial_state(inputs)
    states, random_states, outputs, scores = [], [], [], []
    empty = inputs[:, :, 0].bool()
    for start in range(0, starts[-1] + length, length):
        if start in starts:
            states.append(tuple(value.detach() for value in state))
            random_states.append(rng_state())
        state, logits = advance(*state)
        if start in starts:
            outputs.append(logits.detach())
            scores.append(float(confidence_score(logits, empty)))
    return Scan(starts, states, random_states, outputs, scores, rng_state())


@torch.no_grad()
def window_diagnostics(logits, empty_mask, targets):
    probabilities = logits.float().softmax(-1)
    confidence, predictions = probabilities.max(-1)
    correct = predictions == targets
    mask = empty_mask.expand_as(correct)
    count = int(mask.sum())
    return {
        "cell_accuracy": float((correct & mask).sum() / max(count, 1)),
        "confident_wrong_fraction": float(((confidence >= 0.9) & ~correct & mask).sum() / max(count, 1)),
        "solved_fraction": float((correct | ~empty_mask).all(-1).float().mean()),
    }
