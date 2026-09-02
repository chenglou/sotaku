"""Ordinary recurrent inference using the exact model's recurrent_step."""

import torch
import torch.nn.functional as F

from stabilize.exp_testbed_20k import ROPE_COS, ROPE_SIN


def validate_iterations(iterations):
    values = tuple(iterations)
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Iteration counts must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("Iteration counts must be unique")
    return tuple(sorted(values))


class RecurrentRunner:
    def __init__(self, model, *, compiled=False, track_solutions=False, chunk_size=16):
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.model = model.eval()
        self.compiled = compiled
        self.track_solutions = track_solutions
        self.chunk_size = chunk_size
        self.chunks = {}

    def _chunk(self, steps):
        if steps not in self.chunks:
            model = self.model
            track = self.track_solutions

            def advance(hidden_state, predictions, rope_cos, rope_sin):
                history = []
                for _ in range(steps):
                    hidden_state = model.recurrent_step(hidden_state, predictions, rope_cos, rope_sin)
                    logits = model.output_head(hidden_state)
                    predictions = F.softmax(logits, dim=-1)
                    if track:
                        history.append(logits)
                return hidden_state, predictions, torch.stack(history) if track else logits

            self.chunks[steps] = torch.compile(advance, fullgraph=True) if self.compiled else advance
        return self.chunks[steps]

    @torch.inference_mode()
    def run_batch(self, inputs, iterations, *, targets=None, empty_mask=None):
        iterations = validate_iterations(iterations)
        if self.track_solutions and (targets is None or empty_mask is None):
            raise ValueError("Solution tracking requires targets and empty-cell masks")
        hidden_state = self.model.initial_encoder(inputs)
        predictions = torch.zeros(inputs.shape[0], 81, 9, device=inputs.device, dtype=torch.float32)
        rope_cos, rope_sin = ROPE_COS.to(inputs.device), ROPE_SIN.to(inputs.device)
        outputs, solution_history, finite_checks = {}, [], []
        step = 0
        for horizon in iterations:
            while step < horizon:
                count = min(self.chunk_size, horizon - step)
                hidden_state, predictions, logits = self._chunk(count)(hidden_state, predictions, rope_cos, rope_sin)
                if self.track_solutions:
                    solution_history.append(((logits.argmax(-1) == targets) | ~empty_mask).all(-1))
                    logits = logits[-1]
                step += count
            outputs[horizon] = logits
            finite_checks.append(torch.isfinite(logits).all() & torch.isfinite(hidden_state).all())
        if not torch.stack(finite_checks).all().item():
            raise ValueError("Inference produced non-finite logits or recurrent states")
        diagnostics = None
        if self.track_solutions:
            solved = torch.cat(solution_history, dim=0)
            ever_solved = solved.any(dim=0)
            first = torch.where(ever_solved, solved.long().argmax(dim=0) + 1, 0)
            regressions = solved[:-1] & ~solved[1:]
            times = torch.arange(2, step + 1, device=inputs.device).unsqueeze(1)
            diagnostics = {
                "first_solved_iteration": first.cpu(),
                "regression_count": regressions.sum(0).cpu(),
                "last_regression_iteration": (regressions * times).amax(0).cpu() if step > 1 else torch.zeros_like(first).cpu(),
                "ever_solved": ever_solved.cpu(),
                "stayed_solved_after_first": (ever_solved & ~regressions.any(0)).cpu(),
            }
        return outputs, diagnostics
