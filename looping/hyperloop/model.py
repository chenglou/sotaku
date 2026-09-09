"""One shared transformer with optional loop-level read/write/retention gates."""

import torch
from torch import nn
from torch.nn import functional as F

from stabilize.exp_testbed_20k import ROPE_COS, ROPE_SIN, SudokuTransformer


class LoopGates(nn.Module):
    def __init__(self, streams, width=128, scale=0.01, retention_bias=-8.0, epsilon=1e-6):
        super().__init__()
        if streams not in (1, 4) or epsilon <= 0:
            raise ValueError("The study supports one or four streams and positive epsilon")
        self.streams = streams
        self.epsilon = epsilon
        self.projection = nn.Linear(streams * width, 3 * streams)
        self.scale = nn.Parameter(torch.full((3,), scale))
        with torch.no_grad():
            self.projection.bias.zero_()
            self.projection.bias[2 * streams:].fill_(retention_bias)

    def forward(self, hidden):
        # Normalize only the gate input; never normalize the carried hidden state.
        with torch.autocast(hidden.device.type, enabled=False):
            flattened = hidden.float().flatten(-2)
            normalized = flattened * torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.epsilon)
            projected = F.linear(normalized, self.projection.weight.float()).unflatten(-1, (3, self.streams))
            values = projected * self.scale.float().unsqueeze(-1) + self.projection.bias.float().view(3, self.streams)
            read = values[..., 0, :].sigmoid() * (2.0 / self.streams)
            write = values[..., 1, :].sigmoid() * 2.0
            retain = values[..., 2, :].sigmoid()
        return tuple(value.to(hidden.dtype) for value in (read, write, retain))


class HyperloopTransformer(SudokuTransformer):
    def __init__(self, streams=0, *, gate_scale=0.01, retention_bias=-8.0, gate_epsilon=1e-6):
        if streams not in (0, 1, 4):
            raise ValueError("Expected baseline (0), one, or four streams")
        super().__init__()
        self.streams = streams
        self.gates = LoopGates(streams, scale=gate_scale, retention_bias=retention_bias,
                               epsilon=gate_epsilon) if streams else None
        self.register_buffer("rope_cos", ROPE_COS.clone(), persistent=False)
        self.register_buffer("rope_sin", ROPE_SIN.clone(), persistent=False)

    def initial_state(self, inputs):
        hidden = self.initial_encoder(inputs)
        if self.streams:
            hidden = hidden.unsqueeze(-2).expand(-1, -1, self.streams, -1).clone()
        return hidden, inputs.new_zeros(inputs.shape[0], 81, 9)

    def step(self, hidden, predictions):
        if self.gates is None:
            hidden = super().recurrent_step(hidden, predictions, self.rope_cos, self.rope_sin)
            readout = hidden
        else:
            read, write, retain = self.gates(hidden)
            combined = (hidden * read.unsqueeze(-1)).sum(-2)
            proposal = super().recurrent_step(combined, predictions, self.rope_cos, self.rope_sin)
            hidden = hidden * retain.unsqueeze(-1) + proposal.unsqueeze(-2) * write.unsqueeze(-1)
            readout = hidden.mean(-2)
        logits = self.output_head(readout)
        return hidden, logits.softmax(-1), logits

    def advance(self, hidden, predictions):
        for _ in range(self.training_iterations):
            hidden, predictions, _ = self.step(hidden, predictions)
        return hidden, predictions

    def forward(self, inputs, targets, initial_state=None):
        hidden, predictions = self.initial_state(inputs) if initial_state is None else initial_state
        mask = inputs[:, :, 0]
        denominator = mask.sum().clamp_min(1)
        losses = []
        for _ in range(self.training_iterations):
            hidden, predictions, logits = self.step(hidden, predictions)
            cell_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="none")
            losses.append((cell_loss.reshape_as(mask) * mask).sum() / denominator)
        return torch.stack(losses).mean(), logits, (hidden, predictions)

    @torch.no_grad()
    def state_diagnostics(self, hidden):
        if not torch.isfinite(hidden).all().item():
            return {"nonfinite": True, "state_rms": None}
        values = hidden.double()
        stats = {"state_rms": float(values.square().mean().sqrt())}
        if self.gates is not None:
            mean = values.mean(-2, keepdim=True)
            stats["stream_relative_spread"] = float((values - mean).square().mean().sqrt()
                                                    / values.square().mean().sqrt().clamp_min(1e-8))
            for name, gate in zip(("read", "write", "retain"), self.gates(hidden)):
                stats[name] = gate.float().mean((0, 1)).tolist() if gate.isfinite().all().item() else None
        return stats
