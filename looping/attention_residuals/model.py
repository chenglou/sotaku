"""Sotaku variants adapted from Inkling 2 Mini.

Two changes, alone and together:

- `attnres`: attention residuals inside an iteration. A sublayer's input is a softmax mix of
  unit-RMS sources (the iteration's input and the block sums so far) instead of their plain sum.
  The state carried between iterations is still one accumulating stream, as in the baseline.
- `message`: no stream is carried between iterations. Every iteration starts again from the
  puzzle embedding, and each sublayer mixes in a gated message read from the previous
  iteration's two block sums. The gates start closed.
- `attnres_message`: both, which is the closest match to Inkling 2 Mini's looped blocks.

The four transformer layers, prediction feedback and output head are the baseline's, and the
baseline's weights start from the same values for the same random seed.
"""

import torch
from torch import nn
from torch.nn import functional as F

from stabilize.exp_testbed_20k import SudokuTransformer, apply_rope, d_model

ARMS = ("attnres", "message", "attnres_message")
BLOCKS = ((0, 1), (2, 3))  # two blocks of two layers, like Inkling's two looped blocks
EPSILON = 1e-6


def unit_rms(values):
    """Scale each cell's vector to a root mean square of one. A zero vector stays zero."""
    scale = torch.rsqrt(values.square().mean(-1, keepdim=True) + EPSILON)
    return values * scale.to(values.dtype)


class SoftmaxRead(nn.Module):
    """Mix unit-RMS sources with softmax weights computed from one learned query vector.

    The query starts at zero, so the first mix is a plain average of the sources.
    """

    def __init__(self, width):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(width))
        self.temperature = width ** 0.5

    def forward(self, sources):
        if len(sources) == 1:
            return sources[0]
        scores = torch.stack([source @ self.query for source in sources], -1) / self.temperature
        weights = scores.softmax(-1).to(sources[0].dtype)
        mixed = sources[0] * weights[..., 0:1]
        for index in range(1, len(sources)):
            mixed = mixed + sources[index] * weights[..., index:index + 1]
        return mixed


class MessageGate(nn.Module):
    """Mix a message into a sublayer input: `keep * input + gain * message`.

    `keep = 1 - tanh(F z)` and `gain = tanh(G z)`, where `z` is a small projection of the
    unit-RMS input and message. F and G start at zero, so the message is blocked at first.
    """

    def __init__(self, width, bottleneck):
        super().__init__()
        self.from_input = nn.Linear(width, bottleneck, bias=False)
        self.from_message = nn.Linear(width, bottleneck, bias=False)
        self.keep = nn.Linear(bottleneck, width, bias=False)
        self.gain = nn.Linear(bottleneck, width, bias=False)
        nn.init.zeros_(self.keep.weight)
        nn.init.zeros_(self.gain.weight)

    def forward(self, current, current_unit, message):
        reduced = self.from_input(current_unit) + self.from_message(unit_rms(message))
        # Written as `current - tanh * current` so that `keep` is not stored as a separate tensor.
        return (current - torch.tanh(self.keep(reduced)) * current
                + torch.tanh(self.gain(reduced)) * message)


class PackedEncoder(nn.Module):
    """Initial state for the message arms: [embedding, iteration input, block sum 1, block sum 2]."""

    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, inputs):
        embedding = self.linear(inputs)
        empty = embedding.new_zeros(*embedding.shape[:-1], 3 * embedding.shape[-1])
        return torch.cat([embedding, empty], -1)


class PackedHead(nn.Module):
    """Read logits from the packed state of the message arms."""

    def __init__(self, linear, read):
        super().__init__()
        self.linear = linear
        self.read = read
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, state):
        width = self.in_features
        parts = [state[..., width * index:width * (index + 1)] for index in (1, 2, 3)]
        if self.read is None:
            return self.linear(parts[0] + parts[1] + parts[2])
        return self.linear(unit_rms(self.read([unit_rms(part) for part in parts])))


class LoopVariantTransformer(SudokuTransformer):
    def __init__(self, arm, *, bottleneck=8, separate_qkv=False, **settings):
        super().__init__(**settings)
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; choose one of {ARMS}")
        if (self.layer_schedule != (0, 1, 2, 3) or self.outer_state_norm
                or self.outer_state_rms_cap is not None or self.residual_scale != 1.0):
            raise ValueError("The variants support only the plain four-layer baseline settings")
        self.arm = arm
        self.attention_residuals = arm != "message"
        self.message = arm != "attnres"
        self.separate_qkv = separate_qkv
        self.depth_reads = None
        if self.attention_residuals:
            self.depth_reads = nn.ModuleList(SoftmaxRead(d_model) for _ in range(2 * len(self.layers)))
        if self.message:
            # Per layer: one message shared by the query, key and value inputs, and one for the
            # feed-forward input. `separate_qkv` gives each of the three its own message, as in
            # Inkling, but needs more than 131 GiB of GPU memory at batch size 2048.
            self.inputs_per_layer = 4 if separate_qkv else 2
            count = self.inputs_per_layer * len(self.layers)
            self.message_reads = nn.ModuleList(SoftmaxRead(d_model) for _ in range(count))
            self.message_gates = nn.ModuleList(MessageGate(d_model, bottleneck) for _ in range(count))
            self.initial_encoder = PackedEncoder(self.initial_encoder)
            self.output_head = PackedHead(
                self.output_head, SoftmaxRead(d_model) if self.attention_residuals else None)

    def mix_message(self, layer_index, slot, current, current_unit, previous_units):
        index = layer_index * self.inputs_per_layer + slot
        message = self.message_reads[index](previous_units)
        return self.message_gates[index](current, current_unit, message)

    def attention_output(self, layer, layer_index, current, previous_units, rope_cos, rope_sin):
        if previous_units is None:
            query_input = key_input = value_input = layer.norm1(current)
        elif self.separate_qkv:
            current_unit = unit_rms(current)
            query_input, key_input, value_input = (
                layer.norm1(self.mix_message(layer_index, slot, current, current_unit, previous_units))
                for slot in range(3))
        else:
            mixed = self.mix_message(layer_index, 0, current, unit_rms(current), previous_units)
            query_input = key_input = value_input = layer.norm1(mixed)
        batch, cells, width = current.shape

        def heads(values):
            return values.view(batch, cells, layer.n_heads, layer.head_dim).transpose(1, 2)

        queries = apply_rope(heads(layer.q_proj(query_input)), rope_cos, rope_sin)
        keys = apply_rope(heads(layer.k_proj(key_input)), rope_cos, rope_sin)
        attended = F.scaled_dot_product_attention(
            queries, keys, heads(layer.v_proj(value_input)),
            dropout_p=layer.attn_dropout_p if self.training else 0.0)
        attended = attended.transpose(1, 2).contiguous().view(batch, cells, width)
        return layer.dropout1(layer.out_proj(attended))

    def feedforward_output(self, layer, layer_index, current, previous_units):
        if previous_units is not None:
            current = self.mix_message(
                layer_index, self.inputs_per_layer - 1, current, unit_rms(current), previous_units)
        hidden = layer.dropout(F.relu(layer.linear1(layer.norm2(current))))
        return layer.dropout2(layer.linear2(hidden))

    def recurrent_step(self, state, predictions, rope_cos, rope_sin):
        width = d_model
        feedback = self.feedback_scale * self.pred_proj(predictions)
        if self.message:
            embedding = state[..., :width]
            previous_units = [unit_rms(state[..., 2 * width:3 * width]), unit_rms(state[..., 3 * width:])]
            base = embedding + feedback
            if self.attention_residuals:
                base = unit_rms(base)
        else:
            previous_units = None
            base = state + feedback

        # `stream` is the plain residual stream. `sources` are what a depth-wise read mixes.
        keep_stream = not (self.attention_residuals and self.message)
        stream = base
        sources = [unit_rms(base)] if self.attention_residuals else None
        block_sums = []
        for block in BLOCKS:
            running = None
            for layer_index in block:
                layer = self.layers[layer_index]
                for sublayer in (0, 1):
                    if self.attention_residuals:
                        available = sources if running is None else [*sources, unit_rms(running)]
                        current = self.depth_reads[2 * layer_index + sublayer](available)
                    else:
                        current = stream
                    if sublayer == 0:
                        output = self.attention_output(
                            layer, layer_index, current, previous_units, rope_cos, rope_sin)
                    else:
                        output = self.feedforward_output(layer, layer_index, current, previous_units)
                    running = output if running is None else running + output
                    if keep_stream:
                        stream = stream + output
            block_sums.append(running)
            if self.attention_residuals:
                sources = [*sources, unit_rms(running)]
        if not self.message:
            return stream
        return torch.cat([embedding, base, block_sums[0], block_sums[1]], -1)
