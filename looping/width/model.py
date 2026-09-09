"""The existing single-state recurrence, with configurable transformer width."""

from torch import nn

from looping.hyperloop.model import HyperloopTransformer
from looping.weight_tying.model import rope_tables
from stabilize.exp_testbed_20k import RoPETransformerLayer


class WidthTransformer(HyperloopTransformer):
    def __init__(self, width=128, *, dropout=0.1):
        if type(width) is not int or width <= 0:
            raise ValueError("Width must be a positive integer")
        cosine, sine = rope_tables(width)
        # Keep the reference initialization order and reuse its recurrence and loss.
        nn.Module.__init__(self)
        self.width = width
        self.streams = 0
        self.gates = None
        self.outer_state_norm = False
        self.outer_state_norm_epsilon = 1e-6
        self.outer_state_rms_cap = None
        self.unique_layers = 4
        self.layer_schedule = (0, 1, 2, 3)
        self.residual_scale = 1.0
        self.feedback_scale = 1.0
        self.training_iterations = 16
        self.initial_encoder = nn.Linear(10, width)
        self.pred_proj = nn.Linear(9, width)
        self.layers = nn.ModuleList([
            RoPETransformerLayer(width, 4, 4 * width, dropout=dropout)
            for _ in range(4)
        ])
        self.output_head = nn.Linear(width, 9)
        self.register_buffer("rope_cos", cosine, persistent=False)
        self.register_buffer("rope_sin", sine, persistent=False)
