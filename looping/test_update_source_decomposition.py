import unittest

import torch

from looping.eval_update_source_decomposition import (
    _apply_tangential_scale,
    source_decomposed_recurrent_step,
)


class _FixedProjection:
    def __init__(self, update):
        self.update = update

    def __call__(self, predictions):
        return self.update.expand(
            predictions.size(0),
            predictions.size(1),
            -1,
        )


class _AddUpdate:
    def __init__(self, update):
        self.update = update

    def __call__(self, hidden_state, rope_cos, rope_sin):
        return hidden_state + self.update


class _SourceModel:
    feedback_scale = 1.0

    def __init__(self, feedback_update, layer_updates):
        self.pred_proj = _FixedProjection(feedback_update)
        self.layers = [
            _AddUpdate(layer_update)
            for layer_update in layer_updates
        ]
        self.layer_schedule = tuple(range(len(self.layers)))

    def normalize_outer_state(self, hidden_state):
        return hidden_state


class UpdateSourceDecompositionTest(unittest.TestCase):
    def test_tangential_scale_preserves_radial_update(self):
        hidden_state = torch.tensor([[[1.0, 0.0]]])
        proposed_state = torch.tensor([[[3.0, 4.0]]])
        result, radial, tangential = _apply_tangential_scale(
            hidden_state,
            proposed_state,
            0.25,
        )
        torch.testing.assert_close(radial, torch.tensor([[[2.0, 0.0]]]))
        torch.testing.assert_close(tangential, torch.tensor([[[0.0, 4.0]]]))
        torch.testing.assert_close(result, torch.tensor([[[3.0, 1.0]]]))

    def test_feedback_and_layer_tangential_scales_are_independent(self):
        hidden_state = torch.tensor([[[1.0, 0.0]]])
        predictions = torch.zeros(1, 1, 9)
        unused = torch.empty(0)
        feedback_model = _SourceModel(
            torch.tensor([0.0, 2.0]),
            [],
        )
        feedback_result, *_ = source_decomposed_recurrent_step(
            feedback_model,
            hidden_state,
            predictions,
            unused,
            unused,
            feedback_tangential_alpha=0.25,
            layer_tangential_alpha=1.0,
        )
        torch.testing.assert_close(
            feedback_result,
            torch.tensor([[[1.0, 0.5]]]),
        )

        layer_model = _SourceModel(
            torch.tensor([0.0, 0.0]),
            [torch.tensor([0.0, 2.0])],
        )
        layer_result, *_ = source_decomposed_recurrent_step(
            layer_model,
            hidden_state,
            predictions,
            unused,
            unused,
            feedback_tangential_alpha=1.0,
            layer_tangential_alpha=0.25,
        )
        torch.testing.assert_close(
            layer_result,
            torch.tensor([[[1.0, 0.5]]]),
        )


if __name__ == "__main__":
    unittest.main()
