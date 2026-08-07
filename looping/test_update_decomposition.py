import unittest

import torch

from looping.eval_update_decomposition import (
    MODEL_PRESETS,
    POLICY_PRESETS,
    decompose_update,
    decomposed_recurrent_step,
)
from looping.modal_update_decomposition import (
    POLICY_PRESET_NAMES,
    PRESET_NAMES,
)


class _ProposeState:
    def __init__(self, proposed_state):
        self.proposed_state = proposed_state

    def apply_recurrent_updates(
        self,
        hidden_state,
        predictions,
        rope_cos,
        rope_sin,
    ):
        return self.proposed_state

    def normalize_outer_state(self, hidden_state):
        return hidden_state


class UpdateDecompositionTest(unittest.TestCase):
    def test_modal_and_evaluator_presets_match(self):
        self.assertEqual(set(PRESET_NAMES), set(MODEL_PRESETS))
        self.assertEqual(
            set(POLICY_PRESET_NAMES),
            set(POLICY_PRESETS),
        )

    def test_decomposition_reconstructs_update_and_is_orthogonal(self):
        hidden_state = torch.tensor([[[1.0, 0.0]]])
        proposed_state = torch.tensor([[[3.0, 4.0]]])
        radial, tangential = decompose_update(
            hidden_state,
            proposed_state,
        )
        torch.testing.assert_close(
            radial,
            torch.tensor([[[2.0, 0.0]]]),
        )
        torch.testing.assert_close(
            tangential,
            torch.tensor([[[0.0, 4.0]]]),
        )
        torch.testing.assert_close(
            radial + tangential,
            proposed_state - hidden_state,
        )
        torch.testing.assert_close(
            (tangential * hidden_state).sum(dim=-1),
            torch.zeros(1, 1),
        )

    def test_component_alphas_change_only_the_selected_update(self):
        hidden_state = torch.tensor([[[1.0, 0.0]]])
        proposed_state = torch.tensor([[[3.0, 4.0]]])
        model = _ProposeState(proposed_state)
        unused = torch.empty(0)

        radial_only, _, _ = decomposed_recurrent_step(
            model,
            hidden_state,
            unused,
            unused,
            unused,
            radial_alpha=1.0,
            tangential_alpha=0.0,
        )
        tangential_only, _, _ = decomposed_recurrent_step(
            model,
            hidden_state,
            unused,
            unused,
            unused,
            radial_alpha=0.0,
            tangential_alpha=1.0,
        )
        torch.testing.assert_close(
            radial_only,
            torch.tensor([[[3.0, 0.0]]]),
        )
        torch.testing.assert_close(
            tangential_only,
            torch.tensor([[[1.0, 4.0]]]),
        )


if __name__ == "__main__":
    unittest.main()
