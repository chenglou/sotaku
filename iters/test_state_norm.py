import unittest

import torch
import torch.nn.functional as F

from iters.state_norm import cap_token_rms, per_token_rms, rms_normalize


class StateNormTest(unittest.TestCase):
    def test_rms_normalize_sets_each_token_to_unit_rms(self):
        hidden_state = torch.tensor(
            [[[3.0, 4.0], [6.0, 8.0]], [[1.0, -1.0], [2.0, -2.0]]]
        )
        normalized = rms_normalize(hidden_state)
        torch.testing.assert_close(
            per_token_rms(normalized),
            torch.ones(hidden_state.shape[:-1]),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_cap_leaves_small_tokens_unchanged(self):
        hidden_state = torch.tensor([[[3.0, 4.0], [60.0, 80.0]]])
        capped = cap_token_rms(hidden_state, maximum_rms=10.0)
        torch.testing.assert_close(capped[:, :1], hidden_state[:, :1])
        self.assertLessEqual(float(per_token_rms(capped).max()), 10.0 + 1e-5)

    def test_cap_preserves_direction(self):
        hidden_state = torch.randn(4, 9, 16) * 100
        capped = cap_token_rms(hidden_state, maximum_rms=32.0)
        cosine = F.cosine_similarity(hidden_state, capped, dim=-1)
        torch.testing.assert_close(cosine, torch.ones_like(cosine), atol=1e-6, rtol=0)

    def test_cap_rejects_nonpositive_threshold(self):
        with self.assertRaises(ValueError):
            cap_token_rms(torch.ones(1, 1, 2), maximum_rms=0)


if __name__ == "__main__":
    unittest.main()
