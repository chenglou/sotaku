import unittest

import torch

from dataset_utils import phase_bucket_keys, balanced_indices
from stabilize.exp_testbed_20k import (
    ROPE_COS, ROPE_SIN, SudokuTransformer, build_detached_advance,
    detached_state_mode, get_lr, update_sample_digest,
)


class BurninDropoutTest(unittest.TestCase):
    def test_disabling_burnin_dropout_restores_all_module_modes(self):
        model = SudokuTransformer(unique_layers=1, training_iterations=1).train()
        model.layers[0].dropout.eval()
        modes = [module.training for module in model.modules()]
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with detached_state_mode(model, False):
                self.assertFalse(torch.is_grad_enabled())
                self.assertFalse(any(module.training for module in model.modules()))
                raise RuntimeError("fixture")
        self.assertEqual(modes, [module.training for module in model.modules()])
        self.assertTrue(torch.is_grad_enabled())

    def test_burnin_is_detached_and_supervised_window_still_has_gradients(self):
        torch.manual_seed(41)
        model = SudokuTransformer(unique_layers=1, training_iterations=1).train()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        state = (model.initial_encoder(inputs), torch.zeros(1, 81, 9))
        advance = build_detached_advance(model, False, compile_chunk=False)
        first = advance(state, 16)
        second = advance(state, 16)
        for left, right in zip(first, second):
            torch.testing.assert_close(left, right, atol=0, rtol=0)
            self.assertFalse(left.requires_grad)
        self.assertTrue(model.training)
        logits = model(inputs, initial_state=first)
        logits.square().mean().backward()
        self.assertIsNone(model.initial_encoder.weight.grad)
        self.assertGreater(model.layers[0].q_proj.weight.grad.abs().sum(), 0)

    def test_default_burnin_keeps_dropout(self):
        model = SudokuTransformer(unique_layers=1).train()
        state = (torch.randn(1, 81, 128), torch.zeros(1, 81, 9))
        advance = build_detached_advance(model, compile_chunk=False)
        torch.manual_seed(42)
        first = advance(state, 16)
        torch.manual_seed(43)
        second = advance(state, 16)
        self.assertFalse(torch.equal(first[0], second[0]))
        self.assertTrue(model.training)

    def test_default_burnin_matches_original_loop(self):
        model = SudokuTransformer(unique_layers=1).train()
        state = (torch.randn(1, 81, 128), torch.zeros(1, 81, 9))
        torch.manual_seed(17)
        with torch.no_grad():
            hidden, predictions = state
            for _ in range(32):
                hidden = model.recurrent_step(hidden, predictions, ROPE_COS, ROPE_SIN)
                predictions = torch.softmax(model.output_head(hidden), dim=-1)
        torch.manual_seed(17)
        actual = build_detached_advance(model, compile_chunk=False)(state, 32)
        for expected, restored in zip((hidden, predictions), actual):
            torch.testing.assert_close(restored, expected, rtol=0, atol=0)
        self.assertTrue(model.training)

    def test_curriculum_preserves_historical_whole_bucket_selection(self):
        keys = [(0, 0), (1, 2), (3, 10), (11, 50), (51, 1000)]
        self.assertEqual(phase_bucket_keys(keys, 21), keys[4:])
        self.assertEqual(phase_bucket_keys(keys, 6), keys[3:])
        self.assertEqual(phase_bucket_keys(keys, 1), keys[1:])
        self.assertEqual(phase_bucket_keys(keys, 0), keys)

    def test_sampling_does_not_mutate_global_rng(self):
        import random
        ratings = [1, 3, 60, 20, 0] * 20
        state = random.getstate()
        first = balanced_indices(ratings, 5)
        self.assertEqual(state, random.getstate())
        self.assertEqual(first, balanced_indices(ratings, 5))
        self.assertEqual(len(first[0]), 25)
        self.assertEqual(first[1][:5], ["0"] * 5)

    def test_continuation_uses_original_schedule(self):
        self.assertAlmostEqual(get_lr(39001, 1400, 50000), 0.0002598714167563864)

    def test_sampling_digest_covers_order_and_horizon_without_using_rng(self):
        rows = torch.tensor([4, 2, 9])
        state = torch.get_rng_state()
        first = update_sample_digest("0" * 64, rows, 39001, 128)
        self.assertEqual(first, update_sample_digest("0" * 64, rows, 39001, 128))
        self.assertNotEqual(first, update_sample_digest("0" * 64, rows.flip(0), 39001, 128))
        self.assertNotEqual(first, update_sample_digest("0" * 64, rows, 39001, 256))
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
