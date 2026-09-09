"""Reference equivalence, wider-state gradients, resumption, and export identity."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from checkpoint_utils import atomic_json_save, atomic_torch_save
from looping.hyperloop.common import protocol as original_protocol
from looping.hyperloop.evaluate import load_export as original_load_export
from looping.hyperloop.model import HyperloopTransformer
from looping.weight_tying.test_study import fixture, make_smoke_data
from looping.weight_tying.train import PairedSampler
from looping.width.common import (
    build_model, protocol, run_config, state_sha256, verify_completed_pair, verify_reference,
)
from looping.width.evaluate import evaluate_arrays, load_export
from looping.width.model import WidthTransformer
from looping.width.train import train_run


class WidthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_width128_matches_reference_initialization_loss_gradients_and_dropout(self):
        inputs, answers = fixture(2)
        for training in (True, False):
            for prefix in (0, 32):
                torch.manual_seed(71)
                reference = HyperloopTransformer().train(training)
                torch.manual_seed(71)
                actual = WidthTransformer().train(training)
                self.assertEqual(state_sha256(reference.state_dict()), state_sha256(actual.state_dict()))
                torch.testing.assert_close(actual.rope_cos, reference.rope_cos, rtol=0, atol=0)
                torch.testing.assert_close(actual.rope_sin, reference.rope_sin, rtol=0, atol=0)
                results = []
                for model in (reference, actual):
                    torch.manual_seed(44)
                    state = None
                    if prefix:
                        with torch.no_grad():
                            state = model.advance(*model.advance(*model.initial_state(inputs)))
                    loss, logits, state = model(inputs, answers, initial_state=state)
                    loss.backward()
                    results.append((loss, logits, state))
                for left, right in zip(results[0][:2], results[1][:2]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for left, right in zip(results[0][2], results[1][2]):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)
                for left, right in zip(reference.parameters(), actual.parameters()):
                    if left.grad is None:
                        self.assertIsNone(right.grad)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_controls_and_numerical_sources_match_the_archived_study(self):
        reference = verify_reference()
        for seed in protocol()["seeds"]:
            torch.manual_seed(seed)
            model = WidthTransformer()
            self.assertEqual(state_sha256(model.state_dict()), reference["runs"][str(seed)]["base_initial_state_sha256"])
        changed = copy.deepcopy(protocol())
        changed["training"]["learning_rate"] = 0.001
        with patch("looping.width.common.protocol", return_value=changed):
            with self.assertRaises(ValueError):
                verify_reference()

    def test_only_width_and_feedforward_width_increase(self):
        expected_counts = {128: 796937, 160: 1241929, 192: 1785225}
        inputs, answers = fixture(2)
        for width in (128, 160, 192):
            model = WidthTransformer(width).eval()
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected_counts[width])
            self.assertEqual(len(model.layers), 4)
            self.assertIsNone(model.gates)
            self.assertFalse(model.outer_state_norm)
            for layer in model.layers:
                self.assertEqual(layer.n_heads, 4)
                self.assertEqual(layer.linear1.out_features, 4 * width)
                self.assertEqual(layer.attn_dropout_p, 0.1)
            with torch.no_grad():
                state = model.advance(*model.initial_state(inputs))
            self.assertEqual(state[0].shape, (2, 81, width))
            self.assertTrue(all(not value.requires_grad for value in state))
            loss, _, _ = model(inputs, answers, initial_state=state)
            loss.backward()
            self.assertIsNone(model.initial_encoder.weight.grad)
            self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for name, p in model.named_parameters()
                                if not name.startswith("initial_encoder.")))
        for width in (0, -16, 129, 132, 3.5):
            with self.assertRaises(ValueError):
                WidthTransformer(width)

    def test_sampling_matches_controls_across_curriculum_changes(self):
        settings = protocol()
        ratings = np.array([0, 1, 2, 3, 10, 11, 50, 51, 100], dtype=np.int16)
        regime = {key: settings[key] for key in ("burnin_probability", "burnin_iterations")}
        for seed in settings["seeds"]:
            reference = PairedSampler(ratings, seed, regime, original_protocol()["training"])
            wider = PairedSampler(ratings, seed, regime, settings["training"])
            for step in (0, 3999, 4000, 7999, 8000, 11999, 12000, 19999):
                expected = reference.sample(step, 2048)
                actual = wider.sample(step, 2048)
                np.testing.assert_array_equal(expected[0], actual[0])
                self.assertEqual(expected[1], actual[1])

    def test_wider_state_evaluation_matches_direct_steps_and_restores_precision(self):
        inputs, targets = fixture(3)
        for width in (160, 192):
            model = WidthTransformer(width).eval()
            with torch.no_grad():
                state = model.initial_state(inputs)
                for _ in range(32):
                    hidden, probabilities, logits = model.step(*state)
                    state = hidden, probabilities
                chunked = model.advance(*model.advance(*model.initial_state(inputs)))
                torch.testing.assert_close(chunked[0], hidden, rtol=0, atol=0)
                torch.testing.assert_close(chunked[1], probabilities, rtol=0, atol=0)
            model.train()
            precision = torch.get_float32_matmul_precision()
            scores, predictions, _ = evaluate_arrays(model, inputs.argmax(-1).numpy(), targets.numpy(),
                                                     [16, 32], batch_size=3, track_solutions=True)
            np.testing.assert_array_equal(predictions["predictions_32"], logits.argmax(-1).numpy())
            self.assertEqual(scores["32"]["total"], 3)
            self.assertTrue(model.training)
            self.assertEqual(torch.get_float32_matmul_precision(), precision)

    def test_exact_resume_selected_recovery_and_wrong_width_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            results = []
            for arm in protocol()["arms"]:
                seed = protocol()["seeds"][0]
                directory = root / arm
                train_run(root / "data", directory, arm, seed, smoke=True, stop_after=2, device="cpu")
                atomic_torch_save({}, directory / "best_validation.pt")
                resumed = train_run(root / "data", directory, arm, seed, smoke=True, device="cpu")
                whole = train_run(root / "data", root / (arm + "_whole"), arm, seed, smoke=True, device="cpu")
                results.append(resumed)
                self.assertEqual(resumed["sample_digest"], whole["sample_digest"])
                self.assertEqual(state_sha256(torch.load(directory / "final.pt", weights_only=True)),
                                 state_sha256(torch.load(root / (arm + "_whole") / "final.pt", weights_only=True)))
                loaded, manifest = load_export(directory / "final.pt")
                _, selected = load_export(directory / "best_validation.pt")
                self.assertEqual(loaded.width, protocol()["arms"][arm])
                self.assertEqual(manifest["updates"], 4)
                self.assertEqual(selected["updates"], resumed["best_validation"]["updates"])
                with self.assertRaises(ValueError):
                    original_load_export(directory / "final.pt")
                with self.assertRaises(ValueError):
                    train_run(root / "data", directory, arm, seed + 1, smoke=True, device="cpu")
                manifest["config"]["arm"] = "width192" if arm == "width160" else "width160"
                atomic_json_save(manifest, str(directory / "final.pt") + ".json")
                with self.assertRaises(RuntimeError):
                    load_export(directory / "final.pt")
            for key in ("sample_digest", "work_counts", "horizon_counts"):
                self.assertEqual(results[0][key], results[1][key])

    def test_completed_runs_must_match_control_sampling(self):
        seed = protocol()["seeds"][0]
        result = copy.deepcopy(verify_reference()["runs"][str(seed)])
        result["config"] = run_config("width160", seed)
        verify_completed_pair(result)
        result["sample_digest"] = "wrong"
        with self.assertRaises(ValueError):
            verify_completed_pair(result)


if __name__ == "__main__":
    unittest.main()
