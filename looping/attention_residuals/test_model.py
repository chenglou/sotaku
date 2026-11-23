import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from inference import RecurrentRunner
from looping.attention_residuals.evaluate import variant_loader
from looping.attention_residuals.model import ARMS, LoopVariantTransformer, unit_rms
from looping.attention_residuals.train import model_variant, run_name
from model_io import load_model, write_model_manifest
from stabilize.exp_testbed_20k import (
    ROPE_COS, ROPE_SIN, SudokuTransformer, build_detached_advance, d_model,
)


def puzzles(count=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return F.one_hot(torch.randint(0, 10, (count, 81), generator=generator), 10).float()


def build(arm, seed=7, **settings):
    torch.manual_seed(seed)
    if arm == "baseline":
        return SudokuTransformer().eval()
    return LoopVariantTransformer(arm, **settings).eval()


class LoopVariantTest(unittest.TestCase):
    def test_unit_rms_keeps_zero_vectors_zero(self):
        values = torch.zeros(2, 5, d_model)
        values[0, 0] = torch.randn(d_model) * 30
        scaled = unit_rms(values)
        self.assertTrue(torch.isfinite(scaled).all())
        self.assertAlmostEqual(float(scaled[0, 0].square().mean().sqrt()), 1.0, places=4)
        self.assertEqual(float(scaled[1].abs().max()), 0.0)

    def test_baseline_weights_start_identical(self):
        baseline = build("baseline").state_dict()
        for arm in ARMS:
            variant = build(arm).state_dict()
            renamed = {name.replace("initial_encoder.linear.", "initial_encoder.")
                       .replace("output_head.linear.", "output_head."): value
                       for name, value in variant.items()}
            for name, value in baseline.items():
                self.assertTrue(torch.equal(value, renamed[name]), f"{arm}: {name}")

    def test_parameter_counts(self):
        baseline = sum(p.numel() for p in build("baseline").parameters())
        extra = {arm: sum(p.numel() for p in build(arm).parameters()) - baseline for arm in ARMS}
        gates = 8 * (4 * d_model * 8)
        self.assertEqual(extra["attnres"], 8 * d_model)
        self.assertEqual(extra["message"], 8 * d_model + gates)
        self.assertEqual(extra["attnres_message"], (8 + 8 + 1) * d_model + gates)
        separate = sum(p.numel() for p in build("message", separate_qkv=True).parameters())
        self.assertEqual(separate - baseline, 16 * d_model + 2 * gates)

    def test_forward_shapes_and_gradients(self):
        inputs = puzzles()
        for arm in ARMS:
            model = build(arm).train()
            logits, (state, predictions) = model(inputs, return_all=True, return_state=True)
            self.assertEqual(len(logits), 16)
            self.assertEqual(tuple(logits[-1].shape), (3, 81, 9))
            self.assertEqual(state.shape[-1], d_model if arm == "attnres" else 4 * d_model)
            self.assertEqual(tuple(predictions.shape), (3, 81, 9))
            torch.stack(logits).square().mean().backward()
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.grad is None or not torch.isfinite(parameter.grad).all()]
            # The first sublayer has a single source (the iteration's input), so its query is unused.
            self.assertEqual(missing, [] if arm == "message" else ["depth_reads.0.query"], arm)
            if arm != "attnres":
                gains = [gate.gain.weight.grad.abs().max() for gate in model.message_gates]
                self.assertTrue(all(float(value) > 0 for value in gains), arm)

    def test_message_is_blocked_at_the_start(self):
        # With closed gates and a plain residual stream, one iteration equals the baseline's
        # layers applied to the embedding plus prediction feedback.
        inputs = puzzles()
        predictions = torch.rand(3, 81, 9).softmax(-1)
        baseline, variant = build("baseline"), build("message")
        expected = baseline.output_head(baseline.recurrent_step(
            baseline.initial_encoder(inputs), predictions, ROPE_COS, ROPE_SIN))
        state = variant.initial_encoder(inputs)
        state[..., 2 * d_model:] = torch.randn(3, 81, 2 * d_model)  # a made-up previous iteration
        actual = variant.output_head(variant.recurrent_step(state, predictions, ROPE_COS, ROPE_SIN))
        self.assertTrue(torch.allclose(expected, actual, atol=1e-5))

    def test_open_gate_passes_the_message(self):
        inputs = puzzles()
        predictions = torch.rand(3, 81, 9).softmax(-1)
        for arm in ("message", "attnres_message"):
            variant = build(arm)
            for gate in variant.message_gates:
                torch.nn.init.normal_(gate.gain.weight, std=0.5)
            state = variant.initial_encoder(inputs)
            first = variant.recurrent_step(state, predictions, ROPE_COS, ROPE_SIN)
            changed = state.clone()
            changed[..., 2 * d_model:] = torch.randn(3, 81, 2 * d_model)
            second = variant.recurrent_step(changed, predictions, ROPE_COS, ROPE_SIN)
            self.assertGreater(float((first - second)[..., 2 * d_model:].detach().abs().max()), 1e-3, arm)

    def test_separate_qkv_messages(self):
        self.assertEqual(len(build("attnres_message").message_gates), 8)
        model = build("attnres_message", separate_qkv=True)
        self.assertEqual(len(model.message_gates), 16)
        logits = model(puzzles())
        self.assertTrue(torch.isfinite(logits).all())

    def test_runner_and_detached_advance(self):
        inputs = puzzles()
        for arm in ARMS:
            model = build(arm)
            outputs, _ = RecurrentRunner(model).run_batch(inputs, [1, 16, 40])
            direct = model(inputs)
            self.assertTrue(torch.allclose(outputs[16], direct, atol=1e-5), arm)
            self.assertTrue(torch.isfinite(outputs[40]).all())
            advance = build_detached_advance(model, compile_chunk=False)
            start = (model.initial_encoder(inputs), torch.zeros(3, 81, 9))
            state, predictions = advance(start, 32)
            self.assertFalse(state.requires_grad or predictions.requires_grad)
            later = model(inputs, initial_state=(state, predictions))
            self.assertTrue(torch.isfinite(later).all())

    def test_trainer_patch_and_weight_round_trip(self):
        import stabilize.exp_testbed_20k as trainer

        with model_variant("attnres_message"):
            torch.manual_seed(3)
            model = trainer.SudokuTransformer(
                outer_state_norm=False, outer_state_norm_epsilon=1e-6, outer_state_rms_cap=None).eval()
        self.assertIs(trainer.SudokuTransformer, SudokuTransformer)
        self.assertIsInstance(model, LoopVariantTransformer)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_test.pt"
            torch.save(model.state_dict(), path)
            write_model_manifest(path, model)
            with self.assertRaises(RuntimeError):
                load_model(path)  # the plain loader must refuse a variant's weights
            with variant_loader("attnres_message"):
                loaded, _ = load_model(path)
        inputs = puzzles()
        self.assertTrue(torch.equal(model(inputs), loaded(inputs)))
        self.assertEqual(run_name("message", 5), "attnres20k_message_seed5")


if __name__ == "__main__":
    unittest.main()
