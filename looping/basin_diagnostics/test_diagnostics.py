"""Metric, perturbation, and plotting checks, including temporary correct answers."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from looping.basin_diagnostics.analysis import (OutcomeTracker, coordinates, inject_state, plane_basis,
                                               saved_unit, save_unit, select_rows, sensitivity_map, trace, uncertainty_curve)
from looping.basin_diagnostics.common import checkpoint_spec, protocol
from looping.basin_diagnostics.plots import comparison, pca_paths, render_map
from looping.hyperloop.model import HyperloopTransformer


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_returning_answer_uses_last_change_not_count(self):
        target = torch.zeros(1, 81, dtype=torch.long)
        tracker = OutcomeTracker(target.clone(), target, torch.ones(1, dtype=torch.bool), 10, 16)
        for step, value in enumerate((1, 0, 2, 0, 0, 0), 11):
            tracker.update(torch.full_like(target, value), torch.ones(1, dtype=torch.bool), step)
        arrays = tracker.arrays(2)
        self.assertEqual(arrays["last_change"].item(), 14)
        self.assertEqual(arrays["first_correct"].item(), 10)
        self.assertEqual(arrays["regressions"].item(), 2)
        self.assertEqual(arrays["outcome"].item(), 2)
        self.assertEqual(arrays["correct_suffix_start"].item(), 14)
        self.assertTrue(arrays["confirmed_answer"].item())

    def test_wrong_unchanging_answer_is_not_correct(self):
        target = torch.zeros(1, 81, dtype=torch.long)
        wrong = torch.ones_like(target)
        tracker = OutcomeTracker(wrong, target, torch.ones(1, dtype=torch.bool), 0, 4)
        for step in range(1, 5):
            tracker.update(wrong, torch.ones(1, dtype=torch.bool), step)
        arrays = tracker.arrays(2)
        self.assertTrue(arrays["confirmed_answer"].item())
        self.assertFalse(arrays["final_correct"].item())
        self.assertEqual(arrays["outcome"].item(), 0)

    def test_nonfinite_and_late_changes_are_not_settled(self):
        target = torch.zeros(1, 81, dtype=torch.long)
        tracker = OutcomeTracker(target.clone(), target, torch.ones(1, dtype=torch.bool), 0, 4)
        tracker.update(target, torch.zeros(1, dtype=torch.bool), 4)
        arrays = tracker.arrays(2)
        self.assertFalse(arrays["confirmed_answer"].item())
        self.assertFalse(arrays["final_correct"].item())
        self.assertEqual(arrays["outcome"].item(), 1)

    def test_plane_is_orthogonal_rms_one_and_repeatable(self):
        plane = plane_basis((81, 128), 10)
        torch.testing.assert_close(plane.square().mean((1, 2)), torch.ones(2))
        self.assertLess(abs(float((plane[0] * plane[1]).mean())), 1e-6)
        self.assertTrue(torch.equal(plane, plane_basis((81, 128), 10)))

    def test_mesh_has_exact_center_and_invalid_even_size(self):
        points = coordinates(17)
        self.assertTrue(torch.equal(points[144], torch.zeros(2)))
        with self.assertRaises(ValueError):
            coordinates(16)

    def test_zero_perturbation_and_feedback(self):
        model = HyperloopTransformer(0).eval()
        inputs = torch.nn.functional.one_hot(torch.zeros(1, 81, dtype=torch.long), 10).float()
        hidden, predictions = model.initial_state(inputs)
        offsets = torch.stack((torch.zeros_like(hidden[0]), torch.full_like(hidden[0], .01)))
        h, p = inject_state(model, hidden, predictions, offsets, anchor=0)
        self.assertTrue(torch.equal(h[0], hidden[0]))
        self.assertTrue(torch.equal(p, torch.zeros_like(p)))
        hidden, predictions, _ = model.step(hidden, predictions)
        h, p = inject_state(model, hidden, predictions, offsets, anchor=1)
        self.assertTrue(torch.equal(p[0], predictions[0]))
        torch.testing.assert_close(p[1], model.output_head(h[1]).softmax(-1))

    def test_selection_is_disjoint_and_avoids_monitoring_rows(self):
        settings = protocol()
        labels = np.repeat(["0", "1-2", "3-10", "11-50", "51+"], 5000)
        discovery, evaluation = select_rows(labels, settings)
        self.assertEqual(len(discovery), 10)
        self.assertEqual(len(evaluation), 50)
        self.assertFalse(set(discovery) & set(evaluation))
        self.assertTrue(np.all(np.concatenate((discovery, evaluation)) % 5000 >= 200))
        self.assertTrue(np.array_equal(evaluation, select_rows(labels, settings)[1]))

    def test_flat_map_null_and_censoring(self):
        field = np.ones((17, 17))
        self.assertTrue(all(row["different_fraction"] == 0 for row in uncertainty_curve(field)))
        self.assertTrue(all(row["different_fraction"] is None for row in uncertainty_curve(field, np.zeros_like(field, dtype=bool))))

    def test_neighbor_measure_ignores_shared_magnitude_growth(self):
        values = torch.randn(27, 81, 8)
        torch.testing.assert_close(sensitivity_map(values, 5), sensitivity_map(values * 10, 5))

    def test_pca_preserves_simple_two_dimensional_paths(self):
        rng = np.random.default_rng(30)
        states = rng.normal(size=(8, 3, 2)) @ rng.normal(size=(2, 20))
        paths, variance = pca_paths(states)
        self.assertEqual(paths.shape, (8, 3, 2))
        self.assertAlmostEqual(variance, 1.0)

    def test_recorded_cohort_has_expected_paths(self):
        spec = checkpoint_spec("20k_20260908")
        self.assertTrue(spec["path"].endswith("baseline_seed20260908/final.pt"))
        self.assertAlmostEqual(spec["archived_scores"]["4096"], .02856)
        self.assertEqual(checkpoint_spec("50k_20260910")["updates"], 50000)

    def test_comparison_renders_both_training_budgets(self):
        from checkpoint_utils import atomic_json_save
        from looping.weight_tying.common import atomic_npz
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for key in ("20k_20260908", "50k_20260910"):
                model_dir = directory / "models" / key
                model_dir.mkdir(parents=True)
                atomic_json_save({"status": "complete"}, model_dir / "completed.json")
                atomic_npz(model_dir / "baseline.npz", steps=np.arange(3), correct_curve=np.ones((3, 2)))
            comparison(directory)
            self.assertGreater((directory / "comparison.png").stat().st_size, 1000)

    def test_small_actual_model_trace(self):
        model = HyperloopTransformer(0).eval()
        given = np.zeros((1, 81), dtype=np.int64)
        target = np.zeros_like(given)
        hidden, probabilities = model.initial_state(torch.nn.functional.one_hot(torch.as_tensor(given), 10).float())
        offsets = torch.zeros(27, 81, 128)
        h, p = inject_state(model, hidden, probabilities, offsets, anchor=0)
        settings = {"horizon": 4, "measurement_interval": 1, "snapshot_interval": 1, "confirmation_window": 2}
        arrays, result = trace(model, h, p, given, target, anchor=0, settings=settings, resolution=5)
        self.assertEqual(arrays["correct_curve"].shape, (5, 27))
        self.assertEqual(arrays["hidden_snapshots"].shape, (5, 3, 81, 128))
        self.assertEqual(result["zero_control_board_mismatches"], 0)
        self.assertEqual(result["zero_control_max_hidden_error"], 0)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            save_unit(directory, "fixture", arrays, result)
            info = {"resolution": 5, "horizon": 4, "anchor": 0, "rms_fraction": .03, "plane_seed": 10,
                    "gallery": {"index": 7, "difficulty": "fixture"}}
            render_map(directory / "fixture.npz", directory / "fixture.png",
                       {"key": "test", "expected_status": "fixture"}, info)
            from PIL import Image
            pixels = np.asarray(Image.open(directory / "fixture.png"))
            self.assertGreater(pixels.std(), 10)
            self.assertGreater(pixels.shape[1], 1000)

    def test_resume_checks_complete_arrays_and_recomputes_partial_unit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.assertIsNone(saved_unit(directory, "fixture"))
            summary = save_unit(directory, "fixture", {"value": np.arange(4)}, {"status": "complete"})
            self.assertEqual(saved_unit(directory, "fixture"), summary)
            (directory / "fixture.npz").write_bytes(b"changed artifact")
            with self.assertRaises(ValueError):
                saved_unit(directory, "fixture")

    def test_loss_at_end_is_distinguished_from_recovery(self):
        target = torch.zeros(2, 81, dtype=torch.long)
        tracker = OutcomeTracker(target.clone(), target, torch.ones(2, dtype=torch.bool), 128, 132)
        wrong = torch.ones_like(target)
        tracker.update(wrong, torch.ones(2, dtype=torch.bool), 129)
        wrong[1] = 0
        tracker.update(wrong, torch.ones(2, dtype=torch.bool), 130)
        arrays = tracker.arrays(2)
        np.testing.assert_array_equal(arrays["outcome"], [1, 2])


def gpu_preflight():
    from stabilize.exp_testbed_20k import SudokuTransformer
    from looping.basin_diagnostics.common import load_model
    model, spec = load_model("20k_20260908", "cuda")
    original = SudokuTransformer().float().cuda().eval()
    original.load_state_dict(model.state_dict(), strict=True)
    digits = torch.arange(81, device="cuda").remainder(10)[None].repeat(3, 1)
    inputs = torch.nn.functional.one_hot(digits, 10).float()
    with torch.inference_mode():
        reference = original(inputs, return_all=True)
        hidden, probabilities = model.initial_state(inputs)
        for expected in reference:
            hidden, probabilities, logits = model.step(hidden, probabilities)
            torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DiagnosticTests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise RuntimeError("Diagnostic unit tests failed in the GPU image")
    return {"status": "passed", "exact_recurrence_match": True, "checkpoint": spec}


if __name__ == "__main__":
    unittest.main()
