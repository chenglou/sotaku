"""Check precision, sampling, outcome selection, and resumable grids."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from looping.basin_diagnostics.slow_puzzles.run import (
    BUCKETS, configure_precision, encode, grid_field64, initial_states,
    load_arrays, plane64, protocol, render_case, require_double, screen, screen_offsets, select_cases,
    select_rows, summarize_grid, trace_states,
)
from looping.hyperloop.model import HyperloopTransformer


class ScriptedModel(torch.nn.Module):
    def __init__(self, answers, bad_dtype=False, nonfinite_at=None):
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.answers = answers
        self.position = 0
        self.bad_dtype = bad_dtype
        self.nonfinite_at = nonfinite_at

    def step(self, hidden, predictions):
        logits = hidden.new_zeros(len(hidden), 81, 9)
        logits[..., self.answers[self.position]] = 1
        self.position += 1
        if self.position == self.nonfinite_at:
            logits.fill_(float("nan"))
        if self.bad_dtype:
            logits = logits.float()
        return hidden, logits.softmax(-1), logits


class SlowPuzzleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_sampling_is_balanced_deterministic_and_excludes_old_puzzle(self):
        settings = protocol()
        labels = np.repeat(BUCKETS, 300)
        indices = np.arange(1500) + 1000
        settings["exclude_indices"] = [int(indices[220])]
        rows = select_rows(labels, indices, settings)
        np.testing.assert_array_equal(rows, select_rows(labels, indices, settings))
        self.assertEqual(len(set(rows)), 250)
        self.assertNotIn(220, rows)
        self.assertTrue(np.all(rows % 300 >= 200))
        for bucket in BUCKETS:
            self.assertEqual(int((labels[rows] == bucket).sum()), 50)

    def test_plane_is_double_orthogonal_and_rms_one(self):
        plane = plane64((81, 128), 2701)
        self.assertEqual(plane.dtype, torch.float64)
        torch.testing.assert_close(plane.flatten(1) @ plane.flatten(1).T / (81 * 128),
                                   torch.eye(2, dtype=torch.float64), rtol=0, atol=1e-12)
        self.assertTrue(torch.equal(plane, plane64((81, 128), 2701)))

    def test_nine_starts_have_zero_and_signed_pairs_on_two_planes(self):
        offsets = screen_offsets((81, 128), protocol())
        self.assertEqual(tuple(offsets.shape), (9, 81, 128))
        self.assertTrue(torch.equal(offsets[0], torch.zeros_like(offsets[0])))
        for index in (1, 3, 5, 7):
            self.assertTrue(torch.equal(offsets[index], -offsets[index + 1]))
        self.assertFalse(torch.equal(offsets[1], offsets[5]))

    def test_encoder_and_zero_offsets_preserve_fp64(self):
        model = HyperloopTransformer(0).double().eval()
        digits = np.zeros((2, 81), dtype=np.int64)
        base = encode(model, digits)
        hidden = initial_states(model, digits, torch.zeros(3, 81, 128, dtype=torch.float64))
        self.assertEqual(hidden.dtype, torch.float64)
        torch.testing.assert_close(hidden, base[:, None].expand(-1, 3, -1, -1).flatten(0, 1), rtol=0, atol=0)

    def test_require_double_rejects_float_buffers_and_training_mode(self):
        model = HyperloopTransformer(0).double().eval()
        require_double(model)
        model.rope_cos = model.rope_cos.float()
        with self.assertRaisesRegex(ValueError, "FP64"):
            require_double(model)
        model.double().train()
        with self.assertRaisesRegex(ValueError, "evaluation mode"):
            require_double(model)

    def test_precision_setting_is_restored_after_model_imports(self):
        torch.set_float32_matmul_precision("high")
        configure_precision()
        self.assertEqual(torch.get_float32_matmul_precision(), "highest")
        self.assertFalse(torch.backends.cuda.matmul.allow_tf32)

    def test_trajectory_matches_direct_recurrence(self):
        model = HyperloopTransformer(0).double().eval()
        hidden = initial_states(model, np.zeros((1, 81), dtype=int), torch.zeros(3, 81, 128, dtype=torch.float64))
        expected, predictions = hidden.clone(), hidden.new_zeros(3, 81, 9)
        with torch.inference_mode():
            for _ in range(4):
                expected, predictions, logits = model.step(expected, predictions)
        result = trace_states(model, hidden, np.zeros((1, 81)), 4, 1, pairs=[(0, 1), (1, 2)])
        np.testing.assert_array_equal(result["final_board"], logits.argmax(-1).numpy())

    def test_duplicate_controls_allow_only_cpu_fp64_roundoff(self):
        class DriftingModel(ScriptedModel):
            def __init__(self, hidden_drift=0., prediction_drift=0., different_answer=False):
                super().__init__([0, 0])
                self.hidden_drift = hidden_drift
                self.prediction_drift = prediction_drift
                self.different_answer = different_answer

            def step(self, hidden, predictions):
                hidden, predictions, logits = super().step(hidden, predictions)
                hidden = hidden.clone()
                hidden[-1, 0, 0] += self.hidden_drift
                predictions[-1, 0, 0] += self.prediction_drift
                if self.different_answer:
                    logits[-1, :, 1] = 2
                return hidden, predictions, logits

        for dtype, arguments, should_pass in (
            (torch.float64, {"hidden_drift": 1e-14, "prediction_drift": 1e-14}, True),
            (torch.float64, {"hidden_drift": 1e-8}, False),
            (torch.float64, {"prediction_drift": 1e-8}, False),
            (torch.float64, {"different_answer": True}, False),
            (torch.float64, {"hidden_drift": float("nan")}, False),
            (torch.float32, {"hidden_drift": 1e-6}, False),
        ):
            with self.subTest(dtype=dtype, arguments=arguments):
                model = DriftingModel(**arguments)
                hidden = torch.zeros(2, 81, 128, dtype=dtype)
                if should_pass:
                    trace_states(model, hidden, np.zeros((1, 81)), 2, 1, pairs=[(0, 1)])
                else:
                    with self.assertRaisesRegex(ValueError, "controls diverged"):
                        trace_states(model, hidden, np.zeros((1, 81)), 2, 1, pairs=[(0, 1)])

    def test_returning_solution_uses_last_change_and_counts_regression(self):
        result = trace_states(ScriptedModel([0, 1, 0, 0, 0]), torch.zeros(1, 81, 128, dtype=torch.float64),
                              np.zeros((1, 81)), 5, 2)
        self.assertEqual(result["first_correct"].item(), 1)
        self.assertEqual(result["last_change"].item(), 3)
        self.assertEqual(result["regressions"].item(), 1)
        self.assertTrue(result["successful"].item())

    def test_unchanging_wrong_answer_is_not_success(self):
        result = trace_states(ScriptedModel([1] * 5), torch.zeros(1, 81, 128, dtype=torch.float64),
                              np.zeros((1, 81)), 5, 2)
        self.assertTrue(result["confirmed_answer"].item())
        self.assertFalse(result["successful"].item())

    def test_correct_but_recent_change_is_not_confirmed(self):
        result = trace_states(ScriptedModel([1, 1, 1, 0, 0]), torch.zeros(1, 81, 128, dtype=torch.float64),
                              np.zeros((1, 81)), 5, 2)
        self.assertTrue(result["final_correct"].item())
        self.assertFalse(result["successful"].item())

    def test_earlier_nonfinite_output_excludes_apparent_recovery(self):
        result = trace_states(ScriptedModel([0] * 5, nonfinite_at=1), torch.zeros(1, 81, 128, dtype=torch.float64),
                              np.zeros((1, 81)), 5, 2)
        self.assertTrue(result["ever_nonfinite"].item())
        self.assertFalse(result["successful"].item())

    def test_cast_inside_recurrence_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "numerical precision"):
            trace_states(ScriptedModel([0], bad_dtype=True), torch.zeros(1, 81, 128, dtype=torch.float64),
                         np.zeros((1, 81)), 1, 0)

    def test_selection_uses_successful_mean_with_index_tiebreak(self):
        rows = np.arange(5)
        indices = np.array([50, 40, 30, 20, 10])
        fields = {"successful": np.ones((5, 9), dtype=bool),
                  "last_change": np.repeat([[10], [100], [100], [1000], [0]], 9, axis=1)}
        fields["successful"][3, 1] = False
        fields["successful"][4, 0] = False
        cases = select_cases(rows, indices, fields)
        self.assertEqual(cases["slow"]["index"], 30)
        self.assertEqual(cases["easy"]["index"], 50)

    def test_selection_does_not_fallback_to_failed_puzzles(self):
        fields = {"successful": np.array([[True] * 9, [False] * 9]), "last_change": np.ones((2, 9))}
        with self.assertRaisesRegex(ValueError, "Fewer than two"):
            select_cases(np.arange(2), np.arange(2), fields)

    def test_equal_time_selection_still_returns_distinct_puzzles(self):
        fields = {"successful": np.ones((3, 9), dtype=bool), "last_change": np.ones((3, 9))}
        cases = select_cases(np.arange(3), np.array([3, 2, 1]), fields)
        self.assertEqual(cases["easy"]["index"], 1)
        self.assertEqual(cases["slow"]["index"], 2)

    def test_double_grid_resumes_and_checks_configuration(self):
        model = HyperloopTransformer(0).double().eval()
        puzzle = {"digits": [0] * 81, "targets": [0] * 81}
        arguments = dict(resolution=9, center=[0., 0.], width=.3, horizon=4,
                         confirmation_window=1, chunk_rows=4, plane_seed=2701)
        with tempfile.TemporaryDirectory() as temporary:
            first = grid_field64(model, puzzle, Path(temporary), **arguments)
            second = grid_field64(model, puzzle, Path(temporary), **arguments)
            for name in first:
                np.testing.assert_array_equal(first[name], second[name])
            self.assertEqual(first["final_board"].shape, (9, 9, 81))
            with self.assertRaises(ValueError):
                grid_field64(model, puzzle, Path(temporary), **{**arguments, "width": .2})

    def test_tampered_chunk_fails_checksum(self):
        from looping.basin_diagnostics.analysis import save_unit
        with tempfile.TemporaryDirectory() as temporary:
            save_unit(Path(temporary), "chunk", {"value": np.ones(2)}, {})
            (Path(temporary) / "chunk.npz").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                load_arrays(temporary, "chunk")

    def test_all_unsuccessful_summary_is_finite_and_explicit(self):
        field = {"successful": np.zeros((3, 3), dtype=bool), "last_change": np.ones((3, 3)),
                 "final_correct": np.zeros((3, 3), dtype=bool), "confirmed_answer": np.ones((3, 3), dtype=bool),
                 "ever_nonfinite": np.zeros((3, 3), dtype=bool)}
        summary = summarize_grid(field)
        self.assertIsNone(summary["mean_successful_time"])
        self.assertEqual(summary["successful_fraction"], 0)

    def test_screen_preserves_puzzle_and_start_axes_and_resumes(self):
        model = HyperloopTransformer(0).double().eval()
        settings = {**protocol(), "screen_batch_puzzles": 2, "horizon": 4, "confirmation_window": 1}
        data = {"digits": np.array([[0] * 81, [1] * 81, [2] * 81]),
                "targets": np.zeros((3, 81), dtype=int), "indices": np.array([10, 20, 30])}
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            first = screen(model, data, np.arange(3), folder, settings)
            second = screen(model, data, np.arange(3), folder, settings)
            self.assertEqual(first["last_change"].shape, (3, 9))
            self.assertEqual(first["final_board"].shape, (3, 9, 81))
            for name in first:
                np.testing.assert_array_equal(first[name], second[name])

    def test_render_pipeline_writes_zoom_and_orientation_figures(self):
        model = HyperloopTransformer(0).double().eval()
        puzzle = {"digits": [0] * 81, "targets": [0] * 81, "index": 123,
                  "model_key": "synthetic", "case": "easy", "mean_last_change": 1.}
        settings = {**protocol(), "resolution": 9, "levels": 2, "horizon": 4,
                    "confirmation_window": 1, "chunk_rows": 4, "orientation_control_resolution": 9}
        with tempfile.TemporaryDirectory() as temporary:
            result = render_case(model, puzzle, Path(temporary), settings)
            self.assertEqual(len(result["levels"]), 2)
            for name in ("recursive_fp64.png", "orientation_control.png"):
                self.assertGreater((Path(temporary) / name).stat().st_size, 1000)

    def test_plot_handles_wrong_and_unsettled_pixels(self):
        import matplotlib.pyplot as plt
        from looping.basin_diagnostics.slow_puzzles.plots import panel
        field = {"successful": np.ones((3, 3), dtype=bool), "last_change": np.arange(9).reshape(3, 3) + 1,
                 "confirmed_answer": np.ones((3, 3), dtype=bool), "final_correct": np.ones((3, 3), dtype=bool),
                 "ever_nonfinite": np.zeros((3, 3), dtype=bool)}
        field["successful"][0, 0] = field["final_correct"][0, 0] = False
        field["successful"][0, 1] = field["confirmed_answer"][0, 1] = False
        figure, axis = plt.subplots()
        panel(axis, field, {"center": [0, 0], "width": .3})
        self.assertEqual(len(axis.images), 2)
        self.assertEqual(axis.images[1].get_array()[0, 0, 3], 1)
        self.assertEqual(axis.images[1].get_array()[0, 1, 3], 0)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
