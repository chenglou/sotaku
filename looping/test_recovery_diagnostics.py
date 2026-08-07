import unittest

import torch

from looping.eval_recovery_diagnostics import (
    MODEL_PRESETS,
    QUALITY_BUCKETS,
    build_recovery_curves,
    capture_trajectory,
    classify_wrong_counts,
    summarize_transition,
)
from looping.modal_recovery_diagnostics import PRESET_NAMES
from stabilize.exp_testbed_20k import SudokuTransformer


def _fake_snapshot(wrong_counts, predictions):
    count = len(wrong_counts)
    return {
        "wrong_counts": torch.tensor(wrong_counts),
        "predictions": torch.tensor(predictions, dtype=torch.uint8),
        "cross_entropy": torch.arange(count, dtype=torch.float32),
        "correct_probability": torch.full((count,), 0.5),
        "mean_target_margin": torch.zeros(count),
        "minimum_target_margin": torch.linspace(-1, 1, count),
        "entropy": torch.ones(count),
        "state_rms": torch.arange(1, count + 1, dtype=torch.float32),
        "state_direction_cosine_to_reference": torch.linspace(
            1,
            0.5,
            count,
        ),
        "directional_wrong_counts": torch.tensor(wrong_counts),
        "directional_minimum_target_margin": torch.linspace(-1, 1, count),
    }


class RecoveryDiagnosticsTest(unittest.TestCase):
    def test_modal_and_evaluator_presets_match(self):
        self.assertEqual(set(PRESET_NAMES), set(MODEL_PRESETS))

    def test_switch_trajectory_marks_internal_checkpoints_as_trusted(self):
        for model_config in MODEL_PRESETS["delayed_switch_trajectory"]:
            self.assertTrue(model_config["trusted_full_checkpoint"])

    def test_quality_buckets_cover_wrong_count_boundaries(self):
        wrong_counts = torch.tensor([0, 1, 2, 3, 10, 11, 81])
        self.assertEqual(
            classify_wrong_counts(wrong_counts).tolist(),
            [0, 1, 1, 2, 2, 3, 3],
        )
        self.assertEqual(
            [name for _, _, name in QUALITY_BUCKETS],
            ["solved", "near_miss", "semi_bad", "bad"],
        )

    def test_transition_summary_separates_recovery_and_regression(self):
        empty_mask = torch.ones(4, 4, dtype=torch.bool)
        start = _fake_snapshot(
            [0, 1, 4, 4],
            [
                [0, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
        )
        current = _fake_snapshot(
            [1, 0, 2, 4],
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 1],
                [1, 1, 1, 1],
            ],
        )
        summary = summarize_transition(
            start,
            current,
            empty_mask,
            torch.ones(4, dtype=torch.bool),
            torch.tensor([0.9, 0.8, 0.7, 0.6]),
        )
        self.assertEqual(summary["count"], 4)
        self.assertAlmostEqual(summary["puzzle_accuracy"], 0.25)
        self.assertAlmostEqual(summary["improved_fraction"], 0.5)
        self.assertAlmostEqual(summary["unchanged_fraction"], 0.25)
        self.assertAlmostEqual(summary["worsened_fraction"], 0.25)
        self.assertAlmostEqual(summary["solved_retention"], 0.0)
        self.assertAlmostEqual(summary["unsolved_recovery"], 1 / 3)
        self.assertAlmostEqual(summary["blank_cell_accuracy"], 9 / 16)

    def test_small_cpu_trajectory_covers_the_complete_recovery_window(self):
        torch.manual_seed(29)
        model = SudokuTransformer(unique_layers=1).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        targets = torch.zeros(1, 81, dtype=torch.long)
        empty_mask = torch.ones(1, 81, dtype=torch.bool)
        snapshots, direction_cosines = capture_trajectory(
            model,
            inputs,
            targets,
            empty_mask,
            horizons=(1,),
            batch_size=1,
            device=torch.device("cpu"),
        )
        self.assertEqual(set(snapshots), set(range(1, 18)))
        self.assertEqual(set(direction_cosines[1]), set(range(17)))
        torch.testing.assert_close(
            direction_cosines[1][0],
            torch.ones(1),
        )
        curves = build_recovery_curves(
            snapshots,
            direction_cosines,
            empty_mask,
            horizons=(1,),
        )
        self.assertEqual(len(curves["1"]["curve"]), 17)


if __name__ == "__main__":
    unittest.main()
