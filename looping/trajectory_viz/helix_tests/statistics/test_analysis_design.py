import unittest

import numpy as np

from looping.trajectory_viz.helix_tests.statistics.analyze_number_helix import (
    CERTAINTY_BIN_EDGES,
    SNAPSHOT_ITERATIONS,
    _fixed_bin_one_hot,
    build_nuisance_designs,
)
from looping.trajectory_viz.helix_tests.statistics.statistics_core import (
    evaluate_digit_geometry,
)


class FixedBinDesignTest(unittest.TestCase):
    def test_fixed_bin_one_hot_includes_edges_and_extremes(self):
        encoded = _fixed_bin_one_hot(
            np.asarray((-100.0, -1.0, 0.0, 1.0, 100.0)),
            (-1.0, 0.0, 1.0),
        )
        np.testing.assert_array_equal(encoded.sum(axis=1), np.ones(5))
        np.testing.assert_array_equal(encoded.argmax(axis=1), (0, 1, 2, 3, 3))

    def test_fixed_bin_one_hot_rejects_bad_input(self):
        with self.assertRaisesRegex(ValueError, "finite vector"):
            _fixed_bin_one_hot((0.0, np.nan), (0.5,))
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _fixed_bin_one_hot((0.0, 1.0), (0.5, 0.5))

    def test_pooled_certainty_bins_extend_iteration_and_position_controls(self):
        row_count = 4
        metadata = {
            "puzzle_index": np.asarray((0, 0, 1, 1)),
            "cell_index": np.asarray((0, 1, 0, 1)),
            "rating_bucket_index": np.asarray((0, 0, 1, 1)),
            "iteration": np.asarray(
                (
                    SNAPSHOT_ITERATIONS[0],
                    SNAPSHOT_ITERATIONS[1],
                    SNAPSHOT_ITERATIONS[0],
                    SNAPSHOT_ITERATIONS[1],
                )
            ),
            "max_confidence": np.asarray((0.4, 0.7, 0.9, 0.999)),
            "top1_top2_margin": np.asarray((0.1, 0.7, 3.0, 10.0)),
            "true_probability": np.asarray((0.3, 0.6, 0.85, 0.995)),
            "target_minus_max_wrong_logit_margin": np.asarray(
                (-10.0, -3.0, 0.5, 10.0)
            ),
        }
        designs = build_nuisance_designs(
            metadata, np.asarray((40.0, 50.0)), pooled=True
        )

        context_columns = 2
        iteration_columns = context_columns + len(SNAPSHOT_ITERATIONS) - 1
        position_columns = iteration_columns + 80
        decision_bins = (
            len(CERTAINTY_BIN_EDGES["max_confidence"])
            + len(CERTAINTY_BIN_EDGES["decision_margin"])
            + 2
        )
        target_bins = (
            len(CERTAINTY_BIN_EDGES["true_probability"])
            + len(CERTAINTY_BIN_EDGES["target_margin"])
            + 2
        )
        self.assertEqual(designs["context"].shape, (row_count, context_columns))
        self.assertEqual(
            designs["position"].shape, (row_count, position_columns)
        )
        self.assertEqual(
            designs["decision_adjusted"].shape,
            (row_count, position_columns + decision_bins),
        )
        self.assertEqual(
            designs["target_adjusted"].shape,
            (row_count, position_columns + decision_bins + target_bins),
        )
        np.testing.assert_array_equal(
            designs["decision_adjusted"][:, :position_columns],
            designs["position"],
        )
        np.testing.assert_array_equal(
            designs["target_adjusted"][:, : position_columns + decision_bins],
            designs["decision_adjusted"],
        )
        self.assertTrue(
            np.all(
                np.isin(
                    designs["target_adjusted"],
                    (0.0, 1.0, 40.0 / 81.0, 50.0 / 81.0),
                )
            )
        )

    def test_unseen_heldout_bin_level_stays_finite(self):
        puzzle_ids = np.repeat(np.arange(10), 2)
        fold_ids = np.repeat(np.arange(10) % 5, 2)
        digits = np.tile(np.arange(10) % 9, 2)
        response = np.column_stack(
            (
                np.linspace(-1.0, 1.0, len(puzzle_ids)),
                np.cos(np.arange(len(puzzle_ids))),
            )
        )
        # The last column is present only in fold zero's held-out rows.  Its
        # coefficient is unidentifiable in that fold's training set, which is
        # the conservative unseen-bin case for full one-hot certainty coding.
        base_design = np.column_stack(
            (
                np.ones(len(puzzle_ids)),
                (fold_ids == 0).astype(np.float64),
            )
        )
        result = evaluate_digit_geometry(
            response,
            digits,
            puzzle_ids,
            base_design=base_design,
            fold_ids=fold_ids,
            models=(),
        )
        self.assertTrue(np.isfinite(result["models"]["base"]["sse"]))


if __name__ == "__main__":
    unittest.main()
