import unittest

import torch

from looping.eval_margin_floor_calibration import (
    _required_horizons,
    summarize_margin_floors,
)


class MarginFloorCalibrationTest(unittest.TestCase):
    def test_required_horizons_include_anchor_and_complete_recheck_window(self):
        horizons = _required_horizons((32,), (16,))
        self.assertIn(48, horizons)
        self.assertEqual(
            set(range(65, 81)),
            horizons - {48},
        )

    def test_floor_summary_reports_activation_and_hinge(self):
        summary = summarize_margin_floors(
            torch.tensor([-1.0, 0.5, 2.0]),
            (0.0, 1.0),
        )
        self.assertAlmostEqual(
            summary["0.0"]["active_fraction"],
            1 / 3,
        )
        self.assertAlmostEqual(
            summary["0.0"]["mean_hinge_loss"],
            1 / 3,
        )
        self.assertAlmostEqual(
            summary["1.0"]["active_fraction"],
            2 / 3,
        )
        self.assertAlmostEqual(
            summary["1.0"]["mean_hinge_loss"],
            2.5 / 3,
        )


if __name__ == "__main__":
    unittest.main()
