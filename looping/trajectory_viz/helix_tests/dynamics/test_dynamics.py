import math
import unittest

import numpy as np
import torch

from looping.trajectory_viz.helix_tests.dynamics.analyze_dynamics import (
    _cycle_catalog,
    _digit_geometry_record,
    _score_predictive_models,
    _transition_code,
    _linear_r2,
    phase_geometry_metrics,
    stratified_puzzle_split,
)


class DynamicsTest(unittest.TestCase):
    def test_stratified_split_keeps_every_bucket_in_both_sets(self):
        buckets = ["easy"] * 4 + ["hard"] * 4
        fit, heldout = stratified_puzzle_split(buckets)
        self.assertEqual(fit, [0, 2, 4, 6])
        self.assertEqual(heldout, [1, 3, 5, 7])
        self.assertEqual({buckets[index] for index in fit}, {"easy", "hard"})
        self.assertEqual({buckets[index] for index in heldout}, {"easy", "hard"})

    def test_phase_metrics_recognize_a_full_turn_and_shuffle_breaks_linearity(self):
        time = torch.linspace(0, 1, 65)
        angle = 2 * math.pi * time
        points = torch.stack(
            [torch.cos(angle), torch.sin(angle), time], dim=-1
        ).view(1, 1, 65, 3)
        ordered = phase_geometry_metrics(points, state_residual=False)
        permutation = torch.randperm(65, generator=torch.Generator().manual_seed(7))
        shuffled = phase_geometry_metrics(points[:, :, permutation], state_residual=False)
        self.assertAlmostEqual(float(ordered["turns_net"]), 1.0, places=3)
        self.assertGreater(float(ordered["phase_linearity_r2"]), 0.99)
        self.assertLess(
            float(shuffled["phase_linearity_r2"]),
            float(ordered["phase_linearity_r2"]) - 0.3,
        )

    def test_prepared_state_control_keeps_the_ordered_origin(self):
        time = torch.linspace(0, 1, 19)
        angle = 1.5 * math.pi * time
        points = torch.stack(
            [4 + torch.cos(angle), -3 + torch.sin(angle), time], dim=-1
        ).view(1, 1, 19, 3)
        ordered = phase_geometry_metrics(points, state_residual=True)
        prepared = phase_geometry_metrics(
            points[:, :, 1:-1],
            state_residual=True,
            prepared_state_residual=True,
        )
        self.assertAlmostEqual(
            float(ordered["turns_net"]), float(prepared["turns_net"]), places=6
        )
        self.assertAlmostEqual(
            float(ordered["phase_linearity_r2"]),
            float(prepared["phase_linearity_r2"]),
            places=6,
        )

    def test_predictive_rotation_beats_translation_on_synthetic_helix(self):
        time = torch.linspace(0, 1, 65)
        angle = 2 * math.pi * time
        base = torch.stack(
            [torch.cos(angle), torch.sin(angle), 0.3 * time], dim=-1
        )
        offsets = torch.linspace(-0.2, 0.2, 8).view(2, 4, 1, 1)
        points = base.view(1, 1, 65, 3) + offsets
        mask = torch.ones(2, 4, dtype=torch.bool)
        scored = _score_predictive_models(points, points.clone(), mask, mask)
        self.assertEqual(scored["rotation_cycles"], 1.0)
        skill = scored["per_puzzle"]["rotation_skill_vs_translation"]
        self.assertTrue(torch.all(skill > 0.9))

    def test_linear_r2_fits_each_cell_with_its_own_x_offset(self):
        base = torch.linspace(-1, 1, 17)
        x = torch.stack([base, base + 100]).view(1, 2, 17)
        y = 3 * x + torch.tensor([[[7.0], [-11.0]]])
        result = _linear_r2(x, y)
        self.assertTrue(torch.all(result > 0.99999))

    def test_exact_cycle_catalog_and_natural_digit_circle(self):
        catalog = _cycle_catalog()
        self.assertEqual(len(catalog["cycles"]), 20160)
        angle = np.arange(9) * 2 * np.pi / 9
        centroids = np.stack([np.cos(angle), np.sin(angle)], axis=1)
        result = _digit_geometry_record(centroids, centroids, catalog)
        self.assertEqual(result["natural_cycle_rank"], 1)
        self.assertLessEqual(result["natural_cycle_exact_p"], 1 / 1000)

    def test_transition_codes(self):
        correct = torch.tensor([[False, False, True, False, True]])
        self.assertEqual(_transition_code(correct).tolist(), [[0, 1, 2, 1]])


if __name__ == "__main__":
    unittest.main()
