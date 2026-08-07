import unittest

import torch

from looping.eval_trajectory_geometry import (
    explained_by_basis,
    feature_spectrum,
    fit_principal_basis,
    participation_ratio,
    trajectory_shape_metrics,
    validate_windows,
)


class TrajectoryGeometryTest(unittest.TestCase):
    def test_window_validation(self):
        validate_windows((16, 128), 16)
        with self.assertRaises(ValueError):
            validate_windows((), 16)
        with self.assertRaises(ValueError):
            validate_windows((128, 16), 16)
        with self.assertRaises(ValueError):
            validate_windows((16,), 1)

    def test_participation_ratio_recovers_known_rank(self):
        basis = torch.eye(3, 8)
        matrix = torch.cat([basis, -basis], dim=0)
        self.assertAlmostEqual(
            participation_ratio(matrix, center=True),
            3.0,
            places=5,
        )

    def test_feature_spectrum_identifies_two_dimensional_updates(self):
        matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ]
        )
        result = feature_spectrum(matrix, (1, 2, 4))
        self.assertAlmostEqual(result["effective_rank"], 2.0, places=5)
        self.assertAlmostEqual(
            result["explained_variance"]["2"],
            1.0,
            places=5,
        )

    def test_basis_explains_heldout_points_in_same_subspace(self):
        generator = torch.Generator().manual_seed(7)
        shared_basis = torch.randn(2, 10, generator=generator)
        training = torch.randn(20, 2, generator=generator) @ shared_basis
        heldout = torch.randn(10, 2, generator=generator) @ shared_basis
        mean, basis = fit_principal_basis(training, 4, center=False)
        explained = explained_by_basis(
            heldout,
            mean,
            basis,
            (1, 2, 4),
        )
        self.assertGreater(explained["2"], 0.999)

    def test_straight_constant_updates_have_zero_curvature(self):
        updates = torch.ones(3, 4, 2, 5)
        result = trajectory_shape_metrics(updates)
        self.assertAlmostEqual(
            result["consecutive_update_cosine_mean"],
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            result["relative_acceleration_mean"],
            0.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
