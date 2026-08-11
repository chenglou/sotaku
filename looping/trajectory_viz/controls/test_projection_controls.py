import unittest

import torch

from looping.trajectory_viz.controls.analyze_projection_controls import (
    _fit_pca,
    _normalize_rows,
    _path_metrics,
    _project,
)


class ProjectionControlsTest(unittest.TestCase):
    def test_pca_recovers_planar_curve(self):
        values = torch.tensor([[i, i * i, 0.0, 0.0] for i in range(8)])
        mean, basis, explained = _fit_pca(values)
        projected = _project(values, mean, basis)
        self.assertEqual(projected.shape, (8, 3))
        self.assertAlmostEqual(explained, 1.0, places=6)

    def test_normalization_removes_magnitude(self):
        values = torch.tensor([[[3.0, 4.0], [6.0, 8.0]]])
        normalized = _normalize_rows(values)
        self.assertTrue(torch.allclose(normalized.norm(dim=-1), torch.ones(1, 2)))
        self.assertTrue(torch.allclose(normalized[:, 0], normalized[:, 1]))

    def test_straight_path_is_efficient(self):
        points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        metrics = _path_metrics(points)
        self.assertAlmostEqual(metrics["path_efficiency"], 1.0)
        self.assertAlmostEqual(metrics["turn_cosine"], 1.0)


if __name__ == "__main__":
    unittest.main()
