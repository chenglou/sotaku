"""Focused checks for grid coordinates, row halos, and nested map rendering."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from looping.basin_diagnostics.recursive.run import grid_field, grid_rows, neighbor_distance, next_window, rectangle_sums
from looping.hyperloop.model import HyperloopTransformer


class RecursiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_chunk_coordinates_overlap_and_include_center(self):
        first = grid_rows([0., 0.], .3, 201, 0, 8).reshape(10, 201, 2)
        second = grid_rows([0., 0.], .3, 201, 8, 8).reshape(10, 201, 2)
        np.testing.assert_array_equal(first[-2:], second[:2])
        center = grid_rows([0., 0.], .3, 201, 96, 8).reshape(10, 201, 2)[5, 100]
        np.testing.assert_allclose(center, 0, atol=1e-16)

    def test_rectangle_sums(self):
        array = np.arange(25).reshape(5, 5)
        result = rectangle_sums(array, 2, 3)
        for row in range(4):
            for col in range(3):
                self.assertEqual(result[row, col], array[row:row + 2, col:col + 3].sum())

    def test_zoom_is_inside_parent_and_tracks_variation(self):
        array = np.zeros((33, 33), dtype=int)
        array[3:11, 3:11] = np.indices((8, 8)).sum(0) % 2 * 10
        result = next_window(array, np.ones_like(array, dtype=bool), [0., 0.], 1., 4)
        self.assertEqual(result["width"], .25)
        self.assertGreater(result["score"], 0)
        self.assertTrue(all(abs(value) + result["width"] <= 1 for value in result["center"]))
        self.assertLess(result["center"][0], 0)
        self.assertLess(result["center"][1], 0)

    def test_flat_or_unconfirmed_maps_use_center(self):
        values = np.ones((33, 33), dtype=int)
        for valid in (np.zeros_like(values, dtype=bool), np.ones_like(values, dtype=bool)):
            result = next_window(values, valid, [.2, -.1], .3, 4)
            self.assertEqual(result["center"], [.2, -.1])
            self.assertEqual(result["score"], 0)

    def test_paper_distance_uses_only_adjacent_boards(self):
        boards = torch.zeros(9, 81, dtype=torch.long)
        boards[4, 0] = 8
        values = neighbor_distance(boards, 3, 3)
        np.testing.assert_array_equal(values.numpy(), [[0, 8, 0], [8, 8, 8], [0, 8, 0]])

    def test_small_model_grid_and_resume(self):
        model = HyperloopTransformer(0).eval()
        puzzle = {"digits": [0] * 81, "targets": [0] * 81}
        with tempfile.TemporaryDirectory() as temporary:
            settings = dict(resolution=9, center=[0., 0.], width=.1, horizon=4, confirmation_window=2, chunk_rows=4)
            first = grid_field(model, puzzle, temporary, **settings)
            second = grid_field(model, puzzle, temporary, **settings)
            self.assertEqual(first["last_change"].shape, (9, 9))
            self.assertEqual(first["final_board"].shape, (9, 9, 81))
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])

    def test_plot_with_local_range_and_gray_pixels(self):
        import matplotlib.pyplot as plt
        from looping.basin_diagnostics.recursive.plots import panel
        field = {"last_change": np.arange(81).reshape(9, 9) % 8 + 3548,
                 "confirmed": np.ones((9, 9), dtype=bool), "separation": np.zeros((9, 9))}
        field["confirmed"][0, 0] = False
        figure, axis = plt.subplots()
        image = panel(axis, field, {"center": [0, 0], "width": .3})
        self.assertEqual(image.get_clim(), (3548, 3555))
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "map.png"
            figure.savefig(target)
            self.assertGreater(target.stat().st_size, 1000)
        plt.close(figure)


if __name__ == "__main__":
    unittest.main()
