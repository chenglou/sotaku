"""Audit arithmetic, nested sampling, and parent-artifact validation."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from looping.basin_diagnostics.analysis import plane_basis
from looping.basin_diagnostics.recursive.deep.run import comparison_stats, initial_grid, protocol, trace_initial, validated_json
from looping.basin_diagnostics.recursive.run import grid_rows, next_window, trace_grid_chunk
from looping.hyperloop.model import HyperloopTransformer
from runtime_utils import file_sha256


class DeepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_new_grid_contains_previous_resolution_and_audit_points(self):
        dense = grid_rows([.132, -.195], .01875, 401, 0, 401).reshape(403, 401, 2)[1:-1]
        old = grid_rows([.132, -.195], .01875, 201, 0, 201).reshape(203, 201, 2)[1:-1]
        audit = grid_rows([.132, -.195], .01875, 21, 0, 21).reshape(23, 21, 2)[1:-1]
        np.testing.assert_allclose(dense[::2, ::2], old, rtol=0, atol=1e-16)
        np.testing.assert_allclose(dense[::20, ::20], audit, rtol=0, atol=1e-16)

    def test_audit_matches_frozen_fp32_recurrence(self):
        model = HyperloopTransformer(0).eval()
        puzzle = {"digits": [0] * 81, "targets": [0] * 81}
        coordinates = grid_rows([.1, -.1], .02, 3, 0, 3).reshape(5, 3, 2)[1:-1].reshape(-1, 2)
        expected = trace_grid_chunk(model, puzzle, coordinates, plane_basis((81, 128), 2701), 3, 3, 4, 2)
        initial = initial_grid(model, puzzle, coordinates, dtype=torch.float32)
        actual = trace_initial(model, initial, puzzle, 3, 4, 2)
        for name in actual:
            np.testing.assert_array_equal(actual[name], expected[name])

    def test_double_precision_path_and_duplicate_controls(self):
        model = HyperloopTransformer(0).eval()
        puzzle = {"digits": [0] * 81, "targets": [0] * 81}
        coordinates = np.array([[0., 0.]])
        initial = initial_grid(model, puzzle, coordinates, dtype=torch.float64)
        self.assertEqual(initial.dtype, torch.float64)
        torch.testing.assert_close(initial[0], initial[-1], atol=0, rtol=0)
        result = trace_initial(copy.deepcopy(model).double(), initial, puzzle, 1, 4, 2)
        self.assertEqual(result["last_change"].shape, (1, 1))

    def test_comparison_preserves_constant_case_without_nan(self):
        first = {"last_change": np.ones((3, 3)), "confirmed": np.ones((3, 3), dtype=bool),
                 "final_board": np.zeros((3, 3, 81))}
        result = comparison_stats(first, first)
        self.assertEqual(result["same_last_change_fraction"], 1)
        self.assertEqual(result["median_absolute_time_difference"], 0)
        self.assertIsNone(result["confirmed_time_correlation"])

    def test_all_deep_zoom_boxes_remain_nested(self):
        settings = protocol()
        center, width = [.132, -.195], .01875
        field = np.indices((401, 401)).sum(0) % 5
        for _ in range(settings["levels"]):
            child = next_window(field, np.ones_like(field, dtype=bool), center, width, 4)
            self.assertTrue(all(abs(a - b) + child["width"] <= width + 1e-15 for a, b in zip(center, child["center"])))
            center, width = child["center"], child["width"]

    def test_changed_parent_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.json"
            path.write_text('{"value": 1}')
            checksum = file_sha256(path)
            self.assertEqual(validated_json(path, checksum), {"value": 1})
            path.write_text('{"value": 2}')
            with self.assertRaises(ValueError):
                validated_json(path, checksum)


if __name__ == "__main__":
    unittest.main()
