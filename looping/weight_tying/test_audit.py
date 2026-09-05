"""Check validation replay without assuming the full set has the same ordering."""

import json
import unittest

import numpy as np

from looping.weight_tying.audit import validation_positions, validation_replay
from looping.weight_tying.common import DIRECTORY
from runtime_utils import file_sha256


class ArtifactAuditTests(unittest.TestCase):
    def test_included_test_puzzles_match_the_frozen_manifest(self):
        manifest = json.loads((DIRECTORY / "results/data_manifest.json").read_text())
        path = DIRECTORY / "test_data/holdout.npz"
        self.assertEqual(file_sha256(path), manifest["files"]["holdout.npz"])
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(data["digits"].shape, (10000, 81))
            self.assertEqual(data["targets"].shape, (10000, 81))
            _, counts = np.unique(data["labels"], return_counts=True)
            np.testing.assert_array_equal(counts, [2500] * 4)

    def test_maps_by_original_index_and_checks_puzzle_contents(self):
        development = {"indices": np.array([30, 10, 20]), "digits": np.array([[3], [1], [2]]),
                       "targets": np.array([[6], [4], [5]]), "labels": np.array(["a", "b", "c"])}
        validation = {key: value[[2, 0]] for key, value in development.items()}
        np.testing.assert_array_equal(validation_positions(development, validation), [2, 0])
        validation["digits"][0, 0] = 9
        with self.assertRaises(AssertionError):
            validation_positions(development, validation)

    def test_missing_and_duplicate_indices_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            validation_positions({"indices": np.array([1])}, {"indices": np.array([2])})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validation_positions({"indices": np.array([1, 1])}, {"indices": np.array([1])})

    def test_replay_reports_differences_instead_of_hiding_them(self):
        predictions = {"solved_16": np.array([True, False, True])}
        result = validation_replay(predictions, np.array([2, 0]), {"16": {"solved": 1}})
        self.assertEqual(result["16"], {"training_probe_solved": 1,
                                      "full_evaluation_subset_solved": 2, "difference": 1})
