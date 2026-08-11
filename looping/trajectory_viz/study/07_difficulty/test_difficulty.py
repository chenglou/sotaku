"""Focused tests for difficulty-geometry analysis."""

import importlib
import unittest

import numpy as np


analysis = importlib.import_module(
    "looping.trajectory_viz.study.07_difficulty.analyze_difficulty"
)


class DifficultyAnalysisTests(unittest.TestCase):
    def test_stratified_splits_are_balanced_and_disjoint(self):
        buckets = [bucket for bucket in ("0", "1-2", "3-10", "11-50", "51+") for _ in range(6)]
        splits = analysis.make_stratified_splits(buckets, per_split=2)
        self.assertTrue(all(len(indices) == 10 for indices in splits.values()))
        self.assertEqual(len(set(np.concatenate(list(splits.values())).tolist())), 30)
        for indices in splits.values():
            self.assertEqual({buckets[index] for index in indices}, {"0", "1-2", "3-10", "11-50", "51+"})

    def test_ranks_average_ties(self):
        ranks = analysis.rank_values([10, 20, 20, 40])
        np.testing.assert_allclose(ranks, [0, 1.5, 1.5, 3])

    def test_ridge_recovers_linear_target(self):
        features = np.arange(20, dtype=np.float64)[:, None]
        targets = 3.0 * features[:, 0] + 2.0
        model = analysis.fit_standardized_ridge(features, targets, alpha=1e-8)
        predictions = analysis.predict_ridge(model, features)
        self.assertGreater(analysis.regression_metrics(targets, predictions)["r_squared"], 0.999999)

    def test_pca_is_fit_only_to_supplied_discovery_rows(self):
        discovery = np.asarray([[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0]])
        mean, basis = analysis.fit_pca(discovery, component_count=1)
        heldout = np.asarray([[0.0, 1000.0]])
        self.assertLess(abs(analysis.project(heldout, mean, basis)[0, 0]), 1e-8)

    def test_categorical_metrics_uses_five_class_argmax(self):
        labels = np.arange(5)
        scores = np.eye(5)
        metrics = analysis.categorical_metrics(labels, scores)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
