import sys
import unittest
from pathlib import Path

import torch


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from analysis_core import (  # noqa: E402
    TASKS,
    build_representations,
    classification_metrics,
    compute_cell_roles,
    exact_iteration_order_test,
    fit_ridge_readout,
    pairwise_centroid_geometry,
    regression_metrics,
    shuffle_within_puzzles,
    stratified_three_way_split,
)


class CellRoleAnalysisTests(unittest.TestCase):
    def test_stratified_split_keeps_every_bucket_in_every_split(self):
        buckets = [bucket for bucket in ("easy", "hard") for _ in range(12)]
        splits = stratified_three_way_split(buckets, seed=17)
        seen = []
        for indices in splits.values():
            self.assertEqual(len(indices), 8)
            selected = [buckets[index] for index in indices]
            self.assertEqual(selected.count("easy"), 4)
            self.assertEqual(selected.count("hard"), 4)
            seen.extend(indices)
        self.assertEqual(sorted(seen), list(range(24)))

    def test_candidate_size_uses_input_row_column_and_box(self):
        solved = (
            "123456789"
            "456789123"
            "789123456"
            "214365897"
            "365897214"
            "897214365"
            "531642978"
            "642978531"
            "978531642"
        )
        one_blank = "." + solved[1:]
        roles = compute_cell_roles([one_blank, "." * 81])
        self.assertEqual(roles["candidate_size"][0, 0].item(), 1)
        self.assertEqual(roles["candidate_size"][1, 0].item(), 9)
        self.assertFalse(roles["clue"][0, 0])
        self.assertTrue(roles["clue"][0, 1])
        self.assertEqual(roles["box"][0, 40].item(), 4)

    def test_input_residual_is_zero_at_iteration_zero_on_unseen_cells(self):
        input_symbols = torch.arange(10).repeat(6, 9)[:, :81]
        embeddings = torch.randn(10, 16, generator=torch.Generator().manual_seed(3))
        states = embeddings[input_symbols]
        representations, counts = build_representations(
            states,
            input_symbols,
            [0, 1, 2],
        )
        self.assertTrue((counts > 0).all())
        self.assertLess(representations["input_residual"].abs().max(), 1e-6)

    def test_ridge_classification_transfers_linear_roles(self):
        generator = torch.Generator().manual_seed(5)
        train_labels = torch.arange(3).repeat_interleave(100)
        test_labels = torch.arange(3).repeat_interleave(50)
        centroids = torch.eye(3, 12)
        train = centroids[train_labels] + 0.05 * torch.randn(
            300, 12, generator=generator
        )
        test = centroids[test_labels] + 0.05 * torch.randn(
            150, 12, generator=generator
        )
        readout = fit_ridge_readout(
            train,
            train_labels,
            alpha=0.1,
            kind="classification",
            class_count=3,
        )
        metrics = classification_metrics(test_labels, readout.predict(test), 3)
        self.assertGreater(metrics["balanced_accuracy"], 0.99)

    def test_ridge_regression_recovers_order_within_puzzles(self):
        generator = torch.Generator().manual_seed(7)
        labels = torch.arange(1, 10).float().repeat(20)
        puzzle_ids = torch.arange(20).repeat_interleave(9)
        features = torch.stack(
            (labels, torch.randn(len(labels), generator=generator)),
            dim=1,
        )
        readout = fit_ridge_readout(
            features[:90],
            labels[:90],
            alpha=1e-3,
            kind="regression",
        )
        predictions = readout.predict(features[90:])[:, 0]
        metrics = regression_metrics(labels[90:], predictions, puzzle_ids[90:])
        self.assertGreater(metrics["r_squared"], 0.99)
        self.assertGreater(metrics["within_puzzle_spearman"], 0.99)

    def test_shuffle_preserves_each_puzzles_role_counts(self):
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3])
        puzzles = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        shuffled = shuffle_within_puzzles(labels, puzzles, seed=11)
        self.assertFalse(torch.equal(shuffled, labels))
        for puzzle_id in (0, 1):
            selected = puzzles == puzzle_id
            self.assertTrue(torch.equal(
                torch.sort(shuffled[selected]).values,
                torch.sort(labels[selected]).values,
            ))

    def test_ordered_centroids_beat_shuffled_orders(self):
        labels = torch.arange(6).repeat_interleave(50)
        discovery = labels[:, None].float().repeat(1, 8)
        final = discovery + 0.01 * torch.randn(
            discovery.shape,
            generator=torch.Generator().manual_seed(13),
        )
        geometry = pairwise_centroid_geometry(
            discovery,
            labels,
            final,
            labels,
            class_count=6,
            ordered=True,
            order_permutations=200,
            seed=17,
        )
        self.assertGreater(geometry["discovery_final_distance_correlation"], 0.99)
        self.assertGreater(geometry["ordered_distance_spearman"], 0.95)
        self.assertLess(
            geometry["ordered_distance_control"]["empirical_p_value"],
            0.05,
        )

    def test_iteration_order_control_detects_strict_progression(self):
        control = exact_iteration_order_test(
            (0, 1, 4, 16, 128, 512, 1024),
            (0, 1, 2, 3, 4, 5, 6),
        )
        self.assertAlmostEqual(control["observed_spearman"], 1.0)
        self.assertEqual(control["permutation_count"], 5040)
        self.assertLess(control["empirical_p_value"], 0.001)

    def test_task_specs_include_categorical_and_ordered_candidate_controls(self):
        task_names = {task.name for task in TASKS}
        self.assertIn("candidate_size_categorical", task_names)
        self.assertIn("candidate_size_ordinal", task_names)


if __name__ == "__main__":
    unittest.main()
