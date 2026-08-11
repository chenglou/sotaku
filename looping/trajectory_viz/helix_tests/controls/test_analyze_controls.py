import math
import unittest

import torch

from looping.trajectory_viz.helix_tests.controls.analyze_controls import (
    CYCLE_CONTROLS,
    DIGIT_COUNT,
    _midrank_percentile,
    _position_time_center,
    intrinsic_cycle_metrics,
    output_digit_basis,
    periodic_code,
    remove_basis,
    residualize_train_test,
    shuffled_trajectory_labels,
    stratified_puzzle_folds,
)


def repeated_geometry(centroids, puzzle_count=20, noise=0.0):
    values = []
    labels = []
    puzzles = []
    generator = torch.Generator().manual_seed(7)
    for puzzle_index in range(puzzle_count):
        sample = centroids.clone()
        if noise:
            sample += noise * torch.randn(
                sample.shape,
                generator=generator,
            )
        values.append(sample)
        labels.append(torch.arange(DIGIT_COUNT))
        puzzles.append(torch.full((DIGIT_COUNT,), puzzle_index))
    return torch.cat(values), torch.cat(labels), torch.cat(puzzles)


class HelixControlTests(unittest.TestCase):
    def test_cycle_enumeration_quotients_rotation_and_reflection(self):
        self.assertEqual(tuple(CYCLE_CONTROLS.orders.shape), (20160, 9))
        self.assertTrue(torch.equal(
            CYCLE_CONTROLS.orders[CYCLE_CONTROLS.natural_index],
            torch.arange(9),
        ))

    def test_midrank_gives_tied_null_half_percentile(self):
        values = torch.ones(100)
        self.assertAlmostEqual(_midrank_percentile(values, 1.0), 0.5)

    def test_natural_circle_wins_exact_order_test(self):
        values, labels, puzzles = repeated_geometry(
            periodic_code(),
            noise=0.01,
        )
        train = puzzles < 10
        test = ~train
        metrics = intrinsic_cycle_metrics(
            values[train], labels[train], puzzles[train],
            values[test], labels[test], puzzles[test],
        )
        self.assertGreater(metrics["natural_first_harmonic_fraction"], 0.98)
        self.assertGreater(metrics["natural_harmonic_energy_percentile"], 0.999)
        self.assertGreater(metrics["natural_rdm_correlation_percentile"], 0.999)
        self.assertLess(metrics["natural_cycle_shorter_percentile"], 0.001)

    def test_regular_simplex_has_two_eighths_harmonic_energy(self):
        simplex = torch.eye(9) - torch.ones(9, 9) / 9
        values, labels, puzzles = repeated_geometry(simplex)
        train = puzzles < 10
        test = ~train
        metrics = intrinsic_cycle_metrics(
            values[train], labels[train], puzzles[train],
            values[test], labels[test], puzzles[test],
        )
        self.assertAlmostEqual(
            metrics["natural_first_harmonic_fraction"],
            2 / 8,
            places=6,
        )
        self.assertAlmostEqual(
            metrics["natural_harmonic_energy_percentile"],
            0.5,
            places=3,
        )
        self.assertAlmostEqual(
            metrics["natural_cycle_shorter_percentile"],
            0.5,
            places=3,
        )

    def test_stratified_folds_hold_out_each_bucket(self):
        buckets = [bucket for bucket in ("a", "b", "c") for _ in range(10)]
        folds = stratified_puzzle_folds(buckets, fold_count=5, seed=3)
        seen = []
        for fold in folds:
            self.assertFalse(
                set(fold["train_puzzles"]) & set(fold["test_puzzles"])
            )
            test_buckets = [buckets[index] for index in fold["test_puzzles"]]
            self.assertEqual(test_buckets.count("a"), 2)
            self.assertEqual(test_buckets.count("b"), 2)
            self.assertEqual(test_buckets.count("c"), 2)
            seen.extend(fold["test_puzzles"])
        self.assertEqual(sorted(seen), list(range(30)))

    def test_nuisance_regression_removes_linear_design(self):
        generator = torch.Generator().manual_seed(11)
        train_design = torch.randn(1000, 12, generator=generator)
        test_design = torch.randn(400, 12, generator=generator)
        coefficients = torch.randn(12, 16, generator=generator)
        train_values = train_design @ coefficients
        test_values = test_design @ coefficients
        train_residual, test_residual = residualize_train_test(
            train_values,
            test_values,
            train_design,
            test_design,
            ridge=1e-6,
        )
        self.assertLess(train_residual.square().mean(), 1e-8)
        self.assertLess(test_residual.square().mean(), 1e-8)

    def test_remove_basis_eliminates_decoder_span(self):
        generator = torch.Generator().manual_seed(13)
        basis, _ = torch.linalg.qr(torch.randn(20, 8, generator=generator))
        values = torch.randn(200, 20, generator=generator)
        residual = remove_basis(values, basis)
        self.assertLess((residual @ basis).abs().max(), 2e-6)

    def test_position_time_center_uses_train_group_means(self):
        train_position = torch.tensor([0, 0, 1, 1])
        train_time = torch.tensor([0, 1, 0, 1])
        test_position = train_position.clone()
        test_time = train_time.clone()
        group_value = train_position.float() * 10 + train_time.float()
        train_values = group_value[:, None].repeat(1, 3)
        test_values = train_values + 2
        arrays = {
            "position": torch.cat((train_position, test_position)),
            "time_index": torch.cat((train_time, test_time)),
        }
        train_mask = torch.tensor([True] * 4 + [False] * 4)
        test_mask = ~train_mask
        train_residual, test_residual = _position_time_center(
            train_values,
            test_values,
            arrays,
            train_mask,
            test_mask,
        )
        self.assertLess(train_residual.abs().max(), 1e-7)
        self.assertTrue(torch.allclose(test_residual, torch.full_like(test_residual, 2)))

    def test_random_trajectory_labels_preserve_counts_and_time_strands(self):
        puzzle_count, time_count, cell_count = 5, 4, 9
        puzzle = torch.arange(puzzle_count)[:, None, None].expand(
            -1, time_count, cell_count
        ).reshape(-1)
        position = torch.arange(cell_count)[None, None, :].expand(
            puzzle_count, time_count, -1
        ).reshape(-1)
        true_by_trajectory = (
            torch.arange(puzzle_count)[:, None] + torch.arange(cell_count)[None]
        ) % DIGIT_COUNT
        true_digit = true_by_trajectory[:, None, :].expand(
            -1, time_count, -1
        ).reshape(-1)
        arrays = {
            "puzzle": puzzle,
            "position": position,
            "true_digit": true_digit,
        }
        mask = torch.ones_like(true_digit, dtype=torch.bool)
        shuffled = shuffled_trajectory_labels(
            arrays,
            mask,
            "random_fixed_trajectory_labels",
            seed=17,
        )
        self.assertTrue(torch.equal(
            torch.bincount(shuffled, minlength=9),
            torch.bincount(true_digit, minlength=9),
        ))
        keys = puzzle * cell_count + position
        for key in torch.unique(keys):
            self.assertEqual(torch.unique(shuffled[keys == key]).numel(), 1)


if __name__ == "__main__":
    unittest.main()
