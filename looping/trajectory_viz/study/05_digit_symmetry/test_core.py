"""Focused tests for the digit-symmetry study statistics."""

import pathlib
import sys
import unittest

import numpy as np


sys.path.insert(0, str(pathlib.Path(__file__).parent))

from core import (  # noqa: E402
    all_digit_permutations,
    category_centroids,
    category_subspace,
    digit_code,
    fit_ridge_decoder,
    pairwise_distance_matrix,
    procrustes_similarity,
    puzzle_equal_weights,
    puzzle_mean_accuracy,
    rdm_correlation,
    select_rdm_permutation,
    stratified_three_way_split,
    subspace_overlap,
)


class DigitSymmetryCoreTest(unittest.TestCase):
    def test_stratified_split_has_balanced_disjoint_whole_puzzles(self):
        buckets = ["easy"] * 6 + ["hard"] * 6
        split = stratified_three_way_split(buckets, seed=7)
        self.assertEqual(set(split), {"discovery", "validation", "final"})
        for name in set(split):
            selected = np.flatnonzero(split == name)
            self.assertEqual(len(selected), 4)
            self.assertEqual(sum(index < 6 for index in selected), 2)

    def test_ridge_decoder_transfers_simple_digit_categories(self):
        generator = np.random.default_rng(3)
        labels = np.tile(np.arange(9), 8)
        puzzle_ids = np.repeat(np.arange(8), 9)
        category_vectors = np.eye(9)
        values = category_vectors[labels] + 0.02 * generator.normal(size=(72, 9))
        decoder = fit_ridge_decoder(
            values[:54],
            labels[:54],
            puzzle_equal_weights(puzzle_ids[:54]),
            alpha=0.01,
        )
        accuracy = puzzle_mean_accuracy(
            decoder.predict(values[54:]), labels[54:], puzzle_ids[54:]
        )
        self.assertEqual(accuracy, 1.0)

    def test_optimal_permutation_recovers_relabeling_without_numeric_order(self):
        generator = np.random.default_rng(9)
        centroids = generator.normal(size=(9, 6))
        relabeling = np.array([4, 1, 8, 0, 6, 2, 7, 5, 3])
        target = centroids[np.argsort(relabeling)]
        selected, score = select_rdm_permutation(
            centroids, target, all_digit_permutations()
        )
        self.assertTrue(np.array_equal(selected, relabeling))
        self.assertGreater(score, 0.999999)
        self.assertGreater(
            rdm_correlation(centroids, target, selected), 0.999999
        )

    def test_procrustes_ignores_rotation_scale_and_translation(self):
        generator = np.random.default_rng(13)
        centroids = generator.normal(size=(9, 5))
        rotation, _ = np.linalg.qr(generator.normal(size=(5, 5)))
        transformed = 3.2 * centroids @ rotation + generator.normal(size=(1, 5))
        self.assertGreater(procrustes_similarity(centroids, transformed), 0.999999)
        self.assertGreater(rdm_correlation(centroids, transformed), 0.999999)

    def test_category_subspace_and_centroids_are_puzzle_balanced(self):
        values = []
        labels = []
        puzzles = []
        for puzzle in range(3):
            for digit in range(9):
                repeats = 1 + puzzle
                values.extend([[digit, puzzle, digit * puzzle]] * repeats)
                labels.extend([digit] * repeats)
                puzzles.extend([puzzle] * repeats)
        centroids = category_centroids(values, labels, puzzles)
        self.assertTrue(np.allclose(centroids[:, 1], 1.0))
        basis = category_subspace(centroids)
        self.assertEqual(basis.shape[1], 1)
        self.assertAlmostEqual(subspace_overlap(basis, basis), 1.0)
        self.assertEqual(pairwise_distance_matrix(centroids).shape, (9, 9))

    def test_digit_codes_do_not_assume_order_for_categorical_model(self):
        labels = np.arange(9)
        reversed_order = np.arange(8, -1, -1)
        self.assertTrue(
            np.array_equal(
                digit_code(labels, "categorical"),
                digit_code(labels, "categorical", reversed_order),
            )
        )
        self.assertFalse(
            np.array_equal(
                digit_code(labels, "ordinal"),
                digit_code(labels, "ordinal", reversed_order),
            )
        )


if __name__ == "__main__":
    unittest.main()
