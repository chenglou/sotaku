import unittest

import numpy as np

from es.es_sampling import (
    DEFAULT_SAMPLING_MODE,
    POPULATION_PAIRS,
    POPULATION_SIZE,
    direction_weights,
    generation_direction_seeds,
)


class SamplingAblationTest(unittest.TestCase):
    def test_independent_sampling_is_the_default(self):
        self.assertEqual(DEFAULT_SAMPLING_MODE, "independent")

    def test_modes_share_first_sixteen_directions(self):
        seeds = generation_direction_seeds(123, 7)
        self.assertEqual(len(seeds), POPULATION_SIZE)
        self.assertEqual(seeds[:POPULATION_PAIRS], seeds[:16])
        self.assertEqual(seeds, generation_direction_seeds(123, 7))
        self.assertNotEqual(seeds, generation_direction_seeds(123, 8))

    def test_paired_weights_use_mirrored_score_difference(self):
        scores = np.concatenate((np.arange(16), np.arange(16)[::-1]))
        weights, spread = direction_weights(scores, "paired")
        self.assertGreater(spread, 0)
        self.assertEqual(weights.shape, (POPULATION_PAIRS,))
        self.assertLess(weights[0], 0)
        self.assertGreater(weights[-1], 0)

    def test_independent_weights_are_centered(self):
        weights, spread = direction_weights(np.arange(POPULATION_SIZE), "independent")
        self.assertGreater(spread, 0)
        self.assertEqual(weights.shape, (POPULATION_SIZE,))
        self.assertAlmostEqual(float(weights.sum()), 0.0)

    def test_tied_population_takes_no_step(self):
        for mode, expected_size in (("paired", 16), ("independent", 32)):
            weights, spread = direction_weights(np.ones(POPULATION_SIZE), mode)
            self.assertEqual(spread, 0)
            np.testing.assert_array_equal(weights, np.zeros(expected_size))


if __name__ == "__main__":
    unittest.main()
