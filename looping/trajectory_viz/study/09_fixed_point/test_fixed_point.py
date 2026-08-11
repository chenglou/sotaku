import unittest

import torch

from analyze_fixed_point import (
    endpoint_distances,
    normalized_distance,
    sequence_metrics,
    shuffled_time_control,
    specified_direction_gain,
    stratified_split,
)


class FixedPointAnalysisTest(unittest.TestCase):
    def test_stratified_split_is_balanced_and_disjoint(self):
        buckets = [bucket for bucket in ("a", "b") for _ in range(12)]
        splits = stratified_split(buckets, per_bucket_per_split=4)
        self.assertEqual(
            {name: len(indices) for name, indices in splits.items()},
            {"discovery": 8, "validation": 8, "final_holdout": 8},
        )
        all_indices = [index for indices in splits.values() for index in indices]
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_sequence_metrics_detect_straight_motion(self):
        updates = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        metrics = sequence_metrics(updates)
        self.assertEqual(metrics["cosine_mean"], 1.0)
        self.assertGreater(metrics["relative_acceleration_mean"], 0)

    def test_shuffled_time_disrupts_nonstationary_directions(self):
        angles = torch.linspace(0, 2.8, 20)
        updates = torch.stack((angles.cos(), angles.sin()), dim=1).unsqueeze(0)
        ordered = sequence_metrics(updates)
        shuffled = shuffled_time_control(updates, seed=7)
        self.assertGreater(ordered["cosine_mean"], shuffled["cosine_mean"])

    def test_normalized_distance_ignores_magnitude(self):
        left = torch.tensor([[1.0, 0.0]])
        right = torch.tensor([[10.0, 0.0]])
        self.assertEqual(normalized_distance(left, right).item(), 0.0)

    def test_endpoint_control_uses_other_puzzles(self):
        states = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0]],
                [[0.0, 1.0], [0.0, 2.0]],
            ]
        )
        actual, shuffled = endpoint_distances(states)
        self.assertTrue(torch.allclose(actual[:, -1], torch.zeros(2)))
        self.assertTrue(torch.all(shuffled[:, -1] > 1.0))

    def test_specified_direction_gain_recovers_linear_expansion(self):
        class DoublingMap:
            def output_head(self, state):
                return torch.zeros(state.size(0), state.size(1), 2)

            def recurrent_step(self, state, predictions, rope_cos, rope_sin):
                return 2 * state

        states = torch.ones(2, 3, 4)
        directions = torch.randn_like(states)
        gains = specified_direction_gain(
            DoublingMap(), states, directions, None, None
        )
        self.assertTrue(torch.allclose(gains, torch.full((2,), 2.0), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
