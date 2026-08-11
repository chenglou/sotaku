"""Focused tests for the held-out progress-coordinate study."""

import pathlib
import sys
import unittest

import numpy as np
import torch


TEST_DIRECTORY = pathlib.Path(__file__).parent
sys.path.insert(0, str(TEST_DIRECTORY.parents[3]))
sys.path.insert(0, str(TEST_DIRECTORY))
from analyze_progress import (  # noqa: E402
    REPRESENTATION_NAMES,
    TARGET_NAMES,
    apply_progress_axis,
    build_progress_targets,
    fit_progress_axis,
    run_controls,
    score_coordinate,
    select_axis,
    spearman_correlation,
    stratified_three_way_split,
)


class ProgressStudyTests(unittest.TestCase):
    def test_stratified_split_is_balanced_and_disjoint(self):
        buckets = tuple(
            bucket for bucket in ("0", "1-2", "3-10", "11-50", "51+")
            for _ in range(12)
        )
        splits = stratified_three_way_split(buckets, seed=7)
        self.assertEqual(
            {name: len(indices) for name, indices in splits.items()},
            {"discovery": 20, "validation": 20, "final": 20},
        )
        self.assertEqual(len(set().union(*map(set, splits.values()))), 60)
        for indices in splits.values():
            counts = {bucket: 0 for bucket in set(buckets)}
            for index in indices:
                counts[buckets[index]] += 1
            self.assertEqual(set(counts.values()), {4})

    def test_progress_targets_use_first_solve_and_continuous_streak(self):
        solved = torch.tensor([
            [False, False, True, True, False, True],
            [False, False, False, False, False, False],
        ])
        iterations = (0, 1, 2, 3, 5)
        correct = torch.tensor([
            [0.2, 0.6, 1.0, 1.0, 1.0],
            [0.1, 0.2, 0.4, 0.5, 0.8],
        ])
        targets = build_progress_targets(correct, solved, iterations)
        self.assertEqual(targets["first_solve_iteration"].tolist(), [2, -1])
        self.assertTrue(torch.allclose(
            targets["first_solve_progress"][0],
            torch.tensor([0.0, 0.5, 1.0, 1.0, 1.0]),
        ))
        self.assertTrue(targets["first_solve_progress"][1].eq(0).all())
        self.assertEqual(
            targets["solved_streak_at_snapshots"][0].tolist(), [0, 0, 1, 2, 1]
        )
        self.assertTrue(torch.allclose(
            targets["remaining_incorrect_fraction"], 1 - correct
        ))

    def test_rank_one_axis_recovers_unseen_shared_progress(self):
        generator = torch.Generator().manual_seed(3)
        puzzle_count, snapshots, features = 30, 12, 8
        latent = torch.linspace(-1, 1, snapshots)[None].repeat(puzzle_count, 1)
        latent = latent + 0.08 * torch.randn(
            puzzle_count, snapshots, generator=generator
        )
        direction = torch.randn(features, generator=generator)
        feature_values = latent[..., None] * direction
        feature_values += 0.05 * torch.randn(
            puzzle_count, snapshots, features, generator=generator
        )
        target_values = torch.stack(
            (latent, 0.8 * latent + 0.1, latent.square() * 0.15 + latent),
            dim=-1,
        )
        axis = fit_progress_axis(
            feature_values[:10].flatten(0, 1),
            target_values[:10].flatten(0, 1),
            "synthetic",
            1e-2,
        )
        coordinate, predicted = apply_progress_axis(axis, feature_values[20:])
        model_data = {
            "targets": {
                **{
                    name: target_values[..., index]
                    for index, name in enumerate(TARGET_NAMES)
                },
                "remaining_incorrect_fraction": 1 - target_values[..., 0],
            }
        }
        score = score_coordinate(
            coordinate,
            predicted,
            model_data,
            list(range(20, 30)),
            list(range(snapshots)),
            seed=11,
        )
        self.assertGreater(score["selection_score"], 0.97)
        self.assertGreater(score["nondecreasing_pair_fraction"], 0.85)

    def test_spearman_handles_ties_and_reversal(self):
        self.assertEqual(
            spearman_correlation([0, 1, 1, 2], [0, 3, 3, 5]), 1.0
        )
        self.assertTrue(np.isclose(
            spearman_correlation([0, 1, 2, 3], [3, 2, 1, 0]), -1.0
        ))

    def test_selection_and_controls_run_on_unseen_puzzles(self):
        generator = torch.Generator().manual_seed(19)
        bucket_names = tuple(
            bucket for bucket in ("0", "1-2", "3-10", "11-50", "51+")
            for _ in range(12)
        )
        splits = stratified_three_way_split(bucket_names, seed=23)
        latent = torch.linspace(0, 1, 6)[None].repeat(60, 1)
        latent += 0.03 * torch.randn(60, 6, generator=generator)
        targets = {
            "correct_fraction": latent.clamp(0, 1),
            "remaining_incorrect_fraction": 1 - latent.clamp(0, 1),
            "first_solve_progress": (0.9 * latent).clamp(0, 1),
            "stable_solved_duration": (latent - 0.5).clamp_min(0),
        }
        features = {}
        for representation_index, name in enumerate(REPRESENTATION_NAMES):
            dimension = 8 if name != "blank_mean_and_spread_delta" else 16
            direction = torch.randn(dimension, generator=generator)
            noise = 0.05 + representation_index * 0.03
            features[name] = latent[..., None] * direction
            features[name] += noise * torch.randn(
                60, 6, dimension, generator=generator
            )
        model_data = {"synthetic": {"features": features, "targets": targets}}
        axis, _, _ = select_axis(
            model_data,
            ["synthetic"],
            splits["discovery"],
            splits["validation"],
            tuple(range(6)),
            seed=29,
        )
        controls = run_controls(
            axis,
            model_data,
            ["synthetic"],
            splits["discovery"],
            splits["final"],
            bucket_names,
            tuple(range(6)),
            seed=31,
            repetitions=2,
            random_directions=4,
        )
        self.assertGreater(controls["observed"], 0.8)
        self.assertGreater(controls["iteration_only_baseline"]["score"], 0.7)
        self.assertEqual(controls["shuffled_iteration_fit"]["repetitions"], 2)
        self.assertEqual(controls["random_orthonormal_direction"]["repetitions"], 4)


if __name__ == "__main__":
    unittest.main()
