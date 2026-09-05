from collections import Counter
import json
import math
from pathlib import Path
import struct
import unittest

import torch

from entropy_core import (
    evaluate_probe,
    fit_shuffled_target_controls,
    monotonicity_metrics,
    predictive_targets,
    random_axis_controls,
    rating_balanced_three_way_split,
    select_ridge_alpha,
    shuffled_time_controls,
    shuffled_within_iteration,
)


class PredictiveTargetTests(unittest.TestCase):
    def test_uniform_and_confident_predictions_have_expected_uncertainty(self):
        logits = torch.stack(
            (
                torch.zeros(9),
                torch.tensor([20.0] + [-20.0] * 8),
            )
        )
        targets = predictive_targets(logits)

        self.assertAlmostEqual(float(targets["entropy"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(targets["top_two_gap"][0]), 0.0, places=6)
        self.assertAlmostEqual(
            float(targets["residual_uncertainty"][0]), 8 / 9, places=6
        )
        self.assertLess(float(targets["entropy"][1]), 1e-6)
        self.assertGreater(float(targets["top_two_gap"][1]), 0.999999)
        self.assertLess(float(targets["residual_uncertainty"][1]), 1e-6)


class SplitTests(unittest.TestCase):
    def test_three_way_split_is_balanced_and_disjoint(self):
        buckets = tuple(
            bucket for bucket in ("a", "b", "c", "d", "e") for _ in range(12)
        )
        split = rating_balanced_three_way_split(buckets, seed=7)

        combined = torch.cat((split.discovery, split.validation, split.final))
        self.assertEqual(len(torch.unique(combined)), 60)
        for indices in (split.discovery, split.validation, split.final):
            self.assertEqual(len(indices), 20)
            represented = [buckets[index] for index in indices.tolist()]
            self.assertEqual({name: represented.count(name) for name in set(buckets)}, {
                name: 4 for name in set(buckets)
            })

    def test_three_way_split_rejects_unequal_bucket_size(self):
        with self.assertRaisesRegex(ValueError, "multiple of three"):
            rating_balanced_three_way_split(("a",) * 4 + ("b",) * 3, seed=1)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(12)
        puzzle_count = 30
        time_count = 4
        cells_per_puzzle = 16
        feature_count = 8
        features = torch.randn(
            puzzle_count,
            time_count,
            cells_per_puzzle,
            feature_count,
            generator=generator,
            dtype=torch.float64,
        )
        time_baseline = torch.tensor([0.1, 0.4, 0.7, 1.0]).view(1, -1, 1)
        targets = (
            time_baseline
            + 1.7 * features[..., 0]
            - 0.6 * features[..., 1]
            + 0.01
            * torch.randn(
                puzzle_count,
                time_count,
                cells_per_puzzle,
                generator=generator,
                dtype=torch.float64,
            )
        )

        def flatten(start, end):
            selected_features = features[start:end].reshape(-1, feature_count)
            selected_targets = targets[start:end].reshape(-1)
            iterations = torch.arange(time_count).view(1, -1, 1).expand(
                end - start, time_count, cells_per_puzzle
            ).reshape(-1)
            puzzles = torch.arange(start, end).view(-1, 1, 1).expand(
                end - start, time_count, cells_per_puzzle
            ).reshape(-1)
            return selected_features, selected_targets, iterations, puzzles

        self.discovery = flatten(0, 10)
        self.validation = flatten(10, 20)
        self.final = flatten(20, 30)
        self.time_count = time_count

    def test_discovery_probe_generalizes_after_validation_selection(self):
        probe, candidates = select_ridge_alpha(
            self.discovery,
            self.validation,
            time_count=self.time_count,
            alphas=(0.0, 1e-4, 1e-2, 1.0),
        )
        final = evaluate_probe(probe, *self.final)

        self.assertEqual(len(candidates), 4)
        self.assertGreater(final["partial_r2_over_iteration"], 0.999)
        self.assertGreater(final["within_iteration_correlation"], 0.999)

    def test_shuffle_and_random_axis_controls_are_weaker(self):
        probe, _ = select_ridge_alpha(
            self.discovery,
            self.validation,
            time_count=self.time_count,
            alphas=(0.0, 1e-4, 1e-2),
        )
        observed = evaluate_probe(probe, *self.final)[
            "partial_r2_over_iteration"
        ]
        shuffled = fit_shuffled_target_controls(
            probe,
            self.discovery,
            self.final,
            count=8,
            seed=2,
        )
        random_axes = random_axis_controls(
            probe,
            self.discovery,
            self.final,
            count=8,
            seed=3,
        )

        self.assertGreater(observed, float(shuffled.max()))
        self.assertGreater(observed, float(random_axes.max()))

    def test_shuffle_preserves_each_iteration_multiset(self):
        targets = self.discovery[1]
        iterations = self.discovery[2]
        shuffled = shuffled_within_iteration(
            targets,
            iterations,
            count=3,
            seed=9,
        )
        for iteration in torch.unique(iterations):
            selected = iterations == iteration
            expected = targets[selected].sort().values
            for shuffle_index in range(3):
                actual = shuffled[selected, shuffle_index].sort().values
                self.assertTrue(torch.equal(expected, actual))

    def test_shuffle_changes_the_row_alignment(self):
        targets = torch.arange(24, dtype=torch.float64)
        iterations = torch.tensor([0] * 12 + [1] * 12)
        shuffled = shuffled_within_iteration(
            targets,
            iterations,
            count=4,
            seed=13,
        )
        for shuffle_index in range(shuffled.size(1)):
            self.assertFalse(torch.equal(shuffled[:, shuffle_index], targets))


class MonotonicityTests(unittest.TestCase):
    def test_monotonicity_distinguishes_ordered_and_reversed_paths(self):
        increasing = torch.arange(5, dtype=torch.float64).repeat(6, 1)
        puzzle_indices = torch.tensor([0, 0, 1, 1, 2, 2])
        ordered = monotonicity_metrics(increasing, puzzle_indices, 1)
        reversed_result = monotonicity_metrics(increasing, puzzle_indices, -1)

        self.assertAlmostEqual(ordered["adjacent_direction_fraction"], 1.0, delta=1e-12)
        self.assertAlmostEqual(ordered["all_pairs_direction_fraction"], 1.0, delta=1e-12)
        self.assertAlmostEqual(reversed_result["all_pairs_direction_fraction"], 0.0, delta=1e-12)

    def test_constant_paths_count_as_ties(self):
        constant = torch.ones(4, 3)
        metrics = monotonicity_metrics(constant, torch.arange(4), 1)
        self.assertAlmostEqual(metrics["adjacent_direction_fraction"], 0.5, delta=1e-12)
        self.assertAlmostEqual(metrics["net_direction_fraction"], 0.5, delta=1e-12)

    def test_shuffled_time_control_returns_requested_count(self):
        paths = torch.arange(6, dtype=torch.float64).repeat(4, 1)
        controls = shuffled_time_controls(
            paths,
            torch.arange(4),
            1,
            count=7,
            seed=4,
        )
        self.assertEqual(len(controls["all_pairs_direction_fraction"]), 7)


class DurableArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_directory = (
            Path(__file__).resolve().parent / "entropy_v1_20260811"
        )
        cls.metrics = json.loads(
            (cls.artifact_directory / "metrics.json").read_text()
        )

    def test_json_is_finite_and_contains_the_locked_sample(self):
        def assert_finite(value):
            if isinstance(value, dict):
                for child in value.values():
                    assert_finite(child)
            elif isinstance(value, list):
                for child in value:
                    assert_finite(child)
            elif isinstance(value, float):
                self.assertTrue(math.isfinite(value))

        assert_finite(self.metrics)
        self.assertEqual(self.metrics["config"]["sample_size"], 60)
        self.assertEqual(self.metrics["config"]["control_count"], 64)
        self.assertEqual(len(set(self.metrics["sample"]["puzzle_hashes"])), 60)

        buckets = self.metrics["sample"]["rating_buckets"]
        split_values = []
        for indices in self.metrics["sample"]["splits"].values():
            self.assertEqual(len(indices), 20)
            self.assertEqual(
                sorted(Counter(buckets[index] for index in indices).values()),
                [4, 4, 4, 4, 4],
            )
            split_values.extend(indices)
        self.assertEqual(sorted(split_values), list(range(60)))

    def test_models_targets_and_controls_are_complete(self):
        expected_models = {
            "stable_plain",
            "collapsed_plain",
            "late_state_ce",
            "combined_margin",
        }
        expected_targets = {
            "entropy",
            "top_two_gap",
            "residual_uncertainty",
        }
        self.assertEqual(set(self.metrics["models"]), expected_models)
        for model in self.metrics["models"].values():
            self.assertEqual(
                set(model["primary_cell_centered_hidden"]), expected_targets
            )
            self.assertEqual(
                set(model["output_head_null_sensitivity"]), expected_targets
            )
            self.assertEqual(set(model["controls"]), expected_targets)
            for target_name in expected_targets:
                primary = model["primary_cell_centered_hidden"][target_name]
                best = max(
                    primary["validation_candidates"],
                    key=lambda row: (
                        row["validation_partial_r2"], -row["alpha"]
                    ),
                )
                self.assertEqual(primary["selected_alpha"], best["alpha"])
                self.assertGreater(
                    primary["final"]["partial_r2_over_iteration"], 0
                )
                controls = model["controls"][target_name]
                self.assertEqual(
                    len(controls["shuffled_discovery_targets"]["values"]), 64
                )
                self.assertEqual(
                    len(controls["random_one_dimensional_axes"]["values"]), 64
                )
                self.assertEqual(
                    len(
                        controls["shuffled_iteration_order"]
                        ["all_pairs_direction_fraction"]["values"]
                    ),
                    64,
                )

    def test_png_artifacts_are_nonempty_and_inspectable(self):
        for name in (
            "heldout_encoding.png",
            "trajectory_monotonicity.png",
            "iteration_profiles.png",
        ):
            path = self.artifact_directory / name
            data = path.read_bytes()
            self.assertGreater(len(data), 20_000)
            self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", data[16:24])
            self.assertGreaterEqual(width, 1_000)
            self.assertGreaterEqual(height, 500)


if __name__ == "__main__":
    unittest.main()
