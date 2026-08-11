import json
import pathlib
import sys
import unittest

import numpy as np
import torch


STUDY_DIRECTORY = pathlib.Path(__file__).resolve().parent
if str(STUDY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(STUDY_DIRECTORY))

from core import (  # noqa: E402
    aggregate_rows_by_puzzle_iteration,
    assign_bins,
    deep_temporal_metrics,
    evaluate_categorical_probe,
    evaluate_ordered_probe,
    evaluate_transferred_axis,
    fit_transferred_axis,
    make_bin_thresholds,
    ordered_bin_metrics,
    puzzle_equal_weights,
    random_rank_one_probe,
    select_categorical_alpha,
    select_ordered_alpha,
    shuffle_within_iteration,
    stratified_three_way_split,
)


class MarginCoreTest(unittest.TestCase):
    def test_three_way_split_keeps_puzzles_whole_and_balances_buckets(self):
        buckets = np.repeat(np.asarray(["a", "b", "c", "d", "e"]), 12)
        first = stratified_three_way_split(
            buckets, examples_per_split_bucket=4, seed=19
        )
        second = stratified_three_way_split(
            buckets, examples_per_split_bucket=4, seed=19
        )
        self.assertEqual(first.tolist(), second.tolist())
        for bucket in np.unique(buckets):
            for split in ("discovery", "validation", "final"):
                self.assertEqual(int(np.sum((buckets == bucket) & (first == split))), 4)

    def test_ordered_probe_recovers_axis_beyond_iteration_baseline(self):
        generator = np.random.default_rng(7)
        puzzle_count = 45
        iteration_values = np.asarray([0, 1, 4, 16, 128, 1024])
        puzzle_ids = np.repeat(np.arange(puzzle_count), len(iteration_values))
        iterations = np.tile(iteration_values, puzzle_count)
        features = generator.normal(size=(len(iterations), 12))
        iteration_effect = np.log2(iterations + 1) * 0.4
        target = iteration_effect + 2.5 * features[:, 3]
        target += generator.normal(scale=0.03, size=len(target))
        discovery = puzzle_ids < 15
        validation = (puzzle_ids >= 15) & (puzzle_ids < 30)
        final = puzzle_ids >= 30

        selection = select_ordered_alpha(
            features[discovery],
            target[discovery],
            iterations[discovery],
            puzzle_equal_weights(puzzle_ids[discovery]),
            features[validation],
            target[validation],
            iterations[validation],
            puzzle_equal_weights(puzzle_ids[validation]),
            candidates=(1e-4, 1e-2, 1.0),
        )
        result = evaluate_ordered_probe(
            selection["probe"],
            features[final],
            target[final],
            iterations[final],
            puzzle_equal_weights(puzzle_ids[final]),
        )
        self.assertGreater(result["partial_r2_over_iteration"], 0.995)
        self.assertGreater(abs(selection["probe"].raw_axis[3]), 1.5)

    def test_categorical_probe_is_unconstrained_comparison_on_same_bins(self):
        generator = np.random.default_rng(11)
        puzzle_count = 36
        iterations = np.tile(np.asarray([1, 16, 128, 1024]), puzzle_count)
        puzzle_ids = np.repeat(np.arange(puzzle_count), 4)
        features = generator.normal(size=(len(iterations), 8))
        target = 1.7 * features[:, 0] + 0.1 * np.log2(iterations)
        discovery = puzzle_ids < 12
        validation = (puzzle_ids >= 12) & (puzzle_ids < 24)
        final = puzzle_ids >= 24
        discovery_weights = puzzle_equal_weights(puzzle_ids[discovery])
        validation_weights = puzzle_equal_weights(puzzle_ids[validation])
        final_weights = puzzle_equal_weights(puzzle_ids[final])
        thresholds = make_bin_thresholds(
            target[discovery], discovery_weights, bin_count=5
        )
        ordered = select_ordered_alpha(
            features[discovery], target[discovery], iterations[discovery], discovery_weights,
            features[validation], target[validation], iterations[validation], validation_weights,
            candidates=(1e-4, 1e-2, 1.0),
        )["probe"]
        categorical = select_categorical_alpha(
            features[discovery], target[discovery], iterations[discovery], discovery_weights,
            features[validation], target[validation], iterations[validation], validation_weights,
            thresholds=thresholds,
            candidates=(1e-4, 1e-2, 1.0),
        )["probe"]
        ordered_result = ordered_bin_metrics(
            ordered, thresholds, features[final], target[final], iterations[final], final_weights
        )
        categorical_result = evaluate_categorical_probe(
            categorical, features[final], target[final], iterations[final], final_weights
        )
        self.assertGreater(ordered_result["accuracy"], 0.85)
        self.assertGreater(categorical_result["accuracy"], 0.55)
        self.assertEqual(categorical.coefficients.shape, (8, 5))
        self.assertTrue(np.all(assign_bins(target[discovery], thresholds) < 5))

    def test_label_shuffle_preserves_each_iteration_distribution(self):
        generator = np.random.default_rng(23)
        iterations = np.repeat(np.asarray([0, 16, 128]), 20)
        values = np.arange(len(iterations), dtype=np.float64)
        shuffled = shuffle_within_iteration(values, iterations, generator)
        self.assertFalse(np.array_equal(values, shuffled))
        for iteration in np.unique(iterations):
            selected = iterations == iteration
            self.assertEqual(
                sorted(values[selected].tolist()), sorted(shuffled[selected].tolist())
            )

    def test_random_control_is_one_dimensional_and_usually_worse(self):
        generator = np.random.default_rng(29)
        puzzle_count = 40
        iterations = np.tile(np.asarray([1, 16, 128, 1024]), puzzle_count)
        puzzle_ids = np.repeat(np.arange(puzzle_count), 4)
        features = generator.normal(size=(len(iterations), 32))
        target = 4.0 * features[:, 0] + generator.normal(scale=0.05, size=len(iterations))
        discovery = puzzle_ids < 20
        final = ~discovery
        discovery_weights = puzzle_equal_weights(puzzle_ids[discovery])
        final_weights = puzzle_equal_weights(puzzle_ids[final])
        probe = select_ordered_alpha(
            features[discovery], target[discovery], iterations[discovery], discovery_weights,
            features[final], target[final], iterations[final], final_weights,
            candidates=(1e-4, 1e-2),
        )["probe"]
        observed = evaluate_ordered_probe(
            probe, features[final], target[final], iterations[final], final_weights
        )["partial_r2_over_iteration"]
        random_effects = []
        for _ in range(31):
            control = random_rank_one_probe(
                probe,
                features[discovery],
                target[discovery],
                iterations[discovery],
                discovery_weights,
                generator,
            )
            self.assertEqual(control.coefficient.ndim, 1)
            random_effects.append(
                evaluate_ordered_probe(
                    control,
                    features[final],
                    target[final],
                    iterations[final],
                    final_weights,
                )["partial_r2_over_iteration"]
            )
        self.assertGreater(observed, max(random_effects))

    def test_transfer_calibration_preserves_orientation(self):
        generator = np.random.default_rng(31)
        iteration_values = np.asarray([1, 16, 128, 1024])
        iterations = np.tile(iteration_values, 20)
        puzzle_ids = np.repeat(np.arange(20), len(iteration_values))
        features = generator.normal(size=(len(iterations), 6))
        target = 3.0 * features[:, 2] + 0.2 * np.log2(iterations)
        validation = puzzle_ids < 10
        final = puzzle_ids >= 10
        axis = np.zeros(6)
        axis[2] = 1.0
        calibrated = fit_transferred_axis(
            axis,
            features[validation],
            target[validation],
            iterations[validation],
            np.full(validation.sum(), 1 / validation.sum()),
        )
        result = evaluate_transferred_axis(
            calibrated,
            features[final],
            target[final],
            iterations[final],
            np.full(final.sum(), 1 / final.sum()),
        )
        self.assertGreater(result["partial_r2_over_iteration"], 0.999)
        self.assertTrue(result["orientation_preserved"])

        reversed_features = features.copy()
        reversed_features[:, 2] *= -1
        reversed_calibration = fit_transferred_axis(
            axis,
            reversed_features[validation],
            target[validation],
            iterations[validation],
            np.full(validation.sum(), 1 / validation.sum()),
        )
        self.assertLess(reversed_calibration.unconstrained_slope, 0)
        self.assertEqual(reversed_calibration.slope, 0.0)

    def test_temporal_analysis_uses_puzzle_rows_and_iteration_shuffle(self):
        puzzle_count = 20
        iterations = np.asarray([16, 64, 128, 256, 512, 1024])
        puzzle_scale = np.linspace(0.5, 1.5, puzzle_count)[:, None]
        time = np.log2(iterations + 1)[None, :]
        true_values = puzzle_scale * time
        coordinate_values = true_values * 1.3
        result = deep_temporal_metrics(
            true_values,
            coordinate_values,
            iterations,
            np.repeat(np.asarray(["a", "b", "c", "d", "e"]), 4),
            minimum_iteration=128,
            permutation_repetitions=49,
            bootstrap_repetitions=100,
            seed=37,
        )
        self.assertGreater(result["deep_delta_alignment_pearson"], 0.999)
        self.assertLessEqual(
            result["shuffled_iteration_alignment"]["p_one_sided"], 0.04
        )
        self.assertEqual(
            len(result["shuffled_iteration_alignment"]["null_values"]), 49
        )

        row_puzzles = np.repeat(np.arange(3), 3)
        row_iterations = np.tile(np.asarray([1, 2, 4]), 3)
        values = row_puzzles * 10 + row_iterations
        puzzles, times, matrix = aggregate_rows_by_puzzle_iteration(
            values, row_puzzles, row_iterations
        )
        self.assertEqual(puzzles.tolist(), [0, 1, 2])
        self.assertEqual(times.tolist(), [1, 2, 4])
        self.assertEqual(matrix[2].tolist(), [21.0, 22.0, 24.0])


class DurableArtifactTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact_directory = STUDY_DIRECTORY / "margin_boundary_v1_20260811"
        with (cls.artifact_directory / "metrics.json").open() as handle:
            cls.metrics = json.load(handle)

    def test_authoritative_run_has_balanced_whole_puzzle_splits(self):
        split = self.metrics["split"]
        expected = {
            "discovery": list(range(0, 4)),
            "validation": list(range(4, 8)),
            "final": list(range(8, 12)),
        }
        for split_name, indices in split["indices"].items():
            self.assertEqual(len(indices), 20)
            for bucket_start in range(0, 60, 12):
                offsets = sorted(
                    index - bucket_start
                    for index in indices
                    if bucket_start <= index < bucket_start + 12
                )
                self.assertEqual(offsets, expected[split_name])
        self.assertEqual(
            set(self.metrics["model_accuracy"]),
            {"stable_plain", "collapsed_plain", "late_state_ce", "combined_margin"},
        )

    def test_authoritative_metrics_cover_probes_controls_and_transfer(self):
        primary = self.metrics["primary_result"]
        self.assertEqual(primary["model"], "collapsed_plain")
        self.assertEqual(primary["observation_iteration"], 128)
        self.assertEqual(primary["selected_residual_rank"], 16)
        self.assertEqual(
            set(primary["models"]),
            {
                "current_margin",
                "margin_unordered",
                "margin_ordered",
                "logit_history",
                "raw_hidden",
                "residual_pca",
            },
        )
        controls = self.metrics["primary_controls"]
        self.assertEqual(
            set(controls),
            {
                "shuffled_margin_auc",
                "shuffled_iteration_auc",
                "matched_rank_random_subspace_auc",
            },
        )
        self.assertTrue(all(control["repetitions"] == 32 for control in controls.values()))
        margin_axis = self.metrics["margin_axis"]
        self.assertEqual(set(margin_axis["within_checkpoint"]), set(self.metrics["model_accuracy"]))
        for source, targets in margin_axis["cross_checkpoint"].items():
            self.assertEqual(set(targets), set(self.metrics["model_accuracy"]), source)

    def test_authoritative_html_references_nonempty_pngs(self):
        html = (self.artifact_directory / "index.html").read_text()
        for filename in (
            "primary_boundary_risk.png",
            "margin_trajectories.png",
            "cross_checkpoint_transfer.png",
            "geometry_controls.png",
        ):
            path = self.artifact_directory / filename
            self.assertIn(filename, html)
            self.assertGreater(path.stat().st_size, 100_000)

    def test_modal_wrapper_has_one_spawn_and_no_remote_call(self):
        wrapper = (STUDY_DIRECTORY / "modal_margin.py").read_text()
        self.assertEqual(wrapper.count("analyze.spawn("), 1)
        self.assertNotIn(".remote(", wrapper)


if __name__ == "__main__":
    unittest.main()
