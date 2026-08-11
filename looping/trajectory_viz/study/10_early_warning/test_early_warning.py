import json
import os
import tempfile
import unittest

import numpy as np
import torch

from analyze_early_warning import analyze
from core import (
    DIFFICULTY_FEATURES,
    EARLY_ITERATIONS,
    FEATURE_MANIFEST,
    GEOMETRY_FEATURES,
    OUTPUT_FEATURES,
    SHUFFLED_GEOMETRY_FEATURES,
    apply_platt,
    assign_balanced_splits,
    define_collapse_labels,
    extract_early_features,
    fit_platt,
    fit_ridge_logistic,
    predict_logit,
    probability_metrics,
    stratified_permutation,
)


class SplitAndLeakageTests(unittest.TestCase):
    def test_balanced_whole_puzzle_splits(self):
        buckets = [bucket for bucket in ("a", "b") for _ in range(12)]
        assignments = assign_balanced_splits(buckets, 4)
        for bucket in ("a", "b"):
            selected = [assignments[index] for index, value in enumerate(buckets) if value == bucket]
            self.assertEqual(selected.count("discovery"), 4)
            self.assertEqual(selected.count("validation"), 4)
            self.assertEqual(selected.count("final"), 4)

    def test_feature_extractor_is_target_free_and_rejects_post_128(self):
        generator = torch.Generator().manual_seed(7)
        updates = torch.randn(3, 5, 7, 4, generator=generator) * 0.05
        states = updates.cumsum(dim=1) + torch.randn(3, 1, 7, 4, generator=generator)
        logits = torch.randn(3, 5, 7, 9, generator=generator)
        blank_mask = torch.tensor(
            [[1, 1, 0, 1, 0, 1, 1], [1, 0, 1, 1, 1, 0, 1], [0, 1, 1, 1, 0, 1, 1]],
            dtype=torch.bool,
        )
        features = extract_early_features(
            EARLY_ITERATIONS,
            states,
            updates,
            logits,
            blank_mask,
        )
        self.assertEqual(set(features), set(GEOMETRY_FEATURES + OUTPUT_FEATURES))
        self.assertTrue(all(torch.isfinite(values).all() for values in features.values()))
        with self.assertRaisesRegex(ValueError, "exactly"):
            extract_early_features((64, 80, 96, 112, 129), states, updates, logits, blank_mask)

    def test_late_label_requires_early_solution_and_late_failure(self):
        targets = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        prediction_128 = torch.tensor([[0, 1, 2], [0, 1, 2], [0, 0, 2]])
        prediction_1024 = torch.tensor([[0, 1, 2], [0, 0, 2], [0, 0, 2]])
        labels = define_collapse_labels(prediction_128, prediction_1024, targets, mask)
        self.assertEqual(labels["eligible"].tolist(), [True, True, False])
        self.assertEqual(labels["collapse"].tolist(), [False, True, False])


class StatisticalCoreTests(unittest.TestCase):
    def test_logistic_fit_and_calibration_produce_probabilities(self):
        values = np.asarray([[-2.0], [-1.0], [1.0], [2.0]])
        labels = np.asarray([0, 0, 1, 1])
        coefficients = fit_ridge_logistic(values, labels)
        logits = predict_logit(coefficients, values)
        calibration = fit_platt(logits, labels)
        probabilities = apply_platt(logits, calibration)
        self.assertTrue(np.all((probabilities > 0) & (probabilities < 1)))
        self.assertGreater(probabilities[-1], probabilities[0])
        self.assertGreater(probability_metrics(labels, probabilities)["auroc"], 0.9)

    def test_stratified_permutation_preserves_each_stratum_count(self):
        labels = np.asarray([0, 1, 1, 0, 0, 1])
        strata = np.asarray(["a", "a", "a", "b", "b", "b"], dtype=object)
        permuted = stratified_permutation(labels, strata, np.random.default_rng(4))
        for stratum in ("a", "b"):
            selected = strata == stratum
            self.assertEqual(int(labels[selected].sum()), int(permuted[selected].sum()))

    def test_average_precision_is_invariant_to_tie_order(self):
        first = probability_metrics([1, 0, 0, 1], [0.5, 0.5, 0.5, 0.5])
        second = probability_metrics([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(first["average_precision"], 0.5)
        self.assertEqual(second["average_precision"], 0.5)


class EndToEndAnalysisTest(unittest.TestCase):
    def test_analysis_writes_durable_artifacts(self):
        checkpoints = (
            ("stable_plain", "plain_stable"),
            ("collapsed_plain", "plain_clean_a"),
            ("late_state_ce", "late_state_ce"),
            ("combined_margin", "combined_margin"),
        )
        rows = []
        generator = np.random.default_rng(11)
        for split_index, split in enumerate(("discovery", "validation", "final")):
            for puzzle_index in range(10):
                bucket = str(puzzle_index % 5)
                for checkpoint_index, (checkpoint, family) in enumerate(checkpoints):
                    collapse = int(
                        (checkpoint == "collapsed_plain" and puzzle_index % 3 != 0)
                        or (checkpoint != "collapsed_plain" and puzzle_index == checkpoint_index)
                    )
                    row = {
                        "puzzle_index": split_index * 10 + puzzle_index,
                        "puzzle_hash": f"{split}-{puzzle_index}",
                        "split": split,
                        "rating_bucket": bucket,
                        "rating": puzzle_index,
                        "rating_ordinal": puzzle_index % 5,
                        "clue_fraction": 0.3 + 0.01 * puzzle_index,
                        "checkpoint": checkpoint,
                        "checkpoint_family": family,
                        "solved_128": True,
                        "solved_1024": not bool(collapse),
                        "eligible": True,
                        "collapse": collapse,
                    }
                    for name in OUTPUT_FEATURES:
                        row[name] = float(generator.normal() - 0.8 * collapse)
                    for name in GEOMETRY_FEATURES:
                        row[name] = float(generator.normal() + 1.2 * collapse)
                    for name in SHUFFLED_GEOMETRY_FEATURES:
                        row[name] = float(generator.normal() + 0.3 * collapse)
                    rows.append(row)
        payload = {
            "format_version": 1,
            "config": {
                "models": [
                    {"name": name, "family": family, "filename": name, "sha256": "test"}
                    for name, family in checkpoints
                ],
                "late_outcome_iteration": 1024,
            },
            "feature_manifest": FEATURE_MANIFEST,
            "label_definition": {},
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            collection_path = os.path.join(directory, "collection.json")
            output_directory = os.path.join(directory, "artifacts")
            with open(collection_path, "w") as handle:
                json.dump(payload, handle)
            metrics = analyze(
                collection_path,
                output_directory,
                permutation_repetitions=3,
                bootstrap_repetitions=3,
                seed=5,
            )
            self.assertTrue(metrics["leakage_audit"]["whole_puzzle_splits"])
            for name in (
                "metrics.json",
                "final_predictions.json",
                "event_rates.png",
                "model_comparison.png",
                "calibration.png",
                "geometry_coefficients.png",
                "checkpoint_transfer.png",
                "index.html",
            ):
                self.assertTrue(os.path.isfile(os.path.join(output_directory, name)), name)


if __name__ == "__main__":
    unittest.main()
