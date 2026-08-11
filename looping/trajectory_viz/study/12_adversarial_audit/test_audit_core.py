import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_core import (
    PLANES,
    aggregate_metric,
    benjamini_hochberg,
    build_representations,
    fit_pca_numpy,
    local_pca_coordinates,
    make_three_way_split,
    project,
    random_orthonormal_coefficients,
    select_plane,
    synthetic_selection_experiment,
    time_shuffle_test,
    write_json,
)


class SplitTests(unittest.TestCase):
    def test_three_way_split_is_balanced_and_disjoint(self):
        buckets = tuple(bucket for bucket in ("a", "b", "c") for _ in range(6))
        splits = make_three_way_split(
            buckets, seed=17, puzzles_per_bucket_per_split=2
        )
        all_indices = np.concatenate(tuple(splits.values()))
        self.assertEqual(len(np.unique(all_indices)), len(buckets))
        for indices in splits.values():
            counts = {bucket: 0 for bucket in ("a", "b", "c")}
            for index in indices:
                counts[buckets[index]] += 1
            self.assertEqual(set(counts.values()), {2})

    def test_split_rejects_wrong_bucket_size(self):
        with self.assertRaises(ValueError):
            make_three_way_split(
                ("a",) * 5, seed=1, puzzles_per_bucket_per_split=2
            )


class ProjectionTests(unittest.TestCase):
    def test_pca_fit_does_not_depend_on_unseen_values(self):
        generator = np.random.default_rng(3)
        discovery = generator.normal(size=(4, 10, 7)).astype(np.float32)
        unseen = generator.normal(size=(2, 10, 7)).astype(np.float32)
        first = fit_pca_numpy(discovery, rank=3)
        unseen *= 1000.0
        second = fit_pca_numpy(discovery, rank=3)
        np.testing.assert_allclose(
            np.abs(first["basis"]), np.abs(second["basis"]), atol=1e-7
        )

    def test_random_projection_coefficients_are_orthonormal(self):
        coefficients = random_orthonormal_coefficients(12, 3, 5, seed=9)
        for coefficient in coefficients:
            np.testing.assert_allclose(
                coefficient.T @ coefficient, np.eye(3), atol=1e-5
            )

    def test_representations_anchor_states_but_not_updates(self):
        generator = np.random.default_rng(2)
        states = generator.normal(size=(3, 5, 2, 4)).astype(np.float32)
        updates = generator.normal(size=(3, 5, 2, 4)).astype(np.float32)
        representations = build_representations(states, updates)
        np.testing.assert_allclose(representations["raw_state"][:, 0], 0.0)
        np.testing.assert_allclose(representations["normalized_state"][:, 0], 0.0)
        update_norms = np.linalg.norm(
            representations["normalized_update"], axis=-1
        )
        np.testing.assert_allclose(update_norms, 1.0, atol=1e-6)

    def test_project_preserves_puzzle_and_time_axes(self):
        generator = np.random.default_rng(11)
        values = generator.normal(size=(3, 8, 6)).astype(np.float32)
        fit = fit_pca_numpy(values[:2], rank=3)
        coordinates = project(values[2:], fit["mean"], fit["basis"])
        self.assertEqual(coordinates.shape, (1, 8, 3))


class MetricTests(unittest.TestCase):
    @staticmethod
    def helix(puzzles=12, steps=40):
        time = np.linspace(0.0, 1.0, steps)
        base = np.stack(
            [np.cos(2 * np.pi * time), np.sin(2 * np.pi * time), time],
            axis=1,
        )
        return np.stack([base + 0.005 * index for index in range(puzzles)])

    def test_ordered_helix_beats_shuffled_time(self):
        result = time_shuffle_test(
            self.helix(),
            "helix",
            plane=PLANES[0],
            permutations=99,
            bootstraps=200,
            seed=4,
        )
        self.assertLessEqual(result["time_shuffle_p"], 0.02)
        self.assertGreater(result["ordered_minus_shuffle_effect"], 0.3)
        self.assertGreater(
            result["puzzle_bootstrap_95_percent_effect_interval"][0], 0.0
        )

    def test_validation_plane_selection_recovers_helix_plane(self):
        helix = self.helix()
        coordinates = helix[:, :, [2, 0, 1]]
        plane, score = select_plane(coordinates, "helix")
        self.assertEqual(tuple(plane), (1, 2, 0))
        self.assertGreater(score, 0.5)

    def test_local_pca_returns_requested_rank(self):
        generator = np.random.default_rng(5)
        values = np.cumsum(generator.normal(size=(4, 20, 30)), axis=1)
        self.assertEqual(local_pca_coordinates(values, rank=5).shape, (4, 20, 5))

    def test_synthetic_selection_never_scores_below_fixed_axes(self):
        result = synthetic_selection_experiment(
            seed=8,
            candidates=8,
            time_count=20,
            feature_count=16,
            local_rank=6,
        )
        for metrics in result.values():
            self.assertGreaterEqual(
                metrics["local_pca_selected_axes"]["median"],
                metrics["honest_fixed_axes"]["median"],
            )

    def test_benjamini_hochberg_is_monotone_in_sorted_order(self):
        p_values = np.array([0.04, 0.001, 0.02, 0.5])
        adjusted = benjamini_hochberg(p_values)
        order = np.argsort(p_values)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= -1e-12))
        self.assertTrue(np.all(adjusted >= p_values))


class SerializationTests(unittest.TestCase):
    def test_write_json_converts_numpy_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"array": np.array([1, 2]), "scalar": np.float32(3)})
            with open(path) as input_file:
                result = json.load(input_file)
            self.assertEqual(result, {"array": [1, 2], "scalar": 3.0})


class ProtocolContractTests(unittest.TestCase):
    def test_preregistered_sample_and_controls_meet_protocol_minima(self):
        criteria_path = Path(__file__).resolve().parent / "acceptance_criteria.json"
        criteria = json.loads(criteria_path.read_text())
        self.assertEqual(criteria["sample"]["puzzles_per_split"], 20)
        self.assertEqual(
            criteria["sample"]["puzzles_per_bucket_per_split"] * 5, 20
        )
        self.assertGreaterEqual(
            criteria["projection"]["matched_rank_random_projections"], 100
        )
        self.assertGreaterEqual(criteria["resampling"]["time_shuffles"], 499)
        self.assertTrue(
            criteria["projection"]["per_puzzle_full_trajectory_pca_is_control_only"]
        )

    def test_final_runner_refuses_a_second_holdout_pass(self):
        import run_audit

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "robustness_metrics.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                run_audit.run_final(
                    output_dir,
                    checkpoints={},
                    arrow_path=Path("unused.arrow"),
                )


if __name__ == "__main__":
    unittest.main()
