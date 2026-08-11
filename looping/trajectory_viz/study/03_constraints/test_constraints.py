"""Focused tests for the Sudoku-constraint trajectory study."""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch


STUDY_DIR = Path(__file__).parent
sys.path.insert(0, str(STUDY_DIR))
sys.path.insert(0, str(STUDY_DIR.parents[3]))

import analyze_constraints as analysis  # noqa: E402
import collect_constraints as collection  # noqa: E402


SOLVED_BOARD = (
    "534678912"
    "672195348"
    "198342567"
    "859761423"
    "426853791"
    "713924856"
    "961537284"
    "287419635"
    "345286179"
)


def _empty_inputs(puzzle_count=1):
    inputs = np.zeros((puzzle_count, 81, 10), dtype=np.float32)
    inputs[:, :, 0] = 1.0
    return inputs


def _hard_logits(board):
    digits = np.asarray([int(character) - 1 for character in board])
    logits = np.full((1, 1, 81, 9), -30.0, dtype=np.float32)
    logits[0, 0, np.arange(81), digits] = 30.0
    return logits


class _FakeModel:
    def initial_encoder(self, inputs):
        return inputs[..., :2]

    def output_head(self, hidden_state):
        return torch.cat(
            [hidden_state, torch.zeros(*hidden_state.shape[:-1], 7)],
            dim=-1,
        )

    def recurrent_step(self, hidden_state, predictions, rope_cos, rope_sin):
        del predictions, rope_cos, rope_sin
        return hidden_state + 1


class ConstraintStudyTest(unittest.TestCase):
    def test_peer_matrices_have_sudoku_peer_counts(self):
        matrices = analysis.build_peer_matrices()
        np.testing.assert_array_equal(matrices["row"].sum(axis=1), 8)
        np.testing.assert_array_equal(matrices["column"].sum(axis=1), 8)
        np.testing.assert_array_equal(matrices["box"].sum(axis=1), 8)
        np.testing.assert_array_equal(matrices["unique"].sum(axis=1), 20)
        for matrix in matrices.values():
            np.testing.assert_array_equal(np.diag(matrix), 0)
            np.testing.assert_array_equal(matrix, matrix.T)

    def test_solved_board_has_no_conflicts_and_one_candidate(self):
        targets = np.asarray([
            [int(character) - 1 for character in SOLVED_BOARD]
        ])
        quantities = analysis.derive_constraint_quantities(
            _hard_logits(SOLVED_BOARD),
            _empty_inputs(),
            targets,
        )
        for group in ("row", "column", "box", "unique"):
            np.testing.assert_array_equal(
                quantities[f"hard_{group}_conflicts"],
                0,
            )
        for group in ("row", "column", "box", "joint"):
            np.testing.assert_array_equal(
                quantities[f"{group}_candidate_count"],
                1,
            )
        np.testing.assert_array_equal(quantities["target_candidate_legal"], 1)
        np.testing.assert_array_equal(quantities["prediction_correct"], 1)
        self.assertLess(quantities["expected_unique_conflicts"].max(), 1e-20)

    def test_repeated_digit_board_reports_each_constraint_component(self):
        repeated_board = "1" * 81
        targets = np.zeros((1, 81), dtype=np.int64)
        quantities = analysis.derive_constraint_quantities(
            _hard_logits(repeated_board),
            _empty_inputs(),
            targets,
        )
        np.testing.assert_array_equal(quantities["hard_row_conflicts"], 8)
        np.testing.assert_array_equal(quantities["hard_column_conflicts"], 8)
        np.testing.assert_array_equal(quantities["hard_box_conflicts"], 8)
        np.testing.assert_array_equal(quantities["hard_unique_conflicts"], 20)
        np.testing.assert_array_equal(quantities["row_candidate_count"], 8)
        np.testing.assert_array_equal(quantities["joint_candidate_count"], 8)
        np.testing.assert_array_equal(quantities["target_candidate_legal"], 0)

    def test_fixed_clues_override_model_predictions(self):
        inputs = _empty_inputs()
        inputs[0, 0] = 0
        inputs[0, 0, 5] = 1.0
        targets = np.asarray([
            [int(character) - 1 for character in SOLVED_BOARD]
        ])
        logits = _hard_logits("1" * 81)
        quantities = analysis.derive_constraint_quantities(logits, inputs, targets)
        self.assertEqual(quantities["hard_predictions"][0, 0, 0], 4)

    def test_balanced_split_assignment_is_per_bucket(self):
        buckets = [name for name in ("easy", "hard") for _ in range(12)]
        splits = collection.assign_balanced_splits(buckets, 4)
        expected = ["discovery"] * 4 + ["validation"] * 4 + ["final"] * 4
        self.assertEqual(splits[:12], expected)
        self.assertEqual(splits[12:], expected)

    def test_snapshot_collection_uses_completed_iteration_count(self):
        inputs = torch.zeros(2, 81, 10)
        states, logits = collection.collect_snapshots(
            _FakeModel(),
            inputs,
            snapshots=(0, 2, 4),
        )
        self.assertEqual(states.shape, (2, 3, 81, 2))
        self.assertEqual(logits.shape, (2, 3, 81, 9))
        self.assertTrue(torch.all(states[:, 0] == 0))
        self.assertTrue(torch.all(states[:, 1] == 2))
        self.assertTrue(torch.all(states[:, 2] == 4))

    def test_numeric_probe_transfers_linear_quantity_and_beats_controls(self):
        generator = np.random.default_rng(3)
        coefficient = generator.normal(size=6)
        features = [generator.normal(size=(300, 6)) for _ in range(3)]
        targets = [
            np.einsum("nd,d->n", values, coefficient)
            for values in features
        ]
        standardized = analysis._standardize_features(*features)
        result, predictions = analysis.fit_numeric_probe(
            *standardized,
            *targets,
            control_key="linear-test",
        )
        self.assertGreater(result["final"]["r2"], 0.99)
        self.assertGreater(
            result["final"]["r2"],
            result["label_shuffle"]["maximum_r2"],
        )
        self.assertGreater(np.corrcoef(targets[-1], predictions)[0, 1], 0.99)

    def test_numeric_probe_marks_constant_holdout_r2_undefined(self):
        generator = np.random.default_rng(12)
        features = [generator.normal(size=(100, 4)) for _ in range(3)]
        targets = [np.ones(100) for _ in range(3)]
        standardized = analysis._standardize_features(*features)
        result, _ = analysis.fit_numeric_probe(
            *standardized,
            *targets,
            control_key="constant-test",
        )
        self.assertIsNone(result["final"]["r2"])
        self.assertIsNone(result["label_shuffle"]["mean_r2"])
        self.assertIsNone(
            result["matched_rank_random_subspace"]["maximum_r2"]
        )

    def test_categorical_probe_handles_nonordered_classes(self):
        generator = np.random.default_rng(4)
        features = [generator.normal(size=(500, 5)) for _ in range(3)]
        targets = [
            np.where(
                values[:, 0] > 0.5,
                2,
                np.where(values[:, 1] > 0, 1, 0),
            )
            for values in features
        ]
        standardized = analysis._standardize_features(*features)
        result, _ = analysis.fit_categorical_probe(
            *standardized,
            *targets,
            classes=(0, 1, 2),
            control_key="categorical-test",
        )
        self.assertGreater(result["final"]["balanced_accuracy"], 0.85)
        self.assertGreater(
            result["final"]["balanced_accuracy"],
            result["label_shuffle"]["maximum_balanced_accuracy"],
        )

    def test_iteration_shuffle_detects_monotonic_reduction(self):
        result = analysis.iteration_shuffle_test(
            np.arange(12),
            np.arange(12, 0, -1),
            desired_direction=-1,
            seed=8,
        )
        self.assertEqual(result["ordered_spearman"], 1.0)
        self.assertLess(result["one_sided_p"], 0.01)

    def test_time_row_permutation_preserves_puzzle_cell_trajectories(self):
        dataset = {
            "puzzle_ids": np.repeat([4, 7], 6),
            "time_slots": np.tile(np.repeat([0, 1, 2], 2), 2),
            "cell_ids": np.tile([10, 20], 6),
        }
        values = np.column_stack((
            dataset["puzzle_ids"],
            dataset["time_slots"],
            dataset["cell_ids"],
        ))
        shuffled = analysis._permute_time_rows(
            values, dataset, np.asarray([2, 0, 1])
        )
        np.testing.assert_array_equal(shuffled[:, 0], values[:, 0])
        np.testing.assert_array_equal(shuffled[:, 2], values[:, 2])
        np.testing.assert_array_equal(
            shuffled[:, 1], np.asarray([2, 2, 0, 0, 1, 1] * 2)
        )

    def test_residual_probe_predicts_signal_beyond_nuisance(self):
        generator = np.random.default_rng(19)
        datasets = {}
        for split_index, split_name in enumerate(
            ("discovery", "validation", "final")
        ):
            puzzle_count, time_count, cells = 8, 3, 12
            row_count = puzzle_count * time_count * cells
            puzzle_ids = np.repeat(np.arange(puzzle_count), time_count * cells)
            time_slots = np.tile(
                np.repeat(np.arange(time_count), cells), puzzle_count
            )
            cell_ids = np.tile(np.arange(cells), puzzle_count * time_count)
            nuisance = np.column_stack((
                np.eye(time_count)[time_slots],
                generator.normal(size=row_count),
            ))
            states = generator.normal(size=(row_count, 8))
            target = (
                0.6 * nuisance[:, -1]
                + 1.8 * states[:, 0]
                - 1.1 * states[:, 1]
                + generator.normal(scale=0.03, size=row_count)
            )
            weights = analysis._equal_puzzle_time_weights(
                puzzle_ids, time_slots
            )
            datasets[split_name] = {
                "states": states,
                "nuisance": nuisance,
                "targets": target,
                "puzzle_ids": puzzle_ids,
                "time_slots": time_slots,
                "cell_ids": cell_ids,
                "weights": weights,
            }
        result, cache = analysis.fit_residual_temporal_probe(
            datasets, control_key=("synthetic", "future_improvement")
        )
        self.assertGreater(result["final"]["state_partial_r2"], 0.9)
        self.assertGreater(
            result["final"]["state_partial_r2"],
            result["matched_rank_random_subspace"]["maximum_partial_r2"],
        )
        transfer = analysis.evaluate_cross_checkpoint_probe(
            cache, datasets["final"]
        )
        self.assertAlmostEqual(
            transfer["state_partial_r2"],
            result["final"]["state_partial_r2"],
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
