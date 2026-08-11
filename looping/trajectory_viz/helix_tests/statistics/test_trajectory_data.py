import unittest
from unittest import mock

import torch
from torch import nn

from looping.trajectory_viz.helix_tests.statistics import trajectory_data
from looping.trajectory_viz.helix_tests.statistics.trajectory_data import (
    CELL_COUNT,
    CollectedTrajectory,
    SNAPSHOT_ITERATIONS,
    build_cell_table,
    build_response_representations,
    collect_trajectory,
    make_rating_stratified_folds,
    output_head_contrast_basis,
    puzzle_equal_weights,
    rating_stratified_fold_ids,
)


class AdditiveDummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Identity()
        self.output_head = nn.Linear(3, 9, bias=False)
        with torch.no_grad():
            self.output_head.weight.copy_(torch.arange(27).view(9, 3) / 10)

    def recurrent_step(self, hidden, predictions, rope_cos, rope_sin):
        del predictions, rope_cos, rope_sin
        return hidden + torch.tensor([1.0, 2.0, 3.0])


def _trajectory(logits, targets, blank_mask, hidden_dimension=3):
    puzzle_count, time_count = logits.shape[:2]
    hidden = torch.zeros(puzzle_count, time_count, CELL_COUNT, hidden_dimension)
    return CollectedTrajectory(
        iterations=torch.arange(time_count),
        hidden_states=hidden,
        one_step_updates=hidden.clone(),
        logits=logits,
        targets=targets,
        originally_blank=blank_mask,
    )


def test_default_snapshots_cover_requested_horizons():
    assert SNAPSHOT_ITERATIONS == (
        0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024
    )


def test_collect_aligns_states_one_step_updates_and_logits():
    model = AdditiveDummyModel()
    inputs = torch.zeros(2, CELL_COUNT, 3)
    targets = torch.zeros(2, CELL_COUNT, dtype=torch.long)
    blank_mask = torch.ones(2, CELL_COUNT, dtype=torch.bool)
    trajectory = collect_trajectory(
        model,
        inputs,
        targets,
        blank_mask,
        iterations=(0, 1, 2, 4),
    )

    expected_iterations = torch.tensor([0.0, 1.0, 2.0, 4.0])
    expected_state = expected_iterations.view(1, 4, 1, 1) * torch.tensor(
        [1.0, 2.0, 3.0]
    )
    assert trajectory.hidden_states.shape == (2, 4, CELL_COUNT, 3)
    assert torch.allclose(trajectory.hidden_states, expected_state.expand_as(trajectory.hidden_states))
    assert torch.allclose(
        trajectory.one_step_updates,
        torch.tensor([1.0, 2.0, 3.0]).expand_as(trajectory.one_step_updates),
    )
    assert torch.allclose(
        trajectory.logits,
        model.output_head(trajectory.hidden_states),
    )


def test_cell_table_derives_requested_outputs_and_fixed_primary_mask():
    logits = torch.zeros(2, 2, CELL_COUNT, 9)
    targets = torch.zeros(2, CELL_COUNT, dtype=torch.long)
    targets[0, 0] = 1
    logits[0, 0, 0, 1] = 3.0
    logits[0, 0, 0, 2] = 2.0
    logits[0, 1, 0, 1] = 1.0
    logits[0, 1, 0, 2] = 4.0
    blank_mask = torch.zeros(2, CELL_COUNT, dtype=torch.bool)
    blank_mask[0, :2] = True
    blank_mask[1, :5] = True
    table = build_cell_table(
        _trajectory(logits, targets, blank_mask),
        rating_buckets=("easy", "hard"),
        fold_id_by_puzzle=torch.tensor([1, 0]),
    )

    assert table.predicted_digit[0, 0, 0] == 1
    assert table.predicted_digit[0, 1, 0] == 2
    assert table.correct[0, 0, 0]
    assert not table.correct[0, 1, 0]
    assert torch.isclose(table.target_minus_max_wrong_logit_margin[0, 0, 0], torch.tensor(1.0))
    assert torch.isclose(table.target_minus_max_wrong_logit_margin[0, 1, 0], torch.tensor(-3.0))
    assert torch.isclose(table.top1_top2_margin[0, 0, 0], torch.tensor(1.0))
    probabilities = logits[0, 0, 0].softmax(0)
    assert torch.isclose(table.max_confidence[0, 0, 0], probabilities.max())
    assert torch.isclose(table.true_probability[0, 0, 0], probabilities[1])
    assert torch.equal(table.primary_mask[:, 0], blank_mask)
    assert torch.equal(table.primary_mask[:, 0], table.primary_mask[:, 1])
    assert (table.row_index[0, 0, 31], table.column_index[0, 0, 31]) == (3, 4)
    assert table.box_index[0, 0, 31] == 4
    assert table.fold_index[0, 0, 0] == 1
    assert table.fold_index[1, 1, 80] == 0

    flattened = table.flatten_primary()
    assert len(flattened["true_digit"]) == 2 * (2 + 5)
    puzzle_totals = torch.stack([
        flattened["sample_weight"][flattened["puzzle_index"] == puzzle].sum()
        for puzzle in range(2)
    ])
    assert torch.allclose(puzzle_totals, torch.tensor([0.5, 0.5], dtype=torch.float64))


def test_puzzle_weights_equalize_unequal_observation_counts():
    puzzle_ids = torch.tensor([10, 10, 20, 20, 20, 20])
    weights = puzzle_equal_weights(puzzle_ids)
    assert torch.isclose(weights[puzzle_ids == 10].sum(), torch.tensor(0.5, dtype=torch.float64))
    assert torch.isclose(weights[puzzle_ids == 20].sum(), torch.tensor(0.5, dtype=torch.float64))
    assert torch.isclose(weights.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_rating_stratified_folds_hold_out_whole_puzzles():
    buckets = tuple(bucket for bucket in ("easy", "medium", "hard") for _ in range(6))
    fold_ids = rating_stratified_fold_ids(buckets, n_splits=3, seed=7)
    folds = make_rating_stratified_folds(buckets, n_splits=3, seed=7)
    assert torch.equal(fold_ids, rating_stratified_fold_ids(buckets, n_splits=3, seed=7))
    assert sorted(torch.cat([fold.test_puzzle_indices for fold in folds]).tolist()) == list(range(18))
    for fold in folds:
        assert not set(fold.train_puzzle_indices.tolist()) & set(fold.test_puzzle_indices.tolist())
        held_out_buckets = [buckets[index] for index in fold.test_puzzle_indices]
        assert held_out_buckets.count("easy") == 2
        assert held_out_buckets.count("medium") == 2
        assert held_out_buckets.count("hard") == 2

    non_divisible = tuple(
        bucket for bucket in ("easy", "medium", "hard") for _ in range(5)
    )
    balanced_fold_ids = rating_stratified_fold_ids(
        non_divisible, n_splits=3, seed=7
    )
    assert torch.bincount(balanced_fold_ids).tolist() == [5, 5, 5]


def test_response_directions_are_cell_centered_and_remove_output_contrasts():
    time = torch.tensor([-1.0, 0.0, 1.0]).view(1, 3, 1, 1)
    cell_scale = torch.cat([
        -torch.ones(40),
        torch.full((41,), 40 / 41),
    ]).view(1, 1, CELL_COUNT, 1)
    hidden_pattern = torch.tensor([2.0, 3.0, 0.0]).view(1, 1, 1, 3)
    update_pattern = torch.tensor([1.0, -1.0, 2.0]).view(1, 1, 1, 3)
    hidden_global_motion = time * torch.tensor([5.0, -2.0, 7.0])
    update_global_motion = time * torch.tensor([-3.0, 4.0, 1.0])
    hidden = cell_scale * hidden_pattern + hidden_global_motion
    updates = cell_scale * update_pattern + update_global_motion
    trajectory = CollectedTrajectory(
        iterations=torch.tensor([0, 1, 2]),
        hidden_states=hidden,
        one_step_updates=updates,
        logits=torch.zeros(1, 3, CELL_COUNT, 9),
        targets=torch.zeros(1, CELL_COUNT, dtype=torch.long),
        originally_blank=torch.ones(1, CELL_COUNT, dtype=torch.bool),
    )
    output_weight = torch.zeros(9, 3)
    output_weight[:, 0] = torch.arange(9)
    responses = build_response_representations(trajectory, output_weight)

    assert responses.output_contrast_basis.shape == (1, 3)
    outside = responses.hidden_direction_outside_output_contrast
    assert torch.allclose(outside[..., 0], torch.zeros_like(outside[..., 0]), atol=1e-6)
    assert torch.allclose(outside.norm(dim=-1), torch.ones(1, 3, CELL_COUNT))
    assert torch.allclose(
        responses.cell_centered_unit_hidden_direction[:, 0],
        responses.cell_centered_unit_hidden_direction[:, 2],
    )
    assert torch.allclose(
        responses.cell_centered_unit_update_direction.norm(dim=-1),
        torch.ones(1, 3, CELL_COUNT),
    )
    scale = responses.scale_statistics
    assert torch.allclose(
        scale["output_null_energy_fraction"],
        torch.full((3,), 9.0 / 13.0),
        atol=1e-6,
    )
    assert torch.all(scale["cell_centered_update_rms_norm"] > 0)
    assert torch.equal(
        scale["update_near_zero_fraction_at_1e-8"], torch.zeros(3)
    )


def test_balanced_loader_wrapper_preserves_rating_and_original_blank():
    inputs = torch.zeros(4, CELL_COUNT, 10)
    targets = torch.zeros(4, CELL_COUNT, dtype=torch.long)
    blank_mask = torch.zeros(4, CELL_COUNT, dtype=torch.bool)
    blank_mask[:, ::2] = True
    returned = (
        inputs,
        targets,
        blank_mask,
        ["p" * CELL_COUNT] * 4,
        ["s" * CELL_COUNT] * 4,
        ["easy", "easy", "hard", "hard"],
    )
    with mock.patch.object(
        trajectory_data,
        "_load_balanced_sample",
        return_value=returned,
    ):
        sample = trajectory_data.load_balanced_trajectory_sample(2, seed=9)
    assert sample.rating_buckets == ("easy", "easy", "hard", "hard")
    assert torch.equal(sample.originally_blank, blank_mask)
    assert sample.originally_blank.data_ptr() == sample.empty_mask.data_ptr()


def test_prepare_bundle_exposes_aligned_flat_fit_inputs():
    inputs = torch.zeros(4, CELL_COUNT, 3)
    targets = torch.arange(CELL_COUNT).remainder(9).repeat(4, 1)
    blank_mask = torch.zeros(4, CELL_COUNT, dtype=torch.bool)
    blank_mask[:, :3] = True
    returned = (
        inputs,
        targets,
        blank_mask,
        ["p" * CELL_COUNT] * 4,
        ["s" * CELL_COUNT] * 4,
        ["easy", "easy", "hard", "hard"],
    )
    with mock.patch.object(
        trajectory_data,
        "_load_balanced_sample",
        return_value=returned,
    ):
        prepared = trajectory_data.prepare_trajectory_data(
            AdditiveDummyModel(),
            examples_per_bucket=2,
            n_splits=2,
            iterations=(0, 1),
            device="cpu",
        )

    flat = prepared.flatten_primary()
    expected_rows = 4 * 2 * 3
    assert flat["true_digit"].shape == (expected_rows,)
    assert flat["cell_centered_unit_hidden_direction"].shape == (expected_rows, 3)
    assert flat["sample_weight"].shape == (expected_rows,)
    assert set(flat["fold_index"].tolist()) == {0, 1}
    for puzzle in range(4):
        chosen = flat["puzzle_index"] == puzzle
        assert torch.isclose(
            flat["sample_weight"][chosen].sum(),
            torch.tensor(0.25, dtype=torch.float64),
        )


def test_centered_output_head_basis_is_a_contrast_space():
    weight = torch.randn(9, 12)
    basis = output_head_contrast_basis(weight)
    centered = weight - weight.mean(0, keepdim=True)
    residual = centered - (centered @ basis.T) @ basis
    assert basis.size(0) <= 8
    assert torch.linalg.matrix_norm(residual) < 1e-5


def load_tests(loader, standard_tests, pattern):
    del loader, standard_tests, pattern
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value))
    return suite


if __name__ == "__main__":
    unittest.main()
