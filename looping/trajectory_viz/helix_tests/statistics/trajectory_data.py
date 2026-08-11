"""In-memory trajectory data for held-out helix tests.

The primary observations are cells that were blank in the original puzzle.
Nothing in this module changes that mask based on a later prediction.  Hidden
states, one-step updates, and logits use the common layout
``[puzzle, snapshot, cell, feature]`` so every derived array remains aligned.

This module deliberately has no serialization functions.  The hidden states
are analysis inputs, not artifacts that should be copied into result files.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

import stabilize.exp_testbed_20k as model_module
from looping.eval_loop_diagnostics import _load_balanced_sample


SNAPSHOT_ITERATIONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
GRID_SIZE = 9
CELL_COUNT = GRID_SIZE * GRID_SIZE
DIGIT_COUNT = 9


@dataclass(frozen=True)
class BalancedTrajectorySample:
    """A rating-balanced test sample returned by the canonical loader."""

    inputs: torch.Tensor
    targets: torch.Tensor
    originally_blank: torch.Tensor
    puzzles: tuple[str, ...]
    solutions: tuple[str, ...]
    rating_buckets: tuple[str, ...]

    @property
    def puzzle_count(self):
        return self.inputs.size(0)

    @property
    def empty_mask(self):
        """Compatibility alias for the canonical loader's name."""

        return self.originally_blank

    def to(self, device):
        return BalancedTrajectorySample(
            inputs=self.inputs.to(device),
            targets=self.targets.to(device),
            originally_blank=self.originally_blank.to(device),
            puzzles=self.puzzles,
            solutions=self.solutions,
            rating_buckets=self.rating_buckets,
        )


@dataclass(frozen=True)
class PuzzleFold:
    """One whole-puzzle train/test split."""

    fold_index: int
    train_puzzle_indices: torch.Tensor
    test_puzzle_indices: torch.Tensor


@dataclass(frozen=True)
class CollectedTrajectory:
    """Snapshot tensors kept in memory and moved to ``output_device``."""

    iterations: torch.Tensor
    hidden_states: torch.Tensor
    one_step_updates: torch.Tensor
    logits: torch.Tensor
    targets: torch.Tensor
    originally_blank: torch.Tensor

    @property
    def states(self):
        """Short compatibility alias used by older trajectory diagnostics."""

        return self.hidden_states

    @property
    def updates(self):
        return self.one_step_updates


@dataclass(frozen=True)
class ResponseRepresentations:
    """Direction-valued responses for the statistical fits."""

    cell_centered_unit_hidden_direction: torch.Tensor
    hidden_direction_outside_output_contrast: torch.Tensor
    cell_centered_unit_update_direction: torch.Tensor
    output_contrast_basis: torch.Tensor
    scale_statistics: dict[str, torch.Tensor]

    def flatten(self, mask):
        """Return aligned response matrices selected by a cell/snapshot mask."""

        _validate_mask_shape(mask, self.cell_centered_unit_hidden_direction.shape[:3])
        return {
            "cell_centered_unit_hidden_direction": (
                self.cell_centered_unit_hidden_direction[mask]
            ),
            "hidden_direction_outside_output_contrast": (
                self.hidden_direction_outside_output_contrast[mask]
            ),
            "cell_centered_unit_update_direction": (
                self.cell_centered_unit_update_direction[mask]
            ),
        }


@dataclass(frozen=True)
class CellTrajectoryTable:
    """Scalar cell metadata and model outputs aligned to trajectory tensors."""

    iteration: torch.Tensor
    puzzle_index: torch.Tensor
    cell_index: torch.Tensor
    row_index: torch.Tensor
    column_index: torch.Tensor
    box_index: torch.Tensor
    rating_bucket_index: torch.Tensor
    fold_index: torch.Tensor
    true_digit: torch.Tensor
    predicted_digit: torch.Tensor
    max_confidence: torch.Tensor
    true_probability: torch.Tensor
    target_minus_max_wrong_logit_margin: torch.Tensor
    top1_top2_margin: torch.Tensor
    top1_top2_probability_margin: torch.Tensor
    correct: torch.Tensor
    originally_blank: torch.Tensor
    primary_mask: torch.Tensor
    primary_equal_puzzle_weight: torch.Tensor
    rating_bucket_names: tuple[str, ...]

    def flatten(self, mask):
        """Flatten every tensor column using one aligned boolean mask."""

        _validate_mask_shape(mask, self.primary_mask.shape)
        result = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, torch.Tensor):
                result[field.name] = value[mask]
        return result

    def flatten_primary(self):
        """Return the fixed originally-blank observations used by all fits."""

        result = self.flatten(self.primary_mask)
        result["sample_weight"] = result["primary_equal_puzzle_weight"]
        return result


@dataclass(frozen=True)
class PreparedTrajectoryData:
    """High-level bundle consumed by cyclic and linear fit code."""

    sample: BalancedTrajectorySample
    trajectory: CollectedTrajectory
    table: CellTrajectoryTable
    responses: ResponseRepresentations
    folds: tuple[PuzzleFold, ...]
    fold_id_by_puzzle: torch.Tensor

    def flatten_primary(self):
        result = self.table.flatten_primary()
        result.update(self.responses.flatten(self.table.primary_mask))
        return result


def _validate_sample_tensors(inputs, targets, originally_blank):
    if inputs.ndim != 3 or inputs.shape[:2] != (targets.size(0), CELL_COUNT):
        raise ValueError(
            "inputs must have shape [puzzles, 81, input_features]"
        )
    if targets.shape != (inputs.size(0), CELL_COUNT):
        raise ValueError("targets must have shape [puzzles, 81]")
    if originally_blank.shape != targets.shape:
        raise ValueError("originally_blank must have shape [puzzles, 81]")
    if originally_blank.dtype != torch.bool:
        raise ValueError("originally_blank must be a boolean tensor")


def _validate_iterations(iterations):
    iterations = tuple(iterations)
    if not iterations:
        raise ValueError("iterations must not be empty")
    if iterations != tuple(sorted(set(iterations))):
        raise ValueError("iterations must be strictly increasing")
    if iterations[0] < 0:
        raise ValueError("iterations must be non-negative")
    return iterations


def _validate_mask_shape(mask, expected_shape):
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"mask must be boolean with shape {tuple(expected_shape)}"
        )


def load_balanced_trajectory_sample(examples_per_bucket, seed=42):
    """Load the canonical rating-balanced test sample without changing order."""

    if not isinstance(examples_per_bucket, int) or examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be a positive integer")
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    empty_mask = empty_mask.bool()
    _validate_sample_tensors(inputs, targets, empty_mask)
    if len(bucket_names) != len(inputs):
        raise ValueError("the balanced loader returned misaligned rating buckets")
    return BalancedTrajectorySample(
        inputs=inputs,
        targets=targets.long(),
        originally_blank=empty_mask,
        puzzles=tuple(puzzles),
        solutions=tuple(solutions),
        rating_buckets=tuple(bucket_names),
    )


def rating_stratified_fold_ids(rating_buckets, n_splits=5, seed=42):
    """Assign each whole puzzle to one test fold within its rating bucket.

    Every rating bucket must contain at least ``n_splits`` puzzles.  Requiring
    this condition makes each held-out fold genuinely rating-stratified instead
    of silently dropping small strata from some folds.
    """

    rating_buckets = tuple(rating_buckets)
    if not rating_buckets:
        raise ValueError("rating_buckets must not be empty")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("n_splits must be an integer of at least 2")

    fold_ids = torch.full((len(rating_buckets),), -1, dtype=torch.long)
    generator = torch.Generator().manual_seed(seed)
    bucket_order = tuple(dict.fromkeys(rating_buckets))
    for bucket_index, bucket in enumerate(bucket_order):
        indices = torch.tensor(
            [index for index, value in enumerate(rating_buckets) if value == bucket],
            dtype=torch.long,
        )
        if len(indices) < n_splits:
            raise ValueError(
                f"rating bucket {bucket!r} has {len(indices)} puzzles, fewer "
                f"than n_splits={n_splits}"
            )
        shuffled = indices[torch.randperm(len(indices), generator=generator)]
        # Rotate the fold receiving each stratum's first remainder so several
        # non-divisible strata do not all make fold zero larger.
        fold_ids[shuffled] = (
            torch.arange(len(shuffled)) + bucket_index
        ) % n_splits

    if (fold_ids < 0).any():
        raise RuntimeError("some puzzles were not assigned to a fold")
    return fold_ids


def make_rating_stratified_folds(rating_buckets, n_splits=5, seed=42):
    """Build rating-stratified splits with no cell-level puzzle leakage."""

    fold_ids = rating_stratified_fold_ids(rating_buckets, n_splits, seed)
    puzzle_indices = torch.arange(len(fold_ids))
    return tuple(
        PuzzleFold(
            fold_index=fold_index,
            train_puzzle_indices=puzzle_indices[fold_ids != fold_index],
            test_puzzle_indices=puzzle_indices[fold_ids == fold_index],
        )
        for fold_index in range(n_splits)
    )


def puzzle_equal_weights(puzzle_indices, *, normalize=True):
    """Give every represented puzzle the same total observation weight."""

    if puzzle_indices.ndim != 1 or not puzzle_indices.numel():
        raise ValueError("puzzle_indices must be a non-empty one-dimensional tensor")
    _, inverse, counts = torch.unique(
        puzzle_indices.long(), sorted=True, return_inverse=True, return_counts=True
    )
    weights = counts[inverse].double().reciprocal()
    if normalize:
        weights = weights / counts.numel()
    return weights


def collect_trajectory(
    model,
    inputs,
    targets,
    originally_blank,
    *,
    iterations=SNAPSHOT_ITERATIONS,
    output_device="cpu",
):
    """Collect states, next-step updates, and logits at sparse snapshots.

    The update stored at iteration ``t`` is exactly ``hidden[t + 1] -
    hidden[t]``.  Consequently the final snapshot at 1024 performs one extra
    recurrent step to obtain its update, but does not retain state 1025.
    """

    iterations = _validate_iterations(iterations)
    _validate_sample_tensors(inputs, targets, originally_blank)
    if inputs.device != targets.device or inputs.device != originally_blank.device:
        raise ValueError("inputs, targets, and originally_blank must share a device")

    selected = set(iterations)
    maximum_iteration = iterations[-1]
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    states = []
    updates = []
    logits = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            hidden = model.initial_encoder(inputs)
            predictions = torch.zeros(
                inputs.size(0), CELL_COUNT, DIGIT_COUNT,
                device=inputs.device,
                dtype=hidden.dtype,
            )
            snapshot_logits = model.output_head(hidden)
            for iteration in range(maximum_iteration + 1):
                next_hidden = model.recurrent_step(
                    hidden, predictions, rope_cos, rope_sin
                )
                if iteration in selected:
                    states.append(
                        hidden.detach().to(output_device, dtype=torch.float32)
                    )
                    updates.append(
                        (next_hidden - hidden).detach().to(
                            output_device, dtype=torch.float32
                        )
                    )
                    logits.append(
                        snapshot_logits.detach().to(
                            output_device, dtype=torch.float32
                        )
                    )
                hidden = next_hidden
                if iteration < maximum_iteration:
                    snapshot_logits = model.output_head(hidden)
                    predictions = F.softmax(snapshot_logits, dim=-1)
    finally:
        model.train(was_training)

    return CollectedTrajectory(
        iterations=torch.tensor(iterations, dtype=torch.long, device=output_device),
        hidden_states=torch.stack(states, dim=1),
        one_step_updates=torch.stack(updates, dim=1),
        logits=torch.stack(logits, dim=1),
        targets=targets.detach().to(output_device, dtype=torch.long),
        originally_blank=originally_blank.detach().to(
            output_device, dtype=torch.bool
        ),
    )


def output_head_contrast_basis(output_head_or_weight, *, rtol=None, device="cpu"):
    """Orthonormal basis for centered digit-logit contrasts in hidden space."""

    weight = getattr(output_head_or_weight, "weight", output_head_or_weight)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        raise ValueError("output head weight must be a two-dimensional tensor")
    if weight.size(0) != DIGIT_COUNT:
        raise ValueError("output head must contain nine digit rows")
    centered = weight.detach().to(device=device, dtype=torch.float64)
    centered = centered - centered.mean(dim=0, keepdim=True)
    _, singular_values, right_vectors = torch.linalg.svd(
        centered, full_matrices=False
    )
    if rtol is None:
        rtol = max(centered.shape) * torch.finfo(centered.dtype).eps
    if rtol < 0:
        raise ValueError("rtol must be non-negative")
    if singular_values.numel() == 0 or singular_values[0] == 0:
        return centered.new_empty((0, centered.size(1)), dtype=torch.float32)
    rank = int((singular_values > rtol * singular_values[0]).sum().item())
    return right_vectors[:rank].to(dtype=torch.float32)


def project_outside_rowspace(values, row_basis):
    """Remove the projection onto an orthonormal row-space basis."""

    if values.ndim < 2 or row_basis.ndim != 2:
        raise ValueError("values and row_basis must end in one feature dimension")
    if values.size(-1) != row_basis.size(-1):
        raise ValueError("values and row_basis have different feature dimensions")
    basis = row_basis.to(device=values.device, dtype=values.dtype)
    if basis.size(0) == 0:
        return values.clone()
    flat = values.reshape(-1, values.size(-1))
    residual = flat - (flat @ basis.T) @ basis
    return residual.reshape_as(values)


def _cell_center(values):
    if values.ndim != 4:
        raise ValueError("trajectory values must have four dimensions")
    return values.float() - values.float().mean(dim=2, keepdim=True)


def build_response_representations(trajectory, output_head_or_weight, *, eps=1e-12):
    """Construct cell-centered unit hidden and update response directions.

    Cell centering subtracts the mean over all 81 cells independently at each
    puzzle and snapshot.  This removes the board-global state without using
    future snapshots or subtracting a persistent digit representation from a
    fixed cell.  The output-contrast response starts from the same centered
    hidden vectors, removes the rowspace of the digit-centered output-head
    weights, and is then normalized.
    """

    if eps <= 0:
        raise ValueError("eps must be positive")
    centered_hidden = _cell_center(trajectory.hidden_states)
    centered_updates = _cell_center(trajectory.one_step_updates)
    basis = output_head_contrast_basis(
        output_head_or_weight, device=trajectory.hidden_states.device
    )
    outside = project_outside_rowspace(centered_hidden, basis)
    primary_mask = trajectory.originally_blank[:, None, :].expand(
        centered_hidden.shape[:3]
    )

    def energy_by_iteration(values):
        squared_norm = values.square().sum(dim=-1)
        mask = primary_mask.to(dtype=squared_norm.dtype)
        return (squared_norm * mask).sum(dim=(0, 2)) / mask.sum(dim=(0, 2))

    centered_hidden_energy = energy_by_iteration(centered_hidden)
    output_null_energy = energy_by_iteration(outside)
    centered_update_energy = energy_by_iteration(centered_updates)
    update_norm = centered_updates.norm(dim=-1)
    update_near_zero = []
    update_median = []
    for time_index in range(centered_updates.size(1)):
        selected_norm = update_norm[:, time_index][primary_mask[:, time_index]]
        update_near_zero.append((selected_norm <= 1e-8).float().mean())
        update_median.append(selected_norm.median())
    scale_statistics = {
        "cell_centered_hidden_rms_norm": centered_hidden_energy.sqrt(),
        "output_null_hidden_rms_norm": output_null_energy.sqrt(),
        "output_null_energy_fraction": output_null_energy
        / centered_hidden_energy.clamp_min(eps),
        "cell_centered_update_rms_norm": centered_update_energy.sqrt(),
        "relative_update_rms_norm": (
            centered_update_energy / centered_hidden_energy.clamp_min(eps)
        ).sqrt(),
        "cell_centered_update_median_norm": torch.stack(update_median),
        "update_near_zero_fraction_at_1e-8": torch.stack(update_near_zero),
    }
    return ResponseRepresentations(
        cell_centered_unit_hidden_direction=F.normalize(
            centered_hidden, dim=-1, eps=eps
        ),
        hidden_direction_outside_output_contrast=F.normalize(
            outside, dim=-1, eps=eps
        ),
        cell_centered_unit_update_direction=F.normalize(
            centered_updates, dim=-1, eps=eps
        ),
        output_contrast_basis=basis,
        scale_statistics=scale_statistics,
    )


def _expanded_cell_metadata(puzzle_count, time_count, device):
    shape = (puzzle_count, time_count, CELL_COUNT)
    puzzle = torch.arange(puzzle_count, device=device).view(-1, 1, 1).expand(shape)
    cell = torch.arange(CELL_COUNT, device=device).view(1, 1, -1).expand(shape)
    row = cell // GRID_SIZE
    column = cell % GRID_SIZE
    box = (row // 3) * 3 + column // 3
    return puzzle, cell, row, column, box


def build_cell_table(
    trajectory,
    rating_buckets,
    *,
    fold_id_by_puzzle=None,
):
    """Derive aligned digit, confidence, margin, and position columns."""

    logits = trajectory.logits.float()
    if logits.ndim != 4 or logits.size(2) != CELL_COUNT or logits.size(3) != DIGIT_COUNT:
        raise ValueError("logits must have shape [puzzles, snapshots, 81, 9]")
    puzzle_count, time_count = logits.shape[:2]
    if trajectory.targets.shape != (puzzle_count, CELL_COUNT):
        raise ValueError("trajectory targets are misaligned")
    if trajectory.originally_blank.shape != (puzzle_count, CELL_COUNT):
        raise ValueError("trajectory originally_blank is misaligned")
    if tuple(trajectory.iterations.shape) != (time_count,):
        raise ValueError("trajectory iterations are misaligned")
    rating_buckets = tuple(rating_buckets)
    if len(rating_buckets) != puzzle_count:
        raise ValueError("rating_buckets must have one entry per puzzle")

    device = logits.device
    shape = (puzzle_count, time_count, CELL_COUNT)
    puzzle, cell, row, column, box = _expanded_cell_metadata(
        puzzle_count, time_count, device
    )
    iteration = trajectory.iterations.to(device).view(1, -1, 1).expand(shape)
    true_digit = trajectory.targets.to(device).view(
        puzzle_count, 1, CELL_COUNT
    ).expand(shape)
    probabilities = F.softmax(logits, dim=-1)
    predicted_digit = logits.argmax(dim=-1)
    max_confidence = probabilities.max(dim=-1).values
    true_probability = probabilities.gather(
        -1, true_digit.unsqueeze(-1)
    ).squeeze(-1)
    true_logit = logits.gather(-1, true_digit.unsqueeze(-1)).squeeze(-1)
    wrong_mask = F.one_hot(true_digit, num_classes=DIGIT_COUNT).bool()
    max_wrong_logit = logits.masked_fill(wrong_mask, -torch.inf).max(dim=-1).values
    answer_margin = true_logit - max_wrong_logit
    top_two_logits = logits.topk(2, dim=-1).values
    top1_top2_margin = top_two_logits[..., 0] - top_two_logits[..., 1]
    top_two_probabilities = probabilities.topk(2, dim=-1).values
    top1_top2_probability_margin = (
        top_two_probabilities[..., 0] - top_two_probabilities[..., 1]
    )
    correct = predicted_digit == true_digit

    originally_blank = trajectory.originally_blank.to(device).view(
        puzzle_count, 1, CELL_COUNT
    ).expand(shape)
    primary_mask = originally_blank.clone()
    blank_counts = primary_mask[:, 0].sum(dim=1)
    if (blank_counts == 0).any():
        raise ValueError("every puzzle must contain at least one originally blank cell")

    primary_puzzles = puzzle[primary_mask]
    primary_weights = puzzle_equal_weights(primary_puzzles).to(device)
    equal_weights = torch.zeros(shape, dtype=torch.float64, device=device)
    equal_weights[primary_mask] = primary_weights

    bucket_names = tuple(dict.fromkeys(rating_buckets))
    bucket_lookup = {name: index for index, name in enumerate(bucket_names)}
    bucket_by_puzzle = torch.tensor(
        [bucket_lookup[name] for name in rating_buckets],
        dtype=torch.long,
        device=device,
    )
    rating_bucket_index = bucket_by_puzzle.view(-1, 1, 1).expand(shape)

    if fold_id_by_puzzle is None:
        fold_id_by_puzzle = torch.full((puzzle_count,), -1, dtype=torch.long)
    fold_id_by_puzzle = torch.as_tensor(
        fold_id_by_puzzle, dtype=torch.long, device=device
    )
    if fold_id_by_puzzle.shape != (puzzle_count,):
        raise ValueError("fold_id_by_puzzle must have one entry per puzzle")
    fold_index = fold_id_by_puzzle.view(-1, 1, 1).expand(shape)

    return CellTrajectoryTable(
        iteration=iteration,
        puzzle_index=puzzle,
        cell_index=cell,
        row_index=row,
        column_index=column,
        box_index=box,
        rating_bucket_index=rating_bucket_index,
        fold_index=fold_index,
        true_digit=true_digit,
        predicted_digit=predicted_digit,
        max_confidence=max_confidence,
        true_probability=true_probability,
        target_minus_max_wrong_logit_margin=answer_margin,
        top1_top2_margin=top1_top2_margin,
        top1_top2_probability_margin=top1_top2_probability_margin,
        correct=correct,
        originally_blank=originally_blank,
        primary_mask=primary_mask,
        primary_equal_puzzle_weight=equal_weights,
        rating_bucket_names=bucket_names,
    )


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def prepare_trajectory_data(
    model,
    *,
    examples_per_bucket,
    seed=42,
    n_splits=5,
    iterations=SNAPSHOT_ITERATIONS,
    device=None,
):
    """Load, collect, and align all in-memory inputs to held-out fits."""

    sample = load_balanced_trajectory_sample(examples_per_bucket, seed)
    fold_ids = rating_stratified_fold_ids(
        sample.rating_buckets, n_splits=n_splits, seed=seed
    )
    folds = make_rating_stratified_folds(
        sample.rating_buckets, n_splits=n_splits, seed=seed
    )
    resolved_device = torch.device(device) if device is not None else _model_device(model)
    device_sample = sample.to(resolved_device)
    trajectory = collect_trajectory(
        model,
        device_sample.inputs,
        device_sample.targets,
        device_sample.originally_blank,
        iterations=iterations,
        output_device="cpu",
    )
    responses = build_response_representations(trajectory, model.output_head)
    table = build_cell_table(
        trajectory,
        sample.rating_buckets,
        fold_id_by_puzzle=fold_ids,
    )
    return PreparedTrajectoryData(
        sample=sample,
        trajectory=trajectory,
        table=table,
        responses=responses,
        folds=folds,
        fold_id_by_puzzle=fold_ids,
    )
