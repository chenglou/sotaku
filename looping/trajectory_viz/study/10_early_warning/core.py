"""Leakage-safe primitives for the iteration-128 early-warning study.

The feature extractor deliberately has no target or late-state argument.  Every
transition is indexed by its ending iteration, so ``update[128]`` is
``hidden[128] - hidden[127]`` and does not inspect iteration 129.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


MAX_EARLY_ITERATION = 128
EARLY_ITERATIONS = (64, 80, 96, 112, 128)
SPLIT_NAMES = ("discovery", "validation", "final")

GEOMETRY_FEATURES = (
    "log_update_rms_128",
    "update_log_slope_64_128",
    "log_relative_update_128",
    "temporal_update_cosine_112_128",
    "path_efficiency_64_128",
    "turn_cosine_64_96_128",
    "cell_direction_coherence_128",
    "update_norm_cv_128",
    "update_effective_rank_128",
    "state_effective_rank_128",
)
SHUFFLED_GEOMETRY_FEATURES = tuple(
    f"shuffled_iteration_{name}" for name in GEOMETRY_FEATURES
)
OUTPUT_FEATURES = (
    "min_top2_logit_gap_128",
    "mean_top2_logit_gap_128",
    "min_top1_probability_128",
    "mean_entropy_128",
    "prediction_flip_fraction_64_128",
    "mean_top2_gap_change_64_128",
)
DIFFICULTY_FEATURES = ("rating_ordinal", "clue_fraction")


FEATURE_MANIFEST = {
    "maximum_iteration": MAX_EARLY_ITERATION,
    "transition_indexing": "update[t] = hidden[t] - hidden[t - 1]",
    "uses_targets": False,
    "uses_fitted_projection": False,
    "geometry": {
        "log_update_rms_128": "log board RMS norm of the transition ending at 128",
        "update_log_slope_64_128": "OLS slope of log update RMS at 64,80,96,112,128",
        "log_relative_update_128": "log(update RMS / hidden-state RMS) at 128",
        "temporal_update_cosine_112_128": "board-vector cosine between updates ending at 112 and 128",
        "path_efficiency_64_128": "endpoint displacement divided by the sampled path length",
        "turn_cosine_64_96_128": "cosine between state displacements 64->96 and 96->128",
        "cell_direction_coherence_128": "norm of the mean unit cell-update direction",
        "update_norm_cv_128": "coefficient of variation of blank-cell update norms",
        "update_effective_rank_128": "participation ratio of centered blank-cell updates",
        "state_effective_rank_128": "participation ratio of centered blank-cell states",
    },
    "output_baseline": {
        "min_top2_logit_gap_128": "minimum target-free top-1 minus top-2 logit gap",
        "mean_top2_logit_gap_128": "mean target-free top-1 minus top-2 logit gap",
        "min_top1_probability_128": "minimum top-1 probability",
        "mean_entropy_128": "mean digit entropy",
        "prediction_flip_fraction_64_128": "fraction of blank cells whose argmax changes",
        "mean_top2_gap_change_64_128": "mean top-2 gap at 128 minus its value at 64",
    },
    "shuffled_iteration_control": {
        f"shuffled_iteration_{name}": (
            "the matched geometry feature after a fixed label-independent "
            "within-puzzle permutation of the five early snapshots"
        )
        for name in GEOMETRY_FEATURES
    },
}


def assign_balanced_splits(bucket_names, examples_per_split_bucket):
    """Assign whole puzzles to three equal splits within each rating bucket."""

    if examples_per_split_bucket <= 0:
        raise ValueError("examples_per_split_bucket must be positive")
    expected = examples_per_split_bucket * len(SPLIT_NAMES)
    seen = {}
    assignments = []
    for bucket in bucket_names:
        offset = seen.get(bucket, 0)
        split_index = offset // examples_per_split_bucket
        if split_index >= len(SPLIT_NAMES):
            raise ValueError(f"bucket {bucket!r} contains more than {expected} puzzles")
        assignments.append(SPLIT_NAMES[split_index])
        seen[bucket] = offset + 1
    if not seen or any(count != expected for count in seen.values()):
        raise ValueError(f"every rating bucket must contain exactly {expected} puzzles")
    return assignments


def _validate_early_tensors(iterations, states, updates, logits, blank_mask):
    iterations = tuple(int(value) for value in iterations)
    if iterations != EARLY_ITERATIONS:
        raise ValueError(f"iterations must be exactly {EARLY_ITERATIONS}")
    if max(iterations) > MAX_EARLY_ITERATION:
        raise ValueError("post-128 states are forbidden")
    if states.ndim != 4 or updates.shape != states.shape:
        raise ValueError("states and updates must have shape [puzzle,time,cell,feature]")
    if logits.shape[:3] != states.shape[:3] or logits.size(-1) != 9:
        raise ValueError("logits must align with states and have nine digits")
    if blank_mask.dtype != torch.bool or blank_mask.shape != states.shape[:1] + states.shape[2:3]:
        raise ValueError("blank_mask must be boolean with shape [puzzle,cell]")
    if not blank_mask.any(dim=1).all():
        raise ValueError("every puzzle must contain a blank cell")


def _masked_vector_rms(values, mask):
    squared_norm = values.float().square().sum(dim=-1)
    return ((squared_norm * mask).sum(dim=-1) / mask.sum(dim=-1)).clamp_min(1e-24).sqrt()


def _masked_flat_cosine(first, second, mask):
    mask = mask.unsqueeze(-1)
    first = first.float() * mask
    second = second.float() * mask
    numerator = (first * second).sum(dim=(1, 2))
    denominator = first.square().sum(dim=(1, 2)).sqrt() * second.square().sum(dim=(1, 2)).sqrt()
    return numerator / denominator.clamp_min(1e-12)


def _effective_rank(values, mask):
    mask_float = mask.float().unsqueeze(-1)
    count = mask_float.sum(dim=1, keepdim=True)
    mean = (values.float() * mask_float).sum(dim=1, keepdim=True) / count
    centered = (values.float() - mean) * mask_float
    gram = centered @ centered.transpose(1, 2)
    trace = centered.square().sum(dim=(1, 2))
    denominator = gram.square().sum(dim=(1, 2))
    return trace.square() / denominator.clamp_min(1e-24)


def _masked_min(values, mask):
    return values.masked_fill(~mask, torch.inf).min(dim=1).values


def _masked_mean(values, mask):
    return (values * mask).sum(dim=1) / mask.sum(dim=1)


def extract_early_features(iterations, states, updates, logits, blank_mask):
    """Return fixed target-free features computed only through iteration 128."""

    _validate_early_tensors(iterations, states, updates, logits, blank_mask)
    states = states.float()
    updates = updates.float()
    logits = logits.float()
    mask = blank_mask
    time_index = {iteration: index for index, iteration in enumerate(iterations)}
    mask_by_time = mask[:, None, :].expand(states.shape[:3])

    update_rms = _masked_vector_rms(updates, mask_by_time)
    state_rms = _masked_vector_rms(states, mask_by_time)
    x = torch.tensor(iterations, dtype=states.dtype, device=states.device)
    x = (x - 64.0) / 64.0
    centered_x = x - x.mean()
    log_update = update_rms.clamp_min(1e-12).log()
    update_slope = ((log_update - log_update.mean(dim=1, keepdim=True)) * centered_x).sum(dim=1) / centered_x.square().sum()

    index_64 = time_index[64]
    index_96 = time_index[96]
    index_112 = time_index[112]
    index_128 = time_index[128]
    segment_states = states
    segment_mask = mask.unsqueeze(-1)
    segment_lengths = []
    for index in range(segment_states.size(1) - 1):
        difference = (segment_states[:, index + 1] - segment_states[:, index]) * segment_mask
        segment_lengths.append(difference.square().sum(dim=(1, 2)).sqrt())
    endpoint = (states[:, index_128] - states[:, index_64]) * segment_mask
    endpoint_norm = endpoint.square().sum(dim=(1, 2)).sqrt()
    path_length = torch.stack(segment_lengths, dim=1).sum(dim=1)

    displacement_first = (states[:, index_96] - states[:, index_64]) * segment_mask
    displacement_second = (states[:, index_128] - states[:, index_96]) * segment_mask
    turn_cosine = _masked_flat_cosine(displacement_first, displacement_second, mask)

    update_128 = updates[:, index_128]
    update_norms = update_128.norm(dim=-1)
    mean_update_norm = _masked_mean(update_norms, mask)
    variance_update_norm = _masked_mean((update_norms - mean_update_norm[:, None]).square(), mask)
    unit_updates = update_128 / update_norms.unsqueeze(-1).clamp_min(1e-12)
    mean_direction = (unit_updates * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)

    top_values, top_indices = logits.topk(2, dim=-1)
    top_gap = top_values[..., 0] - top_values[..., 1]
    probabilities = logits.softmax(dim=-1)
    top_probability = probabilities.max(dim=-1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)

    features = {
        "log_update_rms_128": update_rms[:, index_128].log(),
        "update_log_slope_64_128": update_slope,
        "log_relative_update_128": (update_rms[:, index_128] / state_rms[:, index_128]).clamp_min(1e-12).log(),
        "temporal_update_cosine_112_128": _masked_flat_cosine(updates[:, index_112], update_128, mask),
        "path_efficiency_64_128": endpoint_norm / path_length.clamp_min(1e-12),
        "turn_cosine_64_96_128": turn_cosine,
        "cell_direction_coherence_128": mean_direction.norm(dim=-1),
        "update_norm_cv_128": variance_update_norm.sqrt() / mean_update_norm.clamp_min(1e-12),
        "update_effective_rank_128": _effective_rank(update_128, mask),
        "state_effective_rank_128": _effective_rank(states[:, index_128], mask),
        "min_top2_logit_gap_128": _masked_min(top_gap[:, index_128], mask),
        "mean_top2_logit_gap_128": _masked_mean(top_gap[:, index_128], mask),
        "min_top1_probability_128": _masked_min(top_probability[:, index_128], mask),
        "mean_entropy_128": _masked_mean(entropy[:, index_128], mask),
        "prediction_flip_fraction_64_128": _masked_mean((top_indices[:, index_64, :, 0] != top_indices[:, index_128, :, 0]).float(), mask),
        "mean_top2_gap_change_64_128": _masked_mean(top_gap[:, index_128] - top_gap[:, index_64], mask),
    }
    for name, values in features.items():
        if values.ndim != 1 or len(values) != states.size(0) or not torch.isfinite(values).all():
            raise ValueError(f"feature {name!r} is invalid")
    return features


def define_collapse_labels(predictions_128, predictions_1024, targets, blank_mask):
    """Define collapse only after feature extraction from the late outcome."""

    if predictions_128.shape != targets.shape or predictions_1024.shape != targets.shape:
        raise ValueError("predictions and targets must align")
    if blank_mask.dtype != torch.bool or blank_mask.shape != targets.shape:
        raise ValueError("blank_mask must align with targets")
    solved_128 = ((predictions_128 == targets) | ~blank_mask).all(dim=1)
    solved_1024 = ((predictions_1024 == targets) | ~blank_mask).all(dim=1)
    eligible = solved_128
    collapse = eligible & ~solved_1024
    return {
        "solved_128": solved_128,
        "solved_1024": solved_1024,
        "eligible": eligible,
        "collapse": collapse,
    }


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values):
        values = np.asarray(values, dtype=np.float64)
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean, scale)

    def transform(self, values):
        return (np.asarray(values, dtype=np.float64) - self.mean) / self.scale


def _sigmoid(values):
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_ridge_logistic(values, labels, *, l2=1.0, max_iterations=100):
    """Fit an unweighted probability model with Newton steps and fixed ridge."""

    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if values.ndim != 2 or labels.shape != (len(values),):
        raise ValueError("invalid logistic-regression shapes")
    design = np.column_stack([np.ones(len(values)), values])
    prevalence = (labels.sum() + 0.5) / (len(labels) + 1.0)
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    coefficients[0] = math.log(prevalence / (1.0 - prevalence))
    penalty = np.full(design.shape[1], float(l2))
    penalty[0] = 1e-6

    def objective(candidate):
        linear = np.einsum("ni,i->n", design, candidate, optimize=False)
        return float(
            np.logaddexp(0.0, linear).sum()
            - np.sum(labels * linear)
            + 0.5 * np.sum(penalty * np.square(candidate))
        )

    for _ in range(max_iterations):
        linear = np.einsum("ni,i->n", design, coefficients, optimize=False)
        probabilities = _sigmoid(linear)
        weights = np.clip(probabilities * (1.0 - probabilities), 1e-7, None)
        gradient = np.einsum(
            "ni,n->i",
            design,
            probabilities - labels,
            optimize=False,
        ) + penalty * coefficients
        # ``einsum`` avoids a spurious divide-by-zero warning emitted by the
        # macOS Accelerate-backed NumPy 2.0 matrix multiply for this weighted
        # cross-product, even when every input is finite and order one.
        hessian = np.einsum(
            "ni,n,nj->ij",
            design,
            weights,
            design,
            optimize=False,
        ) + np.diag(penalty)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        current_objective = objective(coefficients)
        step_scale = 1.0
        accepted = False
        while step_scale >= 1e-6:
            candidate = coefficients - step_scale * step
            candidate_objective = objective(candidate)
            if np.isfinite(candidate_objective) and candidate_objective <= current_objective + 1e-10:
                coefficients = candidate
                accepted = True
                break
            step_scale *= 0.5
        if not accepted or np.max(np.abs(step_scale * step)) < 1e-8:
            break
    return coefficients


def predict_logit(coefficients, values):
    values = np.asarray(values, dtype=np.float64)
    return coefficients[0] + np.einsum(
        "ni,i->n",
        values,
        coefficients[1:],
        optimize=False,
    )


def fit_platt(raw_logits, labels):
    raw_logits = np.asarray(raw_logits, dtype=np.float64).reshape(-1, 1)
    return fit_ridge_logistic(raw_logits, labels, l2=1e-3)


def apply_platt(raw_logits, coefficients):
    return _sigmoid(coefficients[0] + coefficients[1] * np.asarray(raw_logits))


def roc_auc(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return None
    comparisons = positive[:, None] - negative[None, :]
    return float(((comparisons > 0).mean() + 0.5 * (comparisons == 0).mean()))


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.sum() == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    # Evaluate only after complete tie groups.  Treating tied observations one
    # at a time makes average precision depend on their arbitrary input order.
    group_ends = np.flatnonzero(
        np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    )
    true_positives = np.cumsum(ordered_labels)[group_ends]
    retrieved = group_ends + 1
    recall = true_positives / labels.sum()
    precision = true_positives / retrieved
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def probability_metrics(labels, probabilities, *, bin_count=10):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    log_loss = -np.mean(labels * np.log(probabilities) + (1 - labels) * np.log(1 - probabilities))
    brier = np.mean((probabilities - labels) ** 2)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, bin_count + 1)[:-1], np.linspace(0, 1, bin_count + 1)[1:]):
        selected = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if selected.any():
            ece += selected.mean() * abs(probabilities[selected].mean() - labels[selected].mean())
    return {
        "n": int(len(labels)),
        "events": int(labels.sum()),
        "prevalence": float(labels.mean()) if len(labels) else None,
        "auroc": roc_auc(labels, probabilities),
        "average_precision": average_precision(labels, probabilities),
        "brier": float(brier),
        "log_loss": float(log_loss),
        "ece_10": float(ece),
    }


def calibration_bins(labels, probabilities, bin_count=8):
    """Return equal-count reliability bins for plotting, without fitting."""

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(probabilities)
    bins = []
    for indices in np.array_split(order, min(bin_count, len(order))):
        if len(indices):
            bins.append({
                "n": int(len(indices)),
                "mean_probability": float(probabilities[indices].mean()),
                "event_rate": float(labels[indices].mean()),
            })
    return bins


def stratified_permutation(labels, strata, generator):
    labels = np.asarray(labels, dtype=int)
    strata = np.asarray(strata, dtype=object)
    permuted = labels.copy()
    for stratum in dict.fromkeys(strata.tolist()):
        indices = np.flatnonzero(strata == stratum)
        permuted[indices] = labels[generator.permutation(indices)]
    return permuted


def minimum_events_for_auc(auc=0.70, negative_count=200, z=1.96, maximum=10000):
    """Hanley-McNeil approximation for a lower 95% AUROC bound above 0.5."""

    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    for positive_count in range(2, maximum + 1):
        variance = (
            auc * (1 - auc)
            + (positive_count - 1) * (q1 - auc * auc)
            + (negative_count - 1) * (q2 - auc * auc)
        ) / (positive_count * negative_count)
        if auc - z * math.sqrt(max(variance, 0.0)) > 0.5:
            return positive_count
    return None


def minimum_non_events_for_auc(auc=0.70, positive_count=50, z=1.96, maximum=10000):
    """Solve the same approximation for the required negative examples."""

    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    for negative_count in range(2, maximum + 1):
        variance = (
            auc * (1 - auc)
            + (positive_count - 1) * (q1 - auc * auc)
            + (negative_count - 1) * (q2 - auc * auc)
        ) / (positive_count * negative_count)
        if auc - z * math.sqrt(max(variance, 0.0)) > 0.5:
            return negative_count
    return None
