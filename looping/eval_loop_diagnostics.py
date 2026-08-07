"""Measure iteration-loss conflict and a Sudoku-adapted Jacobian lens.

The digit lens averages the gradient of each next-iteration digit logit over
test puzzles and source/target cells. This gives nine causal directions at every
recurrent block boundary. It is deliberately a local, one-iteration lens: the
trajectory is rolled to long horizons without gradients, then only the next
recurrent step is differentiated.
"""

import json
import os
import random
import re
import time
from collections import Counter

import torch
import torch.nn.functional as F
from datasets import load_dataset

import stabilize.exp_testbed_20k as model_module
from iters.state_norm import per_token_rms


DEFAULT_MODELS = (
    {
        "name": "stable_unbounded",
        "path": "/outputs/model_baseline_lr2e3.pt",
        "model_kwargs": {},
    },
    {
        "name": "collapsed_unbounded",
        "path": "/outputs/model_baseline_lr2e3_clean_a.pt",
        "model_kwargs": {},
    },
    {
        "name": "reliable_rmsnorm",
        "path": "/outputs/model_lr2e3_outer_rmsnorm_trial0_best_probe.pt",
        "model_kwargs": {"outer_state_norm": True},
    },
)

CLEAN_A_TRAJECTORY_MODELS = tuple(
    {
        "name": f"clean_a_step_{step}",
        "path": f"/outputs/baseline_lr2e3_clean_a_checkpoint_step{step}.pt",
        "model_kwargs": {},
    }
    for step in (30000, 35000, 40000, 45000, 49999)
)

LATE_STATE_MODELS = (
    {
        "name": "late_random_trial2_final",
        "path": "/outputs/looping/model_loop_late_random_replace_trial2.pt",
        "model_kwargs": {},
    },
    {
        "name": "late_through1024_trial0_final",
        "path": (
            "/outputs/looping/"
            "model_loop_late_random_replace_through_1024_trial0.pt"
        ),
        "model_kwargs": {},
    },
)

MODEL_PRESETS = {
    "reference_models": DEFAULT_MODELS,
    "clean_a_trajectory": CLEAN_A_TRAJECTORY_MODELS,
    "late_state_models": LATE_STATE_MODELS,
}

DEFAULT_HORIZONS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
RATING_BUCKETS = (
    (0, 0, "0"),
    (1, 2, "1-2"),
    (3, 10, "3-10"),
    (11, 50, "11-50"),
    (51, 1000, "51+"),
)


def _load_model(model_config, device):
    model = model_module.SudokuTransformer(
        **model_config.get("model_kwargs", {})
    ).to(device)
    state = torch.load(
        model_config["path"],
        map_location=device,
        weights_only=not model_config.get(
            "trusted_full_checkpoint",
            False,
        ),
    )
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    state = {
        key.removeprefix("_orig_mod."): value
        for key, value in state.items()
    }
    model.load_state_dict(state)
    model.eval()
    return model


def _load_balanced_sample(examples_per_bucket, seed):
    dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    bucket_indices = {name: [] for _, _, name in RATING_BUCKETS}
    for index, example in enumerate(dataset):
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= example["rating"] <= maximum:
                bucket_indices[name].append(index)
                break

    generator = random.Random(seed)
    selected_indices = []
    bucket_names = []
    for _, _, name in RATING_BUCKETS:
        candidates = bucket_indices[name]
        if len(candidates) < examples_per_bucket:
            raise ValueError(
                f"bucket {name!r} has only {len(candidates)} examples"
            )
        selected_indices.extend(generator.sample(candidates, examples_per_bucket))
        bucket_names.extend([name] * examples_per_bucket)

    puzzles = [dataset[index]["question"] for index in selected_indices]
    solutions = [dataset[index]["answer"] for index in selected_indices]
    inputs = model_module.encode_puzzles(puzzles)
    targets = model_module.encode_solutions(solutions).long()
    empty_mask = inputs[:, :, 0].bool()
    return inputs, targets, empty_mask, puzzles, solutions, bucket_names


def _masked_cross_entropy(logits, targets, empty_mask):
    per_cell = F.cross_entropy(
        logits.reshape(-1, 9),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)
    return (per_cell * empty_mask).sum() / empty_mask.sum()


def _cosine_matrix(vectors):
    matrix = torch.stack([vector.float() for vector in vectors])
    normalized = F.normalize(matrix, dim=1, eps=1e-12)
    return normalized @ normalized.T


def _matrix_summary(vectors):
    matrix = torch.stack([vector.float() for vector in vectors])
    cosine = _cosine_matrix(vectors)
    count = cosine.size(0)
    off_diagonal = cosine[~torch.eye(count, dtype=torch.bool)]
    mean_vector = matrix.mean(dim=0)
    norms = matrix.norm(dim=1)
    mean_alignment = F.cosine_similarity(
        matrix,
        mean_vector.unsqueeze(0),
        dim=1,
        eps=1e-12,
    )
    return {
        "gradient_norms": norms.tolist(),
        "cosine_matrix": cosine.tolist(),
        "mean_off_diagonal_cosine": off_diagonal.mean().item(),
        "minimum_off_diagonal_cosine": off_diagonal.min().item(),
        "negative_off_diagonal_fraction": (off_diagonal < 0).float().mean().item(),
        "early_late_cosine": cosine[:4, -4:].mean().item(),
        "mean_gradient_alignment": mean_alignment.tolist(),
        "cancellation_ratio": (
            mean_vector.norm() / norms.mean().clamp_min(1e-12)
        ).item(),
    }


def parameter_gradient_conflict(model, inputs, targets, empty_mask):
    model.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    all_logits = model(inputs, return_all=True)
    losses = [
        _masked_cross_entropy(logits, targets, empty_mask)
        for logits in all_logits
    ]
    named_parameters = list(model.named_parameters())
    parameters = [parameter for _, parameter in named_parameters]
    group_indices = {
        "all": tuple(range(len(parameters))),
        "initial_encoder": tuple(
            index for index, (name, _) in enumerate(named_parameters)
            if name.startswith("initial_encoder.")
        ),
        "prediction_feedback": tuple(
            index for index, (name, _) in enumerate(named_parameters)
            if name.startswith("pred_proj.")
        ),
        "transformer_blocks": tuple(
            index for index, (name, _) in enumerate(named_parameters)
            if name.startswith("layers.")
        ),
        "output_head": tuple(
            index for index, (name, _) in enumerate(named_parameters)
            if name.startswith("output_head.")
        ),
    }
    vectors = {name: [] for name in group_indices}

    for loss_index, loss in enumerate(losses):
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=loss_index < len(losses) - 1,
        )
        gradients = [gradient.detach().float().cpu() for gradient in gradients]
        for group_name, indices in group_indices.items():
            vectors[group_name].append(torch.cat([
                gradients[index].reshape(-1)
                for index in indices
            ]))

    model.zero_grad(set_to_none=True)
    return {
        "losses": [loss.item() for loss in losses],
        "groups": {
            name: _matrix_summary(group_vectors)
            for name, group_vectors in vectors.items()
        },
    }


def linear_cka(first, second):
    first = first.float() - first.float().mean(dim=0, keepdim=True)
    second = second.float() - second.float().mean(dim=0, keepdim=True)
    first_gram = first @ first.T
    second_gram = second @ second.T
    denominator = first_gram.norm() * second_gram.norm()
    if denominator <= torch.finfo(denominator.dtype).tiny:
        return 0.0
    return ((first_gram * second_gram).sum() / denominator).item()


def _effective_rank(matrix):
    singular_values = torch.linalg.svdvals(matrix.float())
    eigenvalues = singular_values.square()
    denominator = eigenvalues.square().sum()
    if denominator <= 1e-24:
        return 0.0
    return (eigenvalues.sum().square() / denominator).item()


def _stage_name(layer_index, occurrence, total_occurrences):
    layer_name = chr(ord("A") + layer_index)
    if total_occurrences[layer_index] == 1:
        return f"after_{layer_name}"
    return f"after_{layer_name}_{occurrence}"


def recurrent_stages(model, hidden_state, predictions, rope_cos, rope_sin):
    names = ["input_state"]
    stages = [hidden_state]

    hidden_state = (
        hidden_state + model.feedback_scale * model.pred_proj(predictions)
    )
    names.append("after_feedback")
    stages.append(hidden_state)

    total_occurrences = Counter(model.layer_schedule)
    seen = Counter()
    for layer_index in model.layer_schedule:
        seen[layer_index] += 1
        hidden_state = model.layers[layer_index](
            hidden_state,
            rope_cos,
            rope_sin,
        )
        names.append(_stage_name(layer_index, seen[layer_index], total_occurrences))
        stages.append(hidden_state)

    normalized_state = model.normalize_outer_state(hidden_state)
    if normalized_state is not hidden_state:
        names.append("after_outer_state")
        stages.append(normalized_state)
    logits = model.output_head(normalized_state)
    return names, stages, logits


def _gradient_statistics(gradient, activation):
    gradient = gradient.detach().float().cpu()
    activation = activation.detach().float().cpu()
    flat_gradient = gradient.reshape(-1, gradient.size(-1))
    flat_activation = activation.reshape(-1, activation.size(-1))
    token_norms = flat_gradient.norm(dim=1)
    descent_cosine = F.cosine_similarity(
        -flat_gradient,
        flat_activation,
        dim=1,
        eps=1e-12,
    )
    centered = flat_gradient - flat_gradient.mean(dim=0, keepdim=True)
    return {
        "mean_token_norm": token_norms.mean().item(),
        "median_token_norm": token_norms.median().item(),
        "maximum_token_norm": token_norms.max().item(),
        "token_gradient_effective_rank": _effective_rank(centered),
        "mean_descent_state_cosine": descent_cosine.mean().item(),
    }


def one_step_digit_lens(
    model,
    hidden_state,
    targets,
    empty_mask,
    horizon,
    rope_cos,
    rope_sin,
):
    source_state = hidden_state.detach().requires_grad_(True)
    if horizon == 0:
        predictions = torch.zeros(
            source_state.size(0), 81, 9, device=source_state.device
        )
    else:
        predictions = F.softmax(model.output_head(source_state), dim=-1)

    stage_names, stages, logits = recurrent_stages(
        model,
        source_state,
        predictions,
        rope_cos,
        rope_sin,
    )
    digit_vectors = {name: [] for name in stage_names}
    target_count = empty_mask.sum().clamp_min(1)
    for digit in range(9):
        objective = (logits[:, :, digit] * empty_mask).sum() / target_count
        gradients = torch.autograd.grad(
            objective,
            stages,
            retain_graph=True,
        )
        for name, gradient in zip(stage_names, gradients):
            digit_vectors[name].append(
                gradient.detach().float().mean(dim=(0, 1)).cpu()
            )

    correct_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    wrong_mask = F.one_hot(targets, num_classes=9).bool()
    wrong_logits = logits.masked_fill(wrong_mask, -torch.inf)
    margin = correct_logits - torch.logsumexp(wrong_logits, dim=-1)
    margin_objective = (margin * empty_mask).sum() / target_count
    margin_gradients = torch.autograd.grad(margin_objective, stages)

    output_head = model.output_head.weight.detach().float().cpu()
    stage_results = {}
    lens_matrices = {}
    margin_vectors = {}
    for name, stage, gradient in zip(stage_names, stages, margin_gradients):
        lens_matrix = torch.stack(digit_vectors[name])
        lens_matrices[name] = lens_matrix
        margin_vectors[name] = gradient.detach().float().cpu().reshape(-1)
        stage_results[name] = {
            "digit_lens_vectors": lens_matrix.tolist(),
            "digit_lens_row_norms": lens_matrix.norm(dim=1).tolist(),
            "digit_lens_cosine": _cosine_matrix(lens_matrix).tolist(),
            "digit_lens_effective_rank": _effective_rank(
                lens_matrix - lens_matrix.mean(dim=0, keepdim=True)
            ),
            "digit_lens_cka_to_output_head": linear_cka(
                lens_matrix,
                output_head,
            ),
            "answer_margin_gradient": _gradient_statistics(gradient, stage),
        }

    return {
        "next_iteration_loss": _masked_cross_entropy(
            logits,
            targets,
            empty_mask,
        ).item(),
        "next_iteration_mean_margin": (
            (margin * empty_mask).sum() / target_count
        ).item(),
        "stages": stage_results,
    }, lens_matrices, margin_vectors


def _solved_count(logits, targets, empty_mask):
    predictions = logits.argmax(dim=-1)
    return int(((predictions == targets) | ~empty_mask).all(dim=1).sum().item())


def causal_trajectory_probe(model, inputs, targets, empty_mask, horizons):
    model.requires_grad_(False)
    rope_cos = model_module.ROPE_COS.to(inputs.device)
    rope_sin = model_module.ROPE_SIN.to(inputs.device)
    horizon_set = set(horizons)
    maximum_horizon = max(horizons)
    results = {}
    lens_store = {}
    margin_store = {}

    with torch.no_grad():
        hidden_state = model.initial_encoder(inputs)
        predictions = torch.zeros(
            inputs.size(0), 81, 9, device=inputs.device
        )

    for iteration in range(maximum_horizon + 1):
        if iteration in horizon_set:
            if iteration == 0:
                solved = None
            else:
                solved = _solved_count(
                    model.output_head(hidden_state),
                    targets,
                    empty_mask,
                )
            lens_result, lens_matrices, margin_vectors = one_step_digit_lens(
                model,
                hidden_state,
                targets,
                empty_mask,
                iteration,
                rope_cos,
                rope_sin,
            )
            lens_result.update({
                "solved_before_step": solved,
                "total": inputs.size(0),
                "state_token_rms_mean": per_token_rms(
                    hidden_state.detach()
                ).float().mean().item(),
            })
            results[str(iteration)] = lens_result
            lens_store[iteration] = lens_matrices
            margin_store[iteration] = margin_vectors

        if iteration == maximum_horizon:
            break
        with torch.no_grad():
            hidden_state = model.recurrent_step(
                hidden_state,
                predictions,
                rope_cos,
                rope_sin,
            )
            predictions = F.softmax(model.output_head(hidden_state), dim=-1)

    reference_horizon = 16 if 16 in lens_store else horizons[-1]
    for horizon in horizons:
        horizon_result = results[str(horizon)]
        horizon_result["comparison_to_horizon_16"] = {}
        shared_stages = set(lens_store[horizon]) & set(lens_store[reference_horizon])
        for stage_name in sorted(shared_stages):
            horizon_result["comparison_to_horizon_16"][stage_name] = {
                "digit_lens_cka": linear_cka(
                    lens_store[horizon][stage_name],
                    lens_store[reference_horizon][stage_name],
                ),
                "answer_margin_gradient_cosine": F.cosine_similarity(
                    margin_store[horizon][stage_name].unsqueeze(0),
                    margin_store[reference_horizon][stage_name].unsqueeze(0),
                    dim=1,
                    eps=1e-12,
                ).item(),
            }
    return results


def evaluate(
    model_configs=DEFAULT_MODELS,
    horizons=DEFAULT_HORIZONS,
    examples_per_bucket=4,
    seed=42,
    device="cuda",
    output_dir=None,
    output_prefix="loop_diagnostics",
):
    if not horizons or tuple(sorted(set(horizons))) != tuple(horizons):
        raise ValueError("horizons must be strictly increasing")
    if horizons[0] < 0:
        raise ValueError("horizons must be non-negative")
    if examples_per_bucket <= 0:
        raise ValueError("examples_per_bucket must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    inputs, targets, empty_mask, puzzles, solutions, bucket_names = (
        _load_balanced_sample(examples_per_bucket, seed)
    )
    inputs = inputs.to(device)
    targets = targets.to(device)
    empty_mask = empty_mask.to(device)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log_file = open(os.path.join(output_dir, f"{output_prefix}.log"), "w")
    else:
        log_file = None

    def log(message=""):
        print(message, flush=True)
        if log_file:
            log_file.write(message + "\n")
            log_file.flush()

    summary = {
        "config": {
            "models": list(model_configs),
            "horizons": list(horizons),
            "examples_per_bucket": examples_per_bucket,
            "sample_size": len(inputs),
            "seed": seed,
            "device": str(device),
            "diagnostic_precision": "float32",
            "causal_window": "one recurrent Sudoku iteration",
        },
        "sample": {
            "puzzles": puzzles,
            "solutions": solutions,
            "buckets": bucket_names,
        },
        "models": {},
    }
    started_at = time.time()
    log(
        f"Loop diagnostics: {len(inputs)} puzzles, "
        f"models={len(model_configs)}, horizons={list(horizons)}"
    )

    for model_config in model_configs:
        model_name = model_config["name"]
        log(f"\nMODEL {model_name}: {model_config['path']}")
        model_started_at = time.time()
        model = _load_model(model_config, device)
        log("  parameter-gradient conflict across 16 supervised losses")
        gradient_conflict = parameter_gradient_conflict(
            model,
            inputs,
            targets,
            empty_mask,
        )
        log("  local digit Jacobian lens along the 0-1024 trajectory")
        causal_probe = causal_trajectory_probe(
            model,
            inputs,
            targets,
            empty_mask,
            horizons,
        )
        summary["models"][model_name] = {
            "model_config": model_config,
            "parameter_gradient_conflict": gradient_conflict,
            "causal_trajectory": causal_probe,
            "elapsed_seconds": time.time() - model_started_at,
        }
        log(
            f"  done in {summary['models'][model_name]['elapsed_seconds']:.1f}s"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary["elapsed_seconds"] = time.time() - started_at
    if output_dir:
        result_path = os.path.join(output_dir, f"{output_prefix}.json")
        temporary_path = result_path + ".tmp"
        with open(temporary_path, "w") as result_file:
            json.dump(summary, result_file, indent=2)
            result_file.write("\n")
        os.replace(temporary_path, result_path)
        log(f"\nStructured results: {result_path}")
    log(f"Total time: {summary['elapsed_seconds']:.1f}s")
    if log_file:
        log_file.close()
    return summary


if __name__ == "__main__":
    evaluate()
