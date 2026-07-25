"""Estimate a stable tied CD-RGE radius without spending Modal probe jobs.

This uses one local backward pass only to measure the random-gradient estimator's
scale. The actual training experiments remain forward-only.
"""

import glob
import os
from dataclasses import replace

import pyarrow as pa
import pyarrow.ipc as ipc
import torch

from es.cdrge_9x9 import (
    CDRGEConfig,
    SudokuTransformer,
    _direction_seeds,
    _rademacher_like,
    cdrge_step,
    make_rope_tables,
    mean_iterative_ce,
    run_iterations,
)
from iters.exp_baseline_lr2e3 import encode_puzzles, encode_solutions


def load_local_batch(offset, size):
    pattern = os.path.expanduser(
        "~/.cache/huggingface/datasets/sapientinc___sudoku-extreme/"
        "default/0.0.0/*/sudoku-extreme-train-*.arrow"
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError("local sudoku-extreme Arrow cache not found")
    tables = [ipc.open_stream(pa.memory_map(path)).read_all() for path in files]
    rows = pa.concat_tables(tables).slice(offset, size)
    puzzles = rows.column("question").to_pylist()
    solutions = rows.column("answer").to_pylist()
    x = encode_puzzles(puzzles)
    targets = encode_solutions(solutions).long()
    return x, targets, x[:, :, 0].bool()


def estimate_gradient(config, x, targets, empty_mask):
    torch.manual_seed(config.model_seed)
    model = SudokuTransformer(config).eval()
    rope_cos, rope_sin = make_rope_tables(config, torch.device("cpu"))
    logits = run_iterations(model, x, 1, rope_cos, rope_sin, return_all=True)
    loss = mean_iterative_ce(logits, targets, empty_mask)
    loss.backward()
    gradients = [parameter.grad.detach().clone() for parameter in model.parameters()]
    return model, rope_cos, rope_sin, float(loss.item()), gradients


def estimate_random_gradient(parameters, gradients, seeds):
    estimates = [torch.zeros_like(parameter) for parameter in parameters]
    for direction_index, seed in enumerate(seeds, start=1):
        directions = [
            _rademacher_like(parameter, seed + parameter_index)
            for parameter_index, parameter in enumerate(parameters)
        ]
        directional_derivative = sum(
            float((gradient * direction).sum().item())
            for gradient, direction in zip(gradients, directions)
        )
        for estimate, direction in zip(estimates, directions):
            estimate.add_(direction, alpha=directional_derivative)
        if direction_index % 64 == 0:
            print(f"generated {direction_index}/{len(seeds)} directions", flush=True)
    for estimate in estimates:
        estimate.div_(len(seeds))
    return estimates


def score_update(model, rope_cos, rope_sin, x, targets, empty_mask, estimates, epsilon):
    parameters = list(model.parameters())
    snapshots = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter, estimate in zip(parameters, estimates):
            parameter.add_(estimate, alpha=-epsilon)
        logits = run_iterations(model, x, 1, rope_cos, rope_sin, return_all=True)
        loss = mean_iterative_ce(logits, targets, empty_mask)
        for parameter, snapshot in zip(parameters, snapshots):
            parameter.copy_(snapshot)
    return float(loss.item())


def exact_finite_difference_check(
    model,
    rope_cos,
    rope_sin,
    train_x,
    train_targets,
    train_empty,
    validation_x,
    validation_targets,
    validation_empty,
    seeds,
):
    parameters = list(model.parameters())
    snapshots = [parameter.detach().clone() for parameter in parameters]

    def loss_on(x, targets, empty_mask):
        with torch.inference_mode():
            logits = run_iterations(model, x, 1, rope_cos, rope_sin, return_all=True)
            return float(mean_iterative_ce(logits, targets, empty_mask).item())

    for epsilon in (1e-3, 3e-3, 1e-2):
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)

        losses_plus, losses_minus, update_norm = cdrge_step(
            parameters,
            lambda: loss_on(train_x, train_targets, train_empty),
            seeds,
            epsilon,
        )
        print(
            f"exact p={len(seeds)}, epsilon={epsilon:.0e}: "
            f"pair mean={(losses_plus.mean() + losses_minus.mean()) / 2:.6f}, "
            f"diff std={(losses_plus - losses_minus).std():.6f}, "
            f"update L2={update_norm:.6f}, "
            f"train CE={loss_on(train_x, train_targets, train_empty):.6f}, "
            f"validation CE={loss_on(validation_x, validation_targets, validation_empty):.6f}"
        )

    with torch.no_grad():
        for parameter, snapshot in zip(parameters, snapshots):
            parameter.copy_(snapshot)


def exact_trajectory(
    model,
    base_config,
    rope_cos,
    rope_sin,
    train_x,
    train_targets,
    train_empty,
    validation_x,
    validation_targets,
    validation_empty,
):
    parameters = list(model.parameters())
    initial_parameters = [parameter.detach().clone() for parameter in parameters]

    def loss_on(x, targets, empty_mask):
        with torch.inference_mode():
            logits = run_iterations(model, x, 1, rope_cos, rope_sin, return_all=True)
            return float(mean_iterative_ce(logits, targets, empty_mask).item())

    for population in (256, 512):
        with torch.no_grad():
            for parameter, initial in zip(parameters, initial_parameters):
                parameter.copy_(initial)
        config = replace(base_config, population_directions=population)
        for generation in range(5):
            start = (generation * 64) % train_x.size(0)
            batch_slice = slice(start, start + 64)
            _, _, update_norm = cdrge_step(
                parameters,
                lambda: loss_on(
                    train_x[batch_slice],
                    train_targets[batch_slice],
                    train_empty[batch_slice],
                ),
                _direction_seeds(config, generation),
                epsilon=1e-2,
            )
            validation_loss = loss_on(
                validation_x,
                validation_targets,
                validation_empty,
            )
            print(
                f"trajectory p={population}, generation={generation}: "
                f"update L2={update_norm:.6f}, validation CE={validation_loss:.6f}"
            )

    with torch.no_grad():
        for parameter, initial in zip(parameters, initial_parameters):
            parameter.copy_(initial)


def main():
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    x, targets, empty_mask = load_local_batch(2_700_000, 512)
    train_x, validation_x = x[:256], x[256:]
    train_targets, validation_targets = targets[:256], targets[256:]
    train_empty, validation_empty = empty_mask[:256], empty_mask[256:]

    base_config = CDRGEConfig(name="local-calibration", population_directions=512, stages=())
    model, rope_cos, rope_sin, initial_loss, gradients = estimate_gradient(
        base_config,
        train_x,
        train_targets,
        train_empty,
    )
    gradient_norm = sum(float(gradient.square().sum()) for gradient in gradients) ** 0.5
    print(f"initial train CE={initial_loss:.6f}, exact gradient L2={gradient_norm:.6f}")

    all_seeds = _direction_seeds(base_config, generation=0)
    for population in (256, 512):
        estimates = estimate_random_gradient(
            list(model.parameters()),
            gradients,
            all_seeds[:population],
        )
        estimate_norm = sum(float(estimate.square().sum()) for estimate in estimates) ** 0.5
        alignment = sum(
            float((gradient * estimate).sum())
            for gradient, estimate in zip(gradients, estimates)
        )
        print(
            f"p={population}: estimator L2={estimate_norm:.6f}, "
            f"gradient dot estimator={alignment:.6f}"
        )
        for epsilon in (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2):
            validation_loss = score_update(
                model,
                rope_cos,
                rope_sin,
                validation_x,
                validation_targets,
                validation_empty,
                estimates,
                epsilon,
            )
            print(
                f"  epsilon={epsilon:.0e}: update L2={epsilon * estimate_norm:.6f}, "
                f"validation CE={validation_loss:.6f}"
            )

    exact_finite_difference_check(
        model,
        rope_cos,
        rope_sin,
        train_x[:64],
        train_targets[:64],
        train_empty[:64],
        validation_x,
        validation_targets,
        validation_empty,
        all_seeds[:256],
    )
    exact_trajectory(
        model,
        base_config,
        rope_cos,
        rope_sin,
        train_x,
        train_targets,
        train_empty,
        validation_x,
        validation_targets,
        validation_empty,
    )


if __name__ == "__main__":
    main()
