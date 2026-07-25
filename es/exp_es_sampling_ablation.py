"""Matched comparison of antithetic and independent one-sided ES sampling.

Both estimators spend 32 fitness evaluations per generation. Within a trial they
share the checkpoint, data slices, hyperparameters, and first 16 random directions.
The antithetic arm evaluates those directions at both signs; the independent arm
evaluates 32 unrelated positive perturbations.
"""

import json
import os
import re
import time

import numpy as np
import torch
from datasets import load_dataset

from checkpoint_utils import atomic_torch_save, find_latest_checkpoint
from es.es_sampling import (
    POPULATION_PAIRS,
    POPULATION_SIZE,
    direction_weights,
    generation_direction_seeds,
)
from es.exp_es_finetune import (
    count_solved,
    fitness_iters,
    fitness_pool_offset,
    fitness_pool_size,
    fitness_puzzles,
    make_compiled_runner,
    perturb,
    sigma_ladder,
)
from iters.exp_baseline_lr2e3 import RATING_BUCKETS, SudokuTransformer, encode_puzzles


torch.set_float32_matmul_precision("high")

LEARNING_RATE = 3e-4
ANCHOR_LAMBDA = 1e-3
CHECKPOINT_EVERY = 20


def _restore(params, snapshot):
    with torch.no_grad():
        for param, saved in zip(params, snapshot):
            param.copy_(saved)


def _load_seed_state(seed_path, device):
    state = torch.load(seed_path, map_location=device, weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    return {key.replace("_orig_mod.", ""): value for key, value in state.items()}


def train_sampling_ablation(
    output_dir,
    run_name,
    seed_path,
    sampling_mode,
    es_random_seed,
    fitness_dense,
    total_generations,
):
    if sampling_mode not in {"paired", "independent"}:
        raise ValueError("sampling_mode must be 'paired' or 'independent'")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_name):
        raise ValueError(f"unsafe run name: {run_name!r}")
    if total_generations <= 0:
        raise ValueError("total_generations must be positive")

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda")
    config = {
        "experiment": "exp_es_sampling_ablation",
        "run_name": run_name,
        "seed_path": seed_path,
        "sampling_mode": sampling_mode,
        "es_random_seed": es_random_seed,
        "population_evaluations": POPULATION_SIZE,
        "fitness": "dense_cells" if fitness_dense else "solved_puzzles",
        "fitness_puzzles": fitness_puzzles,
        "fitness_iters": fitness_iters,
        "lr": LEARNING_RATE,
        "anchor_lambda": ANCHOR_LAMBDA,
    }

    print("Loading fitness pool...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[fitness_pool_offset:fitness_pool_offset + fitness_pool_size]
    pool_puzzles = pool_rows["question"]
    pool_solutions = pool_rows["answer"]
    pool_x = encode_puzzles(pool_puzzles).to(device)
    pool_targets = torch.tensor(
        [[int(solution[index]) - 1 for index in range(81)] for solution in pool_solutions],
        device=device,
    )
    pool_empty = torch.tensor(
        [[puzzle[index] == "." for index in range(81)] for puzzle in pool_puzzles],
        device=device,
    )

    print("Loading validation probe...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    probe_indices = []
    per_bucket = {name: 0 for _, _, name in RATING_BUCKETS}
    for index in range(len(test_dataset)):
        rating = test_dataset[index]["rating"]
        for min_rating, max_rating, name in RATING_BUCKETS:
            if min_rating <= rating <= max_rating and per_bucket[name] < 200:
                probe_indices.append(index)
                per_bucket[name] += 1
                break
        if len(probe_indices) >= 1000:
            break
    probe_puzzles = [test_dataset[index]["question"] for index in probe_indices]
    probe_solutions = [test_dataset[index]["answer"] for index in probe_indices]
    probe_x = encode_puzzles(probe_puzzles).to(device)

    checkpoint_prefix = f"{run_name}_checkpoint_step"
    checkpoint_path, _ = find_latest_checkpoint(output_dir, checkpoint_prefix)
    model = SudokuTransformer().to(device)
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        for key, expected in config.items():
            actual = checkpoint["config"].get(key)
            if actual != expected:
                raise ValueError(f"resume config mismatch for {key}: {actual!r} != {expected!r}")
        model.load_state_dict(checkpoint["model_state_dict"])
        anchor = [tensor.to(device) for tensor in checkpoint["anchor"]]
        sigma = float(checkpoint["sigma"])
        start_generation = int(checkpoint["step"]) + 1
        print(f"Resumed {run_name} from generation {start_generation - 1}")
    else:
        if not os.path.exists(seed_path):
            raise FileNotFoundError(seed_path)
        model.load_state_dict(_load_seed_state(seed_path, device))
        anchor = [param.detach().clone() for param in model.parameters()]
        sigma = None
        start_generation = 0
        print(f"Loaded seed model from {seed_path}")

    model.eval()
    params = list(model.parameters())
    run_compiled = make_compiled_runner(model, device)
    log_path = os.path.join(output_dir, f"{run_name}.log")
    result_path = os.path.join(output_dir, f"result_{run_name}.json")
    if start_generation < total_generations and os.path.exists(result_path):
        os.remove(result_path)
    log_file = open(log_path, "a")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    def save_checkpoint(generation, probe_solved):
        path = os.path.join(output_dir, f"{checkpoint_prefix}{generation}.pt")
        atomic_torch_save(
            {
                "step": generation,
                "model_state_dict": model.state_dict(),
                "anchor": [tensor.cpu() for tensor in anchor],
                "sigma": sigma,
                "probe_solved": probe_solved,
                "config": config,
            },
            path,
        )
        log(f"Checkpoint saved: {path}")

    def score_slice(offset):
        features = pool_x[offset:offset + fitness_puzzles]
        targets = pool_targets[offset:offset + fitness_puzzles]
        empty = pool_empty[offset:offset + fitness_puzzles]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictions = run_compiled(features, fitness_iters).argmax(dim=-1)
        hits = (predictions == targets) & empty
        if fitness_dense:
            return int(hits.sum().item())
        return int((hits | ~empty).all(dim=1).sum().item())

    starting_probe = count_solved(
        model, probe_x, probe_puzzles, probe_solutions, fitness_iters
    )
    log(
        f"START generation={start_generation} mode={sampling_mode} seed={es_random_seed} "
        f"objective={config['fitness']} validation={starting_probe}/{len(probe_puzzles)}"
    )

    if sigma is None:
        unperturbed_score = score_slice(0)
        snapshot = [param.detach().clone() for param in params]
        sigma = sigma_ladder[-1]
        for candidate in sigma_ladder:
            perturb(params, 777, candidate)
            perturbed_score = score_slice(0)
            _restore(params, snapshot)
            log(
                f"CALIBRATE sigma={candidate:.0e}: {perturbed_score} "
                f"(unperturbed {unperturbed_score})"
            )
            if perturbed_score >= unperturbed_score // 2:
                sigma = candidate
                break
        log(f"CALIBRATE chose sigma={sigma:.0e}")

    final_model_path = os.path.join(output_dir, f"model_{run_name}.pt")
    if start_generation >= total_generations:
        atomic_torch_save(model.state_dict(), final_model_path)
        log(f"Run already complete at generation {start_generation - 1}")
        log_file.close()
        return {
            "run_name": run_name,
            "already_complete": True,
            "final_validation": starting_probe,
        }

    probe_solved = starting_probe
    for generation in range(start_generation, total_generations):
        started_at = time.time()
        offset = (generation * fitness_puzzles) % (fitness_pool_size - fitness_puzzles)
        snapshot = [param.detach().clone() for param in params]
        all_direction_seeds = generation_direction_seeds(es_random_seed, generation)

        if sampling_mode == "paired":
            direction_seeds = all_direction_seeds[:POPULATION_PAIRS]
            scores_plus = []
            scores_minus = []
            for direction_seed in direction_seeds:
                perturb(params, direction_seed, sigma)
                scores_plus.append(score_slice(offset))
                _restore(params, snapshot)
                perturb(params, direction_seed, -sigma)
                scores_minus.append(score_slice(offset))
                _restore(params, snapshot)
            scores = scores_plus + scores_minus
        else:
            direction_seeds = all_direction_seeds
            scores = []
            for direction_seed in direction_seeds:
                perturb(params, direction_seed, sigma)
                scores.append(score_slice(offset))
                _restore(params, snapshot)

        weights, spread = direction_weights(scores, sampling_mode)
        if spread > 1e-9:
            with torch.no_grad():
                for direction_seed, weight in zip(direction_seeds, weights):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(direction_seed)
                    for param in params:
                        noise = torch.randn(
                            param.shape,
                            generator=generator,
                            device=device,
                            dtype=torch.float32,
                        )
                        param.add_(noise, alpha=LEARNING_RATE * float(weight))
                for param, anchor_param in zip(params, anchor):
                    param.add_(param - anchor_param, alpha=-ANCHOR_LAMBDA)

        probe_solved = count_solved(
            model, probe_x, probe_puzzles, probe_solutions, fitness_iters
        )
        log(
            f"GEN {generation:4d} | fitness mean {np.mean(scores):.1f} "
            f"best {max(scores)} spread {spread:.2f} | validation "
            f"{probe_solved}/{len(probe_puzzles)} | {time.time() - started_at:.0f}s"
        )

        if (generation + 1) % CHECKPOINT_EVERY == 0 or generation == total_generations - 1:
            save_checkpoint(generation, probe_solved)

    atomic_torch_save(model.state_dict(), final_model_path)
    result = {
        "run_name": run_name,
        "sampling_mode": sampling_mode,
        "es_random_seed": es_random_seed,
        "generations": total_generations,
        "sigma": sigma,
        "final_validation": probe_solved,
        "final_model_path": final_model_path,
    }
    with open(result_path, "w") as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)
        result_file.write("\n")
    log(f"Final model saved: {final_model_path}")
    log_file.close()
    return result
