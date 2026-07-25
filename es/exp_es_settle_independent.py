"""Independent-sampling ES polish with solved-and-settled fitness at 2048."""

import json
import os
import time

import numpy as np
import torch
from datasets import load_dataset

from checkpoint_utils import atomic_torch_save, find_latest_checkpoint
from es.es_sampling import (
    DEFAULT_SAMPLING_MODE,
    POPULATION_SIZE,
    direction_weights,
    generation_direction_seeds,
)
from es.exp_es_settle import make_two_phase_runner, perturb
from iters.exp_baseline_lr2e3 import (
    RATING_BUCKETS,
    SudokuTransformer,
    encode_puzzles,
)


torch.set_float32_matmul_precision("high")

CHECKPOINT_PREFIX = "es_settle_independent_checkpoint_step"
CONFIG = {
    "experiment": "exp_es_settle_independent",
    "es_generations": 60,
    "sampling_mode": DEFAULT_SAMPLING_MODE,
    "population_evaluations": POPULATION_SIZE,
    "sigma": "calibrated",
    "lr": 3e-4,
    "anchor_lambda": 1e-3,
    "fitness": "solved_and_settled",
    "fitness_puzzles": 384,
    "fitness_iters": 2048,
    "settle_window": 128,
}

total_steps = 60
eval_every = 20
sigma_ladder = (3e-4, 1e-4, 3e-5, 1e-5)
learning_rate = 3e-4
anchor_lambda = 1e-3
fitness_puzzles = 384
fitness_iters = 2048
settle_window = 128
fitness_pool_offset = 2_700_000
fitness_pool_size = 20_000
log_name = "exp_es_settle_independent.log"


def save_best_probe(model, reading, best_both, best_model_path):
    if reading["both"] <= best_both.get("both", -1):
        return best_both, False

    updated_best = dict(reading)
    atomic_torch_save(model.state_dict(), best_model_path)
    return updated_best, True


def train(
    output_dir=".",
    seed_model_path="seed_model.pt",
    generations=total_steps,
):
    if (
        isinstance(generations, bool)
        or not isinstance(generations, int)
        or generations <= 0
    ):
        raise ValueError("generations must be a positive integer")

    os.makedirs(output_dir, exist_ok=True)
    seed_model_path = os.path.abspath(seed_model_path)
    final_model_path = os.path.join(
        output_dir,
        "model_es_settle_independent.pt",
    )
    best_model_path = os.path.join(
        output_dir,
        "model_es_settle_independent_best_probe.pt",
    )
    run_config = {
        **CONFIG,
        "es_generations": generations,
        "seed_model": os.path.basename(seed_model_path),
    }
    device = torch.device("cuda")
    print(
        "ES recipe: "
        f"sampling={CONFIG['sampling_mode']}, "
        f"evaluations={CONFIG['population_evaluations']}, "
        f"fitness={CONFIG['fitness']}, "
        f"fitness_iters={CONFIG['fitness_iters']}, "
        f"settle_window={CONFIG['settle_window']}, "
        f"generations={generations}"
    )
    print(f"Seed model: {seed_model_path}")
    print(f"Output directory: {output_dir}")

    print("Loading fitness pool...")
    dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = dataset[
        fitness_pool_offset:fitness_pool_offset + fitness_pool_size
    ]
    pool_puzzles = pool_rows["question"]
    pool_solutions = pool_rows["answer"]
    pool_x = encode_puzzles(pool_puzzles).to(device)
    pool_targets = torch.tensor(
        [
            [int(solution[index]) - 1 for index in range(81)]
            for solution in pool_solutions
        ],
        device=device,
    )
    pool_empty = torch.tensor(
        [
            [puzzle[index] == "." for index in range(81)]
            for puzzle in pool_puzzles
        ],
        device=device,
    )
    print(f"Fitness pool: {len(pool_puzzles)} puzzles")

    print("Loading validation probe...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    probe_indices = []
    per_bucket = {name: 0 for _, _, name in RATING_BUCKETS}
    for row_index in range(len(test_dataset)):
        rating = test_dataset[row_index]["rating"]
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= rating <= maximum and per_bucket[name] < 200:
                probe_indices.append(row_index)
                per_bucket[name] += 1
                break
        if len(probe_indices) == 1000:
            break

    probe_puzzles = [
        test_dataset[row_index]["question"]
        for row_index in probe_indices
    ]
    probe_solutions = [
        test_dataset[row_index]["answer"]
        for row_index in probe_indices
    ]
    probe_x = encode_puzzles(probe_puzzles).to(device)
    probe_targets = torch.tensor(
        [
            [int(solution[index]) - 1 for index in range(81)]
            for solution in probe_solutions
        ],
        device=device,
    )
    probe_empty = torch.tensor(
        [
            [puzzle[index] == "." for index in range(81)]
            for puzzle in probe_puzzles
        ],
        device=device,
    )
    print(f"Validation probe: {len(probe_indices)} puzzles")

    model = SudokuTransformer().to(device)
    checkpoint_path, _ = find_latest_checkpoint(
        output_dir,
        CHECKPOINT_PREFIX,
    )
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        for key, expected in run_config.items():
            actual = checkpoint.get("config", {}).get(key)
            if actual != expected:
                raise ValueError(
                    f"resume config mismatch for {key}: "
                    f"{actual!r} != {expected!r}"
                )
        model.load_state_dict(checkpoint["model_state_dict"])
        anchor = [
            tensor.to(device)
            for tensor in checkpoint["anchor"]
        ]
        sigma = float(checkpoint["sigma"])
        start_generation = int(checkpoint["step"]) + 1
        probe_history = list(checkpoint.get("probe_history", []))
        best_both = dict(
            checkpoint.get(
                "best_both",
                {"generation": -1, "both": -1},
            )
        )
        print(
            "Resumed ES state from generation "
            f"{start_generation - 1}"
        )
    else:
        if not os.path.isfile(seed_model_path):
            raise FileNotFoundError(seed_model_path)
        state = torch.load(
            seed_model_path,
            map_location=device,
            weights_only=True,
        )
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        anchor = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]
        sigma = None
        start_generation = 0
        probe_history = []
        best_both = {"generation": -1, "both": -1}
        print(f"Loaded seed model from {seed_model_path}")

    model.eval()
    parameters = list(model.parameters())
    run_two_phases = make_two_phase_runner(model, device)
    log_path = os.path.join(output_dir, log_name)
    log_file = open(log_path, "a")

    def log(message):
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    def solved_settled(inputs, targets, empty_mask, chunk_size):
        solved = 0
        settled = 0
        both = 0
        with torch.no_grad(), torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            for start in range(0, inputs.size(0), chunk_size):
                end = start + chunk_size
                answer_early, answer_final = run_two_phases(
                    inputs[start:end],
                    fitness_iters,
                    settle_window,
                )
                target_batch = targets[start:end]
                empty_batch = empty_mask[start:end]
                is_solved = (
                    ((answer_final == target_batch) & empty_batch)
                    | ~empty_batch
                ).all(dim=1)
                is_settled = (
                    ((answer_final == answer_early) & empty_batch)
                    | ~empty_batch
                ).all(dim=1)
                solved += int(is_solved.sum().item())
                settled += int(is_settled.sum().item())
                both += int((is_solved & is_settled).sum().item())
        return solved, settled, both

    def score_slice(offset):
        _, _, both = solved_settled(
            pool_x[offset:offset + fitness_puzzles],
            pool_targets[offset:offset + fitness_puzzles],
            pool_empty[offset:offset + fitness_puzzles],
            fitness_puzzles,
        )
        return both

    def record_probe(generation):
        nonlocal best_both
        solved, settled, both = solved_settled(
            probe_x,
            probe_targets,
            probe_empty,
            500,
        )
        reading = {
            "generation": generation,
            "solved": solved,
            "settled": settled,
            "both": both,
            "total": probe_x.size(0),
        }
        probe_history.append(reading)
        best_both, saved_best = save_best_probe(
            model,
            reading,
            best_both,
            best_model_path,
        )
        log(
            f"GEN {generation:4d} | validation 2048-iter: "
            f"solved {solved}/1000, settled {settled}/1000, "
            f"both {both}/1000"
        )
        if saved_best:
            log(
                f"GEN {generation:4d} | new best probe saved: "
                f"{best_model_path}"
            )
        return reading

    def save_checkpoint(generation):
        path = os.path.join(
            output_dir,
            f"{CHECKPOINT_PREFIX}{generation}.pt",
        )
        atomic_torch_save(
            {
                "step": generation,
                "model_state_dict": model.state_dict(),
                "anchor": [
                    tensor.cpu()
                    for tensor in anchor
                ],
                "sigma": sigma,
                "config": run_config,
                "probe_history": probe_history,
                "best_both": best_both,
            },
            path,
        )
        log(f"Checkpoint saved: {path}")

    if (
        not probe_history
        or probe_history[-1]["generation"] != start_generation - 1
    ):
        final_reading = record_probe(start_generation - 1)
    else:
        final_reading = probe_history[-1]

    if sigma is None:
        unperturbed = score_slice(0)
        snapshot = [
            parameter.detach().clone()
            for parameter in parameters
        ]
        sigma = sigma_ladder[-1]
        for candidate in sigma_ladder:
            perturb(parameters, 777, candidate)
            score = score_slice(0)
            with torch.no_grad():
                for parameter, saved in zip(parameters, snapshot):
                    parameter.copy_(saved)
            log(
                f"CALIBRATE sigma={candidate:.0e}: {score} "
                f"(unperturbed {unperturbed})"
            )
            if score >= unperturbed // 2:
                sigma = candidate
                break
        log(f"CALIBRATE chose sigma={sigma:.0e}")

    if start_generation >= generations:
        atomic_torch_save(model.state_dict(), final_model_path)
        log(
            "Run already complete at generation "
            f"{start_generation - 1}; final model saved: "
            f"{final_model_path}"
        )
        log_file.close()
        return {
            "already_complete": True,
            "best_both": best_both,
        }

    for generation in range(start_generation, generations):
        start_time = time.time()
        offset = (
            generation * fitness_puzzles
            % (fitness_pool_size - fitness_puzzles)
        )
        snapshot = [
            parameter.detach().clone()
            for parameter in parameters
        ]
        direction_seeds = generation_direction_seeds(1234, generation)
        scores = []
        for direction_seed in direction_seeds:
            perturb(parameters, direction_seed, sigma)
            scores.append(score_slice(offset))
            with torch.no_grad():
                for parameter, saved in zip(parameters, snapshot):
                    parameter.copy_(saved)

        weights, spread = direction_weights(
            scores,
            DEFAULT_SAMPLING_MODE,
        )
        if spread > 1e-9:
            with torch.no_grad():
                for direction_seed, weight in zip(
                    direction_seeds,
                    weights,
                ):
                    generator = torch.Generator(device=device)
                    generator.manual_seed(direction_seed)
                    for parameter in parameters:
                        noise = torch.randn(
                            parameter.shape,
                            generator=generator,
                            device=device,
                            dtype=torch.float32,
                        )
                        parameter.add_(
                            noise,
                            alpha=learning_rate * float(weight),
                        )
                for parameter, anchor_parameter in zip(
                    parameters,
                    anchor,
                ):
                    parameter.add_(
                        parameter - anchor_parameter,
                        alpha=-anchor_lambda,
                    )

        if generation % 5 == 0 or generation == generations - 1:
            final_reading = record_probe(generation)
            log(
                f"GEN {generation:4d} | fitness mean "
                f"{np.mean(scores):.0f} best {max(scores):.0f} | "
                f"{time.time() - start_time:.0f}s"
            )
        else:
            log(
                f"GEN {generation:4d} | fitness mean "
                f"{np.mean(scores):.0f} best {max(scores):.0f} | "
                f"{time.time() - start_time:.0f}s"
            )

        if (
            (generation + 1) % eval_every == 0
            or generation == generations - 1
        ):
            save_checkpoint(generation)

    atomic_torch_save(model.state_dict(), final_model_path)
    result = {
        "config": run_config,
        "seed_model_path": seed_model_path,
        "final_model_path": final_model_path,
        "best_model_path": best_model_path,
        "final_probe": final_reading,
        "best_both": best_both,
        "probe_history": probe_history,
    }
    with open(
        os.path.join(output_dir, "result_es_settle_independent.json"),
        "w",
    ) as result_file:
        json.dump(result, result_file, indent=2, sort_keys=True)
    log(f"Final model saved: {final_model_path}")
    log_file.close()
    return result


if __name__ == "__main__":
    train()
