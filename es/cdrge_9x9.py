"""Central-difference random-gradient training for full 9x9 Sudoku.

This intentionally follows the raw CD-RGE update used by zero_order_rnn:

    theta <- theta - sum_i((L+_i - L-_i) * delta_i) / (2 * n)

The learning rate and perturbation radius are tied, so their ratio cancels from
the update. Unlike the project's earlier ES experiments, this uses raw losses,
Rademacher directions, and no ranking or score normalization.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from checkpoint_utils import atomic_torch_save, find_latest_checkpoint
from iters.exp_baseline_lr2e3 import (
    COL_IDX,
    RATING_BUCKETS,
    ROW_IDX,
    RoPETransformerLayer,
    encode_puzzles,
    encode_solutions,
)

torch.set_float32_matmul_precision("highest")


@dataclass(frozen=True)
class TrainingStage:
    name: str
    generations: int
    horizon: int
    data_slice: str


@dataclass(frozen=True)
class CDRGEConfig:
    name: str
    population_directions: int
    stages: tuple[TrainingStage, ...]
    epsilon: float = 0.01
    batch_size: int = 1024
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 4
    pool_offset: int = 2_700_000
    pool_size: int = 100_000
    high_clue_pool_size: int = 4096
    diagnostic_puzzles: int = 1000
    diagnostic_every: int = 10
    checkpoint_every: int = 25
    model_seed: int = 20260718
    data_seed: int = 31001
    direction_seed: int = 71003
    compile_fitness: bool = True
    final_probe_horizons: tuple[int, ...] = (1, 4, 16, 128, 1024)


def fixed_horizon_config(name: str, population_directions: int) -> CDRGEConfig:
    return CDRGEConfig(
        name=name,
        population_directions=population_directions,
        stages=(TrainingStage("all-h1", 500, 1, "all"),),
    )


def curriculum_config(name: str, population_directions: int) -> CDRGEConfig:
    return CDRGEConfig(
        name=name,
        population_directions=population_directions,
        stages=(
            TrainingStage("high-clue-h1", 150, 1, "high_clue"),
            TrainingStage("easy-h4", 150, 4, "easy"),
            TrainingStage("all-h16", 150, 16, "all"),
        ),
    )


class SudokuTransformer(nn.Module):
    def __init__(self, config: CDRGEConfig):
        super().__init__()
        self.initial_encoder = nn.Linear(10, config.d_model)
        self.pred_proj = nn.Linear(9, config.d_model)
        self.layers = nn.ModuleList(
            [
                RoPETransformerLayer(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.output_head = nn.Linear(config.d_model, 9)


def make_rope_tables(config: CDRGEConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = config.d_model // config.n_heads
    if head_dim % 4 != 0:
        raise ValueError("head_dim must be divisible by 4 for two-axis RoPE")
    rope_half = head_dim // 2
    rope_pairs = rope_half // 2
    frequencies = 1.0 / (10.0 ** (torch.arange(rope_pairs).float() * 2 / rope_half))
    row_angles = ROW_IDX.float().unsqueeze(1) * frequencies.unsqueeze(0)
    col_angles = COL_IDX.float().unsqueeze(1) * frequencies.unsqueeze(0)
    angles = torch.cat([row_angles, col_angles], dim=-1)
    return angles.cos().to(device), angles.sin().to(device)


def run_iterations(
    model: SudokuTransformer,
    x: torch.Tensor,
    horizon: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    return_all: bool,
) -> torch.Tensor | list[torch.Tensor]:
    hidden = model.initial_encoder(x)
    probabilities = torch.zeros(x.size(0), 81, 9, device=x.device, dtype=x.dtype)
    all_logits = []
    for _ in range(horizon):
        current = hidden + model.pred_proj(probabilities)
        for layer in model.layers:
            current = layer(current, rope_cos, rope_sin)
        hidden = current
        logits = model.output_head(hidden)
        probabilities = F.softmax(logits, dim=-1)
        if return_all:
            all_logits.append(logits)
    return all_logits if return_all else logits


def mean_iterative_ce(
    all_logits: Sequence[torch.Tensor],
    targets: torch.Tensor,
    empty_mask: torch.Tensor,
) -> torch.Tensor:
    denominator = empty_mask.sum().clamp_min(1)
    total = torch.zeros((), device=targets.device, dtype=torch.float32)
    for logits in all_logits:
        per_cell = F.cross_entropy(
            logits.reshape(-1, 9),
            targets.reshape(-1),
            reduction="none",
        ).view_as(targets)
        total = total + (per_cell * empty_mask).sum() / denominator
    return total / len(all_logits)


def make_fitness_function(
    model: SudokuTransformer,
    horizon: int,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    compile_fitness: bool,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def fitness(x: torch.Tensor, targets: torch.Tensor, empty_mask: torch.Tensor) -> torch.Tensor:
        logits = run_iterations(model, x, horizon, rope_cos, rope_sin, return_all=True)
        return mean_iterative_ce(logits, targets, empty_mask)

    return torch.compile(fitness, dynamic=False) if compile_fitness else fitness


def _rademacher_like(reference: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=reference.device)
    generator.manual_seed(seed)
    return torch.empty_like(reference).bernoulli_(0.5, generator=generator).mul_(2).sub_(1)


def apply_rademacher_(
    parameters: Sequence[torch.nn.Parameter],
    base_seed: int,
    scale: float,
    accumulators: Sequence[torch.Tensor] | None = None,
    accumulator_scale: float = 0.0,
) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(parameters):
            direction = _rademacher_like(parameter, base_seed + parameter_index)
            parameter.add_(direction, alpha=scale)
            if accumulators is not None:
                accumulators[parameter_index].add_(direction, alpha=accumulator_scale)


def cdrge_step(
    parameters: Sequence[torch.nn.Parameter],
    loss_function: Callable[[], float],
    direction_seeds: Iterable[int],
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Take one raw central-difference step and return probe losses/update norm."""
    seeds = list(direction_seeds)
    if not seeds:
        raise ValueError("CD-RGE needs at least one direction")

    snapshots = [parameter.detach().clone() for parameter in parameters]
    updates = [torch.zeros_like(parameter) for parameter in parameters]
    losses_plus = []
    losses_minus = []

    try:
        for seed in seeds:
            apply_rademacher_(parameters, seed, epsilon)
            loss_plus = float(loss_function())

            apply_rademacher_(parameters, seed, -2.0 * epsilon)
            loss_minus = float(loss_function())

            coefficient = -(loss_plus - loss_minus) / (2.0 * len(seeds))
            apply_rademacher_(
                parameters,
                seed,
                epsilon,
                accumulators=updates,
                accumulator_scale=coefficient,
            )
            losses_plus.append(loss_plus)
            losses_minus.append(loss_minus)

        if not np.isfinite(losses_plus).all() or not np.isfinite(losses_minus).all():
            raise FloatingPointError("non-finite perturbation loss")

        with torch.no_grad():
            update_squared_norm = torch.zeros((), device=parameters[0].device)
            for parameter, snapshot, update in zip(parameters, snapshots, updates):
                parameter.copy_(snapshot).add_(update)
                update_squared_norm.add_(update.float().square().sum())
        return (
            np.asarray(losses_plus, dtype=np.float64),
            np.asarray(losses_minus, dtype=np.float64),
            math.sqrt(float(update_squared_norm.item())),
        )
    except BaseException:
        with torch.no_grad():
            for parameter, snapshot in zip(parameters, snapshots):
                parameter.copy_(snapshot)
        raise


def _balanced_probe_indices(dataset, limit: int) -> list[int]:
    per_bucket_limit = max(1, limit // len(RATING_BUCKETS))
    counts = {name: 0 for _, _, name in RATING_BUCKETS}
    indices = []
    for index in range(len(dataset)):
        rating = dataset[index]["rating"]
        for minimum, maximum, name in RATING_BUCKETS:
            if minimum <= rating <= maximum and counts[name] < per_bucket_limit:
                indices.append(index)
                counts[name] += 1
                break
        if len(indices) >= limit:
            break
    return indices


def _stage_for_generation(config: CDRGEConfig, generation: int) -> tuple[TrainingStage, int]:
    stage_start = 0
    for stage in config.stages:
        stage_end = stage_start + stage.generations
        if generation < stage_end:
            return stage, stage_start
        stage_start = stage_end
    raise IndexError(f"generation {generation} is past the configured schedule")


def _sample_indices(eligible: torch.Tensor, batch_size: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    choices = torch.randint(0, eligible.numel(), (batch_size,), generator=generator)
    return eligible[choices]


def _direction_seeds(config: CDRGEConfig, generation: int) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.direction_seed + generation)
    return torch.randint(
        0,
        2**31 - 1 - 100,
        (config.population_directions,),
        generator=generator,
    ).tolist()


def evaluate_probe(
    model: SudokuTransformer,
    x: torch.Tensor,
    targets: torch.Tensor,
    empty_mask: torch.Tensor,
    horizons: Sequence[int],
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    batch_size: int = 128,
) -> dict[int, dict[str, float | int]]:
    results = {}
    with torch.inference_mode():
        for horizon in dict.fromkeys(horizons):
            loss_total = 0.0
            correct_cells = 0
            empty_cells = 0
            solved = 0
            for start in range(0, x.size(0), batch_size):
                batch_x = x[start : start + batch_size]
                batch_targets = targets[start : start + batch_size]
                batch_mask = empty_mask[start : start + batch_size]
                all_logits = run_iterations(
                    model,
                    batch_x,
                    horizon,
                    rope_cos,
                    rope_sin,
                    return_all=True,
                )
                loss = mean_iterative_ce(all_logits, batch_targets, batch_mask)
                predictions = all_logits[-1].argmax(dim=-1)
                correctness = (predictions == batch_targets) | ~batch_mask.bool()
                loss_total += float(loss.item()) * batch_x.size(0)
                correct_cells += int(((predictions == batch_targets) * batch_mask).sum().item())
                empty_cells += int(batch_mask.sum().item())
                solved += int(correctness.all(dim=1).sum().item())
            results[horizon] = {
                "ce": loss_total / x.size(0),
                "cell_accuracy": correct_cells / max(1, empty_cells),
                "solved": solved,
                "total": x.size(0),
            }
    return results


def _format_probe(results: dict[int, dict[str, float | int]]) -> str:
    return " | ".join(
        f"h{horizon}: ce={metrics['ce']:.4f}, cells={metrics['cell_accuracy']:.2%}, "
        f"solved={metrics['solved']}/{metrics['total']}"
        for horizon, metrics in results.items()
    )


def train_cdrge(config: CDRGEConfig, output_dir: str = ".") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CD-RGE training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    run_dir = os.path.join(output_dir, "cdrge_9x9", config.name)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "train.log")
    log_file = open(log_path, "a")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    config_dict = asdict(config)
    total_generations = sum(stage.generations for stage in config.stages)
    log(f"CONFIG {config_dict}")

    log("Loading full 9x9 Sudoku training pool...")
    train_dataset = load_dataset("sapientinc/sudoku-extreme", split="train")
    pool_rows = train_dataset[config.pool_offset : config.pool_offset + config.pool_size]
    pool_x = encode_puzzles(pool_rows["question"])
    pool_targets = encode_solutions(pool_rows["answer"])
    pool_empty = pool_x[:, :, 0].bool()
    ratings = torch.as_tensor(pool_rows["rating"], dtype=torch.int16)

    all_indices = torch.arange(pool_x.size(0))
    high_clue_indices = torch.argsort(pool_empty.sum(dim=1), stable=True)[: config.high_clue_pool_size]
    easy_indices = torch.nonzero(ratings <= 2, as_tuple=False).flatten()
    eligible_indices = {
        "all": all_indices,
        "high_clue": high_clue_indices,
        "easy": easy_indices,
    }
    log(
        f"POOL all={all_indices.numel()}, easy(rating<=2)={easy_indices.numel()}, "
        f"high_clue={high_clue_indices.numel()} "
        f"(empty cells {int(pool_empty[high_clue_indices].sum(dim=1).min())}-"
        f"{int(pool_empty[high_clue_indices].sum(dim=1).max())})"
    )

    log("Loading balanced 9x9 Sudoku test probe...")
    test_dataset = load_dataset("sapientinc/sudoku-extreme", split="test")
    probe_indices = _balanced_probe_indices(test_dataset, config.diagnostic_puzzles)
    probe_puzzles = [test_dataset[index]["question"] for index in probe_indices]
    probe_solutions = [test_dataset[index]["answer"] for index in probe_indices]
    probe_x = encode_puzzles(probe_puzzles).to(device)
    probe_targets = encode_solutions(probe_solutions).long().to(device)
    probe_empty = probe_x[:, :, 0].bool()

    torch.manual_seed(config.model_seed)
    torch.cuda.manual_seed_all(config.model_seed)
    model = SudokuTransformer(config).to(device).eval()
    parameters = list(model.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    rope_cos, rope_sin = make_rope_tables(config, device)
    log(
        f"MODEL parameters={parameter_count:,}, directions={config.population_directions}, "
        f"epsilon=learning_rate={config.epsilon}, batch={config.batch_size}, precision=float32"
    )

    checkpoint_path, checkpoint_generation = find_latest_checkpoint(run_dir, "checkpoint_step")
    start_generation = 0
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("config") != config_dict:
            raise ValueError(f"Config mismatch in {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        start_generation = checkpoint_generation + 1
        log(f"RESUME generation={checkpoint_generation} from {checkpoint_path}")

    fitness_functions = {
        horizon: make_fitness_function(
            model,
            horizon,
            rope_cos,
            rope_sin,
            config.compile_fitness,
        )
        for horizon in sorted({stage.horizon for stage in config.stages})
    }

    def save_checkpoint(generation: int) -> None:
        checkpoint_file = os.path.join(run_dir, f"checkpoint_step{generation}.pt")
        atomic_torch_save(
            {
                "step": generation,
                "model_state_dict": model.state_dict(),
                "config": config_dict,
            },
            checkpoint_file,
        )
        log(f"CHECKPOINT {checkpoint_file}")

    if start_generation >= total_generations:
        log(f"Run already complete at generation {start_generation - 1}")
        log_file.close()
        return {"already_complete": True, "run_dir": run_dir}

    initial_stage, _ = _stage_for_generation(config, start_generation)
    initial_probe = evaluate_probe(
        model,
        probe_x,
        probe_targets,
        probe_empty,
        (1, initial_stage.horizon),
        rope_cos,
        rope_sin,
    )
    log(f"PROBE start | {_format_probe(initial_probe)}")

    previous_stage_name = None
    final_clean_loss = float("nan")
    for generation in range(start_generation, total_generations):
        stage, stage_start = _stage_for_generation(config, generation)
        stage_generation = generation - stage_start
        if stage.name != previous_stage_name:
            log(
                f"STAGE {stage.name} | generation={generation}, horizon={stage.horizon}, "
                f"data={stage.data_slice}, eligible={eligible_indices[stage.data_slice].numel()}"
            )
            previous_stage_name = stage.name

        selected = _sample_indices(
            eligible_indices[stage.data_slice],
            config.batch_size,
            config.data_seed + generation,
        )
        batch_x = pool_x[selected].to(device)
        batch_targets = pool_targets[selected].long().to(device)
        batch_empty = pool_empty[selected].to(device)
        fitness = fitness_functions[stage.horizon]

        def current_loss() -> float:
            with torch.inference_mode():
                return float(fitness(batch_x, batch_targets, batch_empty).item())

        started = time.time()
        losses_plus, losses_minus, update_norm = cdrge_step(
            parameters,
            current_loss,
            _direction_seeds(config, generation),
            config.epsilon,
        )
        final_clean_loss = current_loss()
        pair_losses = np.concatenate([losses_plus, losses_minus])
        directional_differences = losses_plus - losses_minus
        log(
            f"GEN {generation:04d}/{total_generations - 1:04d} "
            f"stage={stage.name}:{stage_generation:03d} h={stage.horizon} "
            f"probe_loss={pair_losses.mean():.6f}+/-{pair_losses.std():.6f} "
            f"diff_std={directional_differences.std():.6f} "
            f"clean={final_clean_loss:.6f} update_l2={update_norm:.6f} "
            f"seconds={time.time() - started:.1f}"
        )

        stage_ends = stage_generation == stage.generations - 1
        if generation % config.diagnostic_every == 0 or stage_ends:
            probe = evaluate_probe(
                model,
                probe_x,
                probe_targets,
                probe_empty,
                (1, stage.horizon),
                rope_cos,
                rope_sin,
            )
            log(f"PROBE generation={generation} | {_format_probe(probe)}")

        if generation % config.checkpoint_every == 0 or stage_ends:
            save_checkpoint(generation)

    final_probe = evaluate_probe(
        model,
        probe_x,
        probe_targets,
        probe_empty,
        config.final_probe_horizons,
        rope_cos,
        rope_sin,
    )
    log(f"FINAL PROBE | {_format_probe(final_probe)}")
    final_path = os.path.join(run_dir, "model.pt")
    atomic_torch_save(model.state_dict(), final_path)
    log(f"FINAL MODEL {final_path}")
    log_file.close()
    return {
        "run_dir": run_dir,
        "final_clean_loss": final_clean_loss,
        "final_probe": final_probe,
    }
