"""Shared sampling rules for evolution-strategies experiments."""

import random

import numpy as np


POPULATION_SIZE = 32
POPULATION_PAIRS = POPULATION_SIZE // 2
DEFAULT_SAMPLING_MODE = "independent"
SAMPLING_MODES = {"independent", "paired"}


def generation_direction_seeds(es_random_seed, generation):
    """Return one generation's deterministic directions, independent of resume."""
    rng = random.Random(es_random_seed * 1_000_003 + generation)
    return [rng.randrange(2**62) for _ in range(POPULATION_SIZE)]


def direction_weights(scores, mode=DEFAULT_SAMPLING_MODE):
    """Convert 32 scores to one normalized weight per sampled direction."""
    if mode not in SAMPLING_MODES:
        raise ValueError(f"unknown sampling mode: {mode}")
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (POPULATION_SIZE,):
        raise ValueError(f"expected {POPULATION_SIZE} scores, got {scores.shape}")
    spread = scores.std()
    direction_count = POPULATION_PAIRS if mode == "paired" else POPULATION_SIZE
    if spread <= 1e-9:
        return np.zeros(direction_count, dtype=np.float64), spread

    utilities = (scores - scores.mean()) / spread
    if mode == "paired":
        weights = (
            utilities[:POPULATION_PAIRS] - utilities[POPULATION_PAIRS:]
        ) / POPULATION_SIZE
    else:
        weights = utilities / POPULATION_SIZE
    return weights, spread
