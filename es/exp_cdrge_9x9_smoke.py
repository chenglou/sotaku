from es.cdrge_9x9 import CDRGEConfig, TrainingStage, train_cdrge


CONFIG = CDRGEConfig(
    name="smoke",
    population_directions=4,
    stages=(
        TrainingStage("high-clue-h1", 1, 1, "high_clue"),
        TrainingStage("all-h16", 1, 16, "all"),
    ),
    batch_size=64,
    pool_size=4096,
    high_clue_pool_size=1024,
    diagnostic_puzzles=100,
    diagnostic_every=1,
    checkpoint_every=1,
    final_probe_horizons=(1, 16),
)


def train(output_dir="."):
    return train_cdrge(CONFIG, output_dir)
