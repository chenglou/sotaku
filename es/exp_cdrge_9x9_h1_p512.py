from es.cdrge_9x9 import fixed_horizon_config, train_cdrge


CONFIG = fixed_horizon_config("h1_p512", population_directions=512)


def train(output_dir="."):
    return train_cdrge(CONFIG, output_dir)
