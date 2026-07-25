from es.cdrge_9x9 import curriculum_config, train_cdrge


CONFIG = curriculum_config("curriculum_p512", population_directions=512)


def train(output_dir="."):
    return train_cdrge(CONFIG, output_dir)
