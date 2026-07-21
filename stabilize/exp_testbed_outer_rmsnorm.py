"""20K SOTA testbed with parameter-free RMS normalization after every loop."""

from stabilize.exp_testbed_20k import train as train_testbed


EXPERIMENT_NAME = "exp_testbed_outer_rmsnorm"


def train(output_dir=".", run_name="testbed_outer_rmsnorm_trial0", random_seed=20_260_720):
    return train_testbed(
        output_dir=output_dir,
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        outer_state_norm=True,
        random_seed=random_seed,
        checkpoint_on_probe=True,
    )


if __name__ == "__main__":
    train()
