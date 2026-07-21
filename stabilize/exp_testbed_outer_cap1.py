"""20K SOTA testbed with a per-token RMS-1 cap after every recurrent loop."""

from stabilize.exp_testbed_20k import train as train_testbed


EXPERIMENT_NAME = "exp_testbed_outer_cap1"


def train(output_dir=".", run_name="testbed_outer_cap1_trial0", random_seed=20_260_720):
    return train_testbed(
        output_dir=output_dir,
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        outer_state_rms_cap=1.0,
        random_seed=random_seed,
        checkpoint_on_probe=True,
    )


if __name__ == "__main__":
    train()
