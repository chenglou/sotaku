"""Full 50K SOTA schedule with parameter-free recurrent-state RMSNorm."""

from stabilize.exp_lr2e3_outer_cap1 import FULL_50K_SCHEDULE
from stabilize.exp_testbed_20k import train as train_scheduled


EXPERIMENT_NAME = "exp_lr2e3_outer_rmsnorm"


def train(output_dir=".", run_name="lr2e3_outer_rmsnorm_trial0", random_seed=20_260_720):
    return train_scheduled(
        output_dir=output_dir,
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        outer_state_norm=True,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        schedule=FULL_50K_SCHEDULE,
    )


if __name__ == "__main__":
    train()
