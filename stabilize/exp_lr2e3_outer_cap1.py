"""Full 50K SOTA schedule with a per-token recurrent-state RMS cap of 1."""

from stabilize.exp_testbed_20k import train as train_scheduled


EXPERIMENT_NAME = "exp_lr2e3_outer_cap1"
FULL_50K_SCHEDULE = {
    'warmup_steps': 1400,
    'total_steps': 50000,
    'eval_every': 5000,
    'probe_every': 2000,
    'phases': (
        (0, 10000, 21, "Phase 1: Hard only (rating 21+)"),
        (10000, 20000, 6, "Phase 2: Medium+ (rating 6+)"),
        (20000, 30000, 1, "Phase 3: Easy+ (rating 1+)"),
        (30000, 50000, 0, "Phase 4: All (rating 0+)"),
    ),
}


def train(output_dir=".", run_name="lr2e3_outer_cap1_trial0", random_seed=20_260_720):
    return train_scheduled(
        output_dir=output_dir,
        experiment_name=EXPERIMENT_NAME,
        run_name=run_name,
        outer_state_rms_cap=1.0,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        schedule=FULL_50K_SCHEDULE,
    )


if __name__ == "__main__":
    train()
