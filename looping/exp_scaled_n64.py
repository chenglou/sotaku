"""Train for 64 recurrent iterations with linear tied-update scaling."""

from stabilize.exp_testbed_20k import train as train_testbed


TRAINING_ITERATIONS = 64
LINEAR_SCALE = 1.0 / TRAINING_ITERATIONS
RUN_BATCH_SIZE = 2048
MICROBATCH_SIZE = 1024


def train(
    output_dir=".",
    *,
    run_name="loop_scaled_n64_trial0",
    random_seed=20_260_720,
):
    return train_testbed(
        output_dir=output_dir,
        experiment_name="exp_loop_scaled_n64",
        run_name=run_name,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        training_iterations=TRAINING_ITERATIONS,
        run_batch_size=RUN_BATCH_SIZE,
        microbatch_size=MICROBATCH_SIZE,
        residual_scale=LINEAR_SCALE,
        feedback_scale=LINEAR_SCALE,
    )


if __name__ == "__main__":
    train()
