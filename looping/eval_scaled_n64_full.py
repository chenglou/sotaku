"""Full-horizon evaluation for the 64-iteration scaled checkpoint."""

from iters.eval_state_rms_cap import evaluate
from looping.exp_scaled_n64 import LINEAR_SCALE, TRAINING_ITERATIONS


def evaluate_run(
    run_name="loop_scaled_n64_trial0",
    *,
    examples_per_bucket=5000,
    output_dir=".",
):
    model_name = f"{run_name}_best"
    model_path = f"/outputs/looping/model_{run_name}_best_probe.pt"
    return evaluate(
        model_configs=((model_name, model_path),),
        experiment_module="stabilize.exp_testbed_20k",
        caps=(None,),
        checkpoints=(16, 64, 128, 1024, 2048),
        examples_per_bucket=examples_per_bucket,
        batch_size=250,
        seed=42,
        output_dir=output_dir,
        output_prefix=f"{run_name}_best_full_horizon",
        model_kwargs={
            "training_iterations": TRAINING_ITERATIONS,
            "residual_scale": LINEAR_SCALE,
            "feedback_scale": LINEAR_SCALE,
        },
    )


if __name__ == "__main__":
    evaluate_run()
