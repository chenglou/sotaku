"""Full-horizon evaluation for the cap-1 recurrent-state testbed model."""

from iters.eval_state_rms_cap import evaluate as evaluate_rms_cap


MODEL_PATH = "/outputs/model_testbed_outer_cap1_trial0.pt"
CHECKPOINTS = (16, 128, 1024, 2048)


def evaluate(
    output_dir=".",
    output_prefix="testbed_outer_cap1_trial0_full_horizon",
    examples_per_bucket=5000,
    batch_size=250,
):
    return evaluate_rms_cap(
        model_configs=(("cap1_trial0", MODEL_PATH),),
        experiment_module="stabilize.exp_testbed_20k",
        caps=(1.0,),
        checkpoints=CHECKPOINTS,
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
        seed=42,
        device="cuda",
        output_dir=output_dir,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    evaluate()
