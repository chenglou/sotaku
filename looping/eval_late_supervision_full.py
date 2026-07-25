"""Full-horizon evaluation for a completed late-supervision run."""

from iters.eval_state_rms_cap import evaluate
from looping.exp_late_supervision import get_late_supervision_config


MODEL_CONFIG_KEYS = {
    "feedback_scale",
    "layer_schedule",
    "outer_state_norm",
    "outer_state_rms_cap",
    "residual_scale",
    "training_iterations",
    "unique_layers",
}


def evaluate_run(
    run_name,
    *,
    arm=None,
    outer_state_norm=False,
    examples_per_bucket=5000,
    output_dir=".",
):
    model_name = f"{run_name}_best"
    model_path = f"/outputs/looping/model_{run_name}_best_probe.pt"
    model_kwargs = {"outer_state_norm": outer_state_norm}
    if arm is not None:
        arm_config = get_late_supervision_config(arm)
        model_kwargs.update({
            key: value
            for key, value in arm_config.items()
            if key in MODEL_CONFIG_KEYS
        })
    return evaluate(
        model_configs=((model_name, model_path),),
        experiment_module="stabilize.exp_testbed_20k",
        caps=(None,),
        checkpoints=(16, 128, 1024, 2048),
        examples_per_bucket=examples_per_bucket,
        batch_size=250,
        seed=42,
        output_dir=output_dir,
        output_prefix=f"{run_name}_best_full_horizon",
        model_kwargs=model_kwargs,
    )


if __name__ == "__main__":
    evaluate_run("loop_late_random_replace_trial0")
