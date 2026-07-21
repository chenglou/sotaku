"""Full-horizon evaluation of the harvested 50K recurrent-RMSNorm checkpoints."""

from iters.eval_state_rms_cap import evaluate as evaluate_outer_state


BEST_MODEL_CONFIGS = tuple(
    (
        f"rmsnorm_50k_trial{trial}_best",
        f"/outputs/model_lr2e3_outer_rmsnorm_trial{trial}_best_probe.pt",
    )
    for trial in range(3)
)
CHECKPOINTS = (16, 128, 1024, 2048)


def evaluate(
    output_dir=".",
    output_prefix="lr2e3_outer_rmsnorm_best_full_horizon",
    examples_per_bucket=5000,
    batch_size=250,
):
    return evaluate_outer_state(
        model_configs=BEST_MODEL_CONFIGS,
        experiment_module="stabilize.exp_testbed_20k",
        caps=(None,),
        checkpoints=CHECKPOINTS,
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
        seed=42,
        device="cuda",
        output_dir=output_dir,
        output_prefix=output_prefix,
        model_kwargs={"outer_state_norm": True},
    )


if __name__ == "__main__":
    evaluate()
