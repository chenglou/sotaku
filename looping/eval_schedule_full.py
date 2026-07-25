"""Full 25K-puzzle evaluation of compact ABAB and AABB checkpoints."""

from iters.eval_state_rms_cap import evaluate


SCHEDULE_MODELS = (
    {
        "name": "schedule_abab_trial0_best",
        "path": "/outputs/looping/model_loop_schedule_abab_trial0_best_probe.pt",
        "model_kwargs": {
            "outer_state_norm": True,
            "unique_layers": 2,
            "layer_schedule": (0, 1, 0, 1),
        },
    },
    {
        "name": "schedule_aabb_trial0_best",
        "path": "/outputs/looping/model_loop_schedule_aabb_trial0_best_probe.pt",
        "model_kwargs": {
            "outer_state_norm": True,
            "unique_layers": 2,
            "layer_schedule": (0, 0, 1, 1),
        },
    },
)


def evaluate_all(
    output_dir=".",
    *,
    examples_per_bucket=5000,
    checkpoints=(16, 128, 1024, 2048),
):
    results = {}
    for model_config in SCHEDULE_MODELS:
        name = model_config["name"]
        results[name] = evaluate(
            model_configs=((name, model_config["path"]),),
            experiment_module="stabilize.exp_testbed_20k",
            caps=(None,),
            checkpoints=checkpoints,
            examples_per_bucket=examples_per_bucket,
            batch_size=250,
            seed=42,
            output_dir=output_dir,
            output_prefix=f"{name}_full_horizon",
            model_kwargs=model_config["model_kwargs"],
        )
    return results


if __name__ == "__main__":
    evaluate_all()
