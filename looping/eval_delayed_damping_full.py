"""Full 25K-puzzle evaluation of the strongest delayed-damping policy."""

from looping.eval_delayed_damping import evaluate
from looping.eval_damping_robustness import (
    DELAYED_LATE_CHECKPOINT_MODEL,
    LATE_CHECKPOINT_MODEL,
    LATE_CHECKPOINT_TRIAL1_MODEL,
)
from looping.eval_loop_diagnostics import DEFAULT_MODELS


MODEL_CONFIGS = {
    model_config["name"]: model_config
    for model_config in (
        *DEFAULT_MODELS[:2],
        LATE_CHECKPOINT_MODEL,
        DELAYED_LATE_CHECKPOINT_MODEL,
        LATE_CHECKPOINT_TRIAL1_MODEL,
    )
}
STANDARD_POLICIES = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
)
LATE_CHECKPOINT_FULL_POLICIES = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm512_a025", "alpha": 0.25, "warmup_iterations": 512},
)
DELAYED_LATE_CHECKPOINT_FULL_POLICIES = (
    {"name": "warm512_a025", "alpha": 0.25, "warmup_iterations": 512},
)


def evaluate_model(
    model_name,
    *,
    examples_per_bucket=5000,
    horizons=None,
    output_dir=".",
):
    if model_name not in MODEL_CONFIGS:
        choices = ", ".join(MODEL_CONFIGS)
        raise ValueError(f"unknown model {model_name!r}; expected one of: {choices}")
    if model_name in {
        LATE_CHECKPOINT_MODEL["name"],
        DELAYED_LATE_CHECKPOINT_MODEL["name"],
        LATE_CHECKPOINT_TRIAL1_MODEL["name"],
    }:
        policies = (
            DELAYED_LATE_CHECKPOINT_FULL_POLICIES
            if model_name != LATE_CHECKPOINT_MODEL["name"]
            else LATE_CHECKPOINT_FULL_POLICIES
        )
        default_horizons = (16, 128, 512, 1024, 2048, 4096)
    else:
        policies = STANDARD_POLICIES
        default_horizons = (16, 128, 1024, 2048)
    if horizons is None:
        horizons = default_horizons
    return evaluate(
        model_configs=(MODEL_CONFIGS[model_name],),
        policies=policies,
        examples_per_bucket=examples_per_bucket,
        batch_size=250,
        horizons=horizons,
        output_dir=output_dir,
        output_prefix=f"delayed-damping-full-{model_name}",
    )


if __name__ == "__main__":
    evaluate_model("collapsed_unbounded")
