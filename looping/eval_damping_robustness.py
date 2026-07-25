"""Check delayed damping across collapse checkpoints and nearby policies."""

from looping.eval_delayed_damping import evaluate
from looping.eval_loop_diagnostics import (
    CLEAN_A_TRAJECTORY_MODELS,
    DEFAULT_MODELS,
)


RESCUE_POLICIES = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
)

POLICY_GRID = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm64_a025", "alpha": 0.25, "warmup_iterations": 64},
    {"name": "warm128_a05", "alpha": 0.5, "warmup_iterations": 128},
    {"name": "warm128_a0375", "alpha": 0.375, "warmup_iterations": 128},
    {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
    {"name": "warm128_a0125", "alpha": 0.125, "warmup_iterations": 128},
    {"name": "warm256_a025", "alpha": 0.25, "warmup_iterations": 256},
)

STEP35_POLICY_GRID = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm64_a025", "alpha": 0.25, "warmup_iterations": 64},
    {"name": "warm64_a0125", "alpha": 0.125, "warmup_iterations": 64},
    {"name": "warm64_a00625", "alpha": 0.0625, "warmup_iterations": 64},
    {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
    {"name": "warm128_a0125", "alpha": 0.125, "warmup_iterations": 128},
    {"name": "warm128_a00625", "alpha": 0.0625, "warmup_iterations": 128},
)

ES_BOUNDARY_MODELS = (
    {
        "name": "cohort_a_step_40000",
        "path": "/outputs/baseline_lr2e3_cohort_a_checkpoint_step40000.pt",
        "model_kwargs": {},
    },
    {
        "name": "cohort_d_step_45000",
        "path": "/outputs/baseline_lr2e3_cohort_d_checkpoint_step45000.pt",
        "model_kwargs": {},
    },
    {
        "name": "cohort_g_step_40000",
        "path": "/outputs/baseline_lr2e3_cohort_g_checkpoint_step40000.pt",
        "model_kwargs": {},
    },
    {
        "name": "clean_b_step_35000",
        "path": "/outputs/bs2048_baseline_clean_b_checkpoint_step35000.pt",
        "model_kwargs": {},
    },
)

LATE_CHECKPOINT_MODEL = {
    "name": "late_random_replace_trial0_best",
    "path": (
        "/outputs/looping/"
        "model_loop_late_random_replace_trial0_best_probe.pt"
    ),
    "model_kwargs": {},
}

DELAYED_LATE_CHECKPOINT_MODEL = {
    "name": "late_random_replace_after_2k_trial0_best",
    "path": (
        "/outputs/looping/"
        "model_loop_late_random_replace_after_2k_trial0_best_probe.pt"
    ),
    "model_kwargs": {},
}

LATE_CHECKPOINT_TRIAL1_MODEL = {
    "name": "late_random_replace_trial1_best",
    "path": (
        "/outputs/looping/"
        "model_loop_late_random_replace_trial1_best_probe.pt"
    ),
    "model_kwargs": {},
}

LATE_THROUGH_1024_CHECKPOINT_MODEL = {
    "name": "late_random_replace_through_1024_trial0_best",
    "path": (
        "/outputs/looping/"
        "model_loop_late_random_replace_through_1024_trial0_best_probe.pt"
    ),
    "model_kwargs": {},
}

LATE_THROUGH_1024_FINAL_MODELS = (
    {
        "name": "late_random_replace_through_1024_trial0_final",
        "path": (
            "/outputs/looping/"
            "model_loop_late_random_replace_through_1024_trial0.pt"
        ),
        "model_kwargs": {},
    },
    {
        "name": "late_random_replace_through_1024_trial1_final",
        "path": (
            "/outputs/looping/"
            "model_loop_late_random_replace_through_1024_trial1.pt"
        ),
        "model_kwargs": {},
    },
)

LATE_CHECKPOINT_POLICIES = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm512_a05", "alpha": 0.5, "warmup_iterations": 512},
    {"name": "warm512_a025", "alpha": 0.25, "warmup_iterations": 512},
    {"name": "warm1024_a05", "alpha": 0.5, "warmup_iterations": 1024},
    {"name": "warm1024_a025", "alpha": 0.25, "warmup_iterations": 1024},
    {"name": "warm1024_a0125", "alpha": 0.125, "warmup_iterations": 1024},
)

LATE_THROUGH_1024_FINAL_POLICIES = (
    {"name": "undamped", "alpha": 1.0, "warmup_iterations": 0},
    {"name": "warm128_a025", "alpha": 0.25, "warmup_iterations": 128},
    {"name": "warm128_a0125", "alpha": 0.125, "warmup_iterations": 128},
    {"name": "warm256_a025", "alpha": 0.25, "warmup_iterations": 256},
    {"name": "warm256_a0125", "alpha": 0.125, "warmup_iterations": 256},
    {"name": "warm512_a025", "alpha": 0.25, "warmup_iterations": 512},
)


def evaluate_trajectory(*, examples_per_bucket=200, output_dir="."):
    return evaluate(
        model_configs=CLEAN_A_TRAJECTORY_MODELS,
        policies=RESCUE_POLICIES,
        examples_per_bucket=examples_per_bucket,
        horizons=(16, 128, 1024, 2048),
        output_dir=output_dir,
        output_prefix="delayed-damping-clean-a-trajectory-n1000",
    )


def evaluate_policy_grid(*, examples_per_bucket=200, output_dir="."):
    return evaluate(
        model_configs=(DEFAULT_MODELS[1],),
        policies=POLICY_GRID,
        examples_per_bucket=examples_per_bucket,
        horizons=(16, 64, 128, 256, 1024, 2048),
        output_dir=output_dir,
        output_prefix="delayed-damping-policy-grid-n1000",
    )


def evaluate_step35_grid(*, examples_per_bucket=200, output_dir="."):
    return evaluate(
        model_configs=(CLEAN_A_TRAJECTORY_MODELS[1],),
        policies=STEP35_POLICY_GRID,
        examples_per_bucket=examples_per_bucket,
        horizons=(16, 64, 128, 256, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-step35-grid-n1000",
    )


def evaluate_es_boundary(*, examples_per_bucket=200, output_dir="."):
    policies = (
        RESCUE_POLICIES[0],
        RESCUE_POLICIES[1],
        {"name": "warm128_a0125", "alpha": 0.125, "warmup_iterations": 128},
    )
    return evaluate(
        model_configs=ES_BOUNDARY_MODELS,
        policies=policies,
        examples_per_bucket=examples_per_bucket,
        horizons=(16, 128, 1024, 2048),
        output_dir=output_dir,
        output_prefix="delayed-damping-es-boundary-n1000",
    )


def evaluate_es_boundary_strong(*, examples_per_bucket=200, output_dir="."):
    policies = (
        {"name": "warm128_a00625", "alpha": 0.0625, "warmup_iterations": 128},
        {"name": "warm128_a003125", "alpha": 0.03125, "warmup_iterations": 128},
    )
    return evaluate(
        model_configs=ES_BOUNDARY_MODELS,
        policies=policies,
        examples_per_bucket=examples_per_bucket,
        horizons=(128, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-es-boundary-strong-n1000",
    )


def evaluate_late_checkpoint(*, examples_per_bucket=200, output_dir="."):
    return evaluate(
        model_configs=(LATE_CHECKPOINT_MODEL,),
        policies=LATE_CHECKPOINT_POLICIES,
        examples_per_bucket=examples_per_bucket,
        horizons=(128, 512, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-late-checkpoint-n1000",
    )


def evaluate_delayed_late_checkpoint(
    *,
    examples_per_bucket=200,
    output_dir=".",
):
    return evaluate(
        model_configs=(DELAYED_LATE_CHECKPOINT_MODEL,),
        policies=LATE_CHECKPOINT_POLICIES,
        examples_per_bucket=examples_per_bucket,
        horizons=(128, 512, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-late-after2k-checkpoint-n1000",
    )


def evaluate_late_checkpoint_trial1(
    *,
    examples_per_bucket=200,
    output_dir=".",
):
    return evaluate(
        model_configs=(LATE_CHECKPOINT_TRIAL1_MODEL,),
        policies=LATE_CHECKPOINT_POLICIES,
        examples_per_bucket=examples_per_bucket,
        horizons=(128, 512, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-late-trial1-checkpoint-n1000",
    )


def evaluate_late_checkpoint_through_1024(
    *,
    examples_per_bucket=200,
    output_dir=".",
):
    return evaluate(
        model_configs=(LATE_THROUGH_1024_CHECKPOINT_MODEL,),
        policies=LATE_CHECKPOINT_POLICIES,
        examples_per_bucket=examples_per_bucket,
        horizons=(128, 512, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-late-through1024-checkpoint-n1000",
    )


def evaluate_late_through_1024_final_grid(
    *,
    examples_per_bucket=200,
    output_dir=".",
):
    return evaluate(
        model_configs=LATE_THROUGH_1024_FINAL_MODELS,
        policies=LATE_THROUGH_1024_FINAL_POLICIES,
        examples_per_bucket=examples_per_bucket,
        batch_size=250,
        horizons=(128, 256, 512, 1024, 2048, 4096),
        output_dir=output_dir,
        output_prefix="delayed-damping-late-through1024-final-grid-n1000",
    )


EXPERIMENTS = {
    "trajectory": evaluate_trajectory,
    "policy_grid": evaluate_policy_grid,
    "step35_grid": evaluate_step35_grid,
    "es_boundary": evaluate_es_boundary,
    "es_boundary_strong": evaluate_es_boundary_strong,
    "late_checkpoint": evaluate_late_checkpoint,
    "delayed_late_checkpoint": evaluate_delayed_late_checkpoint,
    "late_checkpoint_trial1": evaluate_late_checkpoint_trial1,
    "late_checkpoint_through_1024": evaluate_late_checkpoint_through_1024,
    "late_through_1024_final_grid": evaluate_late_through_1024_final_grid,
}


def run_experiment(name, *, examples_per_bucket=200, output_dir="."):
    if name not in EXPERIMENTS:
        choices = ", ".join(EXPERIMENTS)
        raise ValueError(f"unknown experiment {name!r}; expected one of: {choices}")
    return EXPERIMENTS[name](
        examples_per_bucket=examples_per_bucket,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    run_experiment("trajectory")
