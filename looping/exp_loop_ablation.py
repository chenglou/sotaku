"""Controlled 20K ablations for loop order and recurrent residual scaling."""

from copy import deepcopy

from stabilize.exp_testbed_20k import n_iterations, train as train_testbed


LINEAR_LOOP_SCALE = 1.0 / n_iterations
SQRT_LOOP_SCALE = 1.0 / (n_iterations ** 0.5)

ARM_CONFIGS = {
    # Matched unnormalized control for the residual-scaling arms.
    "baseline_unscaled": {},
    # The paper's prescription applies 1/N to each attention and MLP branch.
    # Sotaku's separately repeated prediction injection remains unchanged here.
    "residual_linear_branches": {
        "residual_scale": LINEAR_LOOP_SCALE,
    },
    # Adapt the same tied-update argument to every recurrent addition in Sotaku.
    "residual_linear_all": {
        "residual_scale": LINEAR_LOOP_SCALE,
        "feedback_scale": LINEAR_LOOP_SCALE,
    },
    # Standard deep-residual 1/sqrt(N) scaling, included as the paper's control.
    "residual_sqrt_all": {
        "residual_scale": SQRT_LOOP_SCALE,
        "feedback_scale": SQRT_LOOP_SCALE,
    },
    # Same two stored blocks, parameters, four block calls, and outer RMSNorm.
    "schedule_abab": {
        "outer_state_norm": True,
        "unique_layers": 2,
        "layer_schedule": (0, 1, 0, 1),
    },
    "schedule_aabb": {
        "outer_state_norm": True,
        "unique_layers": 2,
        "layer_schedule": (0, 0, 1, 1),
    },
}


def get_arm_config(arm):
    try:
        return deepcopy(ARM_CONFIGS[arm])
    except KeyError as error:
        choices = ", ".join(sorted(ARM_CONFIGS))
        raise ValueError(f"unknown arm {arm!r}; choose one of: {choices}") from error


def train(
    output_dir=".",
    *,
    arm="baseline_unscaled",
    run_name=None,
    random_seed=20_260_720,
):
    arm_config = get_arm_config(arm)
    if run_name is None:
        run_name = f"loop_{arm}_trial0"
    return train_testbed(
        output_dir=output_dir,
        experiment_name=f"exp_loop_{arm}",
        run_name=run_name,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        **arm_config,
    )


if __name__ == "__main__":
    train()
