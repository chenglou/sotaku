"""Random late-state supervision experiments on the 20K testbed."""

from copy import deepcopy

from stabilize.exp_testbed_20k import train as train_testbed


LATE_SUPERVISION_CONFIGS = {
    # Preserve an ordinary 1-16 loss on every batch and mix in a late-state loss.
    "random_aux": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 0.5,
    },
    # Same late-state distribution, using the historical burn-in replacement rule.
    "random_replace": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 1.0,
    },
    # Test whether denser coverage of late states reduces checkpoint volatility.
    "random_replace_p50": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.5,
        "late_supervision_mix": 1.0,
    },
    # Maximize recovery training without starving the ordinary input path:
    # every batch receives equal-weight ordinary and detached late-state losses.
    "random_aux_p100": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 1.0,
        "late_supervision_mix": 0.5,
    },
    # The opposite timing extreme: learn only from ordinary 1-16 trajectories
    # through half of the 20K schedule, then switch abruptly to the same p=1 arm.
    "random_aux_p100_after_10k": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 1.0,
        "late_supervision_mix": 0.5,
        "late_supervision_start_step": 10000,
    },
    # Combine randomized late states with the best compact loop schedule.
    "random_replace_aabb": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 1.0,
        "outer_state_norm": True,
        "unique_layers": 2,
        "layer_schedule": (0, 0, 1, 1),
    },
    # Extend detached training states to the horizon where the first run peaks.
    "random_replace_through_1024": {
        "late_supervision_horizons": (32, 64, 128, 256, 512, 1024),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 1.0,
    },
    # Let ordinary backprop establish useful features before adding late states.
    "random_aux_after_2k": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 0.5,
        "late_supervision_start_step": 2000,
    },
    "random_replace_after_2k": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 1.0,
        "late_supervision_start_step": 2000,
    },
    "random_replace_rmsnorm": {
        "late_supervision_horizons": (32, 64, 128, 256, 512),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 1.0,
        "outer_state_norm": True,
    },
    "fixed128_replace": {
        "late_supervision_horizons": (128,),
        "late_supervision_probability": 0.2,
        "late_supervision_mix": 1.0,
    },
}


def get_late_supervision_config(arm):
    try:
        return deepcopy(LATE_SUPERVISION_CONFIGS[arm])
    except KeyError as error:
        choices = ", ".join(sorted(LATE_SUPERVISION_CONFIGS))
        raise ValueError(f"unknown arm {arm!r}; choose one of: {choices}") from error


def train(
    output_dir=".",
    *,
    arm="random_replace",
    run_name=None,
    random_seed=20_260_720,
):
    if run_name is None:
        run_name = f"loop_late_{arm}_trial0"
    return train_testbed(
        output_dir=output_dir,
        experiment_name=f"exp_loop_late_{arm}",
        run_name=run_name,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        **get_late_supervision_config(arm),
    )


if __name__ == "__main__":
    train()
