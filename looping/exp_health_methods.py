"""Standalone training-time methods for healthy recurrent trajectories."""

from copy import deepcopy

from looping.exp_stay_solved import (
    BASE_REPLACEMENT,
    FULL_50K_SCHEDULE,
    LATE_HORIZONS,
    RECHECK_GAPS,
)
from stabilize.exp_testbed_20k import train as train_testbed


FUTURE_AUXILIARY = {
    "late_supervision_horizons": LATE_HORIZONS,
    "late_supervision_probability": 0.2,
    "late_supervision_mix": 1.0,
    "late_recheck_gaps": RECHECK_GAPS,
    "late_recheck_loss_weight": 0.0,
    "late_auxiliary_only": True,
}

MARGIN_SETTINGS = {
    "late_margin_floor_weight": 0.1,
    "late_margin_floor": 5.0,
}

MARGIN_CAP_SPECS = {
    # Maximum state = burn-in + first 16-step window + gap + recheck window.
    "margin_cap80": {
        "late_supervision_horizons": (32,),
        "late_recheck_gaps": (16,),
    },
    "margin_cap128": {
        "late_supervision_horizons": (32, 64),
        "late_recheck_gaps": (16, 32),
    },
    "margin_cap192": {
        "late_supervision_horizons": (32, 64, 128),
        "late_recheck_gaps": (16, 32),
    },
}

HEALTH_METHOD_CONFIGS = {
    "vanilla": {},
    "rmsnorm": {
        "outer_state_norm": True,
    },
    "late_state_ce": BASE_REPLACEMENT,
    "consistency_only": {
        **FUTURE_AUXILIARY,
        "late_consistency_weight": 0.1,
    },
    "margin_only": {
        **FUTURE_AUXILIARY,
        **MARGIN_SETTINGS,
    },
    **{
        arm: {
            **FUTURE_AUXILIARY,
            **cap_spec,
            **MARGIN_SETTINGS,
        }
        for arm, cap_spec in MARGIN_CAP_SPECS.items()
    },
}


def get_health_method_config(arm):
    try:
        return deepcopy(HEALTH_METHOD_CONFIGS[arm])
    except KeyError as error:
        choices = ", ".join(sorted(HEALTH_METHOD_CONFIGS))
        raise ValueError(
            f"unknown health-method arm {arm!r}; choose one of: {choices}"
        ) from error


def train(
    output_dir=".",
    *,
    arm="vanilla",
    run_name=None,
    random_seed=20_260_730,
):
    config = get_health_method_config(arm)
    return train_testbed(
        output_dir=output_dir,
        experiment_name=f"exp_health_{arm}",
        run_name=run_name,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        schedule=FULL_50K_SCHEDULE,
        **config,
    )
