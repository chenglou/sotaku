"""Stay-solved and staged late-state training experiments."""

from copy import deepcopy

from stabilize.exp_testbed_20k import train as train_testbed


LATE_HORIZONS = (32, 64, 128, 256, 512)
RECHECK_GAPS = (16, 64, 256)
STAGED_START_STEP = 2000
STAGED_RAMP_STEPS = 4000
HORIZON_START_STEPS = (2000, 3000, 4000, 5000, 6000)
SCREEN_SUFFIX = "_healthy_screen"
FULL_50K_SUFFIX = "_50k"

BASE_REPLACEMENT = {
    "late_supervision_horizons": LATE_HORIZONS,
    "late_supervision_probability": 0.2,
    "late_supervision_mix": 1.0,
}

RAMPED_REPLACEMENT = {
    **BASE_REPLACEMENT,
    "late_supervision_start_step": STAGED_START_STEP,
    "late_supervision_probability_ramp_steps": STAGED_RAMP_STEPS,
}

CURRICULUM_REPLACEMENT = {
    **BASE_REPLACEMENT,
    "late_supervision_start_step": STAGED_START_STEP,
    "late_supervision_horizon_start_steps": HORIZON_START_STEPS,
}

CLEAN_REPLACEMENT = {
    **RAMPED_REPLACEMENT,
    "late_supervision_horizon_start_steps": HORIZON_START_STEPS,
}

RECHECK_SETTINGS = {
    "late_recheck_gaps": RECHECK_GAPS,
    "late_recheck_loss_weight": 0.5,
}

STAY_SOLVED_CONFIGS = {
    "control": BASE_REPLACEMENT,
    "ramp_after_2k": RAMPED_REPLACEMENT,
    "curriculum_after_2k": CURRICULUM_REPLACEMENT,
    "clean_curriculum": CLEAN_REPLACEMENT,
    "stay_recheck": {
        **BASE_REPLACEMENT,
        **RECHECK_SETTINGS,
    },
    "stay_consistency": {
        **BASE_REPLACEMENT,
        **RECHECK_SETTINGS,
        "late_consistency_weight": 0.1,
    },
    "stay_consistency_strong": {
        **BASE_REPLACEMENT,
        **RECHECK_SETTINGS,
        "late_consistency_weight": 0.5,
    },
    "clean_stay": {
        **CLEAN_REPLACEMENT,
        **RECHECK_SETTINGS,
        "late_consistency_weight": 0.1,
    },
    "clean_rmsnorm": {
        **CLEAN_REPLACEMENT,
        "outer_state_norm": True,
    },
}

SCREEN_SCHEDULE = {
    # Preserve the full recipe through the delayed start and its first
    # curriculum transition. Compress only the later phases.
    "warmup_steps": 560,
    "total_steps": 10000,
    "eval_every": 1000,
    "probe_every": 1000,
    "phases": (
        (0, 4000, 21, "Phase 1: Hard only (rating 21+)"),
        (4000, 6000, 6, "Phase 2: Medium+ (rating 6+)"),
        (6000, 8000, 1, "Phase 3: Easy+ (rating 1+)"),
        (8000, 10000, 0, "Phase 4: All (rating 0+)"),
    ),
}

FULL_50K_SCHEDULE = {
    "warmup_steps": 1400,
    "total_steps": 50000,
    "eval_every": 5000,
    "probe_every": 1000,
    "phases": (
        (0, 10000, 21, "Phase 1: Hard only (rating 21+)"),
        (10000, 20000, 6, "Phase 2: Medium+ (rating 6+)"),
        (20000, 30000, 1, "Phase 3: Easy+ (rating 1+)"),
        (30000, 50000, 0, "Phase 4: All (rating 0+)"),
    ),
}


def get_stay_solved_config(arm):
    try:
        return deepcopy(STAY_SOLVED_CONFIGS[arm])
    except KeyError as error:
        choices = ", ".join(sorted(STAY_SOLVED_CONFIGS))
        raise ValueError(f"unknown arm {arm!r}; choose one of: {choices}") from error


def train(
    output_dir=".",
    *,
    arm="control",
    run_name=None,
    random_seed=20_260_724,
    screen=False,
    full_50k=False,
    branch_checkpoint_path=None,
):
    if screen and full_50k:
        raise ValueError("screen and full_50k cannot both be enabled")
    if run_name is None:
        if full_50k:
            suffix = FULL_50K_SUFFIX
        else:
            suffix = SCREEN_SUFFIX if screen else ""
        run_name = f"loop_stay_{arm}{suffix}_trial0"
    train_settings = get_stay_solved_config(arm)
    if screen:
        train_settings["schedule"] = SCREEN_SCHEDULE
    elif full_50k:
        train_settings["schedule"] = FULL_50K_SCHEDULE
    if branch_checkpoint_path is not None:
        train_settings["branch_checkpoint_path"] = branch_checkpoint_path
    return train_testbed(
        output_dir=output_dir,
        experiment_name=f"exp_loop_stay_{arm}",
        run_name=run_name,
        random_seed=random_seed,
        checkpoint_on_probe=True,
        **train_settings,
    )


if __name__ == "__main__":
    train()
