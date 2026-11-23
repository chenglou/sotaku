"""Train one arm with the 20K development recipe (later-iteration training)."""

import contextlib
import functools

ARMS = ("baseline", "attnres", "message", "attnres_message")


@contextlib.contextmanager
def model_variant(arm, **variant_settings):
    """Make the shared trainer build a variant model. The trainer file itself is not edited,
    because archived studies pin it by checksum."""
    import stabilize.exp_testbed_20k as trainer
    from looping.attention_residuals.model import LoopVariantTransformer

    if arm == "baseline":
        yield
        return
    original = trainer.SudokuTransformer
    trainer.SudokuTransformer = functools.partial(LoopVariantTransformer, arm, **variant_settings)
    try:
        yield
    finally:
        trainer.SudokuTransformer = original


def run_name(arm, seed, tag=""):
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose one of {ARMS}")
    return f"attnres20k_{arm}{'_' + tag if tag else ''}_seed{seed}"


def train_arm(output_dir, *, arm, random_seed, tag="", **variant_settings):
    from looping.exp_stay_solved import train as train_development

    with model_variant(arm, **variant_settings):
        return train_development(
            output_dir=output_dir, arm="late_state_ce", run_name=run_name(arm, random_seed, tag),
            random_seed=random_seed,
        )
