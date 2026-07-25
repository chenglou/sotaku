"""Evaluate the recommended late-state checkpoint with delayed damping."""

import argparse
import re

from looping.eval_delayed_damping import evaluate


RECOMMENDED_POLICY = (
    {"name": "warm512_a025", "alpha": 0.25, "warmup_iterations": 512},
)
DEFAULT_HORIZONS = (16, 128, 512, 1024, 2048, 4096)


def make_damping_policy(alpha=0.25, warmup_iterations=512):
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    if not isinstance(warmup_iterations, int) or warmup_iterations < 0:
        raise ValueError("warmup_iterations must be a non-negative integer")
    alpha_label = f"{alpha:g}".replace(".", "")
    return {
        "name": f"warm{warmup_iterations}_a{alpha_label}",
        "alpha": float(alpha),
        "warmup_iterations": warmup_iterations,
    }


def evaluate_checkpoint(
    model_path,
    *,
    model_name="late_state_model",
    examples_per_bucket=5000,
    batch_size=250,
    horizons=DEFAULT_HORIZONS,
    output_dir=".",
    output_prefix="late_state_recommended_damped",
    alpha=0.25,
    warmup_iterations=512,
):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", model_name):
        raise ValueError(f"unsafe model name: {model_name!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", output_prefix):
        raise ValueError(f"unsafe output prefix: {output_prefix!r}")
    return evaluate(
        model_configs=({
            "name": model_name,
            "path": model_path,
            "model_kwargs": {},
        },),
        policies=(make_damping_policy(alpha, warmup_iterations),),
        examples_per_bucket=examples_per_bucket,
        batch_size=batch_size,
        horizons=horizons,
        output_dir=output_dir,
        output_prefix=output_prefix,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--model-name", default="late_state_model")
    parser.add_argument("--examples-per-bucket", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--output-prefix", default="late_state_recommended_damped")
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--warmup-iterations", type=int, default=512)
    arguments = parser.parse_args()
    evaluate_checkpoint(
        arguments.model_path,
        model_name=arguments.model_name,
        examples_per_bucket=arguments.examples_per_bucket,
        batch_size=arguments.batch_size,
        horizons=tuple(arguments.horizons),
        output_dir=arguments.output_dir,
        output_prefix=arguments.output_prefix,
        alpha=arguments.alpha,
        warmup_iterations=arguments.warmup_iterations,
    )
