"""Evaluate one arm's weights with the standard evaluator (frozen 25K benchmark, FP32)."""

import contextlib


@contextlib.contextmanager
def variant_loader(arm, **variant_settings):
    """Make `model_io.load_model` build the variant. A variant's manifest lists the baseline's
    settings, so the arm has to be given here."""
    import model_io
    from looping.attention_residuals.model import LoopVariantTransformer

    if arm == "baseline":
        yield
        return
    original = model_io.build_model

    def build_variant(settings):
        original(settings)  # validates the settings
        allowed = ("unique_layers", "layer_schedule", "residual_scale", "feedback_scale",
                   "training_iterations", "outer_state_norm", "outer_state_norm_epsilon",
                   "outer_state_rms_cap")
        chosen = {name: settings[name] for name in allowed}
        chosen["layer_schedule"] = tuple(chosen["layer_schedule"])
        return LoopVariantTransformer(arm, **variant_settings, **chosen)

    model_io.build_model = build_variant
    try:
        yield
    finally:
        model_io.build_model = original


def evaluate_arm(model_path, output_dir, *, arm, iterations=(16, 128, 1024, 4096),
                 device="cuda", benchmark_path="release/benchmark_25k.json", max_test=5000,
                 dataset=None, **variant_settings):
    from iters.eval_more_iters import evaluate

    with variant_loader(arm, **variant_settings):
        return evaluate(
            model_path, iter_counts=list(iterations), device=device, output_dir=output_dir,
            benchmark_path=benchmark_path, max_test=max_test, precision="fp32", batch_size=256,
            compiled=False, matmul_precision="highest", dataset=dataset,
        )
