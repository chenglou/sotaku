"""Tensor-only inference weights with explicit, checksum-verified model settings."""

import json
import pickle
import warnings
from pathlib import Path

import torch

from checkpoint_utils import atomic_json_save
from runtime_utils import file_sha256

V1_SHA256 = "4f2ee45da4296fc2ce860dcf953c466df907d3878d39c8d4fa0afa9ba28df69e"
FIXED_SETTINGS = {
    "architecture": "sotaku-loop-v1", "d_model": 128, "n_heads": 4,
    "d_ff": 512, "grid_side": 9, "input_categories": 10, "output_categories": 9,
    "rope_base": 10.0, "dropout": 0.1,
}
DEFAULT_SETTINGS = {
    **FIXED_SETTINGS,
    "unique_layers": 4, "layer_schedule": [0, 1, 2, 3],
    "residual_scale": 1.0, "feedback_scale": 1.0, "training_iterations": 16,
    "outer_state_norm": False, "outer_state_norm_epsilon": 1e-6,
    "outer_state_rms_cap": None,
}


def model_settings(model):
    model = getattr(model, "_orig_mod", model)
    settings = dict(FIXED_SETTINGS)
    settings.update({
        "d_model": model.initial_encoder.out_features,
        "input_categories": model.initial_encoder.in_features,
        "output_categories": model.output_head.out_features,
        "n_heads": model.layers[0].n_heads,
        "d_ff": model.layers[0].linear1.out_features,
        "dropout": model.layers[0].attn_dropout_p,
    })
    if any(layer.attn_dropout_p != settings["dropout"] for layer in model.layers):
        raise ValueError("Different per-layer dropout settings are unsupported in the inference format")
    for name in DEFAULT_SETTINGS.keys() - FIXED_SETTINGS.keys():
        value = getattr(model, name)
        settings[name] = list(value) if isinstance(value, tuple) else value
    return settings


def settings_from_training_config(config):
    settings = dict(DEFAULT_SETTINGS)
    for name in ("d_model", "n_heads", "d_ff", "unique_layers", "layer_schedule",
                 "residual_scale", "feedback_scale", "training_iterations",
                 "outer_state_norm_epsilon", "outer_state_rms_cap"):
        if name in config:
            settings[name] = config[name]
    settings["unique_layers"] = config.get("unique_layers", config.get("n_layers", 4))
    settings["layer_schedule"] = list(config.get("layer_schedule", range(settings["unique_layers"])))
    normalization = config.get("outer_state_norm", "none")
    if normalization not in (True, False, "none", "rmsnorm_no_affine"):
        raise ValueError(f"Unknown recurrent normalization {normalization!r}")
    settings["outer_state_norm"] = normalization in (True, "rmsnorm_no_affine")
    return settings


def build_model(settings):
    from stabilize.exp_testbed_20k import SudokuTransformer
    if set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError(f"Model settings must contain exactly {sorted(DEFAULT_SETTINGS)}")
    for name, expected in FIXED_SETTINGS.items():
        if settings[name] != expected:
            raise ValueError(f"Unsupported model setting {name}={settings[name]!r}")
    if not isinstance(settings["outer_state_norm"], bool):
        raise ValueError("outer_state_norm must be a boolean")
    for name in ("residual_scale", "feedback_scale", "outer_state_norm_epsilon"):
        value = settings[name]
        if not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value)):
            raise ValueError(f"{name} must be finite")
    if settings["outer_state_norm_epsilon"] <= 0:
        raise ValueError("outer_state_norm_epsilon must be positive")
    cap = settings["outer_state_rms_cap"]
    if cap is not None and (not isinstance(cap, (int, float)) or not torch.isfinite(torch.tensor(cap))):
        raise ValueError("outer_state_rms_cap must be finite")
    return SudokuTransformer(**{
        name: settings[name] for name in DEFAULT_SETTINGS.keys() - FIXED_SETTINGS.keys()
    })


def write_model_manifest(weights_path, model, *, training=None, provenance=None):
    weights_path = Path(weights_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "sotaku-inference-weights",
        "weights": {
            "file": weights_path.name, "sha256": file_sha256(weights_path),
            "bytes": weights_path.stat().st_size,
        },
        "model": model_settings(model),
        "training": training,
        "provenance": provenance,
    }
    atomic_json_save(manifest, str(weights_path) + ".json")
    return manifest


def load_model(weights_path, *, manifest_path=None, legacy_exp=None, legacy_defaults=False,
               device="cpu"):
    weights_path = Path(weights_path)
    checksum = file_sha256(weights_path)
    if manifest_path and not Path(manifest_path).is_file():
        raise FileNotFoundError(manifest_path)
    manifest_path = Path(manifest_path) if manifest_path else Path(str(weights_path) + ".json")
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema_version") != 1 or manifest.get("artifact_type") != "sotaku-inference-weights":
            raise ValueError("Unsupported inference manifest")
        if manifest["weights"]["sha256"] != checksum or manifest["weights"]["bytes"] != weights_path.stat().st_size:
            raise ValueError("Inference weight checksum/size does not match its manifest")
        settings = manifest["model"]
    elif checksum == V1_SHA256:
        settings = dict(DEFAULT_SETTINGS)
        manifest = {"model": settings, "weights": {"sha256": checksum}, "legacy": "verified published v1"}
    elif legacy_defaults:
        if legacy_exp not in ("iters.exp_baseline_lr2e3", "stabilize.exp_testbed_20k"):
            raise ValueError("Legacy defaults are supported only for the plain reference model")
        warnings.warn("Using explicitly requested plain-model defaults for unlabelled legacy weights")
        settings = dict(DEFAULT_SETTINGS)
        manifest = {"model": settings, "weights": {"sha256": checksum}, "legacy": "explicit plain defaults"}
    else:
        raise ValueError(
            "Missing inference manifest (.pt.json). Export metadata from the trusted training "
            "checkpoint, or explicitly opt into --legacy-defaults for a known plain model."
        )
    model = build_model(settings).float()
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(
            "This is not a tensor-only inference file. Training checkpoints contain pickled RNG "
            "state; use the explicit trusted-checkpoint export command instead."
        ) from error
    if not isinstance(state, dict) or not state or any(not isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("Inference files must contain only a model state_dict, not a training bundle")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, manifest
