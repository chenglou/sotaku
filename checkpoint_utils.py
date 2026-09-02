import os
import re
import glob
import tempfile
import json
import torch


def atomic_torch_save(data, path):
    """Save a torch artifact without exposing a partially-written final path.

    Writes to a temporary sibling file and renames only the complete snapshot into
    place. Use this for checkpoints that a concurrent reader may consume mid-write,
    e.g. an upload thread syncing output_dir to remote storage.
    """
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(fd)
    try:
        torch.save(data, temp_path)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def find_latest_checkpoint(output_dir, checkpoint_prefix):
    """Find the latest checkpoint file and return (path, step) or (None, 0)."""
    if not checkpoint_prefix:
        return None, 0

    patterns = [
        os.path.join(output_dir, f"{checkpoint_prefix}*.pt"),
        os.path.join(output_dir, f"{checkpoint_prefix}_step*.pt"),
    ]

    checkpoint_steps = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            match = re.search(r'step(\d+)\.pt$', path)
            if match:
                checkpoint_steps[path] = int(match.group(1))

    if not checkpoint_steps:
        return None, 0

    latest = max(checkpoint_steps, key=checkpoint_steps.get)
    return latest, checkpoint_steps[latest]


def atomic_json_save(data, path):
    """Publish a complete JSON file using a temporary sibling and rename."""
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".metadata.", suffix=".tmp", dir=output_dir)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def _comparable(value):
    if isinstance(value, (list, tuple)):
        return tuple(_comparable(item) for item in value)
    if isinstance(value, dict):
        return {key: _comparable(item) for key, item in value.items()}
    return value


def validate_config(saved, current, *, allowed_changes=(), legacy_defaults=None,
                    label="Config mismatch!"):
    """Reject changed, added, and removed settings unless explicitly permitted."""
    missing = object()
    defaults = legacy_defaults or {}
    for key in sorted(set(saved) | set(current)):
        if key in allowed_changes:
            continue
        default = defaults.get(key, missing)
        saved_value = saved.get(key, default)
        current_value = current.get(key, default)
        if _comparable(saved_value) != _comparable(current_value):
            saved_text = "<missing>" if saved_value is missing else repr(saved_value)
            current_text = "<missing>" if current_value is missing else repr(current_value)
            raise ValueError(
                f"{label} {key}: saved={saved_text}, current={current_text}. "
                "Use the saved settings or a separately named experiment."
            )


def load_checkpoint(path, model, config, *, legacy_defaults=None):
    """Load a trusted training checkpoint, including pickled optimizer/RNG state."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_config(checkpoint.get('config', {}), config, legacy_defaults=legacy_defaults)
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def load_branch_checkpoint(path, model, config, allowed_config_changes, *,
                           legacy_defaults=None, expected_step=None):
    """Load a trusted checkpoint for a branch, permitting only declared changes."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_config(
        checkpoint.get('config', {}), config,
        allowed_changes=set(allowed_config_changes), legacy_defaults=legacy_defaults,
        label="Branch config mismatch!",
    )
    if expected_step is not None and checkpoint.get('step') != expected_step:
        raise ValueError(
            f"Branch step mismatch: expected {expected_step}, saved {checkpoint.get('step')}"
        )
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
