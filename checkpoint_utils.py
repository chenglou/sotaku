import os
import re
import glob
import tempfile
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


def load_checkpoint(path, model, config):
    """Load checkpoint and verify config matches. Returns checkpoint dict."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    # Verify config matches
    saved_config = checkpoint.get('config', {})
    for key, value in config.items():
        saved_value = saved_config.get(key)
        if saved_value != value:
            raise ValueError(
                f"Config mismatch! {key}: saved={saved_value}, current={value}. "
                f"Use a fresh output_dir or delete old checkpoints."
            )

    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint


def load_branch_checkpoint(path, model, config, allowed_config_changes):
    """Load a checkpoint for a new run while validating every unchanged setting."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    saved_config = checkpoint.get('config', {})
    allowed_config_changes = set(allowed_config_changes)

    for key in set(saved_config) | set(config):
        if key in allowed_config_changes:
            continue
        saved_value = saved_config.get(key)
        current_value = config.get(key)
        if saved_value != current_value:
            raise ValueError(
                f"Branch config mismatch! {key}: saved={saved_value}, "
                f"current={current_value}."
            )

    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint
