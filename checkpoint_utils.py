import os
import re
import glob
import torch


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
