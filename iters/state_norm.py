"""Parameter-free recurrent-state normalization helpers."""

import torch


DEFAULT_EPSILON = 1e-6


def per_token_rms(hidden_state):
    """Return the FP32 RMS over each token's feature dimension."""
    return hidden_state.float().square().mean(dim=-1).sqrt()


def rms_normalize(hidden_state, epsilon=DEFAULT_EPSILON):
    """Set every token's feature RMS to approximately one without centering it."""
    squared_mean = hidden_state.float().square().mean(dim=-1, keepdim=True)
    inverse_rms = torch.rsqrt(squared_mean + epsilon).to(hidden_state.dtype)
    return hidden_state * inverse_rms


def cap_token_rms(hidden_state, maximum_rms, epsilon=DEFAULT_EPSILON):
    """Rescale only tokens above maximum_rms, preserving their directions."""
    if maximum_rms <= 0:
        raise ValueError("maximum_rms must be positive")
    token_rms = per_token_rms(hidden_state).unsqueeze(-1)
    scale = (maximum_rms / token_rms.clamp_min(epsilon)).clamp(max=1.0)
    return hidden_state * scale.to(hidden_state.dtype)
