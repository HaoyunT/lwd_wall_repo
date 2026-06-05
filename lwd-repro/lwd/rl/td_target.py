"""N-step chunk-level TD target computation utilities."""

import torch
from typing import Optional


def compute_chunk_reward(
    step_rewards: torch.Tensor,
    gamma: float = 0.9999,
    chunk_horizon: int = 30,
) -> torch.Tensor:
    """Compute discounted chunk reward from per-step rewards.

    r_t = Σ_{i=0}^{H-1} γ^i * r_{t+i}

    In LWD with sparse binary rewards, this simplifies to:
    r_t = γ^k if success happens at step k within the chunk, else 0.

    Args:
        step_rewards: [B, H] per-step rewards within a chunk
        gamma: discount factor
        chunk_horizon: H

    Returns:
        chunk_reward: [B] discounted chunk reward
    """
    device = step_rewards.device
    discounts = gamma ** torch.arange(chunk_horizon, device=device, dtype=torch.float32)
    return (step_rewards * discounts.unsqueeze(0)).sum(dim=-1)
