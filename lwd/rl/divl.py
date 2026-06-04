"""DIVL: Distributional Implicit Value Learning.

Core value-learning component of LWD. Combines:
1. Distributional value model V_ψ trained via C51 cross-entropy on Q̄(s,a) targets
2. τ-quantile extraction for TD bootstrap targets
3. Adaptive τ schedule based on distributional entropy
4. N-step chunk-level TD target computation
"""

import torch
from typing import Optional

from lwd.model.distributional_value import DistributionalValueHead
from lwd.model.critic import CriticWithTarget


def compute_adaptive_tau(
    value_head: DistributionalValueHead,
    next_state_repr: torch.Tensor,
    tau_base: float = 0.6,
    alpha: float = 0.3,
    tau_min: float = 0.4,
    tau_max: float = 0.9,
) -> torch.Tensor:
    """Compute adaptive τ per sample based on value distribution entropy.

    τ(s_{t+H}) = clip(τ_base - α * H(s_{t+H}), τ_min, τ_max)

    Diffuse (uncertain) distributions get lower τ (more conservative).
    Concentrated (confident) distributions retain higher τ (more optimistic).

    Args:
        value_head: distributional value model V_ψ
        next_state_repr: [B, dim] state representations for s_{t+H}
        tau_base: base quantile level for confident states
        alpha: entropy sensitivity coefficient
        tau_min, tau_max: clipping bounds

    Returns:
        tau: [B] adaptive quantile levels (stop-gradient)
    """
    with torch.no_grad():
        entropy = value_head.get_normalized_entropy(next_state_repr)  # [B] in [0, 1]
        tau = tau_base - alpha * entropy
        tau = tau.clamp(tau_min, tau_max)
    return tau


def compute_td_target(
    reward: torch.Tensor,
    next_state_repr: torch.Tensor,
    value_head: DistributionalValueHead,
    gamma: float = 0.9999,
    chunk_horizon: int = 30,
    tau: Optional[torch.Tensor] = None,
    tau_base: float = 0.6,
    alpha: float = 0.3,
    done: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute 1-step chunk-level TD target (Eq. 14).

    y_Q = r_t + γ^H * Quant_τ(V_ψ(s_{t+H}))

    Args:
        reward: [B] chunk reward r_t
        next_state_repr: [B, dim] state representation for s_{t+H}
        value_head: distributional value head V_ψ
        gamma: discount factor
        chunk_horizon: H (action chunk length)
        tau: [B] pre-computed adaptive tau (if None, compute it)
        done: [B] terminal flag (1 if episode ended within chunk)

    Returns:
        td_target: [B] scalar TD targets (detached)
    """
    with torch.no_grad():
        if tau is None:
            tau = compute_adaptive_tau(value_head, next_state_repr, tau_base, alpha)

        bootstrap = value_head.get_quantile(next_state_repr, tau)  # [B]
        discount = gamma ** chunk_horizon
        td_target = reward + discount * bootstrap

        if done is not None:
            td_target = torch.where(done.bool(), reward, td_target)

    return td_target


def compute_nstep_td_target(
    rewards: torch.Tensor,
    next_state_repr: torch.Tensor,
    value_head: DistributionalValueHead,
    gamma: float = 0.9999,
    chunk_horizon: int = 30,
    n_steps: int = 10,
    tau: Optional[torch.Tensor] = None,
    tau_base: float = 0.6,
    alpha: float = 0.3,
    dones: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute n-step chunk-level TD target for offline cold-start (Eq. 19).

    y_Q = Σ_{i=0}^{n-1} γ^{iH} * r_{t+iH} + γ^{nH} * Quant_τ(V_ψ(s_{t+nH}))

    Used in offline stage for long-horizon tasks to accelerate reward propagation.

    Args:
        rewards: [B, n] chunk rewards for n consecutive chunks
        next_state_repr: [B, dim] state at s_{t+nH}
        value_head: distributional value head
        gamma: discount
        chunk_horizon: H
        n_steps: number of chunk-level steps
        tau: [B] adaptive tau
        dones: [B, n] terminal flags per chunk

    Returns:
        td_target: [B]
    """
    with torch.no_grad():
        if tau is None:
            tau = compute_adaptive_tau(value_head, next_state_repr, tau_base, alpha)

        batch_size = rewards.shape[0]
        device = rewards.device

        # Accumulate discounted n-step return
        n_step_return = torch.zeros(batch_size, device=device)
        still_alive = torch.ones(batch_size, device=device)

        for i in range(n_steps):
            discount_i = gamma ** (i * chunk_horizon)
            n_step_return += still_alive * discount_i * rewards[:, i]
            if dones is not None:
                still_alive = still_alive * (1.0 - dones[:, i].float())

        # Bootstrap from V_ψ at s_{t+nH}
        bootstrap_discount = gamma ** (n_steps * chunk_horizon)
        bootstrap = value_head.get_quantile(next_state_repr, tau)
        td_target = n_step_return + still_alive * bootstrap_discount * bootstrap

    return td_target


def divl_value_loss(
    value_head: DistributionalValueHead,
    critic: CriticWithTarget,
    state_repr: torch.Tensor,
    action_chunk: torch.Tensor,
) -> torch.Tensor:
    """Compute DIVL distributional value loss (Eq. 12).

    L_V(ψ) = E[-log p_ψ(Q̄_ϕ(s,a) | s)]

    Trains V_ψ to model the distribution of target Q-values for replay actions.

    Args:
        value_head: distributional value model V_ψ
        critic: critic with target network
        state_repr: [B, dim] state representations
        action_chunk: [B, T, action_dim] replay actions

    Returns:
        loss: scalar distributional cross-entropy loss
    """
    with torch.no_grad():
        target_q = critic.target_q_min(state_repr, action_chunk)  # [B]

    return value_head.compute_loss(state_repr, target_q)
