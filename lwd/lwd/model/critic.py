"""Critic Network (Q_ϕ) for LWD.

Clipped double-Q critic that conditions on state representation and action chunk.
Uses temporal attention pooling to encode the action chunk before concatenation
with the state readout token.
"""

import copy
import torch
import torch.nn as nn
from typing import Tuple


class TemporalAttentionPooling(nn.Module):
    """Attention-based pooling over action chunk temporal dimension."""

    def __init__(self, action_dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.key_proj = nn.Linear(action_dim, hidden_dim)
        self.value_proj = nn.Linear(action_dim, hidden_dim)
        self.scale = hidden_dim ** -0.5

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Pool action chunk [B, T, action_dim] -> [B, hidden_dim]."""
        B = action_chunk.shape[0]
        keys = self.key_proj(action_chunk)  # [B, T, hidden_dim]
        values = self.value_proj(action_chunk)  # [B, T, hidden_dim]
        query = self.query.expand(B, -1, -1)  # [B, 1, hidden_dim]

        attn = (query @ keys.transpose(-1, -2)) * self.scale  # [B, 1, T]
        attn = attn.softmax(dim=-1)
        out = (attn @ values).squeeze(1)  # [B, hidden_dim]
        return out


class QHead(nn.Module):
    """Single scalar Q-value head."""

    def __init__(self, input_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # [B]


class DoubleCritic(nn.Module):
    """Clipped double-Q critic network.

    Architecture:
        state_repr (z_t from VLM readout token) + action_encoding (temporal attention pooling)
        → concat → two independent Q heads → min for TD targets

    The critic conditions on both state and action to predict Q(s, a).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 30,
        hidden_dim: int = 512,
        action_pool_dim: int = 256,
    ):
        super().__init__()
        self.action_pool = TemporalAttentionPooling(action_dim, action_pool_dim)
        combined_dim = state_dim + action_pool_dim
        self.q1 = QHead(combined_dim, hidden_dim)
        self.q2 = QHead(combined_dim, hidden_dim)

    def forward(
        self, state_repr: torch.Tensor, action_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-value estimates.

        Args:
            state_repr: [B, state_dim] readout token from VLM backbone
            action_chunk: [B, T, action_dim] action sequence

        Returns:
            q1, q2: [B] two independent Q-value estimates
        """
        action_encoding = self.action_pool(action_chunk)  # [B, action_pool_dim]
        combined = torch.cat([state_repr, action_encoding], dim=-1)
        return self.q1(combined), self.q2(combined)

    def q_min(
        self, state_repr: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Return the minimum of two Q estimates (for conservative targets)."""
        q1, q2 = self.forward(state_repr, action_chunk)
        return torch.min(q1, q2)


class CriticWithTarget(nn.Module):
    """Critic with EMA target network for stable TD learning."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_horizon: int = 30,
        hidden_dim: int = 512,
        action_pool_dim: int = 256,
        ema_rate: float = 0.995,
    ):
        super().__init__()
        self.critic = DoubleCritic(
            state_dim, action_dim, action_horizon, hidden_dim, action_pool_dim
        )
        self.target_critic = copy.deepcopy(self.critic)
        self.ema_rate = ema_rate

        # Freeze target parameters
        for param in self.target_critic.parameters():
            param.requires_grad = False

    def forward(
        self, state_repr: torch.Tensor, action_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward through online critic."""
        return self.critic(state_repr, action_chunk)

    def target_q_min(
        self, state_repr: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Get min Q from target network (for DIVL value targets)."""
        with torch.no_grad():
            return self.target_critic.q_min(state_repr, action_chunk)

    @torch.no_grad()
    def update_target(self):
        """EMA update: Q̄_ϕ ← ρ * Q̄_ϕ + (1 - ρ) * Q_ϕ"""
        for target_param, online_param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.mul_(self.ema_rate).add_(
                online_param.data, alpha=1.0 - self.ema_rate
            )

    def critic_loss(
        self,
        state_repr: torch.Tensor,
        action_chunk: torch.Tensor,
        td_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute MSE critic loss (Eq. 15).

        L_Q(ϕ) = E[(Q_ϕ(s,a) - y_Q)^2]

        Args:
            state_repr: [B, state_dim]
            action_chunk: [B, T, action_dim]
            td_target: [B] TD target values (detached)

        Returns:
            loss: scalar MSE loss (average of both Q heads)
        """
        q1, q2 = self.critic(state_repr, action_chunk)
        loss1 = ((q1 - td_target) ** 2).mean()
        loss2 = ((q2 - td_target) ** 2).mean()
        return (loss1 + loss2) / 2
