"""Distributional Value Head (V_ψ) for DIVL.

Categorical distribution over K atoms representing the state-conditioned distribution
of replay action-values. Used to extract τ-quantile bootstrap targets for the critic.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DistributionalValueHead(nn.Module):
    """Categorical distributional value function V_ψ(s).

    Outputs a probability distribution over K atoms spanning [v_min, v_max].
    The distribution represents P(v = Q(s,a) | a ~ D(·|s)) — the distribution
    of critic values for replay actions at state s.
    """

    def __init__(
        self,
        input_dim: int,
        num_atoms: int = 201,
        v_min: float = -0.1,
        v_max: float = 1.1,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Fixed atom support
        self.register_buffer(
            "support", torch.linspace(v_min, v_max, num_atoms)
        )
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Value prediction head: z_t -> distribution logits
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_atoms),
        )

    def forward(self, state_repr: torch.Tensor) -> torch.Tensor:
        """Compute value distribution logits.

        Args:
            state_repr: [B, input_dim] readout token representation z_t

        Returns:
            logits: [B, K] unnormalized log-probabilities over atoms
        """
        return self.head(state_repr)

    def get_probs(self, state_repr: torch.Tensor) -> torch.Tensor:
        """Get normalized probability distribution."""
        logits = self.forward(state_repr)
        return F.softmax(logits, dim=-1)

    def get_quantile(self, state_repr: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Extract τ-quantile from the learned distribution.

        Quant_τ(V_ψ(s)) = inf{v : F_ψ(v|s) >= τ}

        Args:
            state_repr: [B, input_dim]
            tau: [B] or scalar — quantile level per sample

        Returns:
            quantile_values: [B] scalar quantile value for each state
        """
        probs = self.get_probs(state_repr)  # [B, K]
        cdf = torch.cumsum(probs, dim=-1)  # [B, K]

        if tau.dim() == 0:
            tau = tau.expand(probs.shape[0])

        # Find first atom where CDF >= tau
        tau_expanded = tau.unsqueeze(-1)  # [B, 1]
        mask = (cdf >= tau_expanded).float()  # [B, K]

        # Use argmax on mask to find first index where CDF exceeds tau
        # If no atom exceeds tau (shouldn't happen), fall back to last atom
        indices = mask.argmax(dim=-1)  # [B]

        return self.support[indices]

    def get_normalized_entropy(self, state_repr: torch.Tensor) -> torch.Tensor:
        """Compute normalized entropy of the predicted distribution.

        H(s) = -1/log(K) * Σ p_ψ(i|s) * log(p_ψ(i|s))

        Args:
            state_repr: [B, input_dim]

        Returns:
            entropy: [B] normalized entropy in [0, 1]
        """
        probs = self.get_probs(state_repr)  # [B, K]
        log_probs = torch.log(probs + 1e-8)
        entropy = -(probs * log_probs).sum(dim=-1)
        max_entropy = math.log(self.num_atoms)
        return entropy / max_entropy

    def compute_loss(
        self,
        state_repr: torch.Tensor,
        target_q_values: torch.Tensor,
    ) -> torch.Tensor:
        """Compute distributional value loss via C51 projection.

        L_V(ψ) = -E[ log p_ψ(Q̄(s,a) | s) ]

        Projects scalar Q̄_ϕ(s,a) targets onto the categorical support
        using the C51 linear interpolation scheme.

        Args:
            state_repr: [B, input_dim] state representations
            target_q_values: [B] scalar target Q-values from EMA critic

        Returns:
            loss: scalar cross-entropy loss
        """
        logits = self.forward(state_repr)  # [B, K]
        target_dist = self._project_to_support(target_q_values)  # [B, K]
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -(target_dist * log_probs).sum(dim=-1).mean()
        return loss

    def _project_to_support(self, values: torch.Tensor) -> torch.Tensor:
        """C51 projection: project scalar values onto categorical support.

        Clips values to [v_min, v_max] and distributes probability mass
        linearly between the two neighboring atoms.

        Args:
            values: [B] scalar values to project

        Returns:
            target_dist: [B, K] target probability distribution
        """
        values = values.clamp(self.v_min, self.v_max)
        b = (values - self.v_min) / self.delta_z  # [B]
        lower = b.floor().long()  # [B]
        upper = lower + 1

        # Handle edge case where value == v_max
        upper = upper.clamp(max=self.num_atoms - 1)
        lower = lower.clamp(min=0, max=self.num_atoms - 1)

        # Linear interpolation weights
        upper_weight = b - lower.float()  # [B]
        lower_weight = 1.0 - upper_weight

        target_dist = torch.zeros(
            values.shape[0], self.num_atoms, device=values.device
        )
        target_dist.scatter_add_(1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target_dist.scatter_add_(1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))

        return target_dist


