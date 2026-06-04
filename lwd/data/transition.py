"""Chunked transition format for LWD replay buffers.

Each transition represents one chunk-level step: (s_t, a_t, r_t, s_{t+H}).
"""

import torch
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Dict, Any


class SourceType(IntEnum):
    """Data source type for replay transitions."""
    DEMONSTRATION = 0
    ROLLOUT_SUCCESS = 1
    ROLLOUT_FAIL = 2
    PLAY = 3
    ONLINE_POLICY = 4
    ONLINE_INTERVENTION = 5


@dataclass
class ChunkedTransition:
    """Single chunk-level transition (s_t, a_t, r_t, s_{t+H}).

    All tensors are stored without batch dimension.
    """
    # State at time t: dict of tensors (images, proprioception, language, etc.)
    state: Dict[str, torch.Tensor]

    # Action chunk: [H, action_dim] normalized to [-1, 1]
    action_chunk: torch.Tensor

    # Chunk reward: scalar, r_t = Σ_{i=0}^{H-1} γ^i * r_{t+i}
    reward: float

    # Next state at time t+H
    next_state: Dict[str, torch.Tensor]

    # Whether episode terminated within this chunk
    done: bool

    # Whether this chunk ended in success (for sparse reward)
    success: bool

    # Episode identifier
    episode_id: str = ""

    # Data source type
    source_type: SourceType = SourceType.ONLINE_POLICY

    # Dataset/robot name for normalization lookup
    dataset_name: str = "default"

    # DOF mask: [H, action_dim] indicating active degrees of freedom
    dof_mask: Optional[torch.Tensor] = None

    # Chunk index within episode
    chunk_index: int = 0


@dataclass
class Episode:
    """Complete episode containing a sequence of chunked transitions.

    Used for episode-level operations like reward annotation and n-step return computation.
    """
    transitions: list = field(default_factory=list)
    episode_id: str = ""
    source_type: SourceType = SourceType.ONLINE_POLICY
    dataset_name: str = "default"
    success: bool = False
    total_steps: int = 0

    def annotate_rewards(self, gamma: float = 0.9999, chunk_horizon: int = 30):
        """Assign sparse binary rewards to transitions.

        In LWD, r=1 only at successful episode termination, r=0 otherwise.
        The chunk reward is: r_t = Σ_{i=0}^{H-1} γ^i * r_{t+i}
        For sparse binary reward, only the final chunk in a successful episode gets r=1.
        """
        for i, t in enumerate(self.transitions):
            if i == len(self.transitions) - 1 and self.success:
                t.reward = 1.0
                t.done = True
                t.success = True
            elif i == len(self.transitions) - 1:
                t.reward = 0.0
                t.done = True
                t.success = False
            else:
                t.reward = 0.0
                t.done = False
                t.success = False

    def compute_nstep_returns(
        self, gamma: float = 0.9999, chunk_horizon: int = 30, n_steps: int = 10
    ) -> list:
        """Compute n-step chunk-level returns for offline cold-start.

        Returns list of (transition_index, rewards_array[n], next_state_at_n, dones_array[n])
        """
        results = []
        num_transitions = len(self.transitions)

        for i in range(num_transitions):
            n = min(n_steps, num_transitions - i)
            rewards = torch.zeros(n_steps)
            dones = torch.zeros(n_steps)

            for j in range(n):
                rewards[j] = self.transitions[i + j].reward
                dones[j] = float(self.transitions[i + j].done)
                if self.transitions[i + j].done:
                    break

            # Next state is at i + n (or terminal)
            next_idx = min(i + n, num_transitions - 1)
            next_state = self.transitions[next_idx].next_state

            results.append((i, rewards, next_state, dones))

        return results
