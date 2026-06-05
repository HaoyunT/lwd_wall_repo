"""Replay Buffer for LWD offline-to-online RL.

Manages separate offline and online buffers with configurable mixed sampling.
"""

import torch
import random
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from lwd.data.transition import ChunkedTransition, SourceType, Episode


@dataclass
class ReplayBatch:
    """Collated batch from replay buffer."""
    states: Dict[str, torch.Tensor]       # {key: [B, ...]}
    action_chunks: torch.Tensor           # [B, H, action_dim]
    rewards: torch.Tensor                 # [B]
    next_states: Dict[str, torch.Tensor]  # {key: [B, ...]}
    dones: torch.Tensor                   # [B]
    dof_masks: Optional[torch.Tensor]     # [B, H, action_dim] or None
    dataset_names: List[str]              # [B]
    source_types: torch.Tensor            # [B] int

    # For n-step returns (offline stage)
    nstep_rewards: Optional[torch.Tensor] = None  # [B, n]
    nstep_dones: Optional[torch.Tensor] = None    # [B, n]
    nstep_next_states: Optional[Dict[str, torch.Tensor]] = None


class ReplayBuffer:
    """Mixed offline-online replay buffer for LWD.

    Maintains separate pools for offline and online data.
    Sampling draws from both with a configurable ratio.
    """

    def __init__(
        self,
        max_offline_size: int = 1_000_000,
        max_online_size: int = 100_000,
        online_ratio: float = 0.5,
    ):
        """
        Args:
            max_offline_size: maximum transitions in offline buffer
            max_online_size: maximum transitions in online buffer
            online_ratio: fraction of each batch drawn from online buffer
                         (0.5 = balanced 1:1 ratio as in paper)
        """
        self.offline_buffer: List[ChunkedTransition] = []
        self.online_buffer: List[ChunkedTransition] = []
        self.max_offline_size = max_offline_size
        self.max_online_size = max_online_size
        self.online_ratio = online_ratio

    @property
    def offline_size(self) -> int:
        return len(self.offline_buffer)

    @property
    def online_size(self) -> int:
        return len(self.online_buffer)

    @property
    def total_size(self) -> int:
        return self.offline_size + self.online_size

    def add_offline_episode(self, episode: Episode):
        """Add a complete episode to the offline buffer."""
        for t in episode.transitions:
            if len(self.offline_buffer) >= self.max_offline_size:
                # FIFO eviction
                self.offline_buffer.pop(0)
            self.offline_buffer.append(t)

    def add_online_transition(self, transition: ChunkedTransition):
        """Add a single transition to the online buffer."""
        if len(self.online_buffer) >= self.max_online_size:
            self.online_buffer.pop(0)
        self.online_buffer.append(transition)

    def add_online_episode(self, episode: Episode):
        """Add a complete episode to the online buffer."""
        for t in episode.transitions:
            self.add_online_transition(t)

    def sample(self, batch_size: int, device: torch.device = None) -> ReplayBatch:
        """Sample a mixed batch from offline and online buffers.

        Uses the configured online_ratio. Falls back to offline-only
        if online buffer is empty.
        """
        if self.online_size == 0:
            transitions = random.sample(
                self.offline_buffer, min(batch_size, self.offline_size)
            )
        elif self.offline_size == 0:
            transitions = random.sample(
                self.online_buffer, min(batch_size, self.online_size)
            )
        else:
            n_online = int(batch_size * self.online_ratio)
            n_offline = batch_size - n_online

            n_online = min(n_online, self.online_size)
            n_offline = min(n_offline, self.offline_size)

            online_samples = random.sample(self.online_buffer, n_online)
            offline_samples = random.sample(self.offline_buffer, n_offline)
            transitions = offline_samples + online_samples
            random.shuffle(transitions)

        return self._collate(transitions, device)

    def sample_offline_only(self, batch_size: int, device: torch.device = None) -> ReplayBatch:
        """Sample from offline buffer only (for Stage 1)."""
        transitions = random.sample(
            self.offline_buffer, min(batch_size, self.offline_size)
        )
        return self._collate(transitions, device)

    def _collate(
        self, transitions: List[ChunkedTransition], device: torch.device = None
    ) -> ReplayBatch:
        """Collate list of transitions into a batched ReplayBatch."""
        if device is None:
            device = torch.device("cpu")

        # Collate state dicts
        state_keys = transitions[0].state.keys()
        states = {
            k: torch.stack([t.state[k] for t in transitions]).to(device)
            for k in state_keys
        }
        next_states = {
            k: torch.stack([t.next_state[k] for t in transitions]).to(device)
            for k in state_keys
        }

        action_chunks = torch.stack([t.action_chunk for t in transitions]).to(device)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
        dones = torch.tensor([float(t.done) for t in transitions], dtype=torch.float32, device=device)
        source_types = torch.tensor([int(t.source_type) for t in transitions], dtype=torch.long, device=device)
        dataset_names = [t.dataset_name for t in transitions]

        # DOF masks
        if transitions[0].dof_mask is not None:
            dof_masks = torch.stack([t.dof_mask for t in transitions]).to(device)
        else:
            dof_masks = None

        return ReplayBatch(
            states=states,
            action_chunks=action_chunks,
            rewards=rewards,
            next_states=next_states,
            dones=dones,
            dof_masks=dof_masks,
            dataset_names=dataset_names,
            source_types=source_types,
        )
