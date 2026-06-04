"""LWD Training Pipeline (Algorithm 1).

Implements the full offline-to-online RL training loop:
- Stage 1: Offline pretraining on B_off
- Stage 2: Online post-training on B_off ∪ B_on with fleet data collection
"""

import os
import time
import yaml
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path

from lwd.model.distributional_value import DistributionalValueHead
from lwd.model.critic import CriticWithTarget
from lwd.model.action_head import FlowActionHead
from lwd.trainer.learner import Learner, LearnerConfig, LearnerOutput
from lwd.rl.replay_buffer import ReplayBuffer, ReplayBatch


@dataclass
class LWDConfig:
    """Full LWD training configuration."""
    # Model dimensions
    state_dim: int = 896
    action_dim: int = 14
    action_horizon: int = 30
    action_hidden_size: int = 896

    # DIVL hyperparameters
    num_atoms: int = 201
    v_min: float = -0.1
    v_max: float = 1.1
    gamma: float = 0.9999
    ema_rate: float = 0.005
    tau_base_offline: float = 0.6
    tau_base_online: float = 0.9
    alpha: float = 0.3
    tau_min: float = 0.4
    tau_max: float = 0.9

    # QAM hyperparameters
    qam_temperature: float = 2.0
    qam_num_quadrature: int = 4
    num_flow_steps: int = 10

    # Training hyperparameters
    policy_lr: float = 2e-5
    value_lr: float = 5e-4
    batch_size: int = 256
    offline_steps: int = 40000
    online_steps: int = 5000
    n_steps_offline_long: int = 10
    n_steps_offline_short: int = 1
    actor_sync_period: int = 50

    # Replay buffer
    max_offline_size: int = 1_000_000
    max_online_size: int = 100_000
    online_ratio: float = 0.5

    # Checkpoint
    save_path: str = "./checkpoints"
    save_interval: int = 1000
    log_interval: int = 10

    # Flow matching
    beta_alpha: float = 1.5
    beta_beta: float = 1.0
    flow_s: float = 0.999

    @classmethod
    def from_yaml(cls, path: str) -> "LWDConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class LWDTrainer:
    """LWD offline-to-online RL trainer.

    Orchestrates the full training pipeline:
    1. Initialize policy from SFT checkpoint
    2. Set reference policy f_β ← f_θ (frozen copy)
    3. Stage 1: Offline RL pretraining
    4. Stage 2: Online RL with fleet data
    """

    def __init__(self, config: LWDConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize models
        self._init_models()

        # Initialize optimizers
        self._init_optimizers()

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            max_offline_size=config.max_offline_size,
            max_online_size=config.max_online_size,
            online_ratio=config.online_ratio,
        )

        # Initialize learner
        learner_config = LearnerConfig(
            gamma=config.gamma,
            chunk_horizon=config.action_horizon,
            tau_base=config.tau_base_offline,
            alpha=config.alpha,
            tau_min=config.tau_min,
            tau_max=config.tau_max,
            qam_temperature=config.qam_temperature,
            qam_num_quadrature=config.qam_num_quadrature,
            num_flow_steps=config.num_flow_steps,
        )
        self.learner = Learner(self.value_head, self.critic, learner_config)
        self.global_step = 0

    def _init_models(self):
        """Initialize value head, critic, and action head."""
        cfg = self.config

        # Distributional value head V_ψ
        self.value_head = DistributionalValueHead(
            input_dim=cfg.state_dim,
            num_atoms=cfg.num_atoms,
            v_min=cfg.v_min,
            v_max=cfg.v_max,
        ).to(self.device)

        # Critic Q_ϕ with EMA target
        self.critic = CriticWithTarget(
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            action_horizon=cfg.action_horizon,
            ema_rate=cfg.ema_rate,
        ).to(self.device)

        # Flow action head (for reference and current policy)
        self.action_head = FlowActionHead(
            action_dim=cfg.action_dim,
            hidden_size=cfg.state_dim,
            action_hidden_size=cfg.action_hidden_size,
            beta_alpha=cfg.beta_alpha,
            beta_beta=cfg.beta_beta,
            s=cfg.flow_s,
        ).to(self.device)

        # Reference policy (frozen copy)
        self.reference_action_head = FlowActionHead(
            action_dim=cfg.action_dim,
            hidden_size=cfg.state_dim,
            action_hidden_size=cfg.action_hidden_size,
            beta_alpha=cfg.beta_alpha,
            beta_beta=cfg.beta_beta,
            s=cfg.flow_s,
        ).to(self.device)

    def _init_optimizers(self):
        """Initialize separate optimizers for policy and value/critic."""
        cfg = self.config

        # Policy optimizer (AdamW with cosine schedule)
        self.policy_optimizer = torch.optim.AdamW(
            self.action_head.parameters(), lr=cfg.policy_lr, weight_decay=0.1
        )

        # Value/Critic optimizer (Adam with cosine schedule)
        value_critic_params = list(self.value_head.parameters()) + list(self.critic.parameters())
        self.value_optimizer = torch.optim.Adam(
            self.value_head.parameters(), lr=cfg.value_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.value_lr
        )

    def freeze_reference_policy(self):
        """Freeze reference policy f_β ← f_θ."""
        self.reference_action_head.load_state_dict(self.action_head.state_dict())
        for param in self.reference_action_head.parameters():
            param.requires_grad = False
        print("[LWD] Reference policy frozen.")

    def _make_policy_flow_fn(self, state_repr: torch.Tensor):
        """Create a callable flow function for the current policy."""
        def flow_fn(noisy_action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            embed, _ = self.action_head.encode_noisy_action(noisy_action, time)
            # Simplified: directly project back (in full model, this goes through transformer)
            return self.action_head.predict_flow(
                embed.reshape(-1, embed.shape[-1])
            ).reshape(noisy_action.shape)
        return flow_fn

    def _make_reference_flow_fn(self, state_repr: torch.Tensor):
        """Create a callable flow function for the frozen reference policy."""
        def flow_fn(noisy_action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                embed, _ = self.reference_action_head.encode_noisy_action(noisy_action, time)
                return self.reference_action_head.predict_flow(
                    embed.reshape(-1, embed.shape[-1])
                ).reshape(noisy_action.shape)
        return flow_fn

    def _get_state_repr(self, states: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract state representation from state dict.

        In the full model, this would pass through the VLM backbone.
        Here we use a placeholder that concatenates available state features.
        Override this method when integrating with a real VLM.
        """
        if "state_repr" in states:
            return states["state_repr"]
        # Fallback: use proprioception as state representation
        if "proprioception" in states:
            propri = states["proprioception"]
            if propri.dim() == 3:
                propri = propri[:, -1, :]  # Take last timestep
            # Pad to state_dim
            if propri.shape[-1] < self.config.state_dim:
                pad = torch.zeros(
                    propri.shape[0], self.config.state_dim - propri.shape[-1],
                    device=propri.device
                )
                return torch.cat([propri, pad], dim=-1)
            return propri[:, :self.config.state_dim]
        raise ValueError("Cannot extract state representation from batch states")

    # =========================================================================
    # Stage 1: Offline RL Pretraining
    # =========================================================================

    def train_offline(self, num_steps: Optional[int] = None):
        """Stage 1: Offline RL pretraining on B_off.

        Args:
            num_steps: override number of training steps
        """
        steps = num_steps or self.config.offline_steps
        print(f"[LWD] Starting offline pretraining for {steps} steps...")

        # Set tau_base for offline stage
        self.learner.config.tau_base = self.config.tau_base_offline
        self.learner.config.use_nstep = True
        self.learner.config.n_steps = self.config.n_steps_offline_long

        for step in range(steps):
            batch = self.replay_buffer.sample_offline_only(
                self.config.batch_size, self.device
            )

            state_repr = self._get_state_repr(batch.states)
            next_state_repr = self._get_state_repr(batch.next_states)

            policy_flow_fn = self._make_policy_flow_fn(state_repr)
            reference_flow_fn = self._make_reference_flow_fn(state_repr)

            output = self.learner.step(
                batch=batch,
                state_repr=state_repr,
                next_state_repr=next_state_repr,
                policy_flow_fn=policy_flow_fn,
                reference_flow_fn=reference_flow_fn,
                value_optimizer=self.value_optimizer,
                critic_optimizer=self.critic_optimizer,
                policy_optimizer=self.policy_optimizer,
            )

            self.global_step += 1

            if step % self.config.log_interval == 0:
                self._log_step(output, stage="offline")

            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint(f"offline_step_{step}")

        print(f"[LWD] Offline pretraining complete. Global step: {self.global_step}")

    # =========================================================================
    # Stage 2: Online RL Post-Training
    # =========================================================================

    def train_online(self, num_steps: Optional[int] = None):
        """Stage 2: Online RL with mixed replay from B_off ∪ B_on.

        In the full system, this runs concurrently with actor fleet data collection.
        Here we implement the learner side; actors feed data via add_online_episode().

        Args:
            num_steps: override number of training steps
        """
        steps = num_steps or self.config.online_steps
        print(f"[LWD] Starting online post-training for {steps} steps...")

        # Set tau_base for online stage (more optimistic)
        self.learner.config.tau_base = self.config.tau_base_online
        self.learner.config.use_nstep = False
        self.learner.config.n_steps = 1

        for step in range(steps):
            batch = self.replay_buffer.sample(self.config.batch_size, self.device)

            state_repr = self._get_state_repr(batch.states)
            next_state_repr = self._get_state_repr(batch.next_states)

            policy_flow_fn = self._make_policy_flow_fn(state_repr)
            reference_flow_fn = self._make_reference_flow_fn(state_repr)

            output = self.learner.step(
                batch=batch,
                state_repr=state_repr,
                next_state_repr=next_state_repr,
                policy_flow_fn=policy_flow_fn,
                reference_flow_fn=reference_flow_fn,
                value_optimizer=self.value_optimizer,
                critic_optimizer=self.critic_optimizer,
                policy_optimizer=self.policy_optimizer,
            )

            self.global_step += 1

            if step % self.config.log_interval == 0:
                self._log_step(output, stage="online")

            if step % self.config.save_interval == 0 and step > 0:
                self.save_checkpoint(f"online_step_{step}")

            # Actor sync: in real system, publish new policy every N_sync steps
            if step % self.config.actor_sync_period == 0:
                self._publish_policy_checkpoint()

        print(f"[LWD] Online post-training complete. Global step: {self.global_step}")

    def _publish_policy_checkpoint(self):
        """Publish updated policy to robot fleet (placeholder).

        In the distributed system, this broadcasts the action_head state_dict
        to all actor processes.
        """
        pass  # Implemented in distributed/sync.py

    def _log_step(self, output: LearnerOutput, stage: str = "offline"):
        """Log training metrics."""
        print(
            f"[{stage}] step={self.global_step} | "
            f"v_loss={output.value_loss:.4f} | "
            f"c_loss={output.critic_loss:.4f} | "
            f"qam_loss={output.qam_loss:.4f} | "
            f"mean_q={output.mean_q:.4f} | "
            f"mean_tau={output.mean_tau:.4f}"
        )

    # =========================================================================
    # Checkpoint Management
    # =========================================================================

    def save_checkpoint(self, name: str):
        """Save all model components."""
        save_dir = Path(self.config.save_path) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.action_head.state_dict(), save_dir / "action_head.pt")
        torch.save(self.reference_action_head.state_dict(), save_dir / "reference_action_head.pt")
        torch.save(self.value_head.state_dict(), save_dir / "value_head.pt")
        torch.save(self.critic.state_dict(), save_dir / "critic.pt")
        torch.save(self.policy_optimizer.state_dict(), save_dir / "policy_optimizer.pt")
        torch.save(self.value_optimizer.state_dict(), save_dir / "value_optimizer.pt")
        torch.save(self.critic_optimizer.state_dict(), save_dir / "critic_optimizer.pt")
        torch.save({"global_step": self.global_step}, save_dir / "training_state.pt")

        print(f"[LWD] Checkpoint saved: {save_dir}")

    def load_checkpoint(self, path: str):
        """Load all model components from checkpoint."""
        load_dir = Path(path)

        self.action_head.load_state_dict(torch.load(load_dir / "action_head.pt", map_location=self.device))
        self.reference_action_head.load_state_dict(torch.load(load_dir / "reference_action_head.pt", map_location=self.device))
        self.value_head.load_state_dict(torch.load(load_dir / "value_head.pt", map_location=self.device))
        self.critic.load_state_dict(torch.load(load_dir / "critic.pt", map_location=self.device))
        self.policy_optimizer.load_state_dict(torch.load(load_dir / "policy_optimizer.pt", map_location=self.device))
        self.value_optimizer.load_state_dict(torch.load(load_dir / "value_optimizer.pt", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(load_dir / "critic_optimizer.pt", map_location=self.device))

        state = torch.load(load_dir / "training_state.pt", map_location=self.device)
        self.global_step = state["global_step"]

        print(f"[LWD] Checkpoint loaded from: {load_dir}, global_step={self.global_step}")
