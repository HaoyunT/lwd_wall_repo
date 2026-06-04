"""LEARNER: Single update step combining DIVL value learning and QAM policy extraction.

Implements Algorithm 2 from the LWD paper.
"""

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional, Tuple
from dataclasses import dataclass

from lwd.model.distributional_value import DistributionalValueHead
from lwd.model.critic import CriticWithTarget
from lwd.rl.divl import divl_value_loss, compute_td_target, compute_nstep_td_target, compute_adaptive_tau
from lwd.rl.qam import compute_action_gradient, compute_qam_loss_simplified
from lwd.rl.replay_buffer import ReplayBatch


@dataclass
class LearnerConfig:
    """Configuration for the LEARNER update step."""
    gamma: float = 0.9999
    chunk_horizon: int = 30
    tau_base: float = 0.6
    alpha: float = 0.3
    tau_min: float = 0.4
    tau_max: float = 0.9
    qam_temperature: float = 2.0
    qam_num_quadrature: int = 4
    num_flow_steps: int = 10
    n_steps: int = 1
    use_nstep: bool = False


@dataclass
class LearnerOutput:
    """Outputs from a single learner step."""
    value_loss: float
    critic_loss: float
    qam_loss: float
    total_loss: float
    mean_q: float
    mean_td_target: float
    mean_tau: float


class Learner(nn.Module):
    """Single LEARNER update step (Algorithm 2).

    Combines:
    1. DIVL: update V_ψ (distributional value loss) and Q_ϕ (critic loss)
    2. QAM: update π_θ (policy extraction via adjoint matching)
    """

    def __init__(
        self,
        value_head: DistributionalValueHead,
        critic: CriticWithTarget,
        config: LearnerConfig,
    ):
        super().__init__()
        self.value_head = value_head
        self.critic = critic
        self.config = config

    def step(
        self,
        batch: ReplayBatch,
        state_repr: torch.Tensor,
        next_state_repr: torch.Tensor,
        policy_flow_fn: Callable,
        reference_flow_fn: Callable,
        value_optimizer: torch.optim.Optimizer,
        critic_optimizer: torch.optim.Optimizer,
        policy_optimizer: torch.optim.Optimizer,
    ) -> LearnerOutput:
        """Execute one complete LEARNER update.

        Args:
            batch: sampled replay batch
            state_repr: [B, dim] state representations for s_t (from VLM backbone)
            next_state_repr: [B, dim] state representations for s_{t+H}
            policy_flow_fn: f_θ(a_w, w) -> velocity (current policy)
            reference_flow_fn: f_β(a_w, w) -> velocity (frozen reference)
            value_optimizer: optimizer for V_ψ parameters
            critic_optimizer: optimizer for Q_ϕ parameters
            policy_optimizer: optimizer for π_θ parameters

        Returns:
            LearnerOutput with loss metrics
        """
        cfg = self.config

        # ============================================
        # Part 1: DIVL - Value Learning
        # ============================================

        # 1a. Update V_ψ: distributional value loss (Eq. 12)
        v_loss = divl_value_loss(
            self.value_head, self.critic, state_repr, batch.action_chunks
        )
        value_optimizer.zero_grad()
        v_loss.backward()
        value_optimizer.step()

        # 1b. Compute TD target y_Q (Eq. 14 or Eq. 19 for n-step)
        with torch.no_grad():
            tau = compute_adaptive_tau(
                self.value_head, next_state_repr,
                cfg.tau_base, cfg.alpha, cfg.tau_min, cfg.tau_max,
            )

            if cfg.use_nstep and batch.nstep_rewards is not None:
                td_target = compute_nstep_td_target(
                    rewards=batch.nstep_rewards,
                    next_state_repr=next_state_repr,
                    value_head=self.value_head,
                    gamma=cfg.gamma,
                    chunk_horizon=cfg.chunk_horizon,
                    n_steps=cfg.n_steps,
                    tau=tau,
                    dones=batch.nstep_dones,
                )
            else:
                td_target = compute_td_target(
                    reward=batch.rewards,
                    next_state_repr=next_state_repr,
                    value_head=self.value_head,
                    gamma=cfg.gamma,
                    chunk_horizon=cfg.chunk_horizon,
                    tau=tau,
                    done=batch.dones,
                )

        # 1c. Update Q_ϕ: critic loss (Eq. 15)
        c_loss = self.critic.critic_loss(state_repr, batch.action_chunks, td_target)
        critic_optimizer.zero_grad()
        c_loss.backward()
        critic_optimizer.step()

        # 1d. EMA update: Q̄_ϕ ← ρ * Q̄_ϕ + (1 - ρ) * Q_ϕ
        self.critic.update_target()

        # ============================================
        # Part 2: QAM - Policy Extraction
        # ============================================

        # 2a. Sample Gaussian noise a_0
        noise = torch.randn_like(batch.action_chunks)

        # 2b. Roll out reference flow to get endpoint a_1
        with torch.no_grad():
            x_t = noise.clone()
            dt = 1.0 / cfg.num_flow_steps
            for i in range(cfg.num_flow_steps):
                t = torch.full(
                    (noise.shape[0],), i * dt,
                    device=noise.device, dtype=torch.float32
                )
                velocity = reference_flow_fn(x_t, t)
                x_t = x_t + velocity * dt
            endpoint = x_t

        # 2c. Compute action gradient ∇_a Q_ϕ(s, a_1) / λ
        action_grad = compute_action_gradient(
            lambda s, a: self.critic.critic.q_min(s, a),
            state_repr.detach(),
            endpoint,
            cfg.qam_temperature,
        )

        # 2d. Compute QAM loss and update policy
        qam_loss = compute_qam_loss_simplified(
            policy_flow_fn=policy_flow_fn,
            reference_flow_fn=reference_flow_fn,
            action_grad=action_grad,
            noise=noise,
            endpoint=endpoint,
            num_quadrature_points=cfg.qam_num_quadrature,
        )

        policy_optimizer.zero_grad()
        qam_loss.backward()
        policy_optimizer.step()

        # ============================================
        # Metrics
        # ============================================
        with torch.no_grad():
            mean_q = self.critic.critic.q_min(state_repr, batch.action_chunks).mean().item()

        return LearnerOutput(
            value_loss=v_loss.item(),
            critic_loss=c_loss.item(),
            qam_loss=qam_loss.item(),
            total_loss=v_loss.item() + c_loss.item() + qam_loss.item(),
            mean_q=mean_q,
            mean_td_target=td_target.mean().item(),
            mean_tau=tau.mean().item(),
        )
