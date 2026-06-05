"""Flow-matching based action head for VLA policy.

Implements the flow matching action generation used in π0.5 / wall-x:
- Training: sample time from Beta distribution, compute noisy interpolation, predict flow velocity
- Inference: Euler integration from noise to action over N steps
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

from torch.distributions import Beta


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion/flow timestep encoding."""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=torch.float32) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FlowActionHead(nn.Module):
    """Flow-matching action generation head.

    Architecture: noisy_action → Linear → (+ time_emb) → MLP → action_proj_back → predicted flow velocity
    The action head is designed to be attached to the end of a VLM transformer's action expert tokens.
    """

    def __init__(
        self,
        action_dim: int = 14,
        hidden_size: int = 896,
        action_hidden_size: int = 896,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        s: float = 0.999,
        use_adarms: bool = False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.action_hidden_size = action_hidden_size
        self.beta_alpha = beta_alpha
        self.beta_beta = beta_beta
        self.s = s
        self.use_adarms = use_adarms

        self.time_embed = SinusoidalPosEmb(action_hidden_size)

        self.w1 = nn.Linear(action_dim, action_hidden_size, bias=False)

        if use_adarms:
            self.time_mlp_in = nn.Linear(action_hidden_size, action_hidden_size)
            self.time_mlp_out = nn.Linear(action_hidden_size, action_hidden_size)
        else:
            self.w2 = nn.Linear(action_hidden_size * 2, action_hidden_size, bias=False)
            self.w3 = nn.Linear(action_hidden_size, action_hidden_size, bias=False)

        self.act_fn = nn.SiLU()
        self.action_proj_back = nn.Linear(action_hidden_size, action_dim, bias=False)
        self.mse_loss = nn.MSELoss(reduction="none")

    def _get_beta_dist(self, device: torch.device):
        alpha = torch.tensor(self.beta_alpha, dtype=torch.float32, device=device)
        beta = torch.tensor(self.beta_beta, dtype=torch.float32, device=device)
        return Beta(alpha, beta)

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample flow matching timestep from Beta distribution, scaled by s."""
        beta_dist = self._get_beta_dist(device)
        sample = beta_dist.sample([batch_size])
        return (1 - sample) * self.s

    def compute_training_targets(
        self, action_chunk: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute noisy action interpolation and flow target for training.

        Args:
            action_chunk: [B, T, action_dim] ground-truth action sequence (normalized to [-1, 1])

        Returns:
            noisy_action: [B, T, action_dim] interpolated noisy action at sampled time
            flow_target: [B, T, action_dim] target flow velocity (action - noise)
            time: [B] sampled timesteps
            noise: [B, T, action_dim] sampled Gaussian noise
        """
        batch_size = action_chunk.shape[0]
        device = action_chunk.device

        noise = torch.randn_like(action_chunk)
        time = self.sample_time(batch_size, device)
        time_expanded = time[:, None, None]  # [B, 1, 1]

        # Flow matching interpolation: x_t = (1-t)*noise + t*action
        noisy_action = (1 - time_expanded) * noise + time_expanded * action_chunk
        flow_target = action_chunk - noise

        return noisy_action, flow_target, time, noise

    def encode_noisy_action(
        self, noisy_action: torch.Tensor, time: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode noisy action + time into hidden representation for transformer input.

        Args:
            noisy_action: [B, T, action_dim]
            time: [B] timesteps

        Returns:
            action_embed: [B, T, action_hidden_size] — to be fed as action tokens into transformer
            adarms_cond: [B, action_hidden_size] or None — time conditioning for AdaRMS norm
        """
        time_embed = self.time_embed(time)  # [B, action_hidden_size]
        action_embed = self.w1(noisy_action)  # [B, T, action_hidden_size]

        if self.use_adarms:
            time_embed = self.act_fn(self.time_mlp_in(time_embed))
            time_embed = self.act_fn(self.time_mlp_out(time_embed))
            adarms_cond = time_embed
        else:
            time_expanded = time_embed.unsqueeze(1).expand(-1, action_embed.shape[1], -1)
            concat = torch.cat([action_embed, time_expanded], dim=-1)
            action_embed = self.w3(self.act_fn(self.w2(concat)))
            adarms_cond = None

        return action_embed, adarms_cond

    def predict_flow(self, action_hidden_states: torch.Tensor) -> torch.Tensor:
        """Project transformer output back to action-space flow velocity.

        Args:
            action_hidden_states: [N, action_hidden_size] — hidden states at action token positions

        Returns:
            flow_pred: [N, action_dim]
        """
        return self.action_proj_back(action_hidden_states[:, :self.action_hidden_size])

    def flow_loss(
        self,
        action_hidden_states: torch.Tensor,
        flow_target: torch.Tensor,
        dof_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute flow matching MSE loss.

        Args:
            action_hidden_states: [B*T, hidden_size] transformer outputs at action positions
            flow_target: [B*T, action_dim] target velocity
            dof_mask: [B*T, action_dim] optional mask for active degrees of freedom

        Returns:
            loss: scalar mean MSE loss
        """
        flow_pred = self.predict_flow(action_hidden_states)
        loss = self.mse_loss(flow_pred, flow_target)
        if dof_mask is not None:
            loss = loss * dof_mask
        return loss.mean()

    @torch.no_grad()
    def generate_action(
        self,
        state_embedding: torch.Tensor,
        forward_fn,
        pred_horizon: int = 30,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Generate action chunk via Euler integration of the learned flow.

        Args:
            state_embedding: [B, hidden_size] — state representation from VLM
            forward_fn: callable that takes (noisy_action_embed, time, adarms_cond)
                        and returns action_hidden_states at action positions
            pred_horizon: number of action timesteps in the chunk
            num_steps: number of Euler integration steps

        Returns:
            action_chunk: [B, pred_horizon, action_dim] generated actions in [-1, 1]
        """
        batch_size = state_embedding.shape[0]
        device = state_embedding.device

        # Start from Gaussian noise
        x_t = torch.randn(batch_size, pred_horizon, self.action_dim, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=torch.float32)

            action_embed, adarms_cond = self.encode_noisy_action(x_t, t)
            action_hidden = forward_fn(action_embed, t, adarms_cond)
            velocity = self.predict_flow(action_hidden.reshape(-1, action_hidden.shape[-1]))
            velocity = velocity.reshape(batch_size, pred_horizon, self.action_dim)

            x_t = x_t + velocity * dt

        return x_t.clamp(-1, 1)

    @torch.no_grad()
    def rollout_reference_flow(
        self,
        state_embedding: torch.Tensor,
        forward_fn,
        noise: torch.Tensor,
        num_steps: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Roll out the reference (frozen) flow from given noise to endpoint.

        Used by QAM to generate reference trajectories for adjoint matching.

        Args:
            state_embedding: [B, hidden_size]
            forward_fn: reference policy's forward function
            noise: [B, T, action_dim] starting noise a_0
            num_steps: integration steps

        Returns:
            endpoint: [B, T, action_dim] final generated action a_1
            trajectory: list of (x_t, t) at each step for adjoint computation
        """
        batch_size = noise.shape[0]
        pred_horizon = noise.shape[1]
        device = noise.device

        x_t = noise.clone()
        dt = 1.0 / num_steps
        trajectory = []

        for i in range(num_steps):
            t = torch.full((batch_size,), i * dt, device=device, dtype=torch.float32)
            trajectory.append((x_t.clone(), t.clone()))

            action_embed, adarms_cond = self.encode_noisy_action(x_t, t)
            action_hidden = forward_fn(action_embed, t, adarms_cond)
            velocity = self.predict_flow(action_hidden.reshape(-1, action_hidden.shape[-1]))
            velocity = velocity.reshape(batch_size, pred_horizon, self.action_dim)

            x_t = x_t + velocity * dt

        return x_t, trajectory
