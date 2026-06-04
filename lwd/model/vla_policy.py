"""VLA Policy Wrapper for LWD.

Wraps a pretrained Vision-Language-Action model (e.g., π0.5 / wall-x) and exposes
the interfaces needed by the LWD training loop:
- State representation extraction (readout token z_t)
- Flow-based action generation
- Reference policy management (frozen f_β)
"""

import copy
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Callable
from pathlib import Path

from lwd.model.action_head import FlowActionHead
from lwd.model.normalizer import Normalizer


class VLAPolicy(nn.Module):
    """VLA Policy wrapper for LWD.

    In the full LWD system, the policy π_θ is a PaliGemma VLM (2B params)
    with a Gemma-300M action expert. For this reproduction, we provide:
    1. A state encoder (VLM backbone → readout token z_t)
    2. A flow-based action head (action expert)
    3. Normalization utilities

    The architecture separates the VLM backbone from the action expert so that
    during online training, the VLM can be frozen while only the action expert
    is updated (as described in Section IV-D of the paper).
    """

    def __init__(
        self,
        state_dim: int = 896,
        action_dim: int = 14,
        action_horizon: int = 30,
        action_hidden_size: int = 896,
        beta_alpha: float = 1.5,
        beta_beta: float = 1.0,
        flow_s: float = 0.999,
        use_adarms: bool = False,
        num_inference_steps: int = 10,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_inference_steps = num_inference_steps

        # Action head (flow-based generation)
        self.action_head = FlowActionHead(
            action_dim=action_dim,
            hidden_size=state_dim,
            action_hidden_size=action_hidden_size,
            beta_alpha=beta_alpha,
            beta_beta=beta_beta,
            s=flow_s,
            use_adarms=use_adarms,
        )

        # Placeholder VLM state encoder
        # In the full model, this is a PaliGemma VLM backbone
        # For reproduction, we use a simple MLP as placeholder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

        # Normalizers (set externally)
        self.normalizer_action: Optional[Normalizer] = None
        self.normalizer_propri: Optional[Normalizer] = None

    def set_normalizers(self, normalizer_action: Normalizer, normalizer_propri: Normalizer):
        """Set action and proprioception normalizers."""
        self.normalizer_action = normalizer_action
        self.normalizer_propri = normalizer_propri

    def encode_state(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract state representation z_t from observations.

        In the full model, this processes images + language + proprioception
        through the VLM backbone and returns the readout token hidden state.

        Args:
            state: dictionary containing observation tensors

        Returns:
            state_repr: [B, state_dim] readout token representation
        """
        if "state_repr" in state:
            return state["state_repr"]

        # Fallback: encode proprioception through the state encoder
        if "proprioception" in state:
            propri = state["proprioception"]
            if propri.dim() == 3:
                propri = propri[:, -1, :]
            if propri.shape[-1] < self.state_dim:
                pad = torch.zeros(
                    propri.shape[0], self.state_dim - propri.shape[-1],
                    device=propri.device, dtype=propri.dtype
                )
                propri = torch.cat([propri, pad], dim=-1)
            else:
                propri = propri[:, :self.state_dim]
            return self.state_encoder(propri)

        raise ValueError("State dict must contain 'state_repr' or 'proprioception'")

    def forward_flow(
        self,
        state_repr: torch.Tensor,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through action expert: predict flow velocity.

        In the full model, this would:
        1. Encode noisy action + time into action tokens
        2. Pass through transformer (action expert MoE branch)
        3. Project back to action space

        Args:
            state_repr: [B, state_dim] (unused in simplified version)
            noisy_action: [B, T, action_dim]
            time: [B] flow time

        Returns:
            velocity: [B, T, action_dim] predicted flow velocity
        """
        embed, _ = self.action_head.encode_noisy_action(noisy_action, time)
        velocity = self.action_head.predict_flow(embed.reshape(-1, embed.shape[-1]))
        return velocity.reshape(noisy_action.shape)

    def get_flow_fn(self) -> Callable:
        """Return a callable flow function for use in QAM/DIVL.

        Returns callable(noisy_action, time) -> velocity
        """
        def flow_fn(noisy_action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            return self.forward_flow(None, noisy_action, time)
        return flow_fn

    @torch.no_grad()
    def generate_action(
        self,
        state: Dict[str, torch.Tensor],
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate action chunk via Euler integration.

        Args:
            state: observation dictionary
            num_steps: override number of integration steps

        Returns:
            action_chunk: [B, action_horizon, action_dim] in [-1, 1]
        """
        state_repr = self.encode_state(state)
        batch_size = state_repr.shape[0]
        device = state_repr.device
        steps = num_steps or self.num_inference_steps

        # Start from Gaussian noise
        x_t = torch.randn(batch_size, self.action_horizon, self.action_dim, device=device)
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((batch_size,), i * dt, device=device)
            velocity = self.forward_flow(state_repr, x_t, t)
            x_t = x_t + velocity * dt

        return x_t.clamp(-1, 1)

    def compute_sft_loss(
        self,
        state: Dict[str, torch.Tensor],
        action_chunk: torch.Tensor,
        dof_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute supervised flow-matching loss for SFT pretraining.

        L_SFT = E[||f_θ(s, a_w, w) - (a_1 - a_0)||^2]

        Args:
            state: observation dictionary
            action_chunk: [B, T, action_dim] ground-truth actions (normalized)
            dof_mask: [B, T, action_dim] active DOF mask

        Returns:
            loss: scalar MSE flow matching loss
        """
        noisy_action, flow_target, time, _ = self.action_head.compute_training_targets(action_chunk)
        embed, _ = self.action_head.encode_noisy_action(noisy_action, time)
        action_hidden = embed.reshape(-1, embed.shape[-1])
        flow_target_flat = flow_target.reshape(-1, self.action_dim)

        dof_mask_flat = None
        if dof_mask is not None:
            dof_mask_flat = dof_mask.reshape(-1, self.action_dim)

        return self.action_head.flow_loss(action_hidden, flow_target_flat, dof_mask_flat)

    def freeze_vlm_backbone(self):
        """Freeze VLM backbone parameters (for online stage)."""
        for param in self.state_encoder.parameters():
            param.requires_grad = False
        print("[VLAPolicy] VLM backbone frozen.")

    def unfreeze_vlm_backbone(self):
        """Unfreeze VLM backbone parameters."""
        for param in self.state_encoder.parameters():
            param.requires_grad = True


class ReferencePolicy:
    """Frozen reference policy f_β for QAM.

    A deep copy of the policy at the start of RL training.
    All parameters are frozen — no gradients flow through f_β.
    """

    def __init__(self, policy: VLAPolicy):
        """Create a frozen reference from a trained policy.

        Args:
            policy: the SFT-trained policy to freeze as reference
        """
        self.action_head = copy.deepcopy(policy.action_head)
        for param in self.action_head.parameters():
            param.requires_grad = False
        self.action_head.eval()

    def get_flow_fn(self) -> Callable:
        """Return frozen flow function for QAM."""
        @torch.no_grad()
        def flow_fn(noisy_action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            embed, _ = self.action_head.encode_noisy_action(noisy_action, time)
            velocity = self.action_head.predict_flow(embed.reshape(-1, embed.shape[-1]))
            return velocity.reshape(noisy_action.shape)
        return flow_fn

    @torch.no_grad()
    def rollout_flow(
        self, noise: torch.Tensor, num_steps: int = 10
    ) -> torch.Tensor:
        """Roll out reference flow from noise to endpoint a_1.

        Args:
            noise: [B, T, action_dim] starting Gaussian noise
            num_steps: integration steps

        Returns:
            endpoint: [B, T, action_dim] generated endpoint
        """
        x_t = noise.clone()
        dt = 1.0 / num_steps
        flow_fn = self.get_flow_fn()

        for i in range(num_steps):
            t = torch.full((noise.shape[0],), i * dt, device=noise.device)
            velocity = flow_fn(x_t, t)
            x_t = x_t + velocity * dt

        return x_t
