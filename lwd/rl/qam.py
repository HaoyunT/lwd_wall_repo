"""QAM: Q-learning with Adjoint Matching for policy extraction.

Converts critic action gradients into step-wise supervision for the flow-based VLA policy.
Instead of backpropagating through the full multi-step ODE (expensive and unstable),
QAM defines a local regression objective along the reference flow trajectory.

Key equations from the paper:
- Terminal adjoint: g̃_1 = -∇_a Q_ϕ(s, a_1) / λ  (Eq. 10)
- QAM loss (Eq. 9): regress f_θ toward f_β + adjoint-derived correction
"""

import torch
import torch.nn as nn
import math
from typing import Callable, Optional, Tuple, List


def compute_action_gradient(
    critic_fn: Callable,
    state_repr: torch.Tensor,
    action_endpoint: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Compute ∇_a Q_ϕ(s, a_1) / λ at the flow endpoint.

    Args:
        critic_fn: callable that takes (state_repr, action_chunk) -> Q value
        state_repr: [B, state_dim] (detached)
        action_endpoint: [B, T, action_dim] generated endpoint a_1 (requires grad)
        temperature: λ, KL regularization temperature

    Returns:
        action_grad: [B, T, action_dim] normalized critic gradient at a_1
    """
    action_endpoint = action_endpoint.detach().requires_grad_(True)

    # Reshape for critic: critic expects [B, T, action_dim]
    q_value = critic_fn(state_repr.detach(), action_endpoint)
    q_sum = q_value.sum()

    action_grad = torch.autograd.grad(q_sum, action_endpoint, create_graph=False)[0]
    return action_grad / temperature


def compute_qam_loss(
    policy_flow_fn: Callable,
    reference_flow_fn: Callable,
    critic_fn: Callable,
    state_repr: torch.Tensor,
    noise: torch.Tensor,
    temperature: float = 2.0,
    num_flow_steps: int = 10,
    num_quadrature_points: int = 4,
) -> torch.Tensor:
    """Compute QAM policy extraction loss (Eq. 9).

    L_QAM(θ) = E[ ∫_0^1 ||f_δ(s, a_w, w) / σ_w + σ_w * g̃_w||^2 dw ]

    where f_δ = f_θ - f_β and g̃_w is the adjoint state.

    Simplified implementation: uses the terminal adjoint condition g̃_1 = -∇_a Q / λ
    and propagates it backward along the flow using the adjoint ODE.
    For practical implementation, we use a few quadrature points along [0, 1].

    Args:
        policy_flow_fn: f_θ(s, a_w, w) -> velocity prediction from current policy
                        callable(noisy_action, time) -> [B, T, action_dim]
        reference_flow_fn: f_β(s, a_w, w) -> velocity from frozen reference policy
                          callable(noisy_action, time) -> [B, T, action_dim]
        critic_fn: Q_ϕ(s, a) -> scalar Q value
                   callable(state_repr, action_chunk) -> [B]
        state_repr: [B, state_dim] state representations (detached)
        noise: [B, T, action_dim] sampled Gaussian noise a_0
        temperature: λ for KL regularization
        num_flow_steps: steps for reference flow rollout
        num_quadrature_points: number of time points for loss integration

    Returns:
        loss: scalar QAM loss
    """
    batch_size = noise.shape[0]
    pred_horizon = noise.shape[1]
    action_dim = noise.shape[2]
    device = noise.device

    # Step 1: Roll out reference flow to get endpoint a_1
    with torch.no_grad():
        x_t = noise.clone()
        dt = 1.0 / num_flow_steps
        trajectory_points = []

        for i in range(num_flow_steps):
            t = torch.full((batch_size,), i * dt, device=device)
            trajectory_points.append((x_t.clone(), t.clone()))
            velocity = reference_flow_fn(x_t, t)
            x_t = x_t + velocity * dt

        endpoint = x_t  # a_1 from reference flow

    # Step 2: Compute terminal adjoint g̃_1 = -∇_a Q_ϕ(s, a_1) / λ
    action_grad = compute_action_gradient(
        critic_fn, state_repr, endpoint, temperature
    )
    g_tilde_1 = -action_grad  # [B, T, action_dim]

    # Step 3: Compute QAM loss at quadrature points along [0, 1]
    # For each time w, compute:
    #   σ_w = sqrt(2 * (1-w) * w)
    #   target = f_β(s, a_w, w) - σ_w * g̃_w
    # In the simplified single-step adjoint approximation,
    # g̃_w ≈ g̃_1 (constant adjoint, valid for small KL deviations)
    loss = torch.tensor(0.0, device=device)

    # Sample quadrature points uniformly in (0, 1)
    w_points = torch.linspace(0.1, 0.9, num_quadrature_points, device=device)

    for w in w_points:
        w_batch = torch.full((batch_size,), w.item(), device=device)
        sigma_w = math.sqrt(2.0 * (1.0 - w.item()) * w.item())

        # Compute interpolated point on reference trajectory
        # a_w = (1-w) * a_0 + w * a_1
        a_w = (1.0 - w) * noise + w * endpoint.detach()

        # Reference flow velocity at (a_w, w)
        with torch.no_grad():
            ref_velocity = reference_flow_fn(a_w, w_batch)

        # Current policy velocity at (a_w, w)
        policy_velocity = policy_flow_fn(a_w, w_batch)

        # f_δ = f_θ - f_β
        f_delta = policy_velocity - ref_velocity

        # QAM regression target
        # ||f_δ / σ_w + σ_w * g̃_1||^2
        if sigma_w > 1e-6:
            target = f_delta / sigma_w + sigma_w * g_tilde_1.detach()
        else:
            target = f_delta + g_tilde_1.detach()

        loss = loss + (target ** 2).mean()

    loss = loss / num_quadrature_points
    return loss


def compute_qam_loss_simplified(
    policy_flow_fn: Callable,
    reference_flow_fn: Callable,
    action_grad: torch.Tensor,
    noise: torch.Tensor,
    endpoint: torch.Tensor,
    num_quadrature_points: int = 4,
) -> torch.Tensor:
    """Simplified QAM loss when action gradient is pre-computed.

    This is the practical version used in the training loop where
    the reference flow rollout and action gradient computation are
    done separately for efficiency.

    Args:
        policy_flow_fn: f_θ(a_w, w) -> [B, T, action_dim]
        reference_flow_fn: f_β(a_w, w) -> [B, T, action_dim]
        action_grad: [B, T, action_dim] pre-computed ∇_a Q / λ
        noise: [B, T, action_dim] starting noise a_0
        endpoint: [B, T, action_dim] reference flow endpoint a_1
        num_quadrature_points: integration points

    Returns:
        loss: scalar
    """
    batch_size = noise.shape[0]
    device = noise.device
    g_tilde_1 = -action_grad  # terminal adjoint

    loss = torch.tensor(0.0, device=device)
    w_points = torch.linspace(0.1, 0.9, num_quadrature_points, device=device)

    for w in w_points:
        w_batch = torch.full((batch_size,), w.item(), device=device)
        sigma_w = math.sqrt(2.0 * (1.0 - w.item()) * w.item())

        a_w = (1.0 - w) * noise + w * endpoint.detach()

        with torch.no_grad():
            ref_velocity = reference_flow_fn(a_w, w_batch)

        policy_velocity = policy_flow_fn(a_w, w_batch)
        f_delta = policy_velocity - ref_velocity

        if sigma_w > 1e-6:
            residual = f_delta / sigma_w + sigma_w * g_tilde_1.detach()
        else:
            residual = f_delta + g_tilde_1.detach()

        loss = loss + (residual ** 2).mean()

    return loss / num_quadrature_points
