"""Robot Actor Process for LWD fleet deployment.

Handles autonomous rollout collection, human intervention injection,
and episode upload to the shared replay buffer.
"""

import os
import time
import torch
import json
from typing import Optional, Dict, Any, Callable
from pathlib import Path
from dataclasses import dataclass

from lwd.data.transition import ChunkedTransition, Episode, SourceType


@dataclass
class ActorConfig:
    """Configuration for a robot actor."""
    actor_id: int = 0
    task_name: str = "default_task"
    max_episode_steps: int = 3000
    chunk_horizon: int = 30
    action_dim: int = 14
    policy_checkpoint_dir: str = "./policy_checkpoints"
    episode_output_dir: str = "./episodes"
    policy_poll_interval: float = 5.0  # seconds between policy checkpoint checks


class Actor:
    """Robot actor that collects rollout data during deployment.

    In the LWD framework, each actor:
    1. Loads the latest policy checkpoint
    2. Executes autonomous rollouts
    3. Accepts human interventions when needed
    4. Uploads completed episodes to shared storage
    """

    def __init__(
        self,
        config: ActorConfig,
        env_step_fn: Callable,
        observe_fn: Callable,
        device: torch.device = None,
    ):
        """
        Args:
            config: actor configuration
            env_step_fn: callable(action) -> (obs, done, info) that steps the environment
            observe_fn: callable() -> state_dict that returns current observation
            device: torch device for policy inference
        """
        self.config = config
        self.env_step_fn = env_step_fn
        self.observe_fn = observe_fn
        self.device = device or torch.device("cpu")

        self.policy = None
        self.policy_version = -1
        self.episode_count = 0

        # Ensure output directory exists
        Path(config.episode_output_dir).mkdir(parents=True, exist_ok=True)

    def load_policy(self, checkpoint_path: str):
        """Load policy action head from checkpoint."""
        from lwd.model.action_head import FlowActionHead

        state_dict = torch.load(checkpoint_path, map_location=self.device)
        if self.policy is None:
            self.policy = FlowActionHead(
                action_dim=self.config.action_dim,
                hidden_size=896,
                action_hidden_size=896,
            ).to(self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def check_for_new_policy(self) -> bool:
        """Check if a newer policy checkpoint is available.

        Returns:
            True if a new policy was loaded.
        """
        ckpt_dir = Path(self.config.policy_checkpoint_dir)
        if not ckpt_dir.exists():
            return False

        # Look for version file
        version_file = ckpt_dir / "version.txt"
        if version_file.exists():
            with open(version_file, "r") as f:
                new_version = int(f.read().strip())
            if new_version > self.policy_version:
                ckpt_path = ckpt_dir / f"action_head_v{new_version}.pt"
                if ckpt_path.exists():
                    self.load_policy(str(ckpt_path))
                    self.policy_version = new_version
                    print(f"[Actor {self.config.actor_id}] Loaded policy v{new_version}")
                    return True
        return False

    def collect_episode(
        self,
        intervention_fn: Optional[Callable] = None,
    ) -> Episode:
        """Run one complete episode and return collected transitions.

        Args:
            intervention_fn: optional callable(state, action) -> (corrected_action, is_intervention)
                           Returns the action to execute and whether human intervened.

        Returns:
            Episode containing all chunked transitions
        """
        episode = Episode(
            episode_id=f"actor{self.config.actor_id}_ep{self.episode_count}",
            source_type=SourceType.ONLINE_POLICY,
            dataset_name=self.config.task_name,
        )

        transitions = []
        step = 0
        done = False
        success = False

        while not done and step < self.config.max_episode_steps:
            # Observe current state
            state = self.observe_fn()

            # Generate action chunk from policy
            with torch.no_grad():
                action_chunk = self._generate_action(state)

            # Check for human intervention
            is_intervention = False
            if intervention_fn is not None:
                corrected_action, is_intervention = intervention_fn(state, action_chunk)
                if is_intervention:
                    action_chunk = corrected_action

            # Execute action chunk (step H times)
            chunk_done = False
            for h in range(self.config.chunk_horizon):
                if step >= self.config.max_episode_steps:
                    done = True
                    break

                action_h = action_chunk[h] if action_chunk.dim() > 1 else action_chunk
                obs, env_done, info = self.env_step_fn(action_h)
                step += 1

                if env_done:
                    done = True
                    success = info.get("success", False)
                    chunk_done = True
                    break

            # Record transition
            next_state = self.observe_fn()
            source = SourceType.ONLINE_INTERVENTION if is_intervention else SourceType.ONLINE_POLICY

            transition = ChunkedTransition(
                state=state,
                action_chunk=action_chunk if action_chunk.dim() > 1 else action_chunk.unsqueeze(0),
                reward=0.0,  # Will be annotated later
                next_state=next_state,
                done=chunk_done or done,
                success=success if done else False,
                episode_id=episode.episode_id,
                source_type=source,
                dataset_name=self.config.task_name,
                chunk_index=len(transitions),
            )
            transitions.append(transition)

        episode.transitions = transitions
        episode.success = success
        episode.total_steps = step

        # Annotate sparse rewards
        episode.annotate_rewards()

        self.episode_count += 1
        return episode

    def _generate_action(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Generate action chunk using the current policy.

        Override this method for full VLM-based action generation.
        """
        if self.policy is None:
            # Random policy fallback
            return torch.randn(self.config.chunk_horizon, self.config.action_dim)

        # Placeholder: in full implementation, this runs the VLM + flow generation
        batch_state = torch.randn(1, 896, device=self.device)  # placeholder state repr
        noise = torch.randn(1, self.config.chunk_horizon, self.config.action_dim, device=self.device)

        # Euler integration
        x_t = noise.clone()
        num_steps = 10
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((1,), i * dt, device=self.device)
            embed, _ = self.policy.encode_noisy_action(x_t, t)
            velocity = self.policy.predict_flow(embed.reshape(-1, embed.shape[-1]))
            velocity = velocity.reshape(1, self.config.chunk_horizon, self.config.action_dim)
            x_t = x_t + velocity * dt

        return x_t.squeeze(0).clamp(-1, 1).cpu()

    def save_episode(self, episode: Episode):
        """Save episode to disk for upload to central learner."""
        output_path = Path(self.config.episode_output_dir) / f"{episode.episode_id}.pt"
        torch.save({
            "transitions": [(
                {k: v.cpu() for k, v in t.state.items()},
                t.action_chunk.cpu(),
                t.reward,
                {k: v.cpu() for k, v in t.next_state.items()},
                t.done,
                t.success,
                int(t.source_type),
                t.dataset_name,
            ) for t in episode.transitions],
            "episode_id": episode.episode_id,
            "success": episode.success,
            "total_steps": episode.total_steps,
            "source_type": int(episode.source_type),
        }, output_path)

    def run_loop(self, num_episodes: int = -1, intervention_fn: Optional[Callable] = None):
        """Main actor loop: collect episodes and upload.

        Args:
            num_episodes: number of episodes to collect (-1 for infinite)
            intervention_fn: optional human intervention callback
        """
        ep = 0
        while num_episodes < 0 or ep < num_episodes:
            # Check for new policy
            self.check_for_new_policy()

            # Collect episode
            episode = self.collect_episode(intervention_fn)

            # Save episode
            self.save_episode(episode)

            print(
                f"[Actor {self.config.actor_id}] Episode {ep}: "
                f"steps={episode.total_steps}, success={episode.success}"
            )

            ep += 1
