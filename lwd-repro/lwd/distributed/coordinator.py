"""Coordinator: Central episode manager and versioned snapshot controller.

Watches for new episodes from actors, maintains the online replay buffer,
and notifies the learner of new data availability.
"""

import os
import time
import torch
from pathlib import Path
from typing import List, Optional
from threading import Thread, Event
from dataclasses import dataclass

from lwd.data.transition import ChunkedTransition, Episode, SourceType
from lwd.rl.replay_buffer import ReplayBuffer


@dataclass
class CoordinatorConfig:
    """Configuration for the central coordinator."""
    episode_input_dir: str = "./episodes"
    poll_interval: float = 2.0  # seconds between polls for new episodes
    snapshot_interval: int = 100  # episodes between snapshot versions


class Coordinator:
    """Central coordinator that manages episode ingestion and replay buffer updates.

    In the LWD distributed system:
    - Actors upload episodes to shared storage (filesystem)
    - Coordinator watches for new episodes
    - Coordinator ingests episodes into the online replay buffer
    - Coordinator maintains versioned snapshot metadata
    """

    def __init__(
        self,
        config: CoordinatorConfig,
        replay_buffer: ReplayBuffer,
    ):
        self.config = config
        self.replay_buffer = replay_buffer
        self.processed_episodes: set = set()
        self.snapshot_version = 0
        self.total_episodes_ingested = 0
        self._stop_event = Event()

    def ingest_episode_file(self, filepath: str) -> Optional[Episode]:
        """Load an episode from disk and add to online replay buffer."""
        try:
            data = torch.load(filepath, map_location="cpu")
        except Exception as e:
            print(f"[Coordinator] Failed to load {filepath}: {e}")
            return None

        episode = Episode(
            episode_id=data["episode_id"],
            success=data["success"],
            total_steps=data["total_steps"],
            source_type=SourceType(data["source_type"]),
        )

        for t_data in data["transitions"]:
            state, action_chunk, reward, next_state, done, success, source_type, dataset_name = t_data
            transition = ChunkedTransition(
                state=state,
                action_chunk=action_chunk,
                reward=reward,
                next_state=next_state,
                done=done,
                success=success,
                source_type=SourceType(source_type),
                dataset_name=dataset_name,
                episode_id=episode.episode_id,
            )
            episode.transitions.append(transition)

        self.replay_buffer.add_online_episode(episode)
        self.total_episodes_ingested += 1

        if self.total_episodes_ingested % self.config.snapshot_interval == 0:
            self.snapshot_version += 1
            print(f"[Coordinator] Snapshot v{self.snapshot_version}, "
                  f"online_size={self.replay_buffer.online_size}")

        return episode

    def poll_for_new_episodes(self) -> List[str]:
        """Check for new episode files in the input directory."""
        input_dir = Path(self.config.episode_input_dir)
        if not input_dir.exists():
            return []

        new_files = []
        for f in input_dir.glob("*.pt"):
            if f.name not in self.processed_episodes:
                new_files.append(str(f))
                self.processed_episodes.add(f.name)

        return sorted(new_files)

    def run_once(self) -> int:
        """Process all pending episodes. Returns count of newly ingested episodes."""
        new_files = self.poll_for_new_episodes()
        count = 0
        for filepath in new_files:
            ep = self.ingest_episode_file(filepath)
            if ep is not None:
                count += 1
        return count

    def run_loop(self):
        """Main coordinator loop: continuously poll and ingest episodes."""
        print(f"[Coordinator] Starting, watching: {self.config.episode_input_dir}")
        while not self._stop_event.is_set():
            count = self.run_once()
            if count > 0:
                print(f"[Coordinator] Ingested {count} episodes, "
                      f"total={self.total_episodes_ingested}")
            time.sleep(self.config.poll_interval)

    def start_background(self) -> Thread:
        """Start coordinator in a background thread."""
        thread = Thread(target=self.run_loop, daemon=True)
        thread.start()
        return thread

    def stop(self):
        """Stop the coordinator loop."""
        self._stop_event.set()
