"""Policy checkpoint synchronization between learner and actors.

Handles publishing updated policy checkpoints from the learner
and distributing them to the actor fleet.
"""

import os
import time
import torch
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SyncConfig:
    """Configuration for policy sync."""
    checkpoint_dir: str = "./policy_checkpoints"
    sync_period: int = 50  # learner steps between publishes


class PolicySync:
    """Manages policy checkpoint publish/subscribe between learner and actors.

    Learner publishes: saves action_head state_dict + increments version
    Actors subscribe: poll for version changes, load new checkpoint
    """

    def __init__(self, config: SyncConfig):
        self.config = config
        self.version = 0
        self.ckpt_dir = Path(config.checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def publish(self, action_head_state_dict: dict) -> int:
        """Publish a new policy checkpoint.

        Called by the learner every sync_period steps.

        Args:
            action_head_state_dict: state dict of the current policy action head

        Returns:
            new version number
        """
        self.version += 1
        ckpt_path = self.ckpt_dir / f"action_head_v{self.version}.pt"
        torch.save(action_head_state_dict, ckpt_path)

        # Update version file atomically
        version_path = self.ckpt_dir / "version.txt"
        tmp_path = self.ckpt_dir / "version.txt.tmp"
        with open(tmp_path, "w") as f:
            f.write(str(self.version))
        os.replace(str(tmp_path), str(version_path))

        # Clean up old checkpoints (keep last 3)
        self._cleanup_old_checkpoints(keep_last=3)

        return self.version

    def get_latest_version(self) -> int:
        """Check the latest available version."""
        version_path = self.ckpt_dir / "version.txt"
        if version_path.exists():
            with open(version_path, "r") as f:
                return int(f.read().strip())
        return 0

    def get_checkpoint_path(self, version: Optional[int] = None) -> Optional[str]:
        """Get path to a specific version's checkpoint."""
        if version is None:
            version = self.get_latest_version()
        path = self.ckpt_dir / f"action_head_v{version}.pt"
        return str(path) if path.exists() else None

    def _cleanup_old_checkpoints(self, keep_last: int = 3):
        """Remove old checkpoint files, keeping only the most recent ones."""
        ckpt_files = sorted(
            self.ckpt_dir.glob("action_head_v*.pt"),
            key=lambda p: int(p.stem.split("_v")[1]),
        )
        for old_ckpt in ckpt_files[:-keep_last]:
            old_ckpt.unlink(missing_ok=True)
