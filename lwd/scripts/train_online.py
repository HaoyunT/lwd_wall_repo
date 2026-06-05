"""Launch script for LWD online RL post-training (Stage 2).

Starts the learner, coordinator, and optionally actor processes.
"""

import argparse
import torch
from threading import Thread

from lwd.trainer.lwd_trainer import LWDTrainer, LWDConfig
from lwd.distributed.coordinator import Coordinator, CoordinatorConfig
from lwd.distributed.sync import PolicySync, SyncConfig


def main():
    parser = argparse.ArgumentParser(description="LWD Online RL Post-Training")
    parser.add_argument("--config", type=str, default="configs/lwd_online.yml")
    parser.add_argument("--offline-checkpoint", type=str, required=True,
                        help="Path to offline stage checkpoint")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = LWDConfig.from_yaml(args.config)

    # Initialize trainer and load offline checkpoint
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer = LWDTrainer(config, device=device)
    trainer.load_checkpoint(args.offline_checkpoint)

    # Initialize coordinator for episode ingestion
    coord_config = CoordinatorConfig(
        episode_input_dir=config.save_path.replace("checkpoints/online", "episodes"),
    )
    coordinator = Coordinator(coord_config, trainer.replay_buffer)

    # Initialize policy sync
    sync_config = SyncConfig(
        checkpoint_dir=config.save_path.replace("checkpoints/online", "policy_checkpoints"),
        sync_period=config.actor_sync_period,
    )
    policy_sync = PolicySync(sync_config)

    # Start coordinator in background
    coord_thread = coordinator.start_background()
    print("[train_online] Coordinator started in background.")

    # Override publish method to use policy_sync
    original_publish = trainer._publish_policy_checkpoint
    def publish_with_sync():
        policy_sync.publish(trainer.action_head.state_dict())
    trainer._publish_policy_checkpoint = publish_with_sync

    # Run online training
    print("[train_online] Starting online post-training.")
    print("[train_online] Actors should be running separately, writing episodes to:", coord_config.episode_input_dir)
    trainer.train_online(num_steps=args.steps)

    # Cleanup
    coordinator.stop()
    trainer.save_checkpoint("online_final")
    print("[train_online] Done.")


if __name__ == "__main__":
    main()
