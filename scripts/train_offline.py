"""Launch script for LWD offline RL pretraining (Stage 1)."""

import argparse
import torch

from lwd.trainer.lwd_trainer import LWDTrainer, LWDConfig


def main():
    parser = argparse.ArgumentParser(description="LWD Offline RL Pretraining")
    parser.add_argument("--config", type=str, default="configs/lwd_offline.yml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()

    # Load config
    config = LWDConfig.from_yaml(args.config)

    # Initialize trainer
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    trainer = LWDTrainer(config, device=device)

    # Freeze reference policy
    trainer.freeze_reference_policy()

    # TODO: Load offline data into replay buffer
    # This requires connecting to the actual data pipeline.
    # For testing, populate with synthetic data:
    print("[train_offline] NOTE: You need to load offline episodes into the replay buffer.")
    print("[train_offline] Use trainer.replay_buffer.add_offline_episode(episode) for each episode.")

    # Run offline pretraining
    trainer.train_offline(num_steps=args.steps)

    # Save final checkpoint
    trainer.save_checkpoint("offline_final")
    print("[train_offline] Done.")


if __name__ == "__main__":
    main()
