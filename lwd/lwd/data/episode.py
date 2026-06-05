"""Episode storage and loading utilities for LWD."""

import torch
from pathlib import Path
from typing import List, Optional

from lwd.data.transition import Episode, ChunkedTransition, SourceType


def save_episode(episode: Episode, output_dir: str) -> str:
    """Serialize and save an episode to disk.

    Args:
        episode: the episode to save
        output_dir: directory to write to

    Returns:
        path: the saved file path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(output_dir) / f"{episode.episode_id}.pt"

    data = {
        "episode_id": episode.episode_id,
        "success": episode.success,
        "total_steps": episode.total_steps,
        "source_type": int(episode.source_type),
        "dataset_name": episode.dataset_name,
        "num_transitions": len(episode.transitions),
        "transitions": [],
    }

    for t in episode.transitions:
        t_data = {
            "state": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.state.items()},
            "action_chunk": t.action_chunk.cpu() if isinstance(t.action_chunk, torch.Tensor) else t.action_chunk,
            "reward": t.reward,
            "next_state": {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in t.next_state.items()},
            "done": t.done,
            "success": t.success,
            "source_type": int(t.source_type),
            "dataset_name": t.dataset_name,
            "chunk_index": t.chunk_index,
        }
        if t.dof_mask is not None:
            t_data["dof_mask"] = t.dof_mask.cpu()
        data["transitions"].append(t_data)

    torch.save(data, output_path)
    return str(output_path)


def load_episode(filepath: str) -> Episode:
    """Load an episode from disk.

    Args:
        filepath: path to the .pt file

    Returns:
        episode: reconstructed Episode object
    """
    data = torch.load(filepath, map_location="cpu")

    episode = Episode(
        episode_id=data["episode_id"],
        success=data["success"],
        total_steps=data["total_steps"],
        source_type=SourceType(data["source_type"]),
        dataset_name=data.get("dataset_name", "default"),
    )

    for t_data in data["transitions"]:
        transition = ChunkedTransition(
            state=t_data["state"],
            action_chunk=t_data["action_chunk"],
            reward=t_data["reward"],
            next_state=t_data["next_state"],
            done=t_data["done"],
            success=t_data["success"],
            source_type=SourceType(t_data["source_type"]),
            dataset_name=t_data["dataset_name"],
            chunk_index=t_data.get("chunk_index", 0),
            dof_mask=t_data.get("dof_mask", None),
        )
        episode.transitions.append(transition)

    return episode


def load_episodes_from_dir(directory: str) -> List[Episode]:
    """Load all episodes from a directory.

    Args:
        directory: path containing .pt episode files

    Returns:
        episodes: list of loaded Episode objects
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return []

    episodes = []
    for f in sorted(dir_path.glob("*.pt")):
        try:
            ep = load_episode(str(f))
            episodes.append(ep)
        except Exception as e:
            print(f"Warning: failed to load {f}: {e}")

    return episodes
