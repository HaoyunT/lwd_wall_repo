"""Action and proprioception normalization for LWD.

Normalizes actions to [-1, 1] range using min/delta statistics per robot embodiment.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional


class Normalizer(nn.Module):
    """Min-delta normalizer that maps raw action/state values to [-1, 1]."""

    def __init__(
        self,
        action_statistic_dof: Dict[str, Dict],
        dof_config: Dict[str, int],
        min_key: str = "min",
        delta_key: str = "delta",
    ):
        super().__init__()
        self.min_key = min_key
        self.delta_key = delta_key

        self.min = nn.ParameterDict()
        self.delta = nn.ParameterDict()

        for robot_name, stats in action_statistic_dof.items():
            all_min = []
            all_delta = []
            for dof_name, dof_dim in dof_config.items():
                if dof_name in stats and min_key in stats[dof_name] and delta_key in stats[dof_name]:
                    all_min.extend(stats[dof_name][min_key])
                    all_delta.extend(stats[dof_name][delta_key])
                else:
                    all_min.extend([0.0] * dof_dim)
                    all_delta.extend([1.0] * dof_dim)

            self.min[robot_name] = nn.Parameter(
                torch.tensor(all_min, dtype=torch.float32), requires_grad=False
            )
            self.delta[robot_name] = nn.Parameter(
                torch.tensor(all_delta, dtype=torch.float32), requires_grad=False
            )

    @classmethod
    def from_checkpoint(cls, ckpt_path: str) -> "Normalizer":
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.min = nn.ParameterDict()
        instance.delta = nn.ParameterDict()
        instance.min_key = "min"
        instance.delta_key = "delta"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        for key, value in ckpt.items():
            try:
                prefix, name = key.split(".", 1)
                if hasattr(instance, prefix):
                    getattr(instance, prefix)[name] = nn.Parameter(value, requires_grad=False)
            except ValueError:
                continue
        return instance

    def normalize(self, xs: torch.Tensor, dataset_names: List[str]) -> torch.Tensor:
        """Normalize raw values to [-1, 1]."""
        new_xs = []
        for x, name in zip(xs, dataset_names):
            x = (x - self.min[name]) / self.delta[name]
            x = x * 2 - 1
            x = torch.clamp(x, -1, 1)
            new_xs.append(x)
        return torch.stack(new_xs)

    def unnormalize(self, xs: torch.Tensor, dataset_names: List[str]) -> torch.Tensor:
        """Map normalized [-1, 1] values back to raw action space."""
        new_xs = []
        for x, name in zip(xs, dataset_names):
            x = (x + 1) / 2
            x = x * self.delta[name] + self.min[name]
            new_xs.append(x)
        return torch.stack(new_xs)
