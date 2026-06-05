# WALL-OSS

<div align="left">

<p align="center">
    <img src="assets/logo.png" width="600"/>
<p>

<div align="center">

[![Paper](https://img.shields.io/badge/📄%20Paper-PDF-EA1B22?style=for-the-badge&logo=adobeacrobatreader&logoColor=fff)](https://x2robot.com/api/files/file/wall_oss_05.pdf)
&nbsp;&nbsp;
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-x--square--robot-FFB000?style=for-the-badge&logo=huggingface&logoColor=000)](https://huggingface.co/x-square-robot)
&nbsp;&nbsp;
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=fff)](https://github.com/X-Square-Robot/wall-x)
&nbsp;&nbsp;
[![Project Page](https://img.shields.io/badge/Project-1E90FF?style=for-the-badge&logo=google-chrome&logoColor=fff)](https://x2robot.com/en/oss#resources)

</div>

</div>

## <a href="https://x2robot.com/api/files/file/wall_oss_05.pdf" target="_blank"><strong>Wall-OSS-0.5: A Deployment-Ready VLA with GradientBridged Pretraining</strong></a>

We introduce Wall-OSS-0.5, an open-source 4B Vision-Language-Action (VLA) foundation model built upon a 3B VLM backbone augmented with dedicated action-generation components. While traditional VLAs are treated merely as optimization initializations, Wall-OSS-0.5 is designed so that pretrained robotic capability is directly executable and measurable on physical hardware without any downstream fine-tuning.

The model is pretrained across more than 20 distinct robot embodiments, processing over one million trajectories per epoch alongside a grounded multimodal corpus. We adopt a novel gradient-bridged co-training recipe optimizing three complementary objectives:
- **Discrete Action Prediction**: Routes strong VLM-native gradients into the backbone.
- **Multimodal Prediction**: Preserves and strengthens grounded vision-language understanding.
- **Continuous Flow Matching**: Serves as the deployment-time continuous action interface.

## 🌟 Key Highlights
1. **Zero-Shot Real-Robot Behavior**: Achieves non-trivial zero-shot completion on a 17-task suite (including held-out deformable manipulation tasks) directly from the pretrained checkpoint.
2. **Markedly Stronger Adaptation Prior**: After task-specific fine-tuning, Wall-OSS-0.5 reaches 60.5% average task progress on 15 real-robot tasks, outperforming $\pi_{0.5}$ by 17.5%.
3. **No Capabilities Erosion**: Multimodal evaluations confirm action-pretraining preserves broad vision-language competence while significantly sharpening embodied grounding.



## 🚀 Quick Start

### Installation

```bash
# Create conda environment
conda create --name wallx python=3.10
conda activate wallx

# Install base requirements
pip install torch torchvision transformers
pip install huggingface_hub

# Install Wall-X from GitHub
git clone https://github.com/X-Square-Robot/wall-x.git
cd wall-x
pip install -e .
```


## 🎯 Supervised Fine-Tuning (SFT)

For training Wall-X on your robotics datasets, please refer to our comprehensive training guide:

**📖 [Training Documentation](https://github.com/X-Square-Robot/wall-x/blob/main/workspace/README.md)**

The training process includes:
- **Dataset Preparation**: How to prepare your robotics datasets in LeRobot format
- **Configuration Setup**: Detailed configuration for GPU setup, model paths, and robot DOF settings
- **Training Scripts**: Ready-to-use training scripts with proper hyperparameters



## 🔮 Inference

For detailed inference examples and model evaluation:

**📖 [Inference Documentation](https://github.com/X-Square-Robot/wall-x/blob/main/scripts/)**

### Basic Inference Example

```python
"""Load checkpoint and run one inference with fake inputs."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

CHECKPOINT = "x-square-robot/wall-oss-0.5"

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import wall_x._vendor.harrix.adapters  # noqa: F401
from wall_x._vendor.harrix.adapters.registry import build_adapter
from wall_x._vendor.harrix.envs.libero_common import encode_proprio
from wall_x._vendor.harrix.eval_config import EvalConfig, autofill_from_checkpoint

# 1) load model
cfg = EvalConfig()
cfg.model.checkpoint_path = CHECKPOINT
cfg.model.norm_key = "x2_normal"
cfg.model.cam_names = ["face_view", "right_wrist_view"]
cfg = autofill_from_checkpoint(cfg)
model = build_adapter(cfg)

# 2) fake input
rng = np.random.default_rng(0)
obs = {
    "eef_pos": rng.normal(size=3).astype(np.float32),
    "eef_axisangle": rng.normal(size=3).astype(np.float32),
    "gripper": rng.normal(size=1).astype(np.float32),
    "face_view": rng.integers(0, 256, (448, 448, 3), dtype=np.uint8),
    "wrist_view": rng.integers(0, 256, (448, 448, 3), dtype=np.uint8),
}
instruction = "pick up the cup"

# 3) infer (return raw action chunk, shape: [horizon, action_dim])
encoded = encode_proprio(obs, model._train_config, model._action_horizon)
prefix, postfix = model._get_flow_prompt(instruction)
batch_inputs = model._construct_model_input([encoded], [prefix], [postfix])
padding = (
    torch.zeros_like(model._normalizer_action.delta[batch_inputs["dataset_names"][0]])
    .unsqueeze(0)
    .to("cpu")
)
padding = model._normalizer_action.normalize_data(
    padding, batch_inputs["dataset_names"]
).to(batch_inputs["input_ids"].device)

out = model._model.generate_flow_action(
    action_horizon=model._action_horizon,
    action_dim=model._action_dim,
    num_inference_timesteps=model._num_inference_timesteps,
    padding_action=padding,
    **batch_inputs,
)
result = out["predict_action"].detach().cpu().numpy()
print("result shape:", result.shape)
print("result:", result)

```

### Advanced Inference Scripts

For production-ready inference and evaluation scripts:

```bash
# Basic inference test
python ./scripts/fake_inference.py

# Generate open-loop comparison plots
python ./scripts/draw_openloop_plot.py
```

**📁 [View all inference scripts](https://github.com/X-Square-Robot/wall-x/tree/main/scripts)**

## 📚 Complete Documentation

For comprehensive setup, training, and inference instructions:

### 🚀 **[Visit our GitHub Repository](https://github.com/X-Square-Robot/wall-x)**

The repository contains:
- **Detailed Installation Guide**: Complete environment setup with all dependencies
- **Training Tutorials**: Step-by-step SFT process with LeRobot datasets
- **Inference Examples**: Multiple inference scripts and evaluation tools
- **Configuration Templates**: Ready-to-use configs for different robot setups
- **Troubleshooting Guide**: Common issues and solutions

## 📄 Cite Us

If you find WALL-OSS models useful, please cite:

```bibtex
@misc{walloss_paper_2025,
  title        = {WALL-OSS: Igniting VLMs toward the Embodied Space},
  author       = {X Square Robot},
  year         = {2025},
  howpublished = {\url{https://x2robot.cn-wlcb.ufileos.com/wall_oss.pdf}},
  note         = {White paper}
}
```
