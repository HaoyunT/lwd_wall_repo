# LWD & Wall-OSS-0.5 Research Repository

Research reproductions of two Vision-Language-Action (VLA) papers for robotic manipulation.

## Papers

| Paper | Key Contribution | Status |
|-------|-----------------|--------|
| **LWD** (Learning While Deploying) | Offline-to-online RL for VLA policies via DIVL + QAM | From-scratch reproduction |
| **Wall-OSS-0.5** (arXiv:2605.30877) | 4B open-source VLA with zero-shot real-robot capability | Official code organized locally |

## Repository Structure

```
.
├── lwd/                    # LWD algorithm reproduction
│   ├── configs/            # Training configs (offline / online stages)
│   ├── lwd/
│   │   ├── data/           # Episode and transition data structures
│   │   ├── distributed/    # Actor-learner fleet coordination
│   │   ├── model/          # Policy, critic, distributional value head
│   │   ├── rl/             # DIVL, QAM, replay buffer, TD targets
│   │   └── trainer/        # Learner step + full training pipeline
│   └── scripts/            # Entry points for training and testing
├── wall-oss-0.5/           # Wall-OSS-0.5 official code (X-Square Robot)
│   ├── csrc/               # Custom CUDA kernels (RoPE, permute, grouped GEMM)
│   ├── wall_x/             # Model (MoT architecture), data, inference, serving
│   ├── scripts/            # Inference and utility scripts
│   └── 3rdparty/cutlass/   # NVIDIA CUTLASS submodule
└── papers/                 # Reference PDFs
    ├── lwd_paper.pdf
    └── wall_oss_0.5.pdf
```

## LWD Reproduction Details

### Algorithm Overview

LWD enables continuous RL improvement of a pretrained VLA policy during real-world deployment:

1. **Stage 1 - Offline RL Pretraining**: Train on a static offline dataset using n-step chunk-level TD targets to bootstrap value estimates before deployment.
2. **Stage 2 - Online Post-Training**: Continuously improve the policy with mixed replay (offline + fleet-collected online data) using 1-step TD with optimistic quantile targets.

### Core Components

- **DIVL** (Distributional Implicit Value Learning): A C51-based distributional value function V_psi that models the distribution of replay Q-values. Adaptive tau-quantile extraction provides conservative (offline) or optimistic (online) bootstrap targets based on value distribution entropy.
- **QAM** (Q-learning with Adjoint Matching): Converts critic action gradients into step-wise flow-matching supervision. Instead of backpropagating through the full ODE, QAM uses the terminal adjoint g_1 = -nabla_a Q / lambda and regresses the policy flow toward a corrected reference.
- **Chunk-level TD**: Operates at the action chunk granularity (H=30 steps), discounting by gamma^H between chunks.
- **Clipped double-Q critic** with EMA target network and temporal attention pooling for action encoding.

### Key Hyperparameters

| Parameter | Offline | Online | Description |
|-----------|---------|--------|-------------|
| tau_base | 0.6 | 0.9 | Base quantile level (conservative vs optimistic) |
| gamma | 0.9999 | 0.9999 | Discount factor |
| n_steps | 10 | 1 | Chunk-level TD lookahead |
| ema_rate | 0.995 | 0.995 | Target network Polyak averaging |
| temperature | 2.0 | 2.0 | QAM KL regularization |

### Correctness Notes

- **EMA rate fixed**: The target network update uses standard Polyak averaging (target <- 0.995 * target + 0.005 * online). An earlier version had the rate inverted.
- **QAM simplification**: Uses constant adjoint approximation (g_w ~ g_1) which is valid for small KL deviations from the reference policy.
- **Flow matching**: Uses Beta(1.5, 1.0) time sampling biased toward early flow steps, matching the pi0 convention.

## Wall-OSS-0.5 Details

### Architecture

Wall-OSS-0.5 is a 4B-parameter VLA built on Qwen2.5-VL with:

- **Mixture of Transformers (MoT)**: Separate expert branches for language/vision and action tokens sharing the same attention layer. Expert-specific QKV projections with joint attention enable cross-modal information flow.
- **Flow-based action expert**: Predicts action sequences via conditional flow matching (not diffusion). Time embedding via sinusoidal positional encoding + SiLU MLP.
- **Gradient-bridged co-training**: Joint optimization of cross-entropy (language) and flow-matching (action) losses, with the gradient bridge propagating task understanding into action predictions.
- **Custom CUDA kernels**: Optimized RoPE, token permutation, and grouped GEMM for MoT efficiency.

### Requirements

- Ubuntu 22.04, Python 3.10, CUDA 12.x
- PyTorch 2.6.0, FlashAttention 2.7.4
- GPU with >= 24 GB VRAM (40+ GB recommended)
- See `wall-oss-0.5/REPRO_NOTES.md` for setup instructions

## Quick Start

### LWD (CPU-compatible for testing)

```bash
cd lwd
pip install -e .
python scripts/test_components.py    # Unit tests for DIVL/QAM/critic
python scripts/train_offline.py --config configs/lwd_offline.yml
```

### Wall-OSS-0.5 (requires GPU)

```bash
cd wall-oss-0.5
bash setup_gpu_env.sh
bash download_wall_oss_0_5.sh
python scripts/fake_inference.py
```

## References

- LWD: "Learning While Deploying: Online Reinforcement Learning for Vision-Language-Action Models"
- Wall-OSS-0.5: "Wall-OSS-0.5: Open-Source 4B VLA with Zero-Shot Real-Robot Capability" (arXiv:2605.30877)
- pi0: "pi0: A Vision-Language-Action Flow Model for General Robot Control"
