# Wall-OSS-0.5 Reproduction Notes

## Local status

- Code repo cloned: `wall-x/`
- CUDA submodule initialized: `wall-x/3rdparty/cutlass`
- Hugging Face model metadata downloaded: `models/wall-oss-0.5/`
- Full model weights are not downloaded yet. The main `model.safetensors` file is about 8.3 GB.

## Sources

- GitHub: https://github.com/X-Square-Robot/wall-x
- Hugging Face: https://huggingface.co/x-square-robot/wall-oss-0.5
- Paper: https://x2robot.com/api/files/file/wall_oss_05.pdf

## What blocks full local inference on this Mac

The current repository is CUDA-first:

- `setup.py` builds a `CUDAExtension` named `wallx_csrc`.
- `scripts/fake_inference.py`, `scripts/infer_robochallenge.py`, and `scripts/draw_openloop_plot.py` use CUDA paths directly.
- `wall_x/model/joint_attention.py` imports `flash_attn` directly.
- README asks for Python 3.10, CUDA 12.x, FlashAttention, PyTorch, and Ubuntu 22.04.

Current local machine state:

- Python is 3.9.6.
- Torch and Transformers are not installed.
- macOS does not provide CUDA or FlashAttention support for this project.

So the Mac is suitable for source review and preparing scripts, but not for a faithful Wall-OSS-0.5 inference run.

## Recommended GPU environment

Use a Linux GPU server:

- Ubuntu 22.04
- NVIDIA GPU with CUDA 12.x
- Python 3.10
- PyTorch 2.6.0
- FlashAttention 2.7.4.post1
- Enough VRAM for a 4B BF16 VLA model. Start with at least a 24 GB GPU; 40 GB or more is safer for experiments.

## Next commands on a GPU server

From this directory:

```bash
bash scripts/setup_gpu_env.sh
bash scripts/download_wall_oss_0_5.sh
```

Then from `wall-x/`:

```bash
conda activate wallx
python scripts/fake_inference.py
```

The upstream `fake_inference.py` still has `model_path = "/path/to/model"`, so change that line to the downloaded model path first:

```text
/path/to/wall-oss-0.5-repro/models/wall-oss-0.5
```

## Documentation mismatch observed

The Hugging Face README includes an example importing `wall_x._vendor.harrix`, but the current GitHub repo clone does not contain `wall_x/_vendor/harrix`. The GitHub README points to `scripts/fake_inference.py`, but that file is still a template with a placeholder model path. Treat the project as partially released and expect small script fixes during reproduction.
