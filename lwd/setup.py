from setuptools import setup, find_packages

setup(
    name="lwd",
    version="0.1.0",
    description="Reproduction of Learning While Deploying (LWD): Fleet-Scale RL for Generalist Robot Policies",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "safetensors>=0.4.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
    ],
)
