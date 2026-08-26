# Nerfstudio Overview and Installation

## Overview

Nerfstudio provides a simple API for an end-to-end process of creating, training, and testing NeRFs. The library supports a **more interpretable implementation of NeRFs by modularizing each component**, aiming for a user-friendly experience in exploring the technology.

It launched as an open-source project by Berkeley students in the KAIR lab at Berkeley AI Research (BAIR) in October 2022 (SIGGRAPH 2023 paper: "Nerfstudio: A Modular Framework for Neural Radiance Field Development"). Sponsors include Luma AI and the BAIR commons.

The documentation is organized into: **Getting Started** (quick tour, installation), **Nerfology** (educational guides on the tech), **Developer Guides** (pipelines, viewer, config, debugging), and **Reference** (API descriptions).

Nerfstudio is built on:

* **tyro** — easy-to-use config system (by Brent Yi)
* **nerfacc** — library for accelerating NeRF renders (by Ruilong Li)

### Supported methods

Included methods: **Nerfacto** (recommended default, integrates multiple methods), Instant-NGP, NeRF (original), Mip-NeRF, TensoRF, and **Splatfacto** (Nerfstudio's Gaussian Splatting implementation).

Third-party methods include: BioNeRF, Instruct-NeRF2NeRF, Instruct-GS2GS, SIGNeRF, K-Planes, LERF, LiveScene, Feature Splatting, Nerfbusters, NeRFPlayer, Tetra-NeRF, PyNeRF, SeaThru-NeRF, Zip-NeRF, NeRFtoGSandBack, OpenNeRF.

## Installation

### Prerequisites

Nerfstudio requires `python >= 3.8`; conda is recommended for dependency management. On Windows the install is more fragile: Visual Studio 2022 with `Desktop Development with C++` must be installed **before CUDA**, and the Visual C++ environment must be activated via `vcvars64.bat` (re-run after closing the terminal or when updating). Linux is recommended.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```

### Dependencies

* **PyTorch**: recommended Torch 2.1.2 with CUDA 11.8 (`pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url ...`). `cuda-toolkit` is also required to build CUDA extensions: `conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit`. If PyTorch < 2.0.1 was installed, first uninstall torch, torchvision, functorch, and tinycudann.
* **tiny-cuda-nn** torch bindings: `pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch`

### Installing nerfstudio

```bash
pip install nerfstudio            # from pip
# or from source for the latest development version:
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio && pip install -e .
```

Optional: `ns-install-cli` installs bash/zsh tab completion (re-run when the CLI changes). Development packages: `pip install -e .[dev]` and `.[docs]`.

### Pixi

Pixi is a fast package manager built on the conda ecosystem (only Linux supported for nerfstudio). `pixi run post-install` installs all environment dependencies including colmap, tinycudann, and hloc; `pixi shell` activates the environment (must be re-run per new shell). `pixi run train-example-nerf` downloads an example dataset and runs nerfacto.

### Docker

A ready-to-use image is at `ghcr.io/nerfstudio-project/nerfstudio:latest` (based on `nvidia/cuda:11.8.0-devel-ubuntu22.04`, Python 3.10). Key run parameters: `--gpus all` (required), `-v /folder/of/your/data:/workspace/` (required), `-p 7007:7007` for the web UI, `--shm-size=12gb` (or `--ipc=host`) to avoid memory limits (default is only 64 MB), plus `-u $(id -u)` and `--rm`.

When building the image, `CUDA_ARCHITECTURES` should match your GPU (e.g. 90 for H100, 89 for 40X0, 86 for 30X0, 80 for A100, 75 for 20X0). Restricting to one architecture speeds up the build significantly. Everything outside mounted folders is lost when the container is destroyed; always use full paths in mounts.

## Installation FAQ

* **ImportError: DLL load failed while importing \_89_C** — tiny-cuda-nn did not compile support for your CUDA architecture. Reinstall with `TCNN_CUDA_ARCHITECTURES=XX pip install git+.../tiny-cuda-nn/...` (e.g. 89 for a 4090).
* **CUDA mismatch while installing tiny-cuda-nn** — detected CUDA version mismatches the one PyTorch was compiled with; reinstall PyTorch with the correct CUDA version.
* **(Windows) No CUDA toolset found** — CUDA's Visual Studio integration is missing; copy the 4 MSBuildExtensions files from the CUDA toolkit into the VS BuildCustomizations directory.
* **File "setup.py" not found with `pip install -e .`** — upgrade pip: `python -m pip install --upgrade pip`.
* **Runtime errors like "len(sources) > 0"** — CUDA version not detected; set `CUDA_HOME=/usr/local/cuda` and extend `LD_LIBRARY_PATH`/`PATH` accordingly.
* Many Windows errors are caused by not having the Visual Studio environment loaded — re-activate it and retry.
