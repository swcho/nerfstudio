# Training Your First Model and CLI Basics

## Train and run viewer

Train **nerfacto**, the recommended model for real-world scenes:

```bash
ns-download-data nerfstudio --capture-name=poster   # test data
ns-train nerfacto --data data/nerfstudio/poster     # train
```

The link printed at the end of the terminal opens the web viewer. On a remote machine, forward the websocket port (default **7007**); change it with `--viewer.websocket-port`. All data(parser) configurations must go at the **end** of the command, after the model and viewer options.

* Resume from checkpoint: `ns-train nerfacto --data ... --load-dir {outputs/.../nerfstudio_models}`
* Visualize an existing run: `ns-viewer --load-config {outputs/.../config.yml}`

## Exporting results

* **Render video**: create a camera path in the viewer's RENDER tab — orient the view, press "ADD CAMERA" for each keyframe, then press "RENDER" to get the command to run (`ns-render`).
* **Point cloud**: use the EXPORT tab → POINT CLOUD (crop with the yellow box if enabled), or `ns-export pointcloud --help` from the CLI.

## CLI structure

```bash
usage: ns-train {method} [method args] {dataparser} [dataparser args]
```

* `ns-train --help` lists supported models.
* `ns-train nerfacto --help` shows model/training-specific arguments.
* `ns-train nerfacto <args> nerfstudio-data --help` shows dataparser-specific arguments. The default dataparser is `nerfstudio-data`; others include Blender, NuScenes, etc. The dataparser subcommand must come **after** the model subcommand, e.g. `ns-train splatfacto [args] nerfstudio-data --eval-mode filename`.

### Main CLI commands

| Command | Description |
| --- | --- |
| `ns-install-cli` | Install tab completion |
| `ns-process-data` | Generate a dataset from your own data (needs COLMAP + FFmpeg) |
| `ns-download-data` | Download existing captures |
| `ns-train` | Train a NeRF |
| `ns-viewer` | View a trained NeRF |
| `ns-eval` | Run evaluation metrics |
| `ns-render` | Render a video of your NeRF (needs FFmpeg) |
| `ns-export` | Export a NeRF into other formats |

## Experiment tracking

Four tracking options: the viewer, TensorBoard, Weights & Biases, and Comet. Select with `--vis {viewer, tensorboard, wandb, comet, viewer+wandb, viewer+tensorboard, viewer+comet}`. Using the viewer together with wandb/tensorboard may stutter during eval steps. The viewer only works well for fast methods (nerfacto, instant-ngp); for slow methods like vanilla NeRF use the other loggers.

## Evaluating runs

```bash
ns-eval --load-config={PATH_TO_CONFIG} --output-path=output.json
```

Computes metrics (e.g. PSNR) and saves them to JSON. A benchmarking workflow for the classical Blender dataset is also provided.

## Multi-GPU training

Nerfstudio uses **PyTorch Distributed Data Parallel (DDP)** — gradients are averaged over devices. Plotting is done only in the first process. Tune the learning rate and `<X>_num_rays_per_batch` when using DDP. Example with `nerfacto-big` (larger model that benefits more from multi-GPU):

```bash
# 1 GPU (8192 rays per GPU per batch) → ~70k rays/sec on V100
CUDA_VISIBLE_DEVICES=0 ns-train nerfacto-big --machine.num-devices 1 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/aspen

# 2 GPUs (4096 rays per GPU, effectively 8192 per batch) → ~100k rays/sec
CUDA_VISIBLE_DEVICES=0,1 ns-train nerfacto --machine.num-devices 2 \
  --pipeline.datamanager.train-num-rays-per-batch 4096 --data data/nerfstudio/aspen
```

"Train Rays / Sec" reports total training-ray throughput; increase the number of GPUs and observe how it improves and eventually saturates.
