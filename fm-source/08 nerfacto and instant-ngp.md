# Nerfacto and Instant-NGP

## Nerfacto

Nerfstudio's *de facto* default method for **real static-scene captures**. It is not published work but a combination of many published methods that work well for real data. Variants:

| Method | Description | Memory | Speed |
| --- | --- | --- | --- |
| `nerfacto` | Default model | ~6GB | Fast |
| `nerfacto-big` | Larger, higher quality | ~12GB | Slower |
| `nerfacto-huge` | Even larger | ~24GB | Slowest |
| `depth-nerfacto` | Supervise on depth | ~6GB | Fast |

Techniques combined in nerfacto: **camera pose refinement, per-image appearance conditioning, proposal sampling, scene contraction, hash encoding**.

### Pose refinement

Predicted camera poses often contain errors (especially from phone apps like Record3D); misaligned poses cause cloudy artifacts and loss of sharpness. NeRF loss gradients can be backpropagated to the pose inputs to optimize and refine the poses.

### Piecewise sampler

Produces the initial scene samples: **half the samples are distributed uniformly up to distance 1** from the camera; the rest use step sizes that increase per sample (frustums are scaled versions of themselves). This samples distant objects while keeping dense sampling near the camera.

### Proposal sampler

Consolidates sample locations to the regions that contribute most to the final render (typically the first surface intersection), greatly improving quality. It needs a **density function** for the scene — a small fused MLP with hash encoding is accurate enough and fast. Density functions can be chained: **two are better than one; more than 2 gives diminishing returns**. The density field only needs a coarse density representation to guide sampling, so the encoding dictionary size and feature levels can be reduced with little quality impact.

## Instant-NGP

"Instant Neural Graphics Primitives with a Multiresolution Hash Encoding" (NVIDIA). Run with `ns-train instant-ngp`. Many of its contributions are built into Nerfacto, which is recommended for real-world scenes.

Instant-NGP breaks NeRF training into **3 pillars** with improvements enabling real-time training:

1. **Improved training/rendering via occupancy-grid ray marching** — skip sampling in empty space and behind high-density areas. Multiscale occupancy grids coarsely mark empty/non-empty space with a **single bit** per cell; samples with too-low occupancy are skipped. The grids are stored separately from the trainable encoding and updated during training. This speeds sampling 10–100×. Nerfstudio uses **NerfAcc** for this sampling.
2. **A smaller, fully-fused neural network** — the network runs entirely in a single CUDA kernel and is only 4 layers × 64 neurons, 5–10× faster than a TensorFlow implementation. Nerfstudio uses **tiny-cuda-nn**.
3. **Multi-resolution hash encoding** — the main contribution. The speedups are multiplicative, reaching **1000×** overall (train NeRFs in seconds).

### Multi-resolution hash encoding

Instead of a fixed positional encoding, coordinates map to **trainable feature vectors**: F-dimensional vectors arranged in **L** grids (resolution levels) with up to **T** vectors each. Steps: (1) find surrounding voxels at L resolution levels and hash their vertices; (2) look up trainable F-dim feature vectors by hashed keys; (3) linearly interpolate the vectors based on the coordinate's position; (4) concatenate features from all levels plus extras like viewing direction; (5) feed the result to the network for RGB and density. Gradients flow through the interpolation back into the feature vectors.

**Hash collisions are not explicitly handled**: multiple vertices may map to the same feature, but vertices most important to the output have the highest gradients and automatically dominate that feature's optimization. T, F, and L control the quality/memory/performance tradeoff. Viewing direction is encoded with **spherical harmonics**.

Nerfstudio's implementation covers the major ideas but differs in details (LR schedulers, sampling hyper-parameters, camera gradient calculation).
