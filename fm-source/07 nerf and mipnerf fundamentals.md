# NeRF and Mip-NeRF Fundamentals

## NeRF (Neural Radiance Fields)

Original paper: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis" (Mildenhall, Srinivasan, Tancik et al., 2020). Run with `ns-train vanilla-nerf`.

For most tasks the original NeRF is not a good choice, but it is useful to understand since most follow-ups have a similar structure — and it doesn't require CUDA, so you can step through the code with a debugger without a GPU.

The goal is to optimize a volumetric representation of a scene, rendered from novel viewpoints, from a set of images with associated camera poses.

### Assumptions (breaking them causes failures or artifacts)

* Camera poses are known
* The scene is static
* Scene appearance is constant (e.g. exposure doesn't change)
* Dense input capture (each scene point visible in multiple images)

### Field representation

A NeRF is a volumetric representation encoded in a neural network — **not a mesh, not voxels**. For each point in space it represents a **view-dependent radiance**: each point has a **density** (how transparent/opaque it is) and a view-dependent **color**. In nerfstudio, coarse and fine `NeRFField`s are instantiated with position and direction encodings.

**Positional encoding**: input coordinates (x, y, z, θ, φ) are encoded to a higher-dimensional space before entering the network — necessary for the network to represent fine details. Nerfstudio's `NeRFEncoding` uses e.g. 10 frequencies for positions and 4 for directions.

### Rendering

A ray is projected from the target pixel and points are evaluated along it; classic **volumetric rendering** (Kajiya 1984) composites the points into a predicted color — similar to layering objects of varying opacity in Photoshop, but accounting for spacing between points. Besides RGB, other outputs such as depth and semantics can be rendered. The `RGBRenderer` combines per-sample RGB with weights computed from densities (`ray_samples.get_weights(density)`).

### Sampling

NeRF uses **hierarchical sampling**: first a **uniform sampler** distributes samples evenly within a predefined distance range; the initial render produces per-sample **weights** correlating with importance. A **PDF sampler** then generates new samples biased toward high-weight regions — in practice near object surfaces. Stratified samples are used during optimization, unmodified samples at inference.

For unbounded scenes the original paper used NDC warping with a linear-in-disparity sampler; nerfstudio does not support NDC — use **spatial distortions** instead.

### Benchmarks

On Blender synthetic, nerfstudio's vanilla NeRF averages **32.14 PSNR**, ahead of TF NeRF (31.04) and JaxNeRF (31.69).

## Mip-NeRF

"A Multiscale Representation for Anti-Aliasing Neural Radiance Fields." Run with `ns-train mipnerf`.

The primary modification is in the **encoding**: the Positional Encoding (PE) is replaced with an **Integrated Positional Encoding (IPE)** that takes the **size of the sample** into account (cone/frustum rather than point). With this change, the **same mip-NeRF field can be used for both the coarse and fine steps** of the rendering hierarchy, and aliasing is reduced when viewing scenes at multiple scales.
