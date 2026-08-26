# Specialized NeRF Variants

Other third-party methods and representations supported through nerfstudio.

## TensoRF (Tensorial Radiance Fields)

Models the radiance field as a **4D tensor** (3D voxel grid with per-voxel multi-channel features) and **factorizes** it into compact low-rank tensor components: **CP decomposition** (rank-one components with compact vectors) or **Vector-Matrix (VM) decomposition** (vector and matrix factors). This greatly reduces memory versus directly optimizing per-voxel features (Plenoxels, PlenOctrees). CP gives fast reconstruction with a smaller model than NeRF; VM gives even better rendering quality while staying compact. Run: `ns-train tensorf`.

## K-Planes

**Explicit radiance fields in space, time, and appearance** — a unified model for static (k=3) and dynamic (k=4) scenes using **k-choose-2 planes** (e.g. 6 planes for 4D). Querying a position means querying each plane with interpolation and **multiplying** the resulting features. This space/time factorization keeps memory low and is flexible for priors and regularizers. Supports hybrid (small MLP) and fully explicit models via the `linear_decoder` key. Configs: `kplanes` (Blender, ~4GB), `kplanes-dynamic` (D-NeRF monocular dynamic, ~5GB). Install: `pip install kplanes-nerfstudio`.

## NeRFPlayer

**Streamable dynamic scene representation** with decomposed neural radiance fields: the 4D spatiotemporal space is decomposed by temporal characteristics — points get probabilities of being **static, deforming, or new** areas, each represented and regularized by a separate field — plus a hybrid-representation feature streaming scheme. Variants: `nerfplayer-nerfacto` and `nerfplayer-ngp` (instant-ngp-bounded backbone).

## Tetra-NeRF

Represents the radiance field using **tetrahedra**: the SfM input point cloud is **triangulated** (Delaunay), rays are sampled, and **barycentric interpolation** of tetrahedra vertices produces features passed to a shallow MLP for density and color. Requires CUDA, OptiX, CGAL. Variants: `tetra-nerf-original` (paper, ~18GB) and `tetra-nerf` (different sampler — faster and better, ~16GB); memory depends on point cloud size.

## PyNeRF (Pyramidal NeRF)

A **fast anti-aliasing strategy**: NeRF is scale-unaware because it reasons about point samples instead of volumes, degrading when camera distances vary. PyNeRF trains a **pyramid of NeRFs** dividing the scene at different resolutions — coarse NeRFs for far-away samples, finer NeRFs for close-ups. Configs: `pynerf` (outdoor, proposal network), `pynerf-synthetic` (proposal network), `pynerf-occupancy-grid` (multiscale Blender). Proposal sampling suits real-world scenes; occupancy grids suit single-object synthetic scenes.

## Zip-NeRF

Combines **mip-NeRF 360's overall framework with Instant-NGP's featurization**: each pixel corresponds to a cone, and a set of **multisamples** approximates the conical frustum shape for the grid-based encoding. It also presents an alternative interlevel loss that is continuous and smooth with respect to distance along the ray, preventing **z-aliasing**. Run: `ns-train zipnerf`.

## BioNeRF

**Biologically plausible** NeRF: fuses inputs from two parallel networks into a **memory-like structure** (via density, color, memory, and modulation "cognitive filters"), mimicking pyramidal cells' use of contextual information. The memory provides context combined into two subsequent blocks — one producing volumetric densities, the other colors. Achieves e.g. 31.45 avg PSNR on Blender and 27.01 on LLFF.

## SeaThru-NeRF

NeRF for **subsea scenes**: differentiates between solid objects and the **water medium** — both object colors and medium colors of samples contribute to the final pixel color. Five scene parameters: object density, object color, backscatter density, attenuation density, and medium color, predicted by **two separate networks** (plus a proposal network). Render options include `rgb` (normal), `J` (clear scene, water removed), `direct`, `bs` (backscatter), `depth`, and `accumulation`. Variants: `seathru-nerf` (~23GB) and `seathru-nerf-lite` (~7GB, runs on 8GB GPUs).

## Nerfbusters

Removes **ghostly artifacts (floaters)** from casually captured NeRFs. It proposes: an evaluation protocol with **two camera trajectories** (one for training, one for evaluation — instead of the common every-8th-frame split, which doesn't measure quality away from training views); a learned **local 3D diffusion prior** that denoises binarized density cubes; and a **density score distillation sampling (DSDS)** loss that penalizes densities where the diffusion model predicts empty voxels and pushes densities above a target where it predicts occupied ones. A **visibility loss** supervises densities to be low where no training view sees them, enabling stepping outside the training frustums.

## NeRFtoGSandBack (nerf2gs2nerf)

Converts **back and forth between NeRF and Gaussian Splats** to get the best of both. **NeRF-SH** (modified Nerfacto) predicts spherical harmonics per RGB channel; **NeRFGS** exports a point cloud from rendered depth with SH coefficients and density, initializing isotropic gaussians whose scale is half the average distance to the three nearest neighbors (clipped at the 0.8 quantile), then fine-tunes with reduced learning rates. **GSNeRF** renders training views from (possibly edited) splats to build a new dataset for retraining implicit models. Methods: `nerfsh` and `nerfgs`, exported via `ns-export-nerfsh`.

## SDFStudio

A separate framework **built on top of nerfstudio** implementing implicit **surface reconstruction** methods: UniSurf, VolSDF, NeuS, MonoSDF, Geo-NeuS, NeuS-acc, NeuS-facto, NeuralReconW, and variants. Nerfstudio core has integrated back **NeuS** and **NeuS-facto**.
