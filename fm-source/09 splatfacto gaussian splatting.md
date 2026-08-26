# Splatfacto — Gaussian Splatting in Nerfstudio

## Overview

**3D Gaussian Splatting** (SIGGRAPH 2023, INRIA) represents radiance fields by explicitly storing a collection of 3D volumetric gaussians. Given a camera pose these are "splatted" (projected) onto a 2D image and **rasterized** to obtain per-pixel colors. Because rasterization is very fast on GPUs, it renders much faster than neural representations.

Nerfstudio calls its implementation **Splatfacto** to avoid confusion with the original paper — like Nerfacto, it is a blend of different gaussian splatting methodologies. It uses **gsplat** as the rasterization backend, an in-house re-implementation designed to be developer friendly (`pip install gsplat`; CUDA code compiles on first run — PyTorch 2.0 issues can be fixed by installing from source or upgrading to torch 2.1).

## Data

Gaussian splatting works much better initialized from pre-existing geometry, e.g. **SfM points from COLMAP**. COLMAP or `ns-process-data` datasets automatically save these points and initialize gaussians on them; other datasets initialize randomly. Because the method trains on **full images** instead of ray bundles, a new datamanager (`full_images_datamanager.py`) undistorts input images, caches them, and provides single images per train step.

## Running

`ns-train splatfacto --data <data>`. The splat can be viewed in the web viewer, loaded from checkpoints, rendered, and exported like NeRF methods. Variants: `splatfacto` (~6GB, fast) and `splatfacto-big` (~12GB, more gaussians, higher quality).

### Quality and regularization

* For quality over speed/size: decrease the alpha cull threshold and disable culling after 15k steps — `--pipeline.model.cull_alpha_thresh=0.005 --pipeline.model.continue_cull_post_densification=False`.
* Long spikey gaussians are a common artifact; **PhysGaussian's scale regularizer** encourages evenly shaped gaussians — enable with `--pipeline.model.use_scale_regularization True`.

### Exporting splats

Splats export as `.ply` via the viewer or `ns-export gaussian-splat --load-config <config> --output-dir exports/splat`. Only trained splats can be exported (not nerfacto). Compatible third-party viewers include Polycam, PlayCanvas SuperSplat, antimatter15 WebGL viewer, Spline, and mkkellogg's Three.js viewer.

### FAQ

* Mesh/point-cloud export from splats is **not** currently supported.
* Fisheye, equirectangular, orthographic rendering is **not** supported — gaussian rasterization assumes a perspective camera.

## Splatfacto-W (Splatfacto in the Wild)

A Nerfstudio implementation of Gaussian Splatting for **unconstrained photo collections** (e.g. PhotoTourism landmarks with varying appearance). Install via `pip install git+https://github.com/KevinXu02/splatfacto-w`, download data with `ns-download-data phototourism`, and train with `ns-train splatfacto-w --data [PATH]` using the nerf-w train/test tsv split. A lighter `splatfacto-w-light` variant works for general datasets without the nerf-w split (with the `colmap` dataparser for phototourism).

## Feature Splatting

Feature Splatting distills **SAM-enhanced CLIP features** into 3DGS via view-independent rasterization, enabling open-vocabulary 2D and 3D segmentation of gaussians directly in 3D space (`ns-train feature-splatting`, ~8GB). Compared to coarse CLIP features (as in LERF), it performs object-level masked average pooling to refine object boundaries; this implementation uses **MobileSAMv2** (faster than the original SAM) and image-level CLIP features, plus **DINOv2 features as joint supervision** to regularize internal object structure.

Because 3DGS is explicit, grouped gaussians are easy to manipulate. Supported editing primitives: rigid operations (floor estimation, translation, transparent-background highlighting, yaw rotation) and a non-rigid **sand-like melting** effect based on the Taichi MPM method.
