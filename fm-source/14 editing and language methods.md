# Editing and Language-Embedded Methods

Third-party methods built on nerfstudio for scene editing, language grounding, and generation.

## Instruct-NeRF2NeRF (in2n)

Edits NeRF scenes with **text instructions**. Given a NeRF and its training images, an image-conditioned diffusion model (**InstructPix2Pix**) iteratively edits the input images while the scene keeps optimizing, yielding a 3D scene that respects the edit. Variants: `in2n` (~15GB, best), `in2n-small` (half precision, ~12GB), `in2n-tiny` (no LPIPS, ~10GB).

**Iterative Dataset Update**: (1) render an image at a training viewpoint; (2) edit it with InstructPix2Pix conditioned on the *original unedited* image plus the text instruction, denoising a noised version of the current render; (3) replace the dataset image with the edit; (4) continue NeRF training. Rays are sampled across the whole dataset, mixing supervision from edited and unedited images. Early edits are inconsistent across views, but since edits use current renders, they converge to a globally consistent scene. The method can be seen as a variant of the SDS loss (DreamFusion), with higher quality and more stable optimization.

## Instruct-GS2GS (igs2gs)

Same idea applied to **3D Gaussian Splatting** scenes (`ns-train igs2gs`, ~15GB). Datasets must be COLMAP-processed. Start from a GS scene trained for 20k iterations, then `ns-train igs2gs --data ... --load-dir ... --pipeline.prompt {"prompt"} --pipeline.guidance-scale 12.5 --pipeline.image-guidance-scale 1.5`. Unlike in2n, it edits **all** training images individually every 2.5k iterations, then trains the GS with L1 and LPIPS losses, up to a maximum of 27.5k iterations (training usually stops when the edit converges).

## SIGNeRF

**Controlled generative editing** of NeRF scenes ("Scene Integrated Generation"). Key insight: depth-conditioned diffusion models (**ControlNet**) can generate 3D-consistent views when asked for a **grid of images** instead of single views — no iterative optimization needed. A multi-view **reference sheet** of modified images is generated once (depth-conditioned inpainting, mask-controlled region), then the image collection is updated consistently and the NeRF is fine-tuned in one go. Object generation is controlled by placing a mesh into the scene. Fully integrated into the viser interface. Variants: `signerf` (40min, best), `signerf_nerfacto` (20min). Requires Stable Diffusion Web UI.

## LERF (Language Embedded Radiance Fields)

Grounds **CLIP vectors volumetrically inside NeRF**, enabling natural-language queries in 3D — pixel-aligned queries of distilled 3D CLIP embeddings without region proposals, masks, or fine-tuning, supporting long-tail open-vocabulary queries. Supervision is **multi-scale**: an image pyramid of CLIP features is precomputed per view, and sampled rays are supervised by interpolating within the pyramid. **DINO features regularize** CLIP features (inspired by Distilled Feature Fields), improving object boundaries. After optimization, 3D relevancy maps for text queries render interactively (set output to `relevancy_0` and type the query in "LERF Positives"). Variants: `lerf-big` (ViT-L/14, ~22GB), `lerf` (ViT-B/16, ~15GB), `lerf-lite` (~8GB, runs on a 2080).

## LiveScene

The first scene-level **language-embedded interactive radiance field** — reconstructs and controls complex physical scenes with multiple articulated objects via natural language. It decomposes the interactive scene into **multiple local deformable fields** (one per interactive object) in a multi-scale factorized 4D space, trained with a feature repulsion loss; an **interaction-aware language embedding** localizes and controls objects under different interactive states. Ships with two dataparsers: `livescene-sim` (OmniSim synthetic) and `livescene-real` (InterReal); together 28 subsets, 70 interactive objects, 2M samples.

## OpenNeRF

**Open-set 3D neural scene segmentation** with pixel-wise VLM features. Unlike LERF's global CLIP features, OpenNeRF encodes **pixel-aligned** VLM features directly in the NeRF, giving a simpler architecture **without DINO regularization**, and exploits NeRF's novel-view rendering to extract features from poorly observed areas. On Replica point-cloud segmentation it beats LERF and OpenScene by ≥ +4.9 mIoU.

## Generfacto

"Generate 3D models from text" — combines generative 3D with nerfstudio's NeRF methods: `ns-train generfacto --prompt "a high quality photo of a pineapple"`. Two diffusion backends: **DeepFloyd IF** (default — trains faster, better results, requires signing a HuggingFace license) and **Stable Diffusion** (`--pipeline.model.diffusion_model`). Install with `pip install -e .[gen]`. 30k steps ≈ 1 hour with DeepFloyd, ~4 hours with Stable Diffusion.
