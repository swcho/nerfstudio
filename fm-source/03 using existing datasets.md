# Using Existing Datasets

Nerfstudio has built-in support for a number of datasets, downloadable with `ns-download-data`. Each built-in dataset is ready to use with various Nerfstudio methods (e.g. the default Nerfacto).

## Examples

```bash
# All scenes from the Blender dataset, including the classic Lego model
ns-download-data blender

# Subset of data used in the SIGGRAPH 2023 Nerfstudio paper
ns-download-data nerfstudio --capture-name nerfstudio-dataset

# Room-scale scenes from the EyefulTower dataset at different resolutions
pip install awscli   # EyefulTower downloads require the AWS CLI
ns-download-data eyefultower --capture-name riverview seating_area apartment --resolution-name jpeg_1k jpeg_2k

# Full D-NeRF dataset of dynamic synthetic scenes
ns-download-data dnerf
```

## Dataset summary

The built-in datasets differ dramatically along axes such as photorealism (synthetic vs real), dynamic range (LDR vs HDR), scale (number of images), and resolution:

* **Blender** — synthetic, LDR, 8 scenes, 250–999 images per scene, resolution < 1000 px. The classic NeRF benchmark.
* **D-NeRF** — synthetic dynamic scenes, LDR, 8 scenes, < 250 images, low resolution.
* **EyefulTower** — real, both LDR and **HDR** (the only listed dataset with HDR), 11 scenes, from 250 up to ≥ 4000 images, resolutions from 1–2k up to ≥ 8000 px.
* **Mill 19** — real, LDR, 2 scenes, 1000–3999 images, 4000–7999 px.
* **NeRF-OSR** — real outdoor scene relighting dataset, 9 scenes, wide range of image counts.
* **Nerfstudio** — real captures, 18 scenes, up to 3999 images, up to 2000–3999 px.
* **PhotoTourism** — real landmark photo collections, 10 scenes, 250–3999 images.
* **Record3D** — real, 1 scene, 1000–3999 images, < 1000 px.
* **SDFStudio** — mixed synthetic and real, 45 scenes (largest scene count).
* **sitcoms3D** — real, 10 scenes, < 250 images.

These datasets are commonly used as baselines to evaluate new research in novel view synthesis, as in the original Nerfstudio paper.
