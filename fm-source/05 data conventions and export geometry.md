# Data Conventions and Exporting Geometry

## Coordinate conventions

* **Camera/view space**: Nerfstudio uses the **OpenGL/Blender (original NeRF) convention** — +X right, +Y up, +Z pointing back and away from the camera; **-Z is the look-at direction**. This differs from the COLMAP/OpenCV convention, where Y and Z are flipped but +X is the same.
* **World space**: the up vector is **+Z**; the XY plane is parallel to the ground plane. In the viewer, red/green/blue vectors correspond to X/Y/Z.
* **Pixel coordinates**: coordinates correspond to the **centers of pixels** (a ray for pixel (0,0) goes through the center of that pixel) — this follows graphics conventions and is distinct from OpenCV, where corners align with the pixel coordinate.

## Dataset format (`transforms.json`)

The nerfstudio data format is similar to Instant NGP's. Shared **camera intrinsics** go at the top of the file: `camera_model` (OPENCV or OPENCV_FISHEYE), focal lengths `fl_x`/`fl_y`, principal point `cx`/`cy`, image size `w`/`h`, radial distortion `k1`–`k4`, and tangential distortion `p1`/`p2` (OPENCV only). Per-frame intrinsics can instead be defined in the `frames` field, but then all images must define that field; per-frame `camera_model` is not supported.

**Camera extrinsics** are given per frame as a 4×4 `transform_matrix`: the first 3 columns are the +X, +Y, +Z axes defining the camera orientation, the 4th column is the origin, and the last row makes it compatible with homogeneous coordinates.

### Depth images

For depth supervision, provide a `depth_file_path` per frame and use a method supporting depth losses (e.g. **depth-nerfacto**). Depths are assumed 16-bit or 32-bit in **millimeters** (consistent with Polyform); zero means unknown depth. The scaling is adjustable via `depth_unit_scale_factor` in `NerfstudioDataParserConfig` (default 1e-3, mm→m). Depth images are resized to match RGB by default.

### Masks

Regions to exclude from training (e.g. moving people) can be masked with a per-frame `mask_path`. Requirements: 1 channel with only black and white pixels; same resolution as the training image; **black = ignore**; if used, **all** images must have a mask. Warning: the current masking implementation is inefficient and causes large memory allocations.

## Exporting geometry (`ns-export`)

Point clouds are exported as `.ply`; textured meshes as `.obj`.

### Meshes

1. **TSDF Fusion** — meshing algorithm using depth maps to extract a surface; **works for all models**:
   `ns-export tsdf --load-config CONFIG.yml --output-dir OUTPUT_DIR`
2. **Poisson surface reconstruction** — **highest quality meshes**, but only works with a model that computes or predicts normals (e.g. nerfacto). Train with `ns-train nerfacto --pipeline.model.predict-normals True`, then `ns-export poisson --load-config CONFIG.yml --output-dir OUTPUT_DIR`.

### Point cloud

`ns-export pointcloud --help`. Other export methods: `ns-export --help`.

### Texturing an existing mesh with NeRF

You can simplify/smooth a mesh offline and texture it with NeRF via `python nerfstudio/scripts/texture.py --load-config CONFIG.yml --input-mesh-filename FILENAME --output-dir OUTPUT_DIR` (supports any mesh filetype PyMeshLab can read). Export dependencies: **xatlas-python** (UV unwrapping) and **pymeshlab** (face reduction).
