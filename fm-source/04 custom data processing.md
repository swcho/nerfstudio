# Using Custom Data

To train on self-captured data, the data must be processed into the nerfstudio format — camera poses are needed for each image:

```bash
ns-process-data {video,images,polycam,record3d} --data {DATA_PATH} --output-dir {PROCESSED_DATA_DIR}
```

Supported custom data types and their pose sources:

* 📷 **Images / Video / 360 data** — any camera, requires **COLMAP** (slow 🐢)
* 📱 **Polycam** — iOS with LiDAR (fast 🐇)
* 📱 **KIRI Engine** — iOS or Android, no LiDAR needed
* 📱 **Record3D** — iOS with LiDAR
* 📱 **Spectacular AI** — iPhone, OAK-D, RealSense D455/D435i, Azure Kinect DK
* 🖥 **Metashape**, **RealityCapture**, **ODM** — any camera
* 👓 **Aria** — Project Aria glasses

## Images or video (COLMAP path)

`ns-process-data {images, video}` uses COLMAP and FFmpeg — both must be installed. COLMAP can be finicky: capture overlapping, non-blurry images. Recommended COLMAP install: `conda install -c conda-forge colmap`, or via vcpkg.

Then train: `ns-train nerfacto --data {PROCESSED_DATA_DIR}`.

Separate train/eval data (as suggested in Nerfbusters): `ns-process-data {images, video} --data {DATA_PATH} --eval-data {EVAL_DATA_PATH} ...`, then `ns-train nerfacto --data ... nerfstudio-data --eval-mode filename`.

## App-based captures

* **Polycam**: avoids COLMAP; poses are globally optimized, more robust to drift than ARKit/SLAM. Requires a LiDAR-enabled iPhone/iPad and Developer mode enabled in the app. Capture in LiDAR or Room mode, export `raw data` as a .zip, then `ns-process-data polycam --data {OUTPUT_FILE.zip} ...`. Reduce motion blur; manual shutter mode helps.
* **KIRI Engine**: enable Developer Mode, capture with `Camera pose` option, export (sent by email). **No `ns-process-data` step is needed** — train directly on the unzipped output.
* **Record3D**: uses iPhone (≥ 12 Pro) LiDAR for poses, no COLMAP. Export with the **EXR + JPG sequence** format, then `ns-process-data record3d --data {dir} ...`. A zipped PLY point-cloud sequence can be added with `--ply {ply directory}` — useful to avoid random initialization when training gaussian splats; `--voxel-size` (default 0.8) controls downsampling sparsity (higher = sparser).
* **Spectacular AI**: SDK records IMU data fused with camera and LiDAR/ToF (VISLAM) — more robust than image-only methods for monotonic environments, fast motions, narrow FoV. Install `pip install spectacularAI[full]` + FFmpeg, then `sai-cli process {data dir} --preview3d --key_frame_distance=0.05 {output dir}` (0.05 for small scans, 0.15 for room-sized; `--fast` trades quality for speed). No separate `ns-process-data` step.

## Desktop tools

* **Metashape**: align photos, export cameras as XML, then `ns-process-data metashape --data {dir} --xml {file} ...`. All images must use the same sensor type.
* **RealityCapture**: align images, export Internal/External camera parameters as CSV, then `ns-process-data realitycapture --data {dir} --csv {file} ...`.
* **ODM**: process with OpenDroneMap, then `ns-process-data odm --data /path/to/dataset ...`. Same camera for all images/videos.
* **Aria**: `pip install projectaria-tools'[all]'`, download a VRS file, run Machine Perception Services for poses, then `ns-process-data aria --vrs-file ... --mps-data-dir ...`.

## 360 (equirectangular) data

For 360 cameras such as Insta360:

```bash
ns-process-data {images,video} --camera-type equirectangular --images-per-equirect {8, 14} \
  --num-frames-target N --crop-factor {top bottom left right} --data {dir} --output-dir {out}
```

* `--images-per-equirect`: number of perspective images sampled per equirectangular image; 8 is usually sufficient, use 14 if COLMAP struggles or detail is lacking.
* `--num-frames-target`: recommended ≈ 3 × seconds of video (e.g. 90 for a 30 s video).
* `--crop-factor` removes unwanted regions — for videos, hold the camera above your head and crop the bottom ~20% (`--crop-factor 0 0.2 0 0`, or shorthand `--crop-bottom 0.2`) to remove the capturer's hand/head.
