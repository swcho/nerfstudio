# VFX Integration (Blender, Maya, Unreal) and VR Rendering

## Blender VFX add-on

The add-on (Blender ≥ 3.0, `nerfstudio_blender.py`, installed via Edit → Preferences → Add-Ons, appears in Render Properties) enables compositing with Nerfstudio renders:

* **Generate a Nerfstudio camera path JSON from the Blender camera**, and conversely **import a Nerfstudio JSON as a baked Blender camera**.
* Workflow: export a mesh or point cloud of the NeRF from Nerfstudio (keep `save_world_frame` False to preserve the coordinate system), import it into Blender as an invisible reference, transform it to fit the scene, then generate `camera_path_blender.json` and render with Nerfstudio. To hide the reference in Blender renders, use "Shadow Catcher" (keeps shadows; needs cycles) or hide it from the render; enable "Transparent" film so the Blender layer composites over the NeRF render.
* Dynamic FOV from the Blender camera is matched in the Nerfstudio render. Perspective, equirectangular, VR180, and omnidirectional stereo cameras are supported; **fisheye and orthographic are not**. Gaussian Splatting scenes are supported, but equirectangular/VR rendering is not available for splats.
* **Compositing NeRF objects into NeRF environments**: crop the NeRF object in the viewer, export the cropped mesh, generate separate camera paths for environment and object, add the crop parameters into the object's camera path JSON, render RGB plus an **accumulation render** (`--rendered-output-names accumulation`) as an alpha mask, and composite (e.g. Premiere "Track Matte Key" with Matte Luma). Convert videos to image sequences to keep frames in sync.
* Implementation: iterates the frame sequence, computing the camera's 4×4 world matrix relative to the NeRF representation's transform per frame (so the NeRF can be repositioned/rotated/scaled/animated in Blender), plus per-frame FOV; keyframes are **baked** (every frame is a keyframe) so interpolation cannot differ between tools.

## Autodesk Maya plug-in

Same concept for Maya (`nerfplugin_maya.py`, loaded via the Plugin Manager, adds a "Nerfstudio" shelf): store a NeRF **mesh** representation and a camera, generate a camera path JSON for Nerfstudio, or create a Maya camera from a Nerfstudio JSON. Only **perspective cameras** are supported; any renderer works (Arnold, Redshift). Maya uses a **Y-up** coordinate system while Nerfstudio is **Z-up**, so the plugin applies the conversion with the transformation matrix. The exported render runs at **24 fps**. Don't delete the history of the NeRF mesh representation.

## Unreal Engine (Volinga)

Nerfstudio models can be used in Unreal Engine by converting them to an **NVOL** file — a new standard format for storing NeRFs fast and efficiently. NVOL currently only supports the **Volinga model** (based on nerfacto): install the Volinga extension, train with `ns-train volinga --data ... --vis viewer`, then drag the checkpoint (.ckpt) into **Volinga Suite** to export NVOL.

## VR video rendering (VR180 and Omni-Directional Stereo)

Stereo equirectangular rendering is supported as Nerfstudio camera types for video/image rendering:

* **Omni-directional stereo (ODS, 360 VR)**: two equirectangular renders stacked **vertically**, one per eye; may introduce slight depth distortion for close objects. Per-eye aspect ratio must be **2:1** (e.g. 4096×2048).
* **VR180**: two 180° equirectangular renders stacked **horizontally**; front-facing content only. Per-eye aspect ratio must be **1:1** (e.g. 4096×4096).

Setup: the NeRF scene must be scaled approximately **true to life** so depth and IPD are correct — use the Blender add-on with an imported point cloud and a real-size reference object (e.g. a 1m cube) to set the scale, place the camera at eye level, then set `camera_type` to `omnidirectional` or `vr180` in the camera path JSON with the per-eye `render_width`/`render_height`. The default **IPD is 64 mm** (variable `vr_ipd` in `cameras.py`), accurate only when the scene is true to scale. If depth looks too close/expanded the scale is too small; if there is almost no depth, it is too large — test with low-res or single-frame renders first. Nerfstudio renders the left eye, then the right eye, then stacks them; render times grow with the doubled high-res views.
