# Viewer Usage and Customization

## Using the viewer

The web-based viewer shows training in real-time and creates content videos from trained models. It launches automatically with each `ns-train` run, or separately via `ns-viewer`. It is built on **Viser** with ThreeJS, packaged as a ReactJS app; the client connects via a **websocket** to a server on your machine.

* **Local**: open the printed link, typically `http://localhost:7007`.
* **Over SSH**: tunnel the port with `ssh -L 7007:127.0.0.1:7007 <user>@<training-host>`; the port stays open as long as that ssh session is alive. If the port is taken, switch with `--viewer.websocket-port`.
* **Share link**: generate a publicly accessible URL from the "Share:" icon in the GUI or with `--viewer.make-share-url True` — useful without SSH access, but adds latency.
* The **legacy viewer** (default before nerfstudio 1.0) can still be enabled with `--vis viewer_legacy`. It could also be self-hosted: a node/yarn app in `nerfstudio/viewer/app` running on port 4000, connected as `http://localhost:4000/?websocket_url=ws://localhost:7007`.

## Custom GUI elements

Custom viewer GUI elements can be defined in any `nn.Module` by creating instances of classes from `nerfstudio.viewer.viewer_elements` as **class variables**:

```python
from nerfstudio.viewer.viewer_elements import ViewerNumber

class MyClass(nn.Module):
    def __init__(self):
        self.custom_value = ViewerNumber(name="My Value", default_value=1.0)
```

* **Hierarchy**: the viewer recursively searches all `nn.Module` children of the base `Pipeline`; an element defined in `pipeline.model.field` appears in the "Custom/model/field" folder.
* **Reading**: access `element.value`. **Writing**: assign `element.value = x` — a convenient way to track values without wandb/tensorboard.
* **Callbacks**: pass `cb_hook=` to be called whenever a new value is available.
* **Thread safety**: values can change asynchronously to model execution — store the value once at the start of a forward pass and use the local copy afterwards. Updating module state during training can have unexpected side effects; condition on `self.training` if needed.
* Available elements: `ViewerButton`, `ViewerNumber`, `ViewerCheckbox`, `ViewerDropdown`, `ViewerSlider`, `ViewerText`, `ViewerVec3`, `ViewerRGB`.

## Python viewer control

`ViewerControl` (also from `viewer_elements`, instantiated as a class variable in an `nn.Module`) provides a Python interface to:

* **Get the camera**: `get_camera(height, width)` returns a `Cameras` object (or `None` if the viewer isn't connected) — from it you can read `camera_to_worlds` (3×4 extrinsics) and `generate_rays(camera_indices=0)`.
* **Set camera pose/FOV**: `set_pose(position=(1,1,1), look_at=(0,0,0), instant=False)` (`instant=False` animates smoothly), and set the scene crop.
* **Scene pointer callbacks**: register with `register_pointer_cb()` for `ViewerClick` (a click as a world-space ray from the camera origin through the click point) and `ViewerRectSelect` (rectangle defined by two corners in normalized OpenCV screen coordinates). Only **one** scene pointer callback can be active at a time — registering a new one unregisters the old. Use `unregister_pointer_cb()` when done, and optionally pass a `removed_cb` to restore GUI state (e.g. re-enable a disabled button) when the callback is removed. Click callbacks are asynchronous to training and can interrupt `get_outputs()`.
