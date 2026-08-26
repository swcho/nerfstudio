# Pipeline Architecture and DataManagers

## Why pipelines?

Nerfstudio's goal is for **any NeRF paper to be implementable as a Pipeline**. A Pipeline has two major components: the **DataManager** and the **Model**. The DataManager loads data and generates `RayBundle` (ray origins and viewing directions — model input for training and inference) and `RayGT` (ground truth needed only during training, e.g. GT pixel values for an L2 loss). The model renders rays into `RayOutputs`.

The Pipeline implements two main functions: `get_train_loss_dict(step)` and `get_eval_loss_dict(step)`. The `VanillaPipeline` simply routes data:

```python
ray_bundle, batch = self.datamanager.next_train(step)
model_outputs = self.model(ray_bundle)
metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
```

The VanillaPipeline works for most methods. A `DynamicBatchPipeline` is used with Instant-NGP to dynamically choose the number of rays per iteration.

## DataManagers

A DataManager batches and returns two things from an input dataset:

1. A **representation of viewpoint** — a `Cameras` object for splatting methods (`FullImageDataManager`) or a `RayBundle` for ray-sampling methods (`VanillaDataManager`).
2. A **dictionary of ground truth** — complete images for splatting; per-ray information for ray sampling.

Abstract methods to implement: `next_train(step)`, `next_eval(step)`, and `next_eval_image(step)`.

`VanillaDataManager` implements the standard logic of most NeRF papers: randomly sample training rays with corresponding GT. Its config includes the `dataparser`, `train_num_rays_per_batch` (default 1024), `eval_num_rays_per_batch`, numbers of images to sample from, and a `camera_optimizer`. `next_train` samples a batch of images, samples pixels from them, generates rays from the pixel indices via the ray generator, and returns the RayBundle plus batch.

### Disk caching for large datasets

As of January 2025, `FullImageDatamanager` and `ParallelImageDatamanager` support parallelized dataloading and loading from disk to avoid OOM on very large datasets:

* NeRF-based methods: `ns-train nerfacto --data ... --pipeline.datamanager.load-from-disk`
* Splatfacto: `ns-train splatfacto --data ... --pipeline.datamanager.cache-images disk`

### Migrating custom DataManagers (`custom_ray_processor`)

Methods that subclass a DataManager to attach extra data (e.g. LERF adding CLIP/DINO features in `next_train`) can support the new parallel features by subclassing `ParallelDataManager` and moving the customization into the **`custom_ray_processor()`** API. It receives a fully populated ray bundle (or Cameras object) and GT batch and can modify or extend them; it runs in a **background process** when disk caching is enabled.

Caveats: any member used inside `custom_ray_processor` must be **picklable** (background processes); compute new information inside the processor or cache only a subset — initialization over the whole dataset can still OOM, and GPU-heavy precomputation on the training GPU can be slow.

## DataParsers

A DataParser converts the various dataset formats into a common `DataparserOutputs` format, which is **lightweight** — filenames and meta information, later processed by actual PyTorch Datasets/Dataloaders. Fields include `image_filenames`, `cameras`, `alpha_color`, `scene_box`, `mask_filenames`, `metadata`, `dataparser_transform`, and `dataparser_scale`.

To add a new dataparser, implement one private method: `_generate_dataparser_outputs(split)`. The `NerfstudioDataParserConfig` exposes options such as `scale_factor`, `downscale_factor` (auto-chosen so max dimension < 1600px if unset), `scene_scale`, `orientation_method` ("pca", "up", "vertical", "none"), `center_method`, `auto_scale_poses` (fit poses in a ±1 box), `train_split_fraction` (default 0.9), and `depth_unit_scale_factor`.

The DataParser generates train and eval outputs depending on the `split` argument; because outputs share a common form, `InputDataset`s are plug-and-play, and components like `RayGenerator` (which generates RayBundles from camera and pixel indices) consume `dataparser_outputs.cameras`.
