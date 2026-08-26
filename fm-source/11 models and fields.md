# Models and Fields

## Models

A Model is what you usually think of when you think of a NeRF paper — "Model" and "Method" are often used interchangeably, and implemented methods typically only change the model code. At a high level, a model takes regions of space described by `RayBundle` objects, samples points along the rays, and returns rendered values for each ray.

### Functions to implement

```python
class Model:
    config: ModelConfig  # typed config gives Python autocomplete

    def populate_modules(self):       # set Fields, Ray Samplers, Colliders, Renderers, Losses, Metrics
    def get_param_groups(self):       # parameter groups for the optimizers
    def get_training_callbacks(self): # e.g. updating the density grid for Instant NGP
    def get_outputs(self, ray_bundle):  # process a RayBundle → RayOutputs per ray
    def get_metrics_dict(self, outputs, batch):  # metrics plotted to comet/wandb/tensorboard
    def get_loss_dict(self, outputs, batch, metrics_dict):  # losses to be summed
    def get_image_metrics_and_images(self, outputs, batch): # images + metrics to plot (apply colormaps)
```

### Pythonic configs

The config system shines with models. `NerfactoModelConfig` exposes options like `near_plane` (0.05), `far_plane` (1000), `background_color` ("last_sample"), `num_proposal_samples_per_ray` (64,), `num_nerf_samples_per_ray` (64), `proposal` iteration/annealing settings, `interlevel_loss_mult`, `distortion_loss_mult` (0.002), and `use_average_appearance_embedding`. Adding a value to the config automatically makes it appear in `ns-train nerfacto --help`, and the typed `config:` attribute gives autocomplete and static checking (`self.config.<value>` inside `populate_modules`).

## Fields

A **Field** associates a region of space with a quantity. Typically the input is a 3D location and viewing direction, and the output is **density and color**. Fields are used in every model.

```python
class Field(nn.Module):
    @abstractmethod
    def get_density(self, ray_samples) -> (density, density_embedding): ...
    @abstractmethod
    def get_outputs(self, ray_samples, density_embedding) -> Dict[FieldHeadNames, Tensor]: ...

    def forward(self, ray_samples):
        density, density_embedding = self.get_density(ray_samples)
        field_outputs = self.get_outputs(ray_samples, density_embedding=density_embedding)
        field_outputs[FieldHeadNames.DENSITY] = density
        return field_outputs
```

`get_density` is called for every field, followed by `get_outputs` — implement `get_outputs` to return custom data. `FieldHead`s produce correctly-dimensioned outputs; `FieldHeadNames` includes RGB, SH, DENSITY, UNCERTAINTY, TRANSIENT_RGB, TRANSIENT_DENSITY, and SEMANTICS (e.g. `SemanticNerfField`). When only density is needed there is a helper `density_fn(positions)`.

### Frustums instead of positions

To query a **region** of space rather than a point, the `RaySamples` data structure contains **Frustums**: `origins`, `directions`, `starts`, `ends`, and `pixel_area` (projected pixel area at distance 1 from the origin). This enables methods like **Mip-NeRF** to be implemented in the framework.
