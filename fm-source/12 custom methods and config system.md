# Adding Custom Methods and the Config System

## Adding a new method

Nerfstudio's vision is for users to create a **separate repository that imports nerfstudio and overrides pipeline components** for the new approach; generally-useful core changes should be PRed. The `nerfstudio-method-template` repo is a minimal guide; **LERF** is a good real example.

Recommended file structure: a `my_method/` package with `__init__.py`, `my_config.py`, and optional `custom_pipeline.py`, `custom_model.py`, `custom_field.py`, `custom_datamanager.py`, `custom_dataparser.py`, plus a `pyproject.toml`.

### Registering a custom method

Define a `MethodSpecification` in your config file:

```python
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

MyMethod = MethodSpecification(
  config=TrainerConfig(method_name="my-method", pipeline=..., ...),
  description="Custom description",
)
```

Register it as a **`nerfstudio.method_configs` entrypoint** in `pyproject.toml`:

```toml
[project.entry-points.'nerfstudio.method_configs']
my-method = 'my_method.my_config:MyMethod'
```

then `pip install -e .`. Nerfstudio automatically finds registered methods for `ns-train`. During development you can skip packaging by setting `export NERFSTUDIO_METHOD_CONFIGS="my-method=my_method.my_config:MyMethod"` — the variable also accepts a function or a `MethodSpecification` subclass. Run with `ns-train my-method --data DATA_DIR`.

### Registering a custom dataparser

Same mechanism with `DataParserSpecification` and the **`nerfstudio.dataparser_configs`** entrypoint (or the `NERFSTUDIO_DATAPARSER_CONFIGS` environment variable).

Researchers can also add their method to the nerf.studio documentation: add a markdown page under `docs/nerfology/methods`, update the methods index and `docs/index.md`, and add an `ExternalMethod` entry to `nerfstudio/configs/external_methods.py`.

## Customizable configs

Dataclass configs let you plug in different permutations of models, dataloaders, and modules, and modify all parameters from a **typed CLI** powered by **tyro**. Base reusable components live in `nerfstudio/configs/base_config.py`; the top-level `Config` class stores all sub-configs needed for training.

### Creating new configs

For a new model, define a config that points at the model class, wrapping `_target` in a `field`:

```python
@dataclass
class NerfactoModelConfig(ModelConfig):
    _target: Type = field(default_factory=lambda: NerfactoModel)

class NerfactoModel(Model):
    config: NerfactoModelConfig  # enables typed autocomplete
```

### Updating method configs

Implemented model configs are housed in `nerfstudio/configs/method_configs.py` — add your `Config` to the `method_configs` dictionary (overriding pipeline and optimizers as needed, e.g. Adam optimizers for `proposal_networks` and `fields` with lr=1e-2) and add a description to the `descriptions` dictionary.

### Modifying from the CLI

* `ns-train --help` — list all models
* `ns-train {METHOD_NAME} --help` — list all configurable parameters of a method
* `ns-train {METHOD_NAME} --data DATA_PATH` — change the dataset
* `ns-train {METHOD_NAME} --vis viewer` — enable the viewer
* `ns-train {METHOD_NAME} {DATA_PARSER} --help` — dataparser options
* `ns-train {METHOD_NAME} --vis viewer {DATA_PARSER} --scale-factor 0.5` — dataparser attributes go at the **end** of the command
