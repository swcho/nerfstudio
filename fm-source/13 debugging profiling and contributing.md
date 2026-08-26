# Debugging Tools, Benchmarking, and Contributing

## Profiling

Three profiling options:

* **In-house profiler** — enabled by default, prints at program termination (disable with `--logging.no-enable-profiler`). It computes the average total execution time of any function decorated with `@profiler.time_function`; code blocks can also be timed with `with profiler.time_function("name"):`. Use it for specific/individual functions.
* **PyTorch profiler** — enable with `--logging.profiler=pytorch` to track memory and CUDA kernel launches for selected steps (run once with `CUDA_LAUNCH_BLOCKING=1` and once with 0). Trace files are stored in `{MODEL_OUTPUT}/profiler_traces` and load in Chrome via `chrome://tracing`.
* **PySpy** — profile the entire codebase (`pip install py-spy`). Generate a flame graph with `py-spy record -o out.svg $program` or a live top-down view with `py-spy top $program`. When defining `$program`, an extra `--` is needed before the program's own arguments.

## Local writer

The `LocalWriter` outputs numerical stats to the terminal. Config attributes: `enable`, `stats_to_track` (choose from the `EventName` enum in `utils/writer.py`), and `max_log_size` (how many lines to print; 0 prints everything without deleting previous lines). CLI examples: `--logging.local-writer.no-enable` (disable), `--logging.local-writer.max-log-size=0` (disable line wrapping). To track a new stat, add it to the `EventName` enum, put the value into `EVENT_STORAGE` (e.g. `put_scalar`), and add the enum to `stats_to_track`.

## Benchmarking workflow

Benchmark a NeRF against the standard **Blender dataset**:

1. **Train** on each Blender object: `./nerfstudio/scripts/benchmarking/launch_train_blender.sh -m {METHOD_NAME} [-s] [-v {VIS}] [{GPU_LIST}]` — `-m` method name, `-s` single job per GPU, `-v` visualizer (default wandb), and an optional space-separated GPU list (auto-detects free GPUs if empty). Checkpoints are saved per object with the experiment name and timestamp.
2. **Evaluate**: `./nerfstudio/scripts/benchmarking/launch_eval_blender.sh -m {METHOD} -o outputs/ -t {timestamp} [{GPU_LIST}]` — runs benchmarking across all Blender objects concurrently, computing PSNR/FPS/other stats saved as JSON files in the output directory.

## Contributing

Bug fixes and documentation improvements are welcome as PRs; for larger features, discuss on Discord (`#contributing`) and open a GitHub issue first.

Tooling: **Ruff** (formatting & linting), **Pyright** (type checking), **pytest** (testing), **Sphinx** (docs), **Google style** docstrings, **eslint** (JS).

Setup: `pip install -e .[dev]`, `pip install -e .[docs]`, `pre-commit install` (pandoc may also be needed).

Before opening a PR, run **`ns-dev-test`**, which performs: formatting/linting, type checking, pytests, a documentation build, and automatic licensing headers. All checks must pass for a PR to be reviewed/merged.

Docs are built with `python nerfstudio/scripts/docs/build_docs.py`; `sphinx-autobuild` can rebuild on save. Jupyter notebooks are supported in the docs with cell tags `# HIDDEN` (hide code and output), `# COLLAPSED` (collapse code in a dropdown), and `# OUTPUT_ONLY`.
