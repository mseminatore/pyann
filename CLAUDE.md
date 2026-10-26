# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyANN is a Python wrapper around the [libann](https://github.com/mseminatore/ann) C neural network library, exposing a Keras-like API. It has two layers:

- **C library** (`ann/` git submodule) — core neural network engine built as a shared library via CMake. CFFI bindings are declared in `src/pyann/_bindings/`.
- **Python package** (`src/pyann/`, src layout) — high-level API wrapping the C library. `Sequential` is the main entry point.

NumPy is optional — `utils/compat.py` provides fallback helpers that work with plain Python lists.

## Build & Test

```bash
# Build the native library and install in dev mode (scikit-build-core runs CMake)
pip install -e ".[dev]"

# Pass CMake options through CMAKE_ARGS, e.g. CUDA or a plain CPU build
CMAKE_ARGS="-DUSE_CUDA=ON" pip install -e .
CMAKE_ARGS="-DUSE_VULKAN=OFF" pip install -e .    # Linux/Windows default is AUTO
python tools/prepare_vulkan.py                     # vulkan_prebuilt/: shaders + headers (needs glslc)
CMAKE_ARGS="-DUSE_METAL=OFF -DUSE_ACCELERATE=OFF" pip install -e .

# Or build only the shared library into src/pyann/lib/ (no install)
python build_lib.py                  # defaults: no BLAS (see flags below)
python build_lib.py --accelerate --metal   # macOS
python build_lib.py --scipy-openblas       # Linux/Windows (pip install scipy-openblas32 first)
python build_lib.py --cblas | --blas | --cuda | --vulkan | --clean

# Run tests
pytest                                                    # all tests
pytest tests/test_layers.py                               # single file
pytest tests/test_layers.py::TestDense::test_basic_dense -v  # single test

# Build a wheel / sdist
pip wheel . --no-deps -w dist
python -m build --sdist
```

The `ann/` submodule must be checked out (`git submodule update --init`). Wheels default to Accelerate+Metal on macOS and the bundled `scipy-openblas32` OpenBLAS + Vulkan on Linux/Windows (see `[[tool.scikit-build.overrides]]` in `pyproject.toml`). CI (`.github/workflows/wheels.yml`) builds and tests wheels with cibuildwheel; `tests/test_backend.py` fails a wheel that lost its BLAS or GPU backend. `USE_VULKAN` (`ON`/`OFF`/`AUTO`) compiles libann's GLSL shaders with `glslc` and embeds them (the same steps as libann's CMakeLists.txt; keep the shader list in sync when bumping `ann/` — `tests/test_binding_coverage.py` fails when it differs from libann's list or `ann/shaders/vk/*.comp`), or, without glslc, uses `vulkan_prebuilt/` (compiled shader header + Vulkan C headers, written by `tools/prepare_vulkan.py`; git-ignored, shipped in the sdist via `sdist.include`). CI's `vulkan-prebuilt` job writes it once for the sdist and every wheel job, and `vulkan-gpu` runs the Linux wheel on lavapipe so the backend must actually start. The library opens the Vulkan loader at run time, so it has no link dependency on it and falls back to the CPU without a driver. `.github/workflows/blas-bench.yml` (manual) compares BLAS libraries using `benchmarks/`.

`build_lib.py` and plain CMake builds put the shared library in `src/pyann/lib/`; `pip install -e .` puts its build in site-packages. `_find_library()` in `_bindings/ffi.py` searches `PYANN_LIB_PATH` (a directory) first, then the package's directories in `pyann.__path__` order, so an editable install loads its own build ahead of a leftover `src/pyann/lib`. When a newer libann sits in another package directory it warns (`RuntimeWarning`) instead of loading it; rebuild or delete the stale one, or set `PYANN_LIB_PATH`. Tests: `tests/test_library_path.py`.

## Releasing

The PyPI distribution is `pylibann` (the name `pyann` is taken by an unrelated project); the import name is still `pyann`. Releases are tag-driven via `.github/workflows/wheels.yml`:

1. Set `version` in `pyproject.toml` (PEP 440) and merge to `main`.
2. Push a tag `vX.Y.Z`. The tag must equal the version or the `release-meta` job fails.
3. A pre-release tag (`vX.Y.ZrcN`) publishes to TestPyPI via the `testpypi` environment; a final tag publishes to PyPI via the `pypi` environment. Both use trusted publishing (no tokens); each index needs a publisher pointing at this repo, `wheels.yml`, and the matching environment name.

Try a release candidate on TestPyPI first and install it in a clean venv before tagging the real release.

## Conventions

- **Enum + string API**: Activations, losses, and optimizers accept both enum values (`Activation.RELU`) and strings (`"relu"`). Parsing helpers (`parse_optimizer`, `parse_loss`, `parse_activation`) handle conversion.
- **Error handling**: C errors are translated via `raise_for_error_code()` in `exceptions.py`. Python-side validation uses `ValueError`/`TypeError`. `_bindings/ffi.py` installs libann's error log callback when the library loads (otherwise libann prints errors to stderr); it records the last error per thread and logs it to the `pyann` logger at DEBUG. `raise_for_error_code` appends that error to the exception message when its code matches; for calls that signal failure by returning `NULL`, call `clear_c_error()` before and wrap the message in `c_error_message()`. The callback is process-wide, so it also replaces the stderr output of any other code in the process that uses the same libann.
- **Test style**: pytest with class-based grouping (`class TestDense:`). Each class covers one component.
- **Type hints**: All public APIs use type hints. `TYPE_CHECKING` guard for numpy imports.
- **C binding prefixes**: `ann_*` for network functions, `tensor_*` for tensor operations (in `_bindings/` CFFI definitions).
- **Python enum values** (`IntEnum`) must match C enum ordinals exactly.
- **Decoder layers and noise**: `UpSample2D` and `Reshape` (in `core/layers.py`) map to `ann_add_upsample2d`/`ann_add_reshape`. Only layers before the first `Reshape` need an image input (`_uses_images`). Targets go through `_targets_as_rows()`, which flattens image-shaped targets by the model's output shape, so `fit(x, x)` works for autoencoders; `predict()` (and `QuantizedModel.predict()`) return `(N, C, H, W)` when the output is an image (`as_images()`), while the int8 helpers stay flat like the C buffers. `set_input_noise()` stores the setting on the model and `_apply_training_settings()` pushes it, like the `compile()` options; `apply_input_noise()` calls `ann_apply_input_noise` on a copy. Tests: `tests/test_decoder.py`.
- **Pooling and normalization layers**: `GlobalAveragePooling2D` (alias `GlobalAvgPool2D`) maps to `ann_add_global_avgpool2d`, needs an image input (it is in `_IMAGE_LAYERS`, which `_uses_images` and `_layer_shapes` check) and outputs `(channels,)`. `BatchNormalization` (alias `BatchNorm`) maps to `ann_add_batchnorm(activation)`, keeps its input's shape and counts 4 parameters per channel (gamma, beta and the running mean and variance, as Keras counts them). `_from_ptr` rebuilds both from a loaded model. Tests: `tests/test_pooling_norm.py`.
- **Conv and pool windows**: conv layers derive from `_ConvBase` (`Conv2D`, `DepthwiseConv2D`, `Conv1D`) and pools from `_PoolBase` (`MaxPool2D`, `MaxPool1D`); each gives `window()` and the conv's `output_filters`/`conv_groups` for an input's channels, and `compile()` adds every one with `ann_add_conv2d_ex`/`ann_add_maxpool2d_ex` (the convenience C functions are recorded as UNUSED). Settings may be `(height, width)` pairs; square ones read back as ints. 1-D layers are 1 x k windows: a 2-D `input_shape` read first by `Conv1D`/`MaxPool1D` is a sequence `(C, L)`, built as libann's `(C, 1, L)` (`_is_sequence`, `_image_input_shape`), and `_public_shape` shows those images one row high as `(C, L)` in summaries and outputs (`predict` returns `(N, C, L)` for sequence outputs). `_from_ptr` rebuilds layer classes from `ann_get_layer_window` and `ann_get_layer_groups` (groups = input channels is `DepthwiseConv2D`; a 1 x k window over input one row high is 1-D). Tests: `tests/test_conv_layers.py`.
- **Text**: `Embedding` (first layer; `_is_sequence` treats the model as a sequence model) and `GlobalMaxPooling2D` (aliases `...1D`, `GlobalMaxPool...`) map to `ann_add_embedding[_masked]`/`ann_add_global_maxpool2d`; `Embedding(weights=, trainable=)` is applied in `_build` with `ann_set_layer_param`/`ann_set_layer_trainable` (`weights` is excluded from the dataclass repr and comparisons). `pyann/text.py` is pure Python except `load_word_vectors` (NumPy): `Tokenizer` (Keras-like ids, 0 kept for padding; `preprocess=` runs first, and `to_json` names it only when it is one of `_BUILTIN_PREPROCESS`, e.g. `strip_html`), `pad_sequences` (pads after by default, which masking needs), `strip_html`, and `load_word_vectors` (GloVe/word2vec/fastText text files, `.gz`; parses only the vocabulary's lines). Tests: `tests/test_text.py`.
- **Layer weights**: `Sequential.get_layer_weights`/`set_layer_weights`/`get_weights`/`set_weights`/`set_trainable`/`is_trainable` (and `layers`) wrap libann's `ann_get_layer_param*`/`ann_set_layer_param`/`ann_set_layer_trainable`; a layer is an index into `layers` or the layer object (`_layer_index`, libann index + 1). libann stores each parameter as a `[rows x cols]` matrix; `_param_shape` gives the PyTorch-layout array shape (conv weights `(F, C/G, kh, kw)` or `(F, C/G, k)` for `Conv1D`, biases and batch norm `(C,)`). Reading an untrained layer initializes it in libann, so values are what training starts from. Setting checks every shape and value before changing anything. Tests: `tests/test_weights.py` (including NumPy forward passes that pin the layouts).
- **Quantization / C export**: `core/quantized.py` holds `QuantizedModel` (made by `Sequential.quantize`) and the shared `export_c_header` helper. Exported headers need libann's `ann_rt.h`: wheels install it to `pyann/include/` (CMake `install(FILES ...)`), development builds use `ann/ann_rt.h`; `runtime_header_path()` finds either. `quantize_input` must round exactly like `ann_rt_quantize` in C (float32, halves away from zero).
- **Examples**: `examples/*.py` are standalone scripts sharing `examples/example_data.py` (Fashion-MNIST download into the git-ignored `examples/data/`, or Kaggle CSVs via `--train-csv`/`--test-csv`). `tests/test_examples.py` runs each one on tiny synthetic CSVs; a new example needs a test there and a row in `examples/README.md`.
- **Binding coverage**: `tests/test_binding_coverage.py` diffs `ann/*.h` against the cdef (functions, signatures, enum values) and against what the package calls. After bumping the `ann/` submodule, a failure means upstream changed: bind the new function, update the cdef, or record the decision in its `UNBOUND` / `UNUSED` / `CDEF_ONLY` tables with a reason.
