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
CMAKE_ARGS="-DUSE_METAL=OFF -DUSE_ACCELERATE=OFF" pip install -e .

# Or build only the shared library into src/pyann/lib/ (no install)
python build_lib.py                  # defaults: no BLAS (see flags below)
python build_lib.py --accelerate --metal   # macOS
python build_lib.py --scipy-openblas       # Linux/Windows (pip install scipy-openblas32 first)
python build_lib.py --cblas | --blas | --cuda | --clean

# Run tests
pytest                                                    # all tests
pytest tests/test_layers.py                               # single file
pytest tests/test_layers.py::TestDense::test_basic_dense -v  # single test

# Build a wheel / sdist
pip wheel . --no-deps -w dist
python -m build --sdist
```

The `ann/` submodule must be checked out (`git submodule update --init`). Wheels default to Accelerate+Metal on macOS and the bundled `scipy-openblas32` OpenBLAS on Linux/Windows (see `[[tool.scikit-build.overrides]]` in `pyproject.toml`). CI (`.github/workflows/wheels.yml`) builds and tests wheels with cibuildwheel; `tests/test_backend.py` fails a wheel that lost its BLAS. `.github/workflows/blas-bench.yml` (manual) compares BLAS libraries using `benchmarks/`.

The built shared library goes to `src/pyann/lib/` for dev builds. Override with `PYANN_LIB_PATH` env var.

## Conventions

- **Enum + string API**: Activations, losses, and optimizers accept both enum values (`Activation.RELU`) and strings (`"relu"`). Parsing helpers (`parse_optimizer`, `parse_loss`, `parse_activation`) handle conversion.
- **Error handling**: C errors are translated via `raise_for_error_code()` in `exceptions.py`. Python-side validation uses `ValueError`/`TypeError`.
- **Test style**: pytest with class-based grouping (`class TestDense:`). Each class covers one component.
- **Type hints**: All public APIs use type hints. `TYPE_CHECKING` guard for numpy imports.
- **C binding prefixes**: `ann_*` for network functions, `tensor_*` for tensor operations (in `_bindings/` CFFI definitions).
- **Python enum values** (`IntEnum`) must match C enum ordinals exactly.
