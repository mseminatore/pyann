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
- **Quantization / C export**: `core/quantized.py` holds `QuantizedModel` (made by `Sequential.quantize`) and the shared `export_c_header` helper. Exported headers need libann's `ann_rt.h`: wheels install it to `pyann/include/` (CMake `install(FILES ...)`), development builds use `ann/ann_rt.h`; `runtime_header_path()` finds either. `quantize_input` must round exactly like `ann_rt_quantize` in C (float32, halves away from zero).
- **Binding coverage**: `tests/test_binding_coverage.py` diffs `ann/*.h` against the cdef (functions, signatures, enum values) and against what the package calls. After bumping the `ann/` submodule, a failure means upstream changed: bind the new function, update the cdef, or record the decision in its `UNBOUND` / `UNUSED` / `CDEF_ONLY` tables with a reason.
