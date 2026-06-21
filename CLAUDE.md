# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyANN is a Python wrapper around the [libann](https://github.com/mseminatore/ann) C neural network library, exposing a Keras-like API. It has two layers:

- **C library** (`ann/` git submodule) — core neural network engine built as a shared library via CMake. CFFI bindings are declared in `src/pyann/_bindings/`.
- **Python package** (`src/pyann/`, src layout) — high-level API wrapping the C library. `Sequential` is the main entry point.

NumPy is optional — `utils/compat.py` provides fallback helpers that work with plain Python lists.

## Build & Test

```bash
# Build the C shared library (requires CMake + C compiler)
python build_lib.py              # Release build
python build_lib.py --cblas      # With CBLAS acceleration
python build_lib.py --clean      # Clean rebuild

# Install Python package in dev mode
pip install -e ".[dev]"

# Run tests
pytest                                                    # all tests
pytest tests/test_layers.py                               # single file
pytest tests/test_layers.py::TestDense::test_basic_dense -v  # single test
```

The built shared library goes to `src/pyann/lib/`. Override with `PYANN_LIB_PATH` env var.

## Conventions

- **Enum + string API**: Activations, losses, and optimizers accept both enum values (`Activation.RELU`) and strings (`"relu"`). Parsing helpers (`parse_optimizer`, `parse_loss`, `parse_activation`) handle conversion.
- **Error handling**: C errors are translated via `raise_for_error_code()` in `exceptions.py`. Python-side validation uses `ValueError`/`TypeError`.
- **Test style**: pytest with class-based grouping (`class TestDense:`). Each class covers one component.
- **Type hints**: All public APIs use type hints. `TYPE_CHECKING` guard for numpy imports.
- **C binding prefixes**: `ann_*` for network functions, `tensor_*` for tensor operations (in `_bindings/` CFFI definitions).
- **Python enum values** (`IntEnum`) must match C enum ordinals exactly.
