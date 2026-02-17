# PyANN - Copilot Instructions

## Architecture

PyANN is a Python wrapper around the [libann](https://github.com/mseminatore/ann) C neural network library, exposing a Keras-like API. It has two distinct layers:

- **C library** (`ann/` submodule) — core neural network engine built as a shared library (`.dll`/`.so`/`.dylib`). Bindings are declared in `src/pyann/_bindings/` using CFFI. The C library has its own guidelines in `ann/.github/copilot-instructions.md`.
- **Python package** (`src/pyann/`) — high-level API wrapping the C library. Key modules:
  - `core/network.py` — `Sequential` model (the main entry point)
  - `core/layers.py` — `Dense`, `Input` layers
  - `core/tensor.py` — Tensor wrapper around C tensors
  - `callbacks.py` — LR schedulers (`StepDecay`, `ExponentialDecay`, `CosineAnnealing`) and `EarlyStopping`
  - `hypertune/` — hyperparameter search (`GridSearch`, `RandomSearch`, `BayesianSearch`, `TPESearch`)
  - `utils/` — data loading, normalization, one-hot encoding, NumPy compatibility (`compat.py`)

NumPy is optional. The `utils/compat.py` module provides `has_numpy`, `to_array`, and `get_shape` helpers that work with plain Python lists as a fallback.

## Build & Test

```bash
# Build the C shared library (requires CMake)
python build_lib.py              # Release build
python build_lib.py --cblas      # With CBLAS acceleration
python build_lib.py --clean      # Clean rebuild

# Install Python package in dev mode
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_layers.py

# Run a single test
pytest tests/test_layers.py::TestDense::test_basic_dense -v
```

The built shared library is copied to `src/pyann/lib/`. You can also set `PYANN_LIB_PATH` to point to a custom library location.

## Conventions

- **src layout**: Python package lives under `src/pyann/`, not at the project root.
- **Enum + string API**: Activations, losses, and optimizers accept both enum values (`Activation.RELU`) and string names (`"relu"`). Parsing helpers (`parse_optimizer`, `parse_loss`) handle conversion.
- **Error handling**: C library errors are translated via `raise_for_error_code()` in `exceptions.py`. Python-side validation uses standard `ValueError`/`TypeError`.
- **Test style**: Tests use pytest with class-based grouping (`class TestDense:`). Each test class covers one component.
- **Type hints**: All public APIs use type hints. `TYPE_CHECKING` guard is used for numpy imports.
- **C function prefixes**: `ann_*` for network functions, `tensor_*` for tensor operations (relevant when modifying `_bindings/` CFFI definitions).
