# PyANN

Python wrapper for the [libann](https://github.com/mseminatore/ann) neural network library with a Keras-like API.

## Features

- **Keras-style API**: Familiar `Sequential` model with `add()`, `fit()`, `predict()`
- **Multiple optimizers**: SGD, Momentum, RMSProp, AdaGrad, Adam
- **Activation functions**: Sigmoid, ReLU, LeakyReLU, Tanh, Softsign, Softmax, GELU
- **Loss functions**: MSE, Categorical Cross-Entropy
- **Online training**: `train_on_batch()` for streaming data, fine-tuning and custom training loops
- **Learning rate schedulers**: Step decay, Exponential decay, Cosine annealing
- **Hyperparameter tuning**: Grid search, Random search, Bayesian optimization, TPE (pure-Python searches built on `Sequential`; libann's native hypertuner is not exposed)
- **NumPy optional**: Works with Python lists, optimized for NumPy arrays
- **Model persistence**: Save/load in text or binary format, ONNX export/import
- **Network visualization**: Export architecture as PIKCHR diagrams (renders to SVG)
- **Regularization**: L1 and L2 (weight decay) regularization, dropout
- **Evaluation**: Accuracy scoring, confusion matrix with MCC
- **Data utilities**: CSV loading, normalization, one-hot encoding, train/val splitting
- **Hardware acceleration**: BLAS-accelerated CPU everywhere (Accelerate on macOS, OpenBLAS on Linux and Windows), plus optional GPU inference and training via Metal (macOS) or CUDA

## Installation

```bash
pip install pylibann
```

The PyPI package is called **`pylibann`**, but you import it as `pyann`. (The unrelated `pyann` project on PyPI is a different package with the same module name, so don't install both in one environment.)

Pre-built wheels include the native library, so no compiler is needed:

| Platform | BLAS | GPU |
|---|---|---|
| macOS (Apple silicon, 12+) | Accelerate | Metal |
| Linux x86_64 and aarch64 (glibc, manylinux2014+) | OpenBLAS | - |
| Windows x86_64 | OpenBLAS | - |

On other platforms `pip` builds from source, which needs CMake and a C compiler (see below). NumPy is optional: `pip install "pylibann[numpy]"`.

### Building from source

```bash
git clone --recurse-submodules https://github.com/mseminatore/pyann.git
cd pyann
pip install -e ".[dev]"      # builds the native library with CMake
pytest
```

The build picks the same acceleration as the wheels. On Linux and Windows it downloads the prebuilt OpenBLAS from the [`scipy-openblas32`](https://pypi.org/project/scipy-openblas32/) package. Choose other options by passing CMake flags through `CMAKE_ARGS`:

```bash
CMAKE_ARGS="-DUSE_CUDA=ON" pip install -e .                                 # NVIDIA CUDA (Linux/Windows)
CMAKE_ARGS="-DUSE_CBLAS=ON" pip install -e .                                # CBLAS from /opt/cblas
CMAKE_ARGS="-DUSE_BLAS=ON" pip install -e .                                 # system OpenBLAS from /opt/OpenBLAS
CMAKE_ARGS="-DUSE_METAL=OFF -DUSE_ACCELERATE=OFF" pip install -e .          # plain CPU build
```

| CMake option | Meaning |
|---|---|
| `USE_ACCELERATE` | Apple Accelerate (macOS; default there) |
| `USE_METAL` | Apple Metal GPU backend (macOS; default there) |
| `USE_SCIPY_OPENBLAS` | OpenBLAS from `scipy-openblas32` (Linux/Windows; default there) |
| `USE_BLAS` | OpenBLAS installed at `/opt/OpenBLAS` |
| `USE_CBLAS` | [CBLAS](https://github.com/mseminatore/cblas) installed at `/opt/cblas` |
| `USE_CUDA` | NVIDIA CUDA GPU backend |

`python build_lib.py [--accelerate | --scipy-openblas | --cblas | --blas] [--metal] [--cuda] [--clean]` builds just the shared library into `src/pyann/lib/` without installing, which is handy when working on the C side. Set `PYANN_LIB_PATH` to load a library from another location.

Check what a given install was built with:

```python
import pyann
pyann.backend_info()   # {'blas': 'accelerate', 'gpu': 'metal', 'gpu_active': None, ...}
```

## Quick Start

```python
from pyann import Sequential, Dense, Adam, Loss

# Create model
model = Sequential(optimizer=Adam(lr=0.001), loss=Loss.MSE)
model.add(Dense(128, activation='relu', input_shape=(784,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Predict
predictions = model.predict(X_test)

# Evaluate
accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## Examples

### XOR Problem

```python
from pyann import Sequential, Dense

# XOR data
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

model = Sequential()
model.add(Dense(4, activation='relu', input_shape=(2,)))
model.add(Dense(1, activation='sigmoid'))

model.fit(X, y, epochs=1000, batch_size=4)
print(model.predict(X))
```

### With NumPy

```python
import numpy as np
from pyann import Sequential, Dense

X = np.random.randn(1000, 20).astype(np.float32)
y = (X.sum(axis=1) > 0).astype(np.float32).reshape(-1, 1)

model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(20,)))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.fit(X, y, epochs=50)
```

### Learning Rate Scheduling

```python
from pyann import Sequential, Dense
from pyann.callbacks import CosineAnnealing

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(10,)))
model.add(Dense(1, activation='sigmoid'))

scheduler = CosineAnnealing(T_max=100, min_lr=0.0001)
model.fit(X, y, epochs=100, lr_scheduler=scheduler)
```

### Early Stopping

```python
from pyann.callbacks import EarlyStopping

stopping = EarlyStopping(patience=10, min_delta=0.0001, restore_best=True)
history = model.fit(X, y, epochs=500, validation_data=(X_val, y_val), callbacks=[stopping])
print(history["best_epoch"], history["val_loss"])
```

libann measures the validation loss after every epoch and stops once it has not
improved for `patience` epochs; with `restore_best=True` the model keeps the
best epoch's weights. Without `validation_data` it monitors the training loss.

### Online Training and Fine-Tuning

`train_on_batch()` runs one training step on a mini-batch and returns its loss. Unlike `fit()` it keeps the weights and the optimizer state (such as Adam's momentum) between calls, so it works for streaming data, custom training loops, and fine-tuning a loaded model.

```python
model = Sequential.load('model.nna')        # or a model you just built and compiled

for x_batch, y_batch in stream():           # any source of (samples, features) batches
    loss = model.train_on_batch(x_batch, y_batch)

model.end_online_training()                 # optional: dropout off, GPU weights refreshed
model.save('model_finetuned.nna')
```

Batches can vary in size, `predict()` and `evaluate()` work between steps, and `fit()` ends any open session itself. The learning rate scheduler and callbacks only apply to `fit()`. Shapes are checked up front: `x` must be `(samples, input_size)` and `y` `(samples, output_size)`, and a mismatch raises `ValueError`.

### Hyperparameter Tuning

```python
from pyann.hypertune import RandomSearch, HyperparamSpace
from pyann.utils import split_data

# Split data
X_train, y_train, X_val, y_val = split_data(X, y, train_ratio=0.8)

# Define search space
space = HyperparamSpace(
    learning_rate=(0.0001, 0.01),
    batch_sizes=[32, 64, 128],
    hidden_layers=(1, 3),
    layer_sizes=[64, 128, 256],
)

# Run search
search = RandomSearch(space, n_trials=20)
best, results = search.run(
    X_train, y_train, X_val, y_val,
    input_size=784, output_size=10
)

# Create best model
model = search.create_model(best, input_size=784, output_size=10)
```

### TPE Hyperparameter Search

```python
from pyann.hypertune import TPESearch, HyperparamSpace

space = HyperparamSpace(
    learning_rate=(0.0001, 0.01),
    batch_sizes=[32, 64, 128],
    hidden_layers=(1, 3),
    layer_sizes=[64, 128, 256],
)

# TPE uses KDE to model good vs bad configs
search = TPESearch(space, n_trials=50, n_startup=10, gamma=0.25)
best, results = search.run(
    X_train, y_train, X_val, y_val,
    input_size=784, output_size=10
)
```

### Save and Load

```python
# Save
model.save('model.nna')  # Text format
model.save('model.nnb', format='binary')  # Binary format
model.export_onnx('model.onnx.json')  # ONNX format

# Load
loaded = Sequential.load('model.nna')
loaded = Sequential.load_onnx('model.onnx.json')  # From ONNX
```

### Network Visualization

```python
# Export architecture as PIKCHR diagram
model.export_pikchr('network.pikchr')

# Then render to SVG with the pikchr CLI tool:
# pikchr network.pikchr > network.svg
```

Small networks (≤10 nodes per layer) get a detailed diagram with individual nodes and connections. Larger networks get a simplified box diagram.

### Data Utilities

```python
from pyann.utils import load_csv, split_data
from pyann.utils.data import normalize, one_hot_encode

# Load CSV data
data, rows, cols = load_csv('data.csv', has_header=True)

# Normalize features
X_norm, mean, std = normalize(X)

# One-hot encode labels
y_encoded = one_hot_encode(labels, num_classes=10)

# Split into train/validation sets
X_train, y_train, X_val, y_val = split_data(X, y, train_ratio=0.8)
```

## API Reference

### Sequential

```python
Sequential(layers=None, optimizer=None, loss=Loss.MSE, name=None)
```

Methods:
- `add(layer)` - Add a layer
- `compile(optimizer, loss, learning_rate, weight_decay, l1_regularization, gradient_clip, weight_init, convergence, dropout)` - Configure training (see below)
- `fit(x, y, epochs, batch_size, ...)` - Train the model
- `train_on_batch(x, y)` - One training step on a mini-batch; returns the loss
- `end_online_training()` - End the online session started by `train_on_batch()`
- `predict(x)` - Generate predictions
- `evaluate(x, y)` - Compute accuracy
- `evaluate_loss(x, y)` - Compute the loss, averaged like the training loss
- `confusion_matrix(x, y)` - Compute TP/FP/TN/FN and MCC (binary classification)
- `save(filepath, format)` - Save model
- `load(filepath, format)` - Load model (classmethod)
- `export_onnx(filepath)` - Export to ONNX JSON format
- `load_onnx(filepath)` - Load from ONNX JSON (classmethod)
- `export_pikchr(filepath)` - Export architecture as PIKCHR diagram
- `export_learning_curve(filepath)` - Export training history as CSV (adds `val_loss` when `fit()` had validation data)
- `clear_history()` - Clear training history to free memory
- `set_weight_decay(lambda_)` - Set L2 regularization coefficient
- `set_l1_regularization(lambda_)` - Set L1 regularization coefficient
- `summary()` - Get model summary

### Backend

```python
pyann.backend_info()   # dict: blas, gpu (compiled in), gpu_active, plus CBLAS details when present
pyann.enable_gpu()     # initialize the GPU backend; returns True if one is active
pyann.gpu_enabled()    # whether a GPU backend is active
```

#### Training options

All are optional keyword arguments to `compile()`; unset options keep libann's defaults.

| Option | Meaning |
|---|---|
| `weight_decay`, `l1_regularization` | L2 / L1 regularization strength (0 = off) |
| `gradient_clip` | Clip gradients to `[-value, value]`; 0 = off. 1.0-5.0 is typical and helps with exploding gradients |
| `weight_init` | `'uniform'`, `'xavier'`, `'he'` or `'auto'` (default: chosen per layer from its activation); also `WeightInit.HE` etc. |
| `convergence` | Stop training once the average loss falls below this value. **libann's default is 0.01**, so a run can end before `epochs` is reached; pass `convergence=0` to always train the full number |
| `dropout` | Default dropout rate in `[0, 1)` for hidden layers that don't set their own with `Dense(dropout=...)` |

```python
model.compile(optimizer='adam', loss='mse', gradient_clip=5.0, weight_init='he', convergence=0.0)
```

### Layers

```python
Dense(units, activation=None, input_shape=None, dropout=0.0)
Input(shape)
```

### Optimizers

```python
SGD(lr=0.05)
Momentum(lr=0.01)
RMSProp(lr=0.001)
AdaGrad(lr=0.01)
Adam(lr=0.001)  # Recommended
```

### Loss Functions

```python
Loss.MSE                        # Mean Squared Error (regression: use a linear output)
Loss.CATEGORICAL_CROSS_ENTROPY  # For classification
Loss.BINARY_CROSS_ENTROPY       # Alias of CATEGORICAL_CROSS_ENTROPY, for sigmoid outputs ('bce')
```

`CATEGORICAL_CROSS_ENTROPY` needs a softmax output layer (one class per sample) or a sigmoid output layer (binary cross-entropy per output). Training with any other output activation fails.

### Activations

```python
Activation.NONE      # Linear
Activation.SIGMOID
Activation.RELU
Activation.LEAKY_RELU
Activation.TANH
Activation.SOFTSIGN
Activation.SOFTMAX
Activation.GELU      # tanh approximation; ONNX export uses opset 20
```

### Callbacks

```python
StepDecay(step_size, gamma=0.5)          # Multiply LR by gamma every step_size epochs
ExponentialDecay(gamma=0.95)             # Multiply LR by gamma each epoch
CosineAnnealing(T_max, min_lr=0.0001)   # Smooth decay to min_lr over T_max epochs
EarlyStopping(patience=10, min_delta=0.0001, restore_best=True)  # Stop when (validation) loss stops improving
```

### Hyperparameter Tuning

```python
GridSearch(space, lr_steps=3)                          # Exhaustive grid search
RandomSearch(space, n_trials=50)                       # Random sampling
BayesianSearch(space, n_trials=50, n_initial=10)       # Bayesian optimization
TPESearch(space, n_trials=50, n_startup=10, gamma=0.25)  # Tree-structured Parzen Estimator
```

All accept a `HyperparamSpace` and return `(best_result, all_results)` from `run()`.

## What's Not Exposed

pyann covers libann's training, inference, persistence, regularization and GPU features. These parts of libann's C API have no Python equivalent yet:

- **libann's native hypertuner** (`ann_hypertune.h`). `pyann.hypertune` is a separate Python implementation, so its search spaces and results differ from the C one (for example topology names, per-layer activation search, and the Gaussian-process settings).
- **Error-log callbacks**: errors surface as Python exceptions instead.
- **Tensor math** (gemm, elementwise ops and so on): use NumPy.

`tests/test_binding_coverage.py` keeps this list honest. It compares libann's headers with the bindings and fails when upstream adds, removes or changes a function without a matching decision recorded there.

## Hardware Acceleration

The CPU path uses the BLAS the library was built with (see `backend_info()["blas"]`). GPU use is opt-in:

```python
import pyann

if pyann.enable_gpu():          # True when a GPU backend is compiled in and a device is available
    print("Using", pyann.backend_info()["gpu_active"])   # 'metal' or 'cuda'

model.fit(X, y, epochs=10)      # trains on the GPU when enabled
model.predict(X)                # batch inference on the GPU when enabled
```

If no GPU backend is built in, or no device is found, `enable_gpu()` returns `False` and everything keeps running on the CPU with BLAS, so the same code works everywhere.

Notes:
- `enable_gpu()` applies to the whole process and cannot be undone.
- GPU speedups depend on problem size. In an informal test on Apple silicon (batch size 128) Metal sped up batch inference but was several times slower than the CPU for training, which is why it is not enabled automatically. Measure your own workload.
- The macOS wheels include Metal. CUDA is only available when building from source with `USE_CUDA=ON`, and has had little testing.
- `predict()` runs the whole input as one batch call.

## License

MIT License - see [LICENSE](https://github.com/mseminatore/pyann/blob/main/LICENSE) for details.

The Linux and Windows wheels also bundle OpenBLAS, LAPACK and the GCC runtime libraries (`libgfortran`, `libquadmath`) under their own licenses, included in the wheel as `pyann/lib/LICENSE-scipy-openblas32.txt`.
