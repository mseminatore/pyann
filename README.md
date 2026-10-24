# PyANN

Python wrapper for the [libann](https://github.com/mseminatore/ann) neural network library with a Keras-like API.

## Features

- **Keras-style API**: Familiar `Sequential` model with `add()`, `fit()`, `predict()`
- **Layers**: Dense, plus Conv2D, MaxPool2D and Flatten for convolutional networks, and UpSample2D and Reshape for decoders (autoencoders)
- **Input noise** for denoising autoencoders: Gaussian, masking or salt-and-pepper
- **Multiple optimizers**: SGD, Momentum, RMSProp, AdaGrad, Adam
- **Activation functions**: Sigmoid, ReLU, LeakyReLU, Tanh, Softsign, Softmax, GELU
- **Loss functions**: MSE, Categorical Cross-Entropy
- **Online training**: `train_on_batch()` for streaming data, fine-tuning and custom training loops
- **Learning rate schedulers**: Step decay, Exponential decay, Cosine annealing
- **Hyperparameter tuning**: Grid search, Random search, Gaussian-process Bayesian optimization, TPE, with early stopping, custom scoring and network-shape search (pure-Python searches built on `Sequential`; libann's native hypertuner is not exposed)
- **NumPy optional**: Works with Python lists, optimized for NumPy arrays
- **Model persistence**: Save/load in text or binary format, ONNX export/import (including CNNs and PyTorch-style ONNX JSON)
- **Deploy to microcontrollers**: Quantize a trained model to int8 and export it as a standalone C header (integer-only inference, no `malloc`), or export float32 weights
- **Network visualization**: Export architecture as PIKCHR diagrams (renders to SVG)
- **Regularization**: L1 and L2 (weight decay) regularization, dropout
- **Evaluation**: Accuracy scoring, confusion matrix with MCC
- **Data utilities**: CSV loading, normalization, one-hot encoding, train/val splitting
- **Hardware acceleration**: multithreaded, BLAS-accelerated CPU training everywhere (Accelerate on macOS, OpenBLAS on Linux and Windows), plus optional GPU inference and training via Metal (macOS), Vulkan (AMD, Intel and NVIDIA on Linux and Windows) or CUDA for dense networks

## Installation

```bash
pip install pylibann
```

The PyPI package is called **`pylibann`**, but you import it as `pyann`. (The unrelated `pyann` project on PyPI is a different package with the same module name, so don't install both in one environment.)

Pre-built wheels include the native library, so no compiler is needed:

| Platform | BLAS | GPU |
|---|---|---|
| macOS (Apple silicon, 12+) | Accelerate | Metal |
| Linux x86_64 and aarch64 (glibc, manylinux2014+) | OpenBLAS | Vulkan |
| Windows x86_64 | OpenBLAS | Vulkan |

GPU use is opt-in (`pyann.enable_gpu()`). Vulkan needs only a GPU driver with Vulkan 1.1, which current AMD, Intel and NVIDIA drivers include; without one, `enable_gpu()` returns `False` and everything runs on the CPU with BLAS.

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
CMAKE_ARGS="-DUSE_VULKAN=OFF" pip install -e .                              # no Vulkan (Linux/Windows)
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
| `USE_VULKAN` | Vulkan GPU backend (AMD, Intel, NVIDIA): `ON`, `OFF` or `AUTO` (Linux/Windows default: built when it can be). Building needs the Vulkan headers and `glslc` (the [Vulkan SDK](https://vulkan.lunarg.com/), or `libvulkan-dev` and `glslc`), or the prebuilt shaders the sdist ships (`python tools/prepare_vulkan.py` writes them to `vulkan_prebuilt/`). The library loads the Vulkan driver at run time and falls back to the CPU without one |

`python build_lib.py [--accelerate | --scipy-openblas | --cblas | --blas] [--metal] [--cuda] [--vulkan] [--clean]` builds just the shared library into `src/pyann/lib/` without installing, which is handy when working on the C side. Set `PYANN_LIB_PATH` to load a library from another location.

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

The [`examples/`](examples/) folder has complete, runnable programs: logic
gates, noisy digit recognition, regression across activation functions,
online learning on drifting data, and on Fashion-MNIST a CNN classifier, a
denoising autoencoder, autoencoder anomaly detection, int8 export to a C
header for microcontrollers, and hyperparameter search. See
[examples/README.md](examples/README.md) for what each one shows.

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

### Convolutional Networks

Images are channels-first, as in PyTorch and ONNX: a batch has shape
`(N, channels, height, width)`. Channels-last arrays (Keras' default) need
`x.transpose(0, 3, 1, 2)` first.

```python
import numpy as np
from pyann import Sequential, Input, Conv2D, MaxPool2D, Flatten, Dense

# x: (N, 1, 28, 28) float32 images, y: (N, 10) one-hot labels
model = Sequential([
    Input((1, 28, 28)),
    Conv2D(8, 3, activation='relu'),                  # -> (8, 26, 26)
    MaxPool2D(2),                                     # -> (8, 13, 13)
    Conv2D(16, 3, activation='relu'),                 # -> (16, 11, 11)
    MaxPool2D(2),                                     # -> (16, 5, 5)
    Flatten(),                                        # -> (400,)
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])
model.compile(optimizer='adam', loss='cce')
model.fit(x, y, epochs=10, batch_size=32)
print(model.summary())
```

On Fashion-MNIST this network reaches about 90% test accuracy. `predict()`,
`evaluate()` and `train_on_batch()` take image batches, a single image, or
flattened rows. With `enable_gpu()`, convolutional networks run on the GPU
with Metal or Vulkan; the CUDA backend is dense-only, so with CUDA they train on
the CPU (BLAS-accelerated).

### Autoencoders and Denoising

`UpSample2D` (nearest-neighbour, the counterpart of `MaxPool2D`) and `Reshape`
(e.g. a Dense bottleneck back to an image) build decoders. Targets can be
images too, so an autoencoder trains with `fit(x, x)`. With input noise the
model sees corrupted inputs and learns to output the clean ones:

```python
from pyann import Sequential, Input, Conv2D, MaxPool2D, UpSample2D

model = Sequential([
    Input((1, 28, 28)),
    Conv2D(16, 3, padding='same', activation='relu'),
    MaxPool2D(2),                                       # -> (16, 14, 14)
    Conv2D(8, 3, padding='same', activation='relu'),
    MaxPool2D(2),                                       # -> (8, 7, 7)
    UpSample2D(2),                                      # -> (8, 14, 14)
    Conv2D(16, 3, padding='same', activation='relu'),
    UpSample2D(2),                                      # -> (16, 28, 28)
    Conv2D(1, 3, padding='same', activation='sigmoid'), # -> (1, 28, 28)
])
model.compile(optimizer='adam', loss='bce', learning_rate=0.002, convergence=0.0)
model.set_input_noise('gaussian', 0.3, clip=(0, 1))     # or 'masking', 'salt_pepper'
model.fit(x_train, x_train, epochs=5, batch_size=32)    # x: (N, 1, 28, 28) in [0, 1]

noisy = model.apply_input_noise(x_test)                 # corrupted like the training inputs
denoised = model.predict(noisy)                         # (N, 1, 28, 28): image outputs come back as images
```

Noise is drawn fresh for every training batch (in `fit()` and
`train_on_batch()`); `predict()` and `evaluate()` use the inputs as given,
and the setting is not saved with the model. All layers, and noise, run on the
GPU with Metal or Vulkan.

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
print(history["epochs_trained"], history["best_epoch"], history["val_loss"])
```

libann measures the validation loss after every epoch and stops once it has not
improved for `patience` epochs; with `restore_best=True` the model keeps the
best epoch's weights. Without `validation_data` it monitors the training loss.

### Training History

`fit()` returns a summary: `epochs_trained`, the last epoch's `loss`, and, with
validation data, `val_loss` and `val_accuracy` for the final weights.
`training_history()` gives every epoch:

```python
model.fit(X, y, epochs=50, validation_data=(X_val, y_val))
h = model.training_history()   # {"loss": [...], "lr": [...], "val_loss": [...]}

import matplotlib.pyplot as plt
plt.plot(h["loss"], label="train")
plt.plot(h["val_loss"], label="validation")
```

- **Accumulation:** epochs accumulate across `fit()` calls until
  `clear_history()`. libann records at most 10,000.
- **Gaps:** `val_loss` is `None` for epochs trained without validation data.
- **Online training:** `train_on_batch()` steps aren't recorded.
- **CSV:** `export_learning_curve()` writes the same history to a CSV file.

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

### Scoring and Early Stopping in Searches

Every search ranks its trials by `result.score`, where higher is better.
`HypertuneOptions.scoring` chooses what that is:

- `"accuracy"` (the default) is the validation accuracy, for classification.
- `"loss"` is the negated validation loss. Use it for regression.
- A function `score(model, x_val, y_val) -> float` is called with each trained model.

```python
from pyann import Activation, Loss
from pyann.hypertune import HypertuneOptions, RandomSearch

# Regression: rank by validation MSE, stop each trial after 5 epochs without improvement
options = HypertuneOptions(scoring="loss", early_stopping=True, patience=5)
search = RandomSearch(space, n_trials=30, options=options)
best, results = search.run(X_train, y_train, X_val, y_val, input_size=8, output_size=1,
                           output_activation=Activation.LINEAR, loss=Loss.MSE)
print(best.val_loss, best.best_epoch, best.training_time)
```

Early stopping is on by default (`patience=10`). Each trial trains for up to
`HyperparamSpace.epochs` epochs, stops once its validation loss stops improving,
and keeps its best epoch's weights. Each result records:

- `accuracy` and `val_loss`: measured on the validation data
- `loss`: the last epoch's training loss
- `score`: the value the search ranked by
- `best_epoch`: the epoch whose weights were kept
- `epochs_trained`: how many epochs the trial ran
- `training_time`: in seconds

`seed` makes the sampled configurations repeatable. It doesn't seed libann's
weight initialization.

### Network Shapes in Searches

`HyperparamSpace.topology` sets how the hidden layers' widths relate. It takes
one pattern or a list of patterns to search over:

| Topology | Widths for base 128, 3 layers |
|---|---|
| `"any"` (default) | each layer picks its own width from `layer_sizes` |
| `"constant"` | 128, 128, 128 |
| `"pyramid"` | 128, 64, 32 |
| `"inverse"` | 32, 64, 128 |
| `"funnel"` | 128, 256, 128 |

The patterns pick one base width from `layer_sizes` and build every layer's
width from it, as libann's hypertuner does: no layer is narrower than 8, and
`"funnel"` widens to at most 4x the base. `"uniform"` and `"diamond"` are
accepted as names for `"constant"` and `"funnel"`. TPE learns which topologies
and widths score well, not just learning rates and batch sizes. `GridSearch`
tries every base width for a pattern, and every combination of widths for
`"any"` (`len(layer_sizes) ** layers` of them).

```python
space = HyperparamSpace(topology=["constant", "pyramid"], hidden_layers=(2, 4),
                        layer_sizes=[64, 128, 256])
best, _ = TPESearch(space, n_trials=40).run(X_train, y_train, X_val, y_val,
                                           input_size=784, output_size=10)
print(best.topology, best.layer_sizes)   # e.g. pyramid [256, 128, 64]
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

### int8 Quantization and C Export

Quantize a trained model to int8, check its accuracy, and export it as a C
header for a microcontroller or any C program. The header runs on `ann_rt.h`
(copied next to it): integer-only math, no `malloc`, no libc needed for the
int8 path. On Fashion-MNIST, int8 accuracy stays within 0.2 points of float32
with 4x less parameter memory.

```python
q = model.quantize(x_train[:1000])        # calibrate on representative inputs
print(model.evaluate(x_test, y_test), q.evaluate(x_test, y_test))

q.export_c_header('model.h')              # writes model.h and ann_rt.h
model.export_c_header('model_f.h')        # float32 weights instead
```

On the device:

```c
#include "model.h"

static int8_t workspace[MODEL_WORKSPACE_SIZE];
int8_t scores[MODEL_OUTPUT_SIZE];
model_predict(input_q, scores, workspace);  /* MODEL_INPUT_SIZE int8 values */
int label = model_argmax(scores);
```

`q.quantize_input(x)` gives the int8 inputs the device expects, and
`q.predict_int8(x_q)` returns exactly what the header computes. See libann's
[C export guide](https://github.com/mseminatore/ann/blob/main/docs/C_EXPORT.md)
for the scheme and generated API.

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
- `predict(x)` - Generate predictions: `(samples, outputs)`, or `(samples, channels, height, width)` when the output layer produces an image (Conv2D, MaxPool2D, UpSample2D or Reshape)
- `evaluate(x, y)` - Compute accuracy
- `evaluate_loss(x, y)` - Compute the loss, averaged like the training loss
- `confusion_matrix(x, y)` - Compute TP/FP/TN/FN and MCC (binary classification)
- `save(filepath, format)` - Save model
- `load(filepath, format)` - Load model (classmethod)
- `export_onnx(filepath)` - Export to ONNX JSON format
- `load_onnx(filepath)` - Load from ONNX JSON (classmethod)
- `quantize(x_calibration)` - Make an int8 `QuantizedModel`
- `export_c_header(filepath, prefix=None, runtime=True)` - Export float32 weights as a C header (and copy `ann_rt.h` next to it)
- `export_pikchr(filepath)` - Export architecture as PIKCHR diagram
- `training_history()` - Per-epoch `loss`, `lr` and `val_loss` lists from `fit()`
- `export_learning_curve(filepath)` - Export training history as CSV (adds `val_loss` when `fit()` had validation data)
- `clear_history()` - Clear training history to free memory
- `set_weight_decay(lambda_)` - Set L2 regularization coefficient
- `set_l1_regularization(lambda_)` - Set L1 regularization coefficient
- `set_input_noise(noise, amount, clip=None)` - Corrupt training inputs: `'gaussian'`, `'masking'`, `'salt_pepper'` or `None`; see [Autoencoders and Denoising](#autoencoders-and-denoising)
- `input_noise` - The noise setting as `(noise, amount, clip)`, or `None`
- `apply_input_noise(x)` - A copy of `x` with the input noise applied
- `summary()` - Get model summary

### QuantizedModel

Made by `Sequential.quantize()`; independent of the model it came from.

- `predict(x)` - Real in and out, int8 inside (matches the header's `<prefix>_predict_float`); shaped like `Sequential.predict()`, so image outputs come back as `(N, C, H, W)`
- `predict_int8(x_q)` - int8 in and out (matches the header's `<prefix>_predict` bit for bit); one row per sample, like the header's buffers
- `quantize_input(x)` / `dequantize_output(y_q)` - Convert between real and int8 values
- `evaluate(x, y)` - Accuracy of the int8 model
- `export_c_header(filepath, prefix=None, runtime=True)` - Export the int8 model as a C header
- `input_scale`, `input_zero_point`, `output_scale`, `output_zero_point` - Quantization: `real = scale * (q - zero_point)`

`pyann.runtime_header_path()` gives the path of the bundled `ann_rt.h`.

### Backend

```python
pyann.backend_info()   # dict: blas, gpu (compiled in), gpu_active, threads, optimized, plus CBLAS details when present
pyann.enable_gpu()     # initialize the GPU backend; returns True if one is active
pyann.gpu_enabled()    # whether a GPU backend is active
pyann.set_num_threads(4)   # CPU threads for training/batched inference (0 = default, 1 = single-threaded)
pyann.get_num_threads()    # current CPU thread count
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
Input(shape)              # (features,) or (channels, height, width)
Conv2D(filters, kernel_size=3, strides=1, padding=0, activation=None, input_shape=None, dropout=0.0)
MaxPool2D(pool_size=2, strides=None)   # strides default to pool_size; alias MaxPooling2D
Flatten()                 # optional: Dense layers read image outputs as vectors
UpSample2D(size=2)        # nearest-neighbour: (C, H, W) -> (C, H*size, W*size); alias UpSampling2D
Reshape(target_shape)     # same values as (channels, height, width), e.g. after a Dense layer
```

`Conv2D` uses square kernels; `padding` is an int, `'valid'` (0) or `'same'`
(keeps the size; odd kernel sizes with stride 1). Output size is
`(size + 2 * padding - kernel_size) // strides + 1`. A model whose layers
don't fit together raises `NetworkError` when it is built.

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
BayesianSearch(space, n_trials=50, n_initial=10,       # Gaussian-process Bayesian optimization
               n_candidates=100, exploration=0.01)
TPESearch(space, n_trials=50, n_startup=10, gamma=0.25)  # Tree-structured Parzen Estimator
```

All accept a `HyperparamSpace` and return `(best_result, all_results)` from `run()`.

`BayesianSearch` uses the same model as libann's hypertuner:

- **The model:** a Gaussian process over every setting. Learning rate, batch
  size, dropout, depth and each layer's width are continuous; optimizer,
  topology and each layer's activation are categorical. Each setting has its
  own length scale, refit every trial, so settings that don't matter stop
  steering the search.
- **Choosing the next trial:** after `n_initial` random trials, it picks the
  candidate with the highest Expected Improvement. Half of the candidates are
  random and half are small changes to the best trial.
- **Cost:** it takes longer than TPE to choose each trial, about a second over
  30 trials in pure Python. That's small next to training. On a 20-trial
  Fashion-MNIST search it found configurations about as good as TPE's and
  random search's, so try both on your problem.

### Errors

Failures raise exceptions derived from `pyann.exceptions.PyANNError` (`NetworkError`, `InvalidParameterError`, `AllocationError`, `IOError`); bad arguments raise `ValueError` or `TypeError`. When libann reports the error, the message names the C function, for example `NetworkError: Failed to load model from model.nna (ann_load_network: File I/O error)`. pyann prints nothing to stderr; to see libann's errors as they happen, enable the `pyann` logger at DEBUG level:

```python
import logging
logging.basicConfig()
logging.getLogger("pyann").setLevel(logging.DEBUG)
```

## What's Not Exposed

pyann covers libann's training, inference, persistence, regularization and GPU features. These parts of libann's C API have no Python equivalent yet:

- **libann's native hypertuner** (`ann_hypertune.h`). `pyann.hypertune` is a separate Python implementation, so its search spaces and results differ from the C one. Its topology patterns produce the same widths as libann's, and `BayesianSearch` uses the same kind of Gaussian process.
- **Error-log callbacks**: pyann installs its own, so libann errors surface in the Python exception instead of on stderr (see [Errors](#errors)).
- **Tensor math** (gemm, elementwise ops and so on): use NumPy.

`tests/test_binding_coverage.py` keeps this list honest. It compares libann's headers with the bindings and fails when upstream adds, removes or changes a function without a matching decision recorded there.

## Hardware Acceleration

The CPU path uses the BLAS the library was built with (see `backend_info()["blas"]`). GPU use is opt-in:

```python
import pyann

if pyann.enable_gpu():          # True when a GPU backend is compiled in and a device is available
    print("Using", pyann.backend_info()["gpu_active"])   # 'metal', 'vulkan' or 'cuda'

model.fit(X, y, epochs=10)      # trains on the GPU when enabled
model.predict(X)                # batch inference on the GPU when enabled
```

If no GPU backend is built in, or no device is found, `enable_gpu()` returns `False` and everything keeps running on the CPU with BLAS, so the same code works everywhere.

Notes:
- `enable_gpu()` applies to the whole process and cannot be undone.
- GPU speedups depend on problem size. In an informal test on Apple silicon (batch size 128) Metal sped up batch inference but was several times slower than the CPU for training, which is why it is not enabled automatically. Measure your own workload.
- The macOS wheels include Metal, and the Linux and Windows wheels Vulkan. CUDA is only available when building from source with `USE_CUDA=ON`, runs dense networks only, and has had little testing.
- A build with several GPU backends tries them in the order Metal, Vulkan, CUDA. Set the `ANN_GPU` environment variable (`metal`, `vulkan`, `cuda` or `none`) before `enable_gpu()` to pick one; `backend_info()["gpu_active"]` names the one that started.
- `predict()` runs the whole input as one batch call.

## License

MIT License - see [LICENSE](https://github.com/mseminatore/pyann/blob/main/LICENSE) for details.

The Linux and Windows wheels also bundle OpenBLAS, LAPACK and the GCC runtime libraries (`libgfortran`, `libquadmath`) under their own licenses, included in the wheel as `pyann/lib/LICENSE-scipy-openblas32.txt`.
