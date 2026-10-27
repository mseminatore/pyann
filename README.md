# PyANN

Python wrapper for the [libann](https://github.com/mseminatore/ann) neural network library with a Keras-like API.

## Features

- **Keras-style API**: Familiar `Sequential` model with `add()`, `fit()`, `predict()`
- **Layers**: Dense, plus Conv2D (any window, grouped), DepthwiseConv2D, MaxPool2D, GlobalAveragePooling2D, BatchNormalization and Flatten for convolutional networks, Conv1D and MaxPool1D for sequences such as time series, Embedding (with padding masks and pre-trained GloVe/word2vec vectors), GlobalMaxPooling1D/2D and a Tokenizer for text, and UpSample2D and Reshape for decoders (autoencoders)
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
    Conv2D(filters=8, kernel_size=3, activation='relu'),   # -> (8, 26, 26)
    MaxPool2D(pool_size=2),                                # -> (8, 13, 13)
    Conv2D(filters=16, kernel_size=3, activation='relu'),  # -> (16, 11, 11)
    MaxPool2D(pool_size=2),                                # -> (16, 5, 5)
    Flatten(),                                             # -> (400,)
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

### Batch Normalization and Global Average Pooling

`BatchNormalization` normalizes each channel (each unit after a `Dense`
layer) over the batch, then scales and shifts it by learned parameters, so
deeper networks train faster and accept higher learning rates. It goes
between a layer and its activation: give the layer before no activation and
the batch norm layer the activation. `GlobalAveragePooling2D` turns each
channel into its mean, replacing `Flatten` + a large `Dense` layer at the end
of a CNN with far fewer parameters:

```python
from pyann import Sequential, Input, Conv2D, BatchNormalization, MaxPool2D, GlobalAveragePooling2D, Dense

model = Sequential([
    Input((1, 28, 28)),
    Conv2D(16, 3, padding='same'),               # no activation here...
    BatchNormalization(activation='relu'),       # ...it comes after normalizing
    MaxPool2D(2),                                # -> (16, 14, 14)
    Conv2D(32, 3, padding='same'),
    BatchNormalization(activation='relu'),
    GlobalAveragePooling2D(),                    # -> (32,)
    Dense(10, activation='softmax'),
])
```

Training uses each batch's mean and variance (train with batches of more
than one sample) and keeps running averages of them (momentum 0.1, as in
PyTorch), which `predict()` and `evaluate()` use; they are saved with the
model and exported to ONNX as `BatchNormalization`. A layer's bias before a
batch norm is redundant, since the batch norm subtracts the mean.
`quantize()` and C export fold each batch norm into the `Dense` or `Conv2D`
layer before it, which must have no activation. Both layers run on the GPU
with Metal or Vulkan.

### Depthwise Separable Convolutions (MobileNet)

`DepthwiseConv2D` filters each channel on its own; a 1x1 `Conv2D` after it
mixes the channels. Together they do the work of one convolution with
several times fewer weights and multiply-adds, which suits small, quantized
models. `Conv2D(..., groups=g)` is the general form, and kernels, strides and
padding may also be `(height, width)` pairs:

```python
from pyann import Sequential, Input, Conv2D, DepthwiseConv2D, BatchNormalization, GlobalAveragePooling2D, Dense

model = Sequential([
    Input((1, 28, 28)),
    Conv2D(32, 3, padding='same'),
    BatchNormalization(activation='relu'),
    DepthwiseConv2D(3, strides=2, padding=1),    # -> (32, 14, 14): 9 weights per channel
    BatchNormalization(activation='relu'),
    Conv2D(64, 1),                               # -> (64, 14, 14): mixes the channels
    BatchNormalization(activation='relu'),
    GlobalAveragePooling2D(),
    Dense(10, activation='softmax'),
])
```

### Sequences and Time Series

`Conv1D` and `MaxPool1D` slide along sequences. A sample of L steps with C
features per step has shape `(C, L)`, channels-first as in PyTorch's
`nn.Conv1d`, so a batch is `(N, C, L)`; channels-last data `(N, L, C)` needs
`x.transpose(0, 2, 1)`:

```python
from pyann import Sequential, Conv1D, BatchNormalization, MaxPool1D, GlobalAveragePooling1D, Dense

# x: (N, 3, 128) float32 sequences of 3 features, y: (N, 4) one-hot labels
model = Sequential([
    Conv1D(16, 5, padding='same', input_shape=(3, 128)),   # -> (16, 128)
    BatchNormalization(activation='relu'),
    MaxPool1D(2),                                          # -> (16, 64)
    Conv1D(32, 3, padding='same', activation='relu'),      # -> (32, 64)
    GlobalAveragePooling1D(),                              # -> (32,)
    Dense(4, activation='softmax'),
])
model.compile(optimizer='adam', loss='cce')
model.fit(x, y, epochs=20, batch_size=32)
```

A model whose output is a sequence (e.g. one that denoises signals) returns
`(N, C, L)` from `predict()`. These layers run on the GPU with Metal or
Vulkan, save and load, export to ONNX, quantize and export to C like the
2-D ones; `load_onnx()` also reads PyTorch `nn.Conv1d` models.

### Text

`Tokenizer` builds a vocabulary and turns texts into lists of word ids
(0 is left for padding; with an `oov_token`, 1 stands for unknown words), and
`pad_sequences` makes them rows of one length. An `Embedding` turns each row
of ids into a sequence of learned vectors `(dim, length)`, so `Conv1D` and the
pooling layers follow; `GlobalMaxPooling1D` keeps each filter's strongest match
wherever it occurs in the text:

```python
from pyann import Sequential, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Tokenizer, pad_sequences

tok = Tokenizer(num_words=10000, oov_token="<unk>").fit_on_texts(train_texts)
x = pad_sequences(tok.texts_to_sequences(train_texts), maxlen=200)    # (N, 200) ids, 0-padded at the end

model = Sequential([
    Embedding(tok.vocab_size, 64, mask_zero=True, input_length=200),  # -> (64, 200)
    Conv1D(128, 5, padding='same', activation='relu'),                # -> (128, 200): 5-word phrases
    GlobalMaxPooling1D(),                                             # -> (128,)
    Dense(2, activation='softmax'),
])
model.compile(optimizer='adam', loss='cce')
model.fit(x, y, epochs=10, batch_size=32)
open("tokenizer.json", "w").write(tok.to_json())   # keep it with the model: new texts need the same ids
```

`mask_zero=True` (or `padding_idx=` with `mask_padding=True`) leaves the
padding out of everything above the embedding: `Conv1D`, `MaxPool1D` and
`BatchNormalization` outputs are zero there, and batch norm statistics and
global pooling skip it, so a text padded at the end gives what it would
unpadded (`pad_sequences` pads at the end by default for this reason; padding
at the start also works when every stride is 1). Without a mask the padding
vector is still zero, but a conv's bias makes the padding steps count. Text
models train on the GPU with Metal or Vulkan, save and load (the mask
included) and export to ONNX; they cannot be quantized yet. See
[`examples/text_classification.py`](examples/text_classification.py).

The `Tokenizer` lower-cases texts and turns ASCII punctuation into spaces,
but knows nothing of HTML: scraped text such as IMDB reviews would give words
like `br` (from `<br />`) and `amp` (from `&amp;`). `preprocess=strip_html`
removes tags, comments and scripts and decodes entities first; any function
from text to text works there (`to_json` saves `strip_html` by name, and your
own function is passed to `from_json` again).

#### Pre-trained word vectors (GloVe, word2vec, fastText)

`load_word_vectors` reads a vector file in text format (GloVe, or word2vec and
fastText `.vec` with their header line; `.gz` is fine) and keeps the vectors of
your vocabulary, row `i` for word id `i`. Give them to the `Embedding` with
`weights=`; `trainable=False` keeps them fixed while the layers above learn:

```python
from pyann import Tokenizer, load_word_vectors, strip_html

tok = Tokenizer(num_words=20000, oov_token="<unk>", preprocess=strip_html).fit_on_texts(train_texts)
matrix, found = load_word_vectors("glove.6B.100d.txt", tok)     # (tok.vocab_size, 100)
print(f"{found} of {tok.vocab_size - 2} words have vectors")    # the rest are zeros (or missing="random")

model = Sequential([
    Embedding(tok.vocab_size, 100, mask_zero=True, input_length=200, weights=matrix, trainable=False),
    Conv1D(128, 5, padding='same', activation='relu'),
    GlobalMaxPooling1D(),
    Dense(2, activation='softmax'),
])
model.compile(optimizer='adam', loss='cce')
model.fit(x, y, epochs=5)

model.set_trainable(0, True)            # then fine-tune the vectors a little
model.compile(learning_rate=1e-4)
model.fit(x, y, epochs=2)
```

The words must match the file's (GloVe's are lower case, as the Tokenizer's
are by default). The binary word2vec format isn't read; convert it to text
first, e.g. with gensim's `save_word2vec_format(path, binary=False)`.

### Layer Weights and Freezing

Every layer's parameters can be read and set as NumPy arrays, to copy layers
between models, start from pre-trained values, or inspect what was learned:

```python
emb_vectors = model.get_layer_weights(0)[0]            # (vocab, dim): row i is id i's vector
np.save("vectors.npy", emb_vectors)                    # keep them on their own

other.set_layer_weights(0, [np.load("vectors.npy")])   # into another model with the same vocabulary
other.set_trainable(0, False)                          # freeze it while the rest trains

clone.set_weights(model.get_weights())                 # every layer (same architecture)
```

`model.layers` lists the layers after the input; a layer is chosen by its
position there (negative counts from the end) or by the layer object itself.
The arrays are float32 copies in PyTorch's layouts:

| Layer | Arrays |
|---|---|
| `Dense` | weights `(units, inputs)`, bias `(units,)` (Keras stores the weights transposed) |
| `Conv2D`, `DepthwiseConv2D` | weights `(filters, channels / groups, kh, kw)`, bias `(filters,)` |
| `Conv1D` | weights `(filters, channels / groups, kernel_size)`, bias `(filters,)` |
| `BatchNormalization` | gamma, beta, running mean, running variance, each `(channels,)` |
| `Embedding` | vectors `(input_dim, output_dim)` |
| others | none (`[]`) |

Before the first `fit()` the weights are already initialized, and are what
training starts from; values you set are kept as the starting point (layers you
didn't set are still initialized randomly). Shapes and finiteness are checked
before anything changes, and a model on the GPU gets the new values too.

A frozen layer (`set_trainable(layer, False)`) keeps its parameters through
`fit()` and `train_on_batch()`, on the CPU or a GPU; gradients still pass
through it to the layers below, and a frozen `BatchNormalization` still
updates its running statistics. `summary()` marks frozen layers. Freezing is
not saved with the model.

### Autoencoders and Denoising

`UpSample2D` (nearest-neighbour, the counterpart of `MaxPool2D`) and `Reshape`
(e.g. a Dense bottleneck back to an image) build decoders. Targets can be
images too, so an autoencoder trains with `fit(x, x)`. With input noise the
model sees corrupted inputs and learns to output the clean ones:

```python
from pyann import Sequential, Input, Conv2D, MaxPool2D, UpSample2D

model = Sequential([
    Input((1, 28, 28)),
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),    # -> (16, 28, 28)
    MaxPool2D(pool_size=2),                                                  # -> (16, 14, 14)
    Conv2D(filters=8, kernel_size=3, padding='same', activation='relu'),     # -> (8, 14, 14)
    MaxPool2D(pool_size=2),                                                  # -> (8, 7, 7)
    UpSample2D(size=2),                                                      # -> (8, 14, 14)
    Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'),    # -> (16, 14, 14)
    UpSample2D(size=2),                                                      # -> (16, 28, 28)
    Conv2D(filters=1, kernel_size=3, padding='same', activation='sigmoid'),  # -> (1, 28, 28)
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
from pyann.utils import load_csv, normalize, one_hot_encode, split_data

# Load CSV data: features, then an integer class label in the last column
data, rows, cols = load_csv('data.csv', has_header=True)
X, labels = data[:, :-1], data[:, -1]

# Normalize features (keep mean and std to apply to test data: normalize(X_test, mean=mean, std=std))
X_norm, mean, std = normalize(X, axis=0)

# One-hot encode labels
y = one_hot_encode(labels, num_classes=10)

# Split into train/validation sets
X_train, y_train, X_val, y_val = split_data(X_norm, y, train_ratio=0.8)
```

`load_csv` reads files of numbers only (like the MNIST CSVs): a field that
isn't a number, or a line with a different number of fields, raises
`ValueError` naming the line and column. For text, such as the IMDB reviews
CSV, `load_text_csv` returns each column as a list of strings, with quoted
commas, quotes and line breaks handled:

```python
from pyann import Tokenizer, pad_sequences, strip_html
from pyann.utils import load_text_csv, one_hot_encode, split_data

data = load_text_csv('IMDB Dataset.csv')            # {'review': [...], 'sentiment': [...]}
labels = [s == 'positive' for s in data['sentiment']]
y = one_hot_encode(labels, num_classes=2)

tok = Tokenizer(num_words=20000, oov_token='<unk>', preprocess=strip_html).fit_on_texts(data['review'])
x = pad_sequences(tok.texts_to_sequences(data['review']), maxlen=200)
x_train, y_train, x_val, y_val = split_data(x, y, train_ratio=0.8, seed=1)
```

`columns=['review']` keeps only some columns (by name, or by index), and
`has_header=False` keys them `0, 1, ...`. `split_data` needs one row per sample
in both `X` and `y`, each row the same length: token-id lists straight from
`texts_to_sequences` raise `ValueError` asking for `pad_sequences` first.

`normalize` and `one_hot_encode` need NumPy; `load_csv`, `load_text_csv` and
`split_data` also work without it (returning lists).

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
- `layers` - The layers after the input, in order
- `get_layer_weights(layer)` / `set_layer_weights(layer, weights)` - One layer's parameters as NumPy arrays; see [Layer Weights and Freezing](#layer-weights-and-freezing)
- `get_weights()` / `set_weights(weights)` - Every layer's parameters, one list
- `set_trainable(layer, trainable=True)` / `is_trainable(layer)` - Freeze a layer, or let it train again

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
Dense(units, activation=None, input_shape=None, dropout=0.0, name=None)
Input(shape)                                     # (features,) or (channels, height, width)
Conv2D(filters, kernel_size=3, strides=1, padding=0, activation=None, input_shape=None, dropout=0.0, name=None)
MaxPool2D(pool_size=2, strides=None, name=None)  # strides default to pool_size; alias MaxPooling2D
Flatten(name=None)                               # optional: Dense layers read image outputs as vectors
UpSample2D(size=2, name=None)                    # nearest-neighbour: (C, H, W) -> (C, H*size, W*size); alias UpSampling2D
Reshape(target_shape, name=None)                 # same values as (channels, height, width), e.g. after a Dense layer
Embedding(input_dim, output_dim, padding_idx=None, mask_padding=False, mask_zero=False, input_length=None, name=None,
          weights=None, trainable=True)
GlobalMaxPooling2D(name=None)                    # each channel's largest value; aliases GlobalMaxPooling1D, GlobalMaxPool2D/1D
GlobalAveragePooling2D(name=None)                # each channel's mean; aliases GlobalAveragePooling1D, GlobalAvgPool2D
```

- `units`: number of neurons (outputs) in the layer.
- `filters`: number of output channels, one learned kernel each.
- `kernel_size`: kernel height and width (kernels are square).
- `strides`: step between kernel or pooling windows.
- `pool_size`: pooling window height and width. `MaxPool2D(pool_size=2)` halves
  height and width (rounding down).
- `size`: upsampling factor. Each value is repeated `size` times along height and width.
- `target_shape`: `(channels, height, width)` to view the incoming values as.
  Its product must equal the previous layer's output size.
- `input_dim`, `output_dim`: an embedding's vocabulary size (ids `0 .. input_dim - 1`)
  and vector length; `padding_idx` pads texts (its vector stays zero);
  `mask_padding` leaves those steps out of the layers above, and `mask_zero`
  means both with padding id 0; `weights` gives starting vectors
  `(input_dim, output_dim)` (e.g. from `load_word_vectors`), and
  `trainable=False` keeps them fixed.
- `name`: optional label for the layer.

`Conv2D` uses square kernels; `padding` is an int, `'valid'` (0) or `'same'`
(keeps the size; odd kernel sizes with stride 1). Output size is
`(size + 2 * padding - kernel_size) // strides + 1`. A model whose layers
don't fit together raises `NetworkError` when it is built.

### Text

```python
Tokenizer(num_words=None, oov_token=None, lower=True, filters=..., split=None, char_level=False, preprocess=None)
pad_sequences(sequences, maxlen=None, padding='post', truncating='post', value=0)
strip_html(text)                                 # tags, comments and scripts removed, entities decoded
load_word_vectors(path, word_index, vocab_size=None, dim=None, missing='zeros', seed=None, padding_idx=0)
```

- `Tokenizer`: `fit_on_texts(texts)`, `texts_to_sequences(texts)`,
  `sequences_to_texts(seqs)`, `vocab_size` (the Embedding's `input_dim`),
  `word_index`, `to_json()` / `Tokenizer.from_json(text, preprocess=None)`.
- `load_word_vectors` returns `(matrix, found)`: a float32 `(vocab_size, dim)`
  array and how many words had vectors. `word_index` is a fitted `Tokenizer`
  or a dict; `missing='random'` draws the absent words' vectors from the
  found ones' mean and standard deviation.

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
