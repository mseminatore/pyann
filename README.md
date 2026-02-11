# PyANN

Python wrapper for the [libann](https://github.com/mseminatore/ann) neural network library with a Keras-like API.

## Features

- **Keras-style API**: Familiar `Sequential` model with `add()`, `fit()`, `predict()`
- **Multiple optimizers**: SGD, Momentum, RMSProp, AdaGrad, Adam
- **Activation functions**: Sigmoid, ReLU, LeakyReLU, Tanh, Softsign, Softmax
- **Loss functions**: MSE, Categorical Cross-Entropy
- **Learning rate schedulers**: Step decay, Exponential decay, Cosine annealing
- **Hyperparameter tuning**: Grid search, Random search, Bayesian optimization
- **NumPy optional**: Works with Python lists, optimized for NumPy arrays
- **Model persistence**: Save/load in text or binary format, ONNX export

## Installation

### Prerequisites

1. Build the libann shared library:

```bash
# Clone pyann and the ann submodule
git clone https://github.com/mseminatore/pyann.git
cd pyann
git submodule add https://github.com/mseminatore/ann.git ann

# Build the shared library
python build_lib.py
```

2. Install the Python package:

```bash
pip install -e .
```

### With BLAS acceleration

```bash
python build_lib.py --cblas  # or --blas for OpenBLAS
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

### Save and Load

```python
# Save
model.save('model.nna')  # Text format
model.save('model.nnb', format='binary')  # Binary format
model.export_onnx('model.onnx.json')  # ONNX format

# Load
loaded = Sequential.load('model.nna')
```

## API Reference

### Sequential

```python
Sequential(layers=None, optimizer=None, loss=Loss.MSE, name=None)
```

Methods:
- `add(layer)` - Add a layer
- `compile(optimizer, loss, learning_rate)` - Configure training
- `fit(x, y, epochs, batch_size, ...)` - Train the model
- `predict(x)` - Generate predictions
- `evaluate(x, y)` - Compute accuracy
- `save(filepath, format)` - Save model
- `load(filepath, format)` - Load model (classmethod)
- `summary()` - Get model summary

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
Loss.MSE                        # Mean Squared Error
Loss.CATEGORICAL_CROSS_ENTROPY  # For classification
```

### Activations

```python
Activation.NONE      # Linear
Activation.SIGMOID
Activation.RELU
Activation.LEAKY_RELU
Activation.TANH
Activation.SOFTSIGN
Activation.SOFTMAX
```

## License

MIT License - see [LICENSE](LICENSE) for details.
