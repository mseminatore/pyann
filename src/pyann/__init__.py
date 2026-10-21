"""
PyANN - Python wrapper for libann neural network library.

A Keras-like API for training and inference with neural networks,
powered by the lightweight libann C library.
"""

try:
    # Single source of truth: the version in pyproject.toml, via installed metadata
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("pylibann")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0+unknown"

from pyann.core.network import Sequential
from pyann.core.layers import Dense, Input, Conv2D, MaxPool2D, MaxPooling2D, Flatten
from pyann.optimizers import Optimizer, SGD, Momentum, RMSProp, AdaGrad, Adam
from pyann.losses import Loss
from pyann.activations import Activation
from pyann.weight_init import WeightInit
from pyann._bindings.ffi import (
    get_backend_info as backend_info,
    enable_gpu,
    gpu_enabled,
    set_num_threads,
    get_num_threads,
)

__all__ = [
    "Sequential",
    "Dense",
    "Input",
    "Conv2D",
    "MaxPool2D",
    "MaxPooling2D",
    "Flatten",
    "Optimizer",
    "SGD",
    "Momentum",
    "RMSProp",
    "AdaGrad",
    "Adam",
    "Loss",
    "Activation",
    "WeightInit",
    "backend_info",
    "enable_gpu",
    "gpu_enabled",
    "set_num_threads",
    "get_num_threads",
]
