"""Layer definitions for neural networks."""

from typing import Optional, Union, Tuple
from dataclasses import dataclass

from pyann.activations import Activation, parse_activation


@dataclass
class Layer:
    """Base class for network layers."""
    pass


@dataclass
class Input(Layer):
    """Input layer specification.
    
    Defines the input shape for the network. This is optional as the
    input shape can be inferred from the first layer's input_shape.
    
    Args:
        shape: Input shape as tuple: (features,) for vectors, or
            (channels, height, width) for images. Images are channels-first,
            as in PyTorch and ONNX, so a batch has shape (N, C, H, W).
    """
    shape: Tuple[int, ...]
    
    @property
    def units(self) -> int:
        """Number of input units."""
        if len(self.shape) == 1:
            return self.shape[0]
        # Flatten multi-dimensional input
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass 
class Dense(Layer):
    """Fully connected (dense) layer.
    
    Args:
        units: Number of neurons in this layer
        activation: Activation function (string or Activation enum)
        input_shape: Optional input shape (only needed for first layer)
        dropout: Dropout rate for this layer (0.0 = disabled)
        name: Optional layer name for debugging
        
    Example:
        >>> # Simple layer with 64 units and ReLU activation
        >>> layer = Dense(64, activation='relu')
        >>> 
        >>> # First layer with explicit input shape
        >>> layer = Dense(128, activation='relu', input_shape=(784,))
        >>> 
        >>> # Output layer with softmax for classification
        >>> layer = Dense(10, activation='softmax')
    """
    units: int
    activation: Union[str, Activation, None] = None
    input_shape: Optional[Tuple[int, ...]] = None
    dropout: float = 0.0
    name: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize parameters."""
        if self.units <= 0:
            raise ValueError(f"units must be positive, got {self.units}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        # Normalize activation
        self._activation_enum = parse_activation(self.activation)
    
    @property
    def activation_type(self) -> Activation:
        """Get the Activation enum value."""
        return self._activation_enum
    
    @property
    def input_units(self) -> Optional[int]:
        """Number of input units (if input_shape specified)."""
        if self.input_shape is None:
            return None
        if len(self.input_shape) == 1:
            return self.input_shape[0]
        result = 1
        for dim in self.input_shape:
            result *= dim
        return result


def parse_layer(layer_spec: Union[Layer, int, Tuple[int, str]]) -> Layer:
    """Parse various layer specifications to Layer objects.
    
    Allows shorthand layer specifications:
    - Dense(64, 'relu')  -> Dense layer as-is
    - 64                 -> Dense(64)
    - (64, 'relu')       -> Dense(64, activation='relu')
    
    Args:
        layer_spec: Layer specification
        
    Returns:
        Layer object
    """
    if isinstance(layer_spec, Layer):
        return layer_spec
    if isinstance(layer_spec, int):
        return Dense(layer_spec)
    if isinstance(layer_spec, tuple) and len(layer_spec) == 2:
        units, activation = layer_spec
        return Dense(units, activation=activation)
    raise TypeError(f"Cannot parse layer specification: {layer_spec}")


def _pair(value, name: str) -> int:
    """A square window setting: an int, or a 2-tuple of equal ints."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"{name} must be an int or a pair of equal ints, got {value}")
        value = value[0]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


@dataclass
class Conv2D(Layer):
    """2-D convolution layer with square kernels.
    
    Inputs and outputs are channels-first: (channels, height, width). The
    output has ``filters`` channels and height
    ``(height + 2 * padding - kernel_size) // strides + 1`` (likewise width).
    
    Args:
        filters: Number of output channels
        kernel_size: Kernel height and width (int, or a pair of equal ints)
        strides: Step between windows (int, or a pair of equal ints)
        padding: Zeros added on each side: an int, 'valid' (0), or 'same'
            ((kernel_size - 1) // 2, which keeps the size; needs an odd
            kernel_size and strides of 1)
        activation: Activation function (not softmax)
        input_shape: (channels, height, width), only for the first layer
        dropout: Dropout rate for this layer (0.0 = disabled)
        name: Optional layer name
        
    Example:
        >>> Conv2D(16, 3, padding='same', activation='relu', input_shape=(1, 28, 28))
    """
    filters: int
    kernel_size: Union[int, Tuple[int, int]] = 3
    strides: Union[int, Tuple[int, int]] = 1
    padding: Union[int, str] = 0
    activation: Union[str, Activation, None] = None
    input_shape: Optional[Tuple[int, ...]] = None
    dropout: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError(f"filters must be a positive int, got {self.filters}")
        self.kernel_size = _pair(self.kernel_size, "kernel_size")
        self.strides = _pair(self.strides, "strides")
        if self.kernel_size <= 0 or self.strides <= 0:
            raise ValueError("kernel_size and strides must be positive")
        if self.padding == "valid":
            self.padding = 0
        elif self.padding == "same":
            if self.kernel_size % 2 == 0 or self.strides != 1:
                raise ValueError("padding='same' needs an odd kernel_size and strides=1; give an int padding instead")
            self.padding = (self.kernel_size - 1) // 2
        elif not isinstance(self.padding, int) or isinstance(self.padding, bool) or self.padding < 0:
            raise ValueError(f"padding must be 'valid', 'same' or an int >= 0, got {self.padding!r}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        self._activation_enum = parse_activation(self.activation)
        if self._activation_enum == Activation.SOFTMAX:
            raise ValueError("softmax is not supported on Conv2D layers")

    @property
    def activation_type(self) -> Activation:
        """Get the Activation enum value."""
        return self._activation_enum

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Output (channels, height, width) for an input of the given shape."""
        _, h, w = input_shape
        size = lambda n: (n + 2 * self.padding - self.kernel_size) // self.strides + 1
        return (self.filters, size(h), size(w))


@dataclass
class MaxPool2D(Layer):
    """2-D max pooling layer (no padding), applied to each channel.
    
    Output height is ``(height - pool_size) // strides + 1`` (likewise width).
    
    Args:
        pool_size: Window height and width (int, or a pair of equal ints)
        strides: Step between windows; defaults to pool_size
        name: Optional layer name
    """
    pool_size: Union[int, Tuple[int, int]] = 2
    strides: Union[int, Tuple[int, int], None] = None
    name: Optional[str] = None

    def __post_init__(self):
        self.pool_size = _pair(self.pool_size, "pool_size")
        self.strides = self.pool_size if self.strides is None else _pair(self.strides, "strides")
        if self.pool_size <= 0 or self.strides <= 0:
            raise ValueError("pool_size and strides must be positive")

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Output (channels, height, width) for an input of the given shape."""
        c, h, w = input_shape
        size = lambda n: (n - self.pool_size) // self.strides + 1
        return (c, size(h), size(w))


#: Keras-style alias
MaxPooling2D = MaxPool2D


@dataclass
class Flatten(Layer):
    """Flatten (channels, height, width) into a vector.
    
    Optional: a Dense layer after a Conv2D or MaxPool2D layer reads its
    output as a vector anyway. It makes the structure explicit and maps to
    an ONNX Flatten node.
    """
    name: Optional[str] = None
