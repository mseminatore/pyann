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


@dataclass
class UpSample2D(Layer):
    """2-D nearest-neighbour upsampling, applied to each channel.
    
    Each value is repeated in a ``size`` x ``size`` block, so the output is
    (channels, height * size, width * size). The decoder's counterpart to
    MaxPool2D, e.g. in autoencoders; maps to an ONNX Resize node.
    
    Args:
        size: Integer scale for height and width (int, or a pair of equal ints)
        name: Optional layer name
    """
    size: Union[int, Tuple[int, int]] = 2
    name: Optional[str] = None

    def __post_init__(self):
        self.size = _pair(self.size, "size")
        if self.size <= 0:
            raise ValueError(f"size must be positive, got {self.size}")

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Output (channels, height, width) for an input of the given shape."""
        c, h, w = input_shape
        return (c, h * self.size, w * self.size)


#: Keras-style alias
UpSampling2D = UpSample2D


@dataclass
class GlobalAveragePooling2D(Layer):
    """Global average pooling: each channel's mean over height and width.
    
    The output is a vector of one value per channel, so a Dense layer can
    follow directly. Used in place of Flatten + a large Dense layer at the
    end of a CNN, it removes most of the parameters, which keeps quantized
    and exported models small. Maps to an ONNX GlobalAveragePool node.
    
    Args:
        name: Optional layer name
    """
    name: Optional[str] = None

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int]:
        """Output (channels,) for an input of the given shape."""
        return (input_shape[0],)


#: Short alias
GlobalAvgPool2D = GlobalAveragePooling2D


@dataclass
class BatchNormalization(Layer):
    """Batch normalization, then an optional activation.
    
    Normalizes each channel (each unit after a Dense layer) to zero mean and
    unit variance, then scales and shifts it by a learned gamma and beta:
    ``activation(gamma * (x - mean) / sqrt(var + 1e-5) + beta)``. Training
    uses each batch's mean and variance (so train with batches of more than
    one sample) and keeps running averages of them (momentum 0.1, as
    PyTorch's default), which ``predict()`` and evaluation use. The output
    has the input's shape.
    
    Put it between a layer and its activation, so the layer before has
    none; that layer's bias is then redundant. Quantization and C export fold
    it into that Dense or Conv2D layer. Maps to an ONNX BatchNormalization
    node.
    
    Args:
        activation: Activation applied after normalizing (default none)
        name: Optional layer name
        
    Example:
        >>> Conv2D(16, 3, padding='same')            # no activation here...
        >>> BatchNormalization(activation='relu')    # ...it comes after normalizing
    """
    activation: Union[str, Activation, None] = None
    name: Optional[str] = None

    def __post_init__(self):
        self._activation_enum = parse_activation(self.activation)

    @property
    def activation_type(self) -> Activation:
        """Get the Activation enum value."""
        return self._activation_enum


#: Short alias
BatchNorm = BatchNormalization


@dataclass
class Reshape(Layer):
    """Give the previous layer's values the shape (channels, height, width).
    
    The values stay in place (channels, then rows, then columns), so this
    turns e.g. a Dense layer of C*H*W units into an image that Conv2D,
    MaxPool2D and UpSample2D layers can use. Maps to an ONNX Reshape node.
    
    Args:
        target_shape: (channels, height, width); its size must equal the
            previous layer's
        name: Optional layer name
    """
    target_shape: Tuple[int, int, int]
    name: Optional[str] = None

    def __post_init__(self):
        shape = tuple(self.target_shape)
        if len(shape) != 3 or not all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in shape):
            raise ValueError(f"target_shape must be (channels, height, width) of positive ints, got {self.target_shape}")
        self.target_shape = shape
