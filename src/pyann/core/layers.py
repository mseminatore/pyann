"""Layer definitions for neural networks."""

from typing import Any, Optional, Union, Tuple
from dataclasses import dataclass, field

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


def _axes(value, name: str) -> Tuple[int, int]:
    """A window setting for (height, width): an int for both, or a pair."""
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a (height, width) pair, got {value}")
        h, w = value
    else:
        h = w = value
    for v in (h, w):
        if not isinstance(v, int) or isinstance(v, bool):
            raise TypeError(f"{name} must be an int or a pair of ints, got {value!r}")
    return h, w


def _compact(pair: Tuple[int, int]) -> Union[int, Tuple[int, int]]:
    """An int when both axes agree, else the (height, width) pair."""
    return pair[0] if pair[0] == pair[1] else tuple(pair)


def _window_size(n: int, kernel: int, stride: int, padding: int) -> int:
    return (n + 2 * padding - kernel) // stride + 1


def _conv_padding(padding, kernel: Tuple[int, int], strides: Tuple[int, int]) -> Tuple[int, int]:
    """'valid' (0), 'same' (keeps the size: odd kernels and strides of 1), an int, or a pair."""
    if padding == "valid":
        return (0, 0)
    if padding == "same":
        if kernel[0] % 2 == 0 or kernel[1] % 2 == 0 or strides != (1, 1):
            raise ValueError("padding='same' needs odd kernel sizes and strides of 1; give an int padding instead")
        return ((kernel[0] - 1) // 2, (kernel[1] - 1) // 2)
    if isinstance(padding, str):
        raise ValueError(f"padding must be 'valid', 'same', an int or a pair, got {padding!r}")
    pads = _axes(padding, "padding")
    if min(pads) < 0:
        raise ValueError(f"padding must be >= 0, got {padding!r}")
    return pads


class _ConvBase(Layer):
    """What Sequential needs from a convolution layer: its window
    (kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w), and its
    filters and groups for an input of a given number of channels."""

    def window(self) -> Tuple[int, int, int, int, int, int]:
        raise NotImplementedError

    def output_filters(self, in_channels: int) -> int:
        return self.filters

    def conv_groups(self, in_channels: int) -> int:
        return self.groups

    @property
    def activation_type(self) -> Activation:
        """Get the Activation enum value."""
        return self._activation_enum

    def _check_common(self):
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        self._activation_enum = parse_activation(self.activation)
        if self._activation_enum == Activation.SOFTMAX:
            raise ValueError(f"softmax is not supported on {type(self).__name__} layers")

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Output (channels, height, width) for an input of the given shape."""
        c, h, w = input_shape
        kh, kw, sh, sw, ph, pw = self.window()
        return (self.output_filters(c), _window_size(h, kh, sh, ph), _window_size(w, kw, sw, pw))

    def param_count(self, in_channels: int) -> int:
        kh, kw = self.window()[:2]
        filters, groups = self.output_filters(in_channels), self.conv_groups(in_channels)
        return filters * (in_channels // groups * kh * kw + 1)


@dataclass
class Conv2D(_ConvBase):
    """2-D convolution layer.
    
    Inputs and outputs are channels-first: (channels, height, width). The
    output has ``filters`` channels and height
    ``(height + 2 * padding - kernel_size) // strides + 1`` (likewise width,
    with its own settings when they are pairs).
    
    Args:
        filters: Number of output channels
        kernel_size: Kernel size: an int, or a (height, width) pair
        strides: Step between windows: an int, or a (height, width) pair
        padding: Zeros added on each side: an int or a (height, width) pair,
            'valid' (0), or 'same' ((kernel_size - 1) // 2 per axis, which keeps
            the size; needs odd kernel sizes and strides of 1)
        groups: Split the input channels and the filters into this many
            groups, each group's filters seeing only its channels (both must
            be divisible by it); groups = channels is a depthwise conv
        activation: Activation function (not softmax)
        input_shape: (channels, height, width), only for the first layer
        dropout: Dropout rate for this layer (0.0 = disabled)
        name: Optional layer name
        
    Square settings read back as ints, non-square ones as (height, width) pairs.
        
    Example:
        >>> Conv2D(16, 3, padding='same', activation='relu', input_shape=(1, 28, 28))
    """
    filters: int
    kernel_size: Union[int, Tuple[int, int]] = 3
    strides: Union[int, Tuple[int, int]] = 1
    padding: Union[int, str, Tuple[int, int]] = 0
    activation: Union[str, Activation, None] = None
    input_shape: Optional[Tuple[int, ...]] = None
    dropout: float = 0.0
    name: Optional[str] = None
    groups: int = 1

    def __post_init__(self):
        if not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError(f"filters must be a positive int, got {self.filters}")
        if not isinstance(self.groups, int) or isinstance(self.groups, bool) or self.groups <= 0:
            raise ValueError(f"groups must be a positive int, got {self.groups}")
        if self.filters % self.groups:
            raise ValueError(f"filters ({self.filters}) must be a multiple of groups ({self.groups})")
        kernel, strides = _axes(self.kernel_size, "kernel_size"), _axes(self.strides, "strides")
        if min(kernel) <= 0 or min(strides) <= 0:
            raise ValueError("kernel_size and strides must be positive")
        pads = _conv_padding(self.padding, kernel, strides)
        self.kernel_size, self.strides, self.padding = _compact(kernel), _compact(strides), _compact(pads)
        self._window = kernel + strides + pads
        self._check_common()

    def window(self) -> Tuple[int, int, int, int, int, int]:
        kh, kw, sh, sw, ph, pw = self._window
        return (kh, kw, sh, sw, ph, pw)


@dataclass
class DepthwiseConv2D(_ConvBase):
    """Depthwise 2-D convolution: each input channel convolved with its own
    ``depth_multiplier`` filters, without mixing channels.
    
    The first half of a depthwise separable convolution (MobileNet): follow
    it with a 1x1 Conv2D to mix the channels, usually with
    BatchNormalization and an activation after each. Output channels are
    ``channels * depth_multiplier``. A Conv2D with groups = channels.
    
    Args:
        kernel_size: Kernel size: an int, or a (height, width) pair
        strides: Step between windows: an int, or a (height, width) pair
        padding: As for Conv2D: an int, a pair, 'valid' or 'same'
        depth_multiplier: Filters per input channel
        activation: Activation function (not softmax)
        dropout: Dropout rate for this layer (0.0 = disabled)
        name: Optional layer name
    """
    kernel_size: Union[int, Tuple[int, int]] = 3
    strides: Union[int, Tuple[int, int]] = 1
    padding: Union[int, str, Tuple[int, int]] = 0
    depth_multiplier: int = 1
    activation: Union[str, Activation, None] = None
    dropout: float = 0.0
    name: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.depth_multiplier, int) or isinstance(self.depth_multiplier, bool) or \
                self.depth_multiplier <= 0:
            raise ValueError(f"depth_multiplier must be a positive int, got {self.depth_multiplier}")
        kernel, strides = _axes(self.kernel_size, "kernel_size"), _axes(self.strides, "strides")
        if min(kernel) <= 0 or min(strides) <= 0:
            raise ValueError("kernel_size and strides must be positive")
        pads = _conv_padding(self.padding, kernel, strides)
        self.kernel_size, self.strides, self.padding = _compact(kernel), _compact(strides), _compact(pads)
        self._window = kernel + strides + pads
        self._check_common()

    def window(self) -> Tuple[int, int, int, int, int, int]:
        kh, kw, sh, sw, ph, pw = self._window
        return (kh, kw, sh, sw, ph, pw)

    def output_filters(self, in_channels: int) -> int:
        return in_channels * self.depth_multiplier

    def conv_groups(self, in_channels: int) -> int:
        return in_channels


@dataclass
class Conv1D(_ConvBase):
    """1-D convolution over sequences, such as time series.
    
    A sequence of L steps with C features (channels) per step is the sample
    shape (C, L), channels-first as in PyTorch's nn.Conv1d (a batch is
    (N, C, L)); give input_shape=(C, L). The layer slides a window of
    kernel_size steps along it; the output is (filters, L_out) with
    ``L_out = (L + 2 * padding - kernel_size) // strides + 1``.
    
    Args:
        filters: Number of output channels
        kernel_size: Window length in steps
        strides: Step between windows
        padding: Zeros added at each end: an int, 'valid' (0), or 'same'
            ((kernel_size - 1) // 2; needs an odd kernel_size and strides of 1)
        groups: Split the channels and filters into groups, as in Conv2D
        activation: Activation function (not softmax)
        input_shape: (channels, steps), only for the first layer
        dropout: Dropout rate for this layer (0.0 = disabled)
        name: Optional layer name
        
    Example:
        >>> Conv1D(16, 5, padding='same', activation='relu', input_shape=(3, 128))
    """
    filters: int
    kernel_size: int = 3
    strides: int = 1
    padding: Union[int, str] = 0
    activation: Union[str, Activation, None] = None
    input_shape: Optional[Tuple[int, ...]] = None
    dropout: float = 0.0
    name: Optional[str] = None
    groups: int = 1

    def __post_init__(self):
        if not isinstance(self.filters, int) or self.filters <= 0:
            raise ValueError(f"filters must be a positive int, got {self.filters}")
        if not isinstance(self.groups, int) or isinstance(self.groups, bool) or self.groups <= 0:
            raise ValueError(f"groups must be a positive int, got {self.groups}")
        if self.filters % self.groups:
            raise ValueError(f"filters ({self.filters}) must be a multiple of groups ({self.groups})")
        for name, v in (("kernel_size", self.kernel_size), ("strides", self.strides)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"{name} must be an int, got {v!r}")
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        if isinstance(self.padding, (tuple, list)):
            raise TypeError("padding must be an int, 'valid' or 'same'")
        self.padding = _conv_padding(self.padding, (1, self.kernel_size), (1, self.strides))[1]
        self._check_common()

    def window(self) -> Tuple[int, int, int, int, int, int]:
        return (1, self.kernel_size, 1, self.strides, 0, self.padding)


class _PoolBase(Layer):
    """What Sequential needs from a max pooling layer: its window
    (pool_h, pool_w, stride_h, stride_w)."""

    def window(self) -> Tuple[int, int, int, int]:
        raise NotImplementedError

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        """Output (channels, height, width) for an input of the given shape."""
        c, h, w = input_shape
        kh, kw, sh, sw = self.window()
        return (c, _window_size(h, kh, sh, 0), _window_size(w, kw, sw, 0))


@dataclass
class MaxPool2D(_PoolBase):
    """2-D max pooling layer (no padding), applied to each channel.
    
    Output height is ``(height - pool_size) // strides + 1`` (likewise width,
    with its own settings when they are pairs).
    
    Args:
        pool_size: Window size: an int, or a (height, width) pair
        strides: Step between windows (an int or a pair); defaults to pool_size
        name: Optional layer name
    """
    pool_size: Union[int, Tuple[int, int]] = 2
    strides: Union[int, Tuple[int, int], None] = None
    name: Optional[str] = None

    def __post_init__(self):
        size = _axes(self.pool_size, "pool_size")
        strides = size if self.strides is None else _axes(self.strides, "strides")
        if min(size) <= 0 or min(strides) <= 0:
            raise ValueError("pool_size and strides must be positive")
        self.pool_size, self.strides = _compact(size), _compact(strides)
        self._window = size + strides

    def window(self) -> Tuple[int, int, int, int]:
        kh, kw, sh, sw = self._window
        return (kh, kw, sh, sw)


@dataclass
class MaxPool1D(_PoolBase):
    """1-D max pooling over sequences (channels, steps), applied to each channel.
    
    Args:
        pool_size: Window length in steps
        strides: Step between windows; defaults to pool_size
        name: Optional layer name
    """
    pool_size: int = 2
    strides: Optional[int] = None
    name: Optional[str] = None

    def __post_init__(self):
        if self.strides is None:
            self.strides = self.pool_size
        for name, v in (("pool_size", self.pool_size), ("strides", self.strides)):
            if not isinstance(v, int) or isinstance(v, bool):
                raise TypeError(f"{name} must be an int, got {v!r}")
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

    def window(self) -> Tuple[int, int, int, int]:
        return (1, self.pool_size, 1, self.strides)


#: Keras-style alias
MaxPooling1D = MaxPool1D


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
        size = _axes(self.size, "size")
        if size[0] != size[1]:
            raise ValueError(f"size must be an int or a pair of equal ints, got {self.size}")
        self.size = size[0]
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

#: Keras-style alias for sequences (channels, steps): the mean over the steps
GlobalAveragePooling1D = GlobalAveragePooling2D


@dataclass
class GlobalMaxPooling2D(Layer):
    """Global max pooling: each channel's largest value over height and width.
    
    The output is a vector of one value per channel, so a Dense layer can
    follow directly. Over a sequence (channels, steps) it keeps each
    feature's strongest response wherever it occurs, the usual pooling for
    text after an Embedding and Conv1D layers. Over a masked sequence (see
    Embedding) only the kept steps count. Maps to an ONNX GlobalMaxPool node.
    
    Args:
        name: Optional layer name
    """
    name: Optional[str] = None

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int]:
        """Output (channels,) for an input of the given shape."""
        return (input_shape[0],)


#: Aliases: Keras's 1-D name (over the steps of a sequence) and short names
GlobalMaxPooling1D = GlobalMaxPooling2D
GlobalMaxPool2D = GlobalMaxPooling2D
GlobalMaxPool1D = GlobalMaxPooling2D


@dataclass
class Embedding(Layer):
    """Token ids to learned vectors: the first layer of a text model.
    
    Each sample is a row of ``input_length`` token ids (integers in
    ``[0, input_dim)``, e.g. from ``pyann.text.Tokenizer`` and
    ``pad_sequences``). The output is the sequence (output_dim, input_length),
    channels-first as for Conv1D, so Conv1D, MaxPool1D, BatchNormalization
    and the global pooling layers follow directly. Vectors start
    uniform(-0.05, 0.05), as in Keras (PyTorch's N(0, 1) trains far slower
    with Adam), or from ``weights``.
    
    ``padding_idx`` (as in PyTorch) is the id that pads texts to the same
    length: its vector is always zero and never trained; ids outside the
    vocabulary also give zeros. With ``mask_padding`` (or Keras's
    ``mask_zero``, which makes 0 the padding id) the padding steps are also
    left out of the layers above: Conv1D, MaxPool1D and BatchNormalization
    outputs are zero there, batch norm statistics and global pooling skip
    them, so a text padded at the end gives what it would unpadded.
    
    Args:
        input_dim: Vocabulary size (number of token ids)
        output_dim: Length of each vector
        padding_idx: Id that pads texts, or None
        mask_padding: Leave the padding steps out of the layers above
            (needs padding_idx)
        mask_zero: Keras's flag: padding_idx = 0 and mask_padding
        input_length: Ids per sample; sets the model's input shape
            (input_length,) when this is the first layer
        name: Optional layer name
        weights: Starting vectors, an (input_dim, output_dim) array whose row
            i is id i's vector, e.g. pre-trained GloVe or word2vec vectors
            from ``pyann.text.load_word_vectors``; None starts them random
        trainable: False keeps the vectors as they are while the rest of the
            model trains (see ``Sequential.set_trainable``)
        
    Example:
        >>> Embedding(10000, 64, mask_zero=True, input_length=200)
        >>> matrix, found = load_word_vectors("glove.6B.100d.txt", tok)
        >>> Embedding(tok.vocab_size, 100, mask_zero=True, weights=matrix, trainable=False)
    """
    input_dim: int
    output_dim: int
    padding_idx: Optional[int] = None
    mask_padding: bool = False
    mask_zero: bool = False
    input_length: Optional[int] = None
    name: Optional[str] = None
    weights: Any = field(default=None, repr=False, compare=False)
    trainable: bool = True

    def __post_init__(self):
        for name in ("input_dim", "output_dim"):
            v = getattr(self, name)
            if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive int, got {v!r}")
        if self.mask_zero:
            if self.padding_idx not in (None, 0):
                raise ValueError(f"mask_zero makes 0 the padding id, but padding_idx is {self.padding_idx}")
            self.padding_idx, self.mask_padding = 0, True
        if self.padding_idx is not None and (not isinstance(self.padding_idx, int) or isinstance(self.padding_idx, bool)
                                             or not 0 <= self.padding_idx < self.input_dim):
            raise ValueError(f"padding_idx must be None or an int in [0, {self.input_dim}), got {self.padding_idx!r}")
        if self.mask_padding and self.padding_idx is None:
            raise ValueError("mask_padding needs a padding_idx (or use mask_zero=True)")
        if self.input_length is not None and (not isinstance(self.input_length, int) or
                                              isinstance(self.input_length, bool) or self.input_length <= 0):
            raise ValueError(f"input_length must be a positive int, got {self.input_length!r}")
        if self.weights is not None:
            from pyann.utils.compat import has_numpy
            if not has_numpy():
                raise RuntimeError("Embedding weights need NumPy")
            import numpy as np
            self.weights = np.array(self.weights, dtype=np.float32)
            if self.weights.shape != (self.input_dim, self.output_dim):
                raise ValueError(f"weights must have shape (input_dim, output_dim) = "
                                 f"{(self.input_dim, self.output_dim)}, got {self.weights.shape}")
            if not np.isfinite(self.weights).all():
                raise ValueError("weights must be finite (no NaN or infinity)")

    @property
    def input_shape(self) -> Optional[Tuple[int]]:
        """(input_length,) when given, else None."""
        return None if self.input_length is None else (self.input_length,)

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, int, int]:
        """The sequence (output_dim, 1, steps) for a flat input of ids."""
        return (self.output_dim, 1, input_shape[0])

    def param_count(self) -> int:
        return self.input_dim * self.output_dim


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
