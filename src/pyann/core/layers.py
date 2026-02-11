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
    input shape can be inferred from the first Dense layer's input_shape.
    
    Args:
        shape: Input shape as tuple (e.g., (784,) for 784 features)
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
