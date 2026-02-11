"""Activation functions for neural network layers."""

from enum import IntEnum
from typing import Union


class Activation(IntEnum):
    """Activation function types matching libann's Activation_type enum."""
    
    NONE = 0        # ACTIVATION_NULL - no activation (linear)
    SIGMOID = 1     # ACTIVATION_SIGMOID
    RELU = 2        # ACTIVATION_RELU
    LEAKY_RELU = 3  # ACTIVATION_LEAKY_RELU
    TANH = 4        # ACTIVATION_TANH
    SOFTSIGN = 5    # ACTIVATION_SOFTSIGN
    SOFTMAX = 6     # ACTIVATION_SOFTMAX
    
    # Aliases for common naming conventions
    LINEAR = 0
    
    @classmethod
    def from_string(cls, name: str) -> "Activation":
        """Convert string name to Activation enum.
        
        Args:
            name: Activation name (case-insensitive)
            
        Returns:
            Corresponding Activation enum value
            
        Raises:
            ValueError: If name is not a valid activation
        """
        name_map = {
            "none": cls.NONE,
            "linear": cls.LINEAR,
            "sigmoid": cls.SIGMOID,
            "relu": cls.RELU,
            "leaky_relu": cls.LEAKY_RELU,
            "leakyrelu": cls.LEAKY_RELU,
            "tanh": cls.TANH,
            "softsign": cls.SOFTSIGN,
            "softmax": cls.SOFTMAX,
        }
        key = name.lower().replace("-", "_")
        if key not in name_map:
            valid = ", ".join(sorted(set(name_map.keys())))
            raise ValueError(f"Unknown activation '{name}'. Valid options: {valid}")
        return name_map[key]


def parse_activation(activation: Union[str, Activation, None]) -> Activation:
    """Parse activation argument to Activation enum.
    
    Args:
        activation: String name, Activation enum, or None
        
    Returns:
        Activation enum value (NONE if None provided)
    """
    if activation is None:
        return Activation.NONE
    if isinstance(activation, Activation):
        return activation
    if isinstance(activation, str):
        return Activation.from_string(activation)
    raise TypeError(f"activation must be str, Activation, or None, got {type(activation)}")
