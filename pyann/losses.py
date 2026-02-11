"""Loss functions for neural network training."""

from enum import IntEnum
from typing import Union


class Loss(IntEnum):
    """Loss function types matching libann's Loss_type enum."""
    
    MSE = 0                         # LOSS_MSE - Mean Squared Error
    CATEGORICAL_CROSS_ENTROPY = 1   # LOSS_CATEGORICAL_CROSS_ENTROPY
    
    # Aliases
    MEAN_SQUARED_ERROR = 0
    CCE = 1
    CROSS_ENTROPY = 1
    
    @classmethod
    def from_string(cls, name: str) -> "Loss":
        """Convert string name to Loss enum.
        
        Args:
            name: Loss function name (case-insensitive)
            
        Returns:
            Corresponding Loss enum value
            
        Raises:
            ValueError: If name is not a valid loss function
        """
        name_map = {
            "mse": cls.MSE,
            "mean_squared_error": cls.MSE,
            "categorical_cross_entropy": cls.CATEGORICAL_CROSS_ENTROPY,
            "cce": cls.CCE,
            "cross_entropy": cls.CROSS_ENTROPY,
            "crossentropy": cls.CROSS_ENTROPY,
        }
        key = name.lower().replace("-", "_")
        if key not in name_map:
            valid = ", ".join(sorted(set(name_map.keys())))
            raise ValueError(f"Unknown loss '{name}'. Valid options: {valid}")
        return name_map[key]


def parse_loss(loss: Union[str, Loss]) -> Loss:
    """Parse loss argument to Loss enum.
    
    Args:
        loss: String name or Loss enum
        
    Returns:
        Loss enum value
    """
    if isinstance(loss, Loss):
        return loss
    if isinstance(loss, str):
        return Loss.from_string(loss)
    raise TypeError(f"loss must be str or Loss, got {type(loss)}")
