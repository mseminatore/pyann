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
    # LOSS_BINARY_CROSS_ENTROPY: the same loss, binary per output with a sigmoid output layer
    BINARY_CROSS_ENTROPY = 1
    BCE = 1
    
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
            "binary_cross_entropy": cls.BINARY_CROSS_ENTROPY,
            "binary_crossentropy": cls.BINARY_CROSS_ENTROPY,
            "bce": cls.BCE,
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


class LossReduction(IntEnum):
    """How training combines a batch's errors, matching libann's Loss_reduction enum.

    The reported loss is always a mean (MSE over the batch and the outputs,
    cross-entropy over the batch). SUM, libann's default, trains on the
    gradient summed over the batch, and for MSE on half the summed squared
    errors over the outputs, so SGD steps grow with the batch size and the
    number of outputs. MEAN trains on the gradient of the reported loss, as
    Keras and PyTorch do.
    """

    SUM = 0     # LOSS_REDUCTION_SUM
    MEAN = 1    # LOSS_REDUCTION_MEAN

    @classmethod
    def from_string(cls, name: str) -> "LossReduction":
        """Convert "sum" or "mean" (any case) to a LossReduction."""
        try:
            return cls[name.upper()]
        except KeyError:
            raise ValueError(f"Unknown loss reduction '{name}'. Valid options: mean, sum") from None


def parse_loss_reduction(reduction: Union[str, LossReduction]) -> LossReduction:
    """Parse a loss reduction argument (string or enum) to a LossReduction."""
    if isinstance(reduction, LossReduction):
        return reduction
    if isinstance(reduction, str):
        return LossReduction.from_string(reduction)
    raise TypeError(f"loss_reduction must be str or LossReduction, got {type(reduction)}")
