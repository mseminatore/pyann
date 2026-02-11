"""Optimizers for neural network training."""

from enum import IntEnum
from typing import Union, Optional
from dataclasses import dataclass


class OptimizerType(IntEnum):
    """Optimizer types matching libann's Optimizer_type enum."""
    
    SGD = 0       # OPT_SGD - Stochastic Gradient Descent
    MOMENTUM = 1  # OPT_MOMENTUM
    RMSPROP = 2   # OPT_RMSPROP
    ADAGRAD = 3   # OPT_ADAGRAD
    ADAM = 4      # OPT_ADAM
    
    @classmethod
    def from_string(cls, name: str) -> "OptimizerType":
        """Convert string name to OptimizerType enum."""
        name_map = {
            "sgd": cls.SGD,
            "momentum": cls.MOMENTUM,
            "rmsprop": cls.RMSPROP,
            "adagrad": cls.ADAGRAD,
            "adam": cls.ADAM,
        }
        key = name.lower()
        if key not in name_map:
            valid = ", ".join(sorted(name_map.keys()))
            raise ValueError(f"Unknown optimizer '{name}'. Valid options: {valid}")
        return name_map[key]


@dataclass
class Optimizer:
    """Base optimizer configuration.
    
    Attributes:
        optimizer_type: The type of optimizer algorithm
        learning_rate: Base learning rate (default varies by optimizer)
    """
    optimizer_type: OptimizerType
    learning_rate: Optional[float] = None
    
    def _get_type(self) -> OptimizerType:
        """Get the optimizer type enum value."""
        return self.optimizer_type


@dataclass
class SGD(Optimizer):
    """Stochastic Gradient Descent optimizer.
    
    Args:
        lr: Learning rate (default: 0.05)
    """
    def __init__(self, lr: float = 0.05):
        super().__init__(OptimizerType.SGD, lr)


@dataclass
class Momentum(Optimizer):
    """SGD with Momentum optimizer.
    
    Args:
        lr: Learning rate (default: 0.01)
        momentum: Momentum factor (not configurable in libann, uses default)
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(OptimizerType.MOMENTUM, lr)


@dataclass
class RMSProp(Optimizer):
    """RMSProp optimizer.
    
    Args:
        lr: Learning rate (default: 0.001)
    """
    def __init__(self, lr: float = 0.001):
        super().__init__(OptimizerType.RMSPROP, lr)


@dataclass
class AdaGrad(Optimizer):
    """AdaGrad optimizer.
    
    Args:
        lr: Learning rate (default: 0.01)
    """
    def __init__(self, lr: float = 0.01):
        super().__init__(OptimizerType.ADAGRAD, lr)


@dataclass
class Adam(Optimizer):
    """Adam optimizer (recommended default).
    
    Args:
        lr: Learning rate (default: 0.001)
    """
    def __init__(self, lr: float = 0.001):
        super().__init__(OptimizerType.ADAM, lr)


def parse_optimizer(optimizer: Union[str, Optimizer, OptimizerType, None]) -> Optimizer:
    """Parse optimizer argument to Optimizer instance.
    
    Args:
        optimizer: String name, Optimizer instance, OptimizerType enum, or None
        
    Returns:
        Optimizer instance (defaults to Adam if None)
    """
    if optimizer is None:
        return Adam()
    if isinstance(optimizer, Optimizer):
        return optimizer
    if isinstance(optimizer, OptimizerType):
        # Create default optimizer for the type
        defaults = {
            OptimizerType.SGD: SGD,
            OptimizerType.MOMENTUM: Momentum,
            OptimizerType.RMSPROP: RMSProp,
            OptimizerType.ADAGRAD: AdaGrad,
            OptimizerType.ADAM: Adam,
        }
        return defaults[optimizer]()
    if isinstance(optimizer, str):
        opt_type = OptimizerType.from_string(optimizer)
        return parse_optimizer(opt_type)
    raise TypeError(f"optimizer must be str, Optimizer, OptimizerType, or None, got {type(optimizer)}")
