"""Learning rate schedulers and training callbacks."""

from typing import Callable, Optional, Any
from dataclasses import dataclass


# Type alias for LR scheduler function signature
LRSchedulerFunc = Callable[[int, float, Any], float]


@dataclass
class LRScheduler:
    """Base class for learning rate schedulers."""
    pass


@dataclass
class StepDecay(LRScheduler):
    """Step decay scheduler - multiply LR by gamma every step_size epochs.
    
    Args:
        step_size: Number of epochs between LR reductions
        gamma: Multiplicative factor (default: 0.5, halves LR)
    
    Example:
        >>> scheduler = StepDecay(step_size=10, gamma=0.5)
        >>> # LR halves every 10 epochs
    """
    step_size: int
    gamma: float = 0.5


@dataclass 
class ExponentialDecay(LRScheduler):
    """Exponential decay scheduler - multiply LR by gamma each epoch.
    
    Args:
        gamma: Decay rate per epoch (default: 0.95, 5% reduction)
    
    Example:
        >>> scheduler = ExponentialDecay(gamma=0.95)
        >>> # LR reduces by 5% each epoch
    """
    gamma: float = 0.95


@dataclass
class CosineAnnealing(LRScheduler):
    """Cosine annealing scheduler - smooth decay from base LR to min LR.
    
    Args:
        T_max: Maximum number of epochs for one cycle
        min_lr: Minimum learning rate (default: 0.0001)
    
    Example:
        >>> scheduler = CosineAnnealing(T_max=100, min_lr=0.0001)
        >>> # LR smoothly decays to 0.0001 over 100 epochs
    """
    T_max: int
    min_lr: float = 0.0001


def step_decay_fn(epoch: int, base_lr: float, params: StepDecay) -> float:
    """Step decay learning rate function."""
    factor = params.gamma ** (epoch // params.step_size)
    return base_lr * factor


def exponential_decay_fn(epoch: int, base_lr: float, params: ExponentialDecay) -> float:
    """Exponential decay learning rate function."""
    return base_lr * (params.gamma ** epoch)


def cosine_annealing_fn(epoch: int, base_lr: float, params: CosineAnnealing) -> float:
    """Cosine annealing learning rate function."""
    import math
    if epoch >= params.T_max:
        return params.min_lr
    cos_val = (1 + math.cos(math.pi * epoch / params.T_max)) / 2
    return params.min_lr + (base_lr - params.min_lr) * cos_val


@dataclass
class EarlyStopping:
    """Early stopping callback to halt training when loss stops improving.
    
    Args:
        patience: Number of epochs with no improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        restore_best: Whether to restore weights from best epoch (not yet supported)
    """
    patience: int = 10
    min_delta: float = 0.0001
    restore_best: bool = True
    
    # Internal state
    _best_loss: Optional[float] = None
    _wait: int = 0
    
    def on_epoch_end(self, epoch: int, loss: float) -> bool:
        """Check if training should stop.
        
        Returns:
            True if training should stop, False otherwise
        """
        if self._best_loss is None or loss < self._best_loss - self.min_delta:
            self._best_loss = loss
            self._wait = 0
            return False
        
        self._wait += 1
        return self._wait >= self.patience
    
    def reset(self) -> None:
        """Reset callback state for new training run."""
        self._best_loss = None
        self._wait = 0
