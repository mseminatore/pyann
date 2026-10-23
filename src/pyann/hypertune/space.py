"""Hyperparameter space configuration."""

import math
from typing import Any, Callable, List, Tuple, Optional, Union
from dataclasses import dataclass, field

from pyann.activations import Activation
from pyann.optimizers import OptimizerType


@dataclass
class HyperparamSpace:
    """Configuration for hyperparameter search space.
    
    Defines the ranges and options for hyperparameters to search over
    during automated tuning.
    
    Args:
        learning_rate: Tuple of (min, max) learning rate range
        batch_sizes: List of batch sizes to try
        optimizers: List of optimizer types to try
        hidden_layers: Tuple of (min, max) number of hidden layers
        layer_sizes: List of possible layer sizes
        activations: List of activation functions to try
        dropout_rates: List of dropout rates to try
        
    Example:
        >>> space = HyperparamSpace(
        ...     learning_rate=(0.0001, 0.1),
        ...     batch_sizes=[16, 32, 64, 128],
        ...     hidden_layers=(1, 4),
        ...     layer_sizes=[32, 64, 128, 256, 512],
        ...     activations=[Activation.RELU, Activation.LEAKY_RELU]
        ... )
    """
    # Learning rate range (log scale typically)
    learning_rate: Tuple[float, float] = (0.0001, 0.1)
    
    # Batch sizes to try
    batch_sizes: List[int] = field(default_factory=lambda: [16, 32, 64])
    
    # Optimizers to try
    optimizers: List[OptimizerType] = field(
        default_factory=lambda: [OptimizerType.ADAM]
    )
    
    # Number of hidden layers range
    hidden_layers: Tuple[int, int] = (1, 3)
    
    # Possible layer sizes
    layer_sizes: List[int] = field(
        default_factory=lambda: [32, 64, 128, 256]
    )
    
    # Activation functions to try
    activations: List[Activation] = field(
        default_factory=lambda: [Activation.RELU]
    )
    
    # Dropout rates to try (0.0 = no dropout)
    dropout_rates: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.5]
    )
    
    # Topology patterns
    topology: str = "uniform"  # "uniform", "pyramid", "funnel", "diamond"
    
    # Maximum epochs for each trial (fewer with early stopping)
    epochs: int = 50
    
    def validate(self) -> None:
        """Validate the hyperparameter space configuration."""
        if self.learning_rate[0] <= 0 or self.learning_rate[1] <= 0:
            raise ValueError("Learning rate must be positive")
        if self.learning_rate[0] >= self.learning_rate[1]:
            raise ValueError("learning_rate[0] must be less than learning_rate[1]")
        
        if not self.batch_sizes:
            raise ValueError("At least one batch size must be specified")
        if any(bs <= 0 for bs in self.batch_sizes):
            raise ValueError("Batch sizes must be positive")
        
        if self.hidden_layers[0] < 1:
            raise ValueError("Must have at least 1 hidden layer")
        if self.hidden_layers[0] > self.hidden_layers[1]:
            raise ValueError("hidden_layers[0] must be <= hidden_layers[1]")
        
        if not self.layer_sizes:
            raise ValueError("At least one layer size must be specified")
        if any(ls <= 0 for ls in self.layer_sizes):
            raise ValueError("Layer sizes must be positive")


@dataclass
class HypertuneResult:
    """Result from a hyperparameter tuning trial.
    
    Attributes:
        learning_rate: Learning rate used
        batch_size: Batch size used
        optimizer: Optimizer type used
        n_hidden_layers: Number of hidden layers
        layer_sizes: List of layer sizes
        activations: List of activations per layer
        dropout_rate: Dropout rate used
        accuracy: Validation accuracy of the trained model
        loss: Training loss of the last epoch
        epochs: Maximum epochs the trial could train (HyperparamSpace.epochs)
        val_loss: Validation loss of the trained model
        score: What the search ranks by, higher is better: the validation
            accuracy, the negated validation loss, or a custom scorer's value
            (see HypertuneOptions.scoring)
        best_epoch: With early stopping, the epoch whose weights the model
            kept; None without it
        training_time: Seconds spent training the trial
    """
    learning_rate: float
    batch_size: int
    optimizer: OptimizerType
    n_hidden_layers: int
    layer_sizes: List[int]
    activations: List[Activation]
    dropout_rate: float
    accuracy: float
    loss: float
    epochs: int
    val_loss: float = math.nan
    score: float = math.nan
    best_epoch: Optional[int] = None
    training_time: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "optimizer": self.optimizer.name,
            "n_hidden_layers": self.n_hidden_layers,
            "layer_sizes": self.layer_sizes,
            "activations": [a.name for a in self.activations],
            "dropout_rate": self.dropout_rate,
            "accuracy": self.accuracy,
            "loss": self.loss,
            "epochs": self.epochs,
            "val_loss": self.val_loss,
            "score": self.score,
            "best_epoch": self.best_epoch,
            "training_time": self.training_time,
        }


@dataclass
class HypertuneOptions:
    """Options for hyperparameter tuning.
    
    Attributes:
        verbose: Verbosity level (0=silent, 1=summary, 2=detailed)
        seed: Seed for the search's own sampling of configurations (Python's
            ``random``); libann's weight initialization is not affected
        early_stopping: Stop each trial once its validation loss has not
            improved for ``patience`` epochs, and keep its best epoch's weights
        patience: Early stopping patience (epochs)
        scoring: What trials are ranked by, higher is better: "accuracy"
            (validation accuracy, for classification), "loss" (the negated
            validation loss; use it for regression), or a function
            ``score(model, x_val, y_val) -> float`` called with each trained
            model
    """
    verbose: int = 1
    seed: Optional[int] = None
    early_stopping: bool = True
    patience: int = 10
    scoring: Union[str, Callable[[Any, Any, Any], float]] = "accuracy"
    
    def validate(self) -> None:
        """Validate options."""
        if not callable(self.scoring) and self.scoring not in ("accuracy", "loss"):
            raise ValueError(f"scoring must be 'accuracy', 'loss' or a function, got {self.scoring!r}")
        if self.early_stopping and self.patience < 1:
            raise ValueError(f"patience must be at least 1, got {self.patience}")
