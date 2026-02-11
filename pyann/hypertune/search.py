"""Hyperparameter search algorithms."""

import math
import random
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

from pyann.hypertune.space import HyperparamSpace, HypertuneResult, HypertuneOptions
from pyann.core.network import Sequential
from pyann.core.layers import Dense
from pyann.activations import Activation
from pyann.optimizers import OptimizerType, parse_optimizer
from pyann.losses import Loss
from pyann.utils.compat import ArrayLike, to_array, has_numpy

if TYPE_CHECKING:
    import numpy as np


class BaseSearch:
    """Base class for hyperparameter search algorithms."""
    
    def __init__(
        self,
        space: HyperparamSpace,
        options: Optional[HypertuneOptions] = None
    ):
        self.space = space
        self.options = options or HypertuneOptions()
        self.results: List[HypertuneResult] = []
        self.best_result: Optional[HypertuneResult] = None
        
        self.space.validate()
        self.options.validate()
        
        if self.options.seed is not None:
            random.seed(self.options.seed)
    
    def _create_model(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int],
        activations: List[Activation],
        output_activation: Activation,
        optimizer_type: OptimizerType,
        learning_rate: float,
        dropout_rate: float,
        loss: Loss
    ) -> Sequential:
        """Create a model with specified configuration."""
        optimizer = parse_optimizer(optimizer_type)
        optimizer.learning_rate = learning_rate
        
        model = Sequential(optimizer=optimizer, loss=loss)
        
        # First hidden layer with input shape
        if hidden_sizes:
            model.add(Dense(
                hidden_sizes[0],
                activation=activations[0] if activations else Activation.RELU,
                input_shape=(input_size,),
                dropout=dropout_rate
            ))
            
            # Additional hidden layers
            for i, size in enumerate(hidden_sizes[1:], 1):
                act = activations[i] if i < len(activations) else activations[-1]
                model.add(Dense(size, activation=act, dropout=dropout_rate))
        else:
            # No hidden layers - just input to output
            model.add(Dense(
                output_size,
                activation=output_activation,
                input_shape=(input_size,)
            ))
            return model
        
        # Output layer
        model.add(Dense(output_size, activation=output_activation))
        
        return model
    
    def _evaluate_config(
        self,
        config: dict,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        input_size: int,
        output_size: int,
        output_activation: Activation,
        loss: Loss
    ) -> HypertuneResult:
        """Evaluate a single configuration."""
        model = self._create_model(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=config["layer_sizes"],
            activations=config["activations"],
            output_activation=output_activation,
            optimizer_type=config["optimizer"],
            learning_rate=config["learning_rate"],
            dropout_rate=config["dropout_rate"],
            loss=loss
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            epochs=self.space.epochs,
            batch_size=config["batch_size"],
            validation_data=(X_val, y_val),
            verbose=0
        )
        
        # Evaluate
        accuracy = model.evaluate(X_val, y_val)
        
        result = HypertuneResult(
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            optimizer=config["optimizer"],
            n_hidden_layers=len(config["layer_sizes"]),
            layer_sizes=config["layer_sizes"],
            activations=config["activations"],
            dropout_rate=config["dropout_rate"],
            accuracy=accuracy,
            loss=history.get("loss", 0.0),
            epochs=self.space.epochs
        )
        
        return result
    
    def _update_best(self, result: HypertuneResult) -> None:
        """Update best result if this one is better."""
        self.results.append(result)
        
        if self.best_result is None:
            self.best_result = result
        elif self.options.scoring == "accuracy":
            if result.accuracy > self.best_result.accuracy:
                self.best_result = result
        else:  # loss
            if result.loss < self.best_result.loss:
                self.best_result = result
    
    def create_model(
        self,
        result: HypertuneResult,
        input_size: int,
        output_size: int,
        output_activation: Activation = Activation.SOFTMAX,
        loss: Loss = Loss.CATEGORICAL_CROSS_ENTROPY
    ) -> Sequential:
        """Create a model from a tuning result."""
        return self._create_model(
            input_size=input_size,
            output_size=output_size,
            hidden_sizes=result.layer_sizes,
            activations=result.activations,
            output_activation=output_activation,
            optimizer_type=result.optimizer,
            learning_rate=result.learning_rate,
            dropout_rate=result.dropout_rate,
            loss=loss
        )


class RandomSearch(BaseSearch):
    """Random search over hyperparameter space.
    
    Randomly samples configurations from the search space.
    Often more efficient than grid search for high-dimensional spaces.
    
    Args:
        space: Hyperparameter search space
        n_trials: Number of random configurations to try
        options: Search options
        
    Example:
        >>> space = HyperparamSpace(
        ...     learning_rate=(0.0001, 0.01),
        ...     batch_sizes=[32, 64],
        ...     hidden_layers=(1, 3)
        ... )
        >>> search = RandomSearch(space, n_trials=50)
        >>> best, results = search.run(X_train, y_train, X_val, y_val, 
        ...                            input_size=784, output_size=10)
    """
    
    def __init__(
        self,
        space: HyperparamSpace,
        n_trials: int = 50,
        options: Optional[HypertuneOptions] = None
    ):
        super().__init__(space, options)
        self.n_trials = n_trials
    
    def _sample_config(self) -> dict:
        """Sample a random configuration from the space."""
        # Sample learning rate (log scale)
        lr_min, lr_max = self.space.learning_rate
        log_lr = random.uniform(math.log(lr_min), math.log(lr_max))
        learning_rate = math.exp(log_lr)
        
        # Sample other parameters
        batch_size = random.choice(self.space.batch_sizes)
        optimizer = random.choice(self.space.optimizers)
        dropout_rate = random.choice(self.space.dropout_rates)
        
        # Sample number of hidden layers
        n_hidden = random.randint(self.space.hidden_layers[0], self.space.hidden_layers[1])
        
        # Sample layer sizes
        layer_sizes = [random.choice(self.space.layer_sizes) for _ in range(n_hidden)]
        
        # Sample activations
        activations = [random.choice(self.space.activations) for _ in range(n_hidden)]
        
        return {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "dropout_rate": dropout_rate,
            "layer_sizes": layer_sizes,
            "activations": activations,
        }
    
    def run(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        input_size: int,
        output_size: int,
        output_activation: Activation = Activation.SOFTMAX,
        loss: Loss = Loss.CATEGORICAL_CROSS_ENTROPY
    ) -> Tuple[HypertuneResult, List[HypertuneResult]]:
        """Run the random search.
        
        Args:
            X_train: Training input data
            y_train: Training target data
            X_val: Validation input data
            y_val: Validation target data
            input_size: Number of input features
            output_size: Number of output classes/values
            output_activation: Activation for output layer
            loss: Loss function
            
        Returns:
            Tuple of (best_result, all_results)
        """
        X_train = to_array(X_train)
        y_train = to_array(y_train)
        X_val = to_array(X_val)
        y_val = to_array(y_val)
        
        for trial in range(self.n_trials):
            if self.options.verbose >= 1:
                print(f"Trial {trial + 1}/{self.n_trials}")
            
            config = self._sample_config()
            result = self._evaluate_config(
                config, X_train, y_train, X_val, y_val,
                input_size, output_size, output_activation, loss
            )
            self._update_best(result)
            
            if self.options.verbose >= 2:
                print(f"  Accuracy: {result.accuracy:.4f}")
        
        if self.options.verbose >= 1:
            print(f"\nBest accuracy: {self.best_result.accuracy:.4f}")
        
        return self.best_result, self.results


class GridSearch(BaseSearch):
    """Exhaustive grid search over hyperparameter space.
    
    Tries all combinations of hyperparameters. Can be expensive for
    large search spaces.
    
    Args:
        space: Hyperparameter search space
        options: Search options
    """
    
    def __init__(
        self,
        space: HyperparamSpace,
        options: Optional[HypertuneOptions] = None,
        lr_steps: int = 3
    ):
        super().__init__(space, options)
        self.lr_steps = lr_steps
    
    def _generate_configs(self) -> List[dict]:
        """Generate all configurations for grid search."""
        configs = []
        
        # Learning rate values (log scale)
        lr_min, lr_max = self.space.learning_rate
        lr_values = [
            math.exp(math.log(lr_min) + i * (math.log(lr_max) - math.log(lr_min)) / (self.lr_steps - 1))
            for i in range(self.lr_steps)
        ]
        
        # Generate all combinations
        for lr in lr_values:
            for batch_size in self.space.batch_sizes:
                for optimizer in self.space.optimizers:
                    for dropout in self.space.dropout_rates:
                        for n_hidden in range(self.space.hidden_layers[0], self.space.hidden_layers[1] + 1):
                            for layer_size in self.space.layer_sizes:
                                for activation in self.space.activations:
                                    configs.append({
                                        "learning_rate": lr,
                                        "batch_size": batch_size,
                                        "optimizer": optimizer,
                                        "dropout_rate": dropout,
                                        "layer_sizes": [layer_size] * n_hidden,
                                        "activations": [activation] * n_hidden,
                                    })
        
        return configs
    
    def run(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        input_size: int,
        output_size: int,
        output_activation: Activation = Activation.SOFTMAX,
        loss: Loss = Loss.CATEGORICAL_CROSS_ENTROPY
    ) -> Tuple[HypertuneResult, List[HypertuneResult]]:
        """Run the grid search."""
        X_train = to_array(X_train)
        y_train = to_array(y_train)
        X_val = to_array(X_val)
        y_val = to_array(y_val)
        
        configs = self._generate_configs()
        n_configs = len(configs)
        
        if self.options.verbose >= 1:
            print(f"Grid search: {n_configs} configurations")
        
        for i, config in enumerate(configs):
            if self.options.verbose >= 1:
                print(f"Config {i + 1}/{n_configs}")
            
            result = self._evaluate_config(
                config, X_train, y_train, X_val, y_val,
                input_size, output_size, output_activation, loss
            )
            self._update_best(result)
        
        if self.options.verbose >= 1:
            print(f"\nBest accuracy: {self.best_result.accuracy:.4f}")
        
        return self.best_result, self.results


class BayesianSearch(BaseSearch):
    """Bayesian optimization for hyperparameter search.
    
    Uses a simple surrogate model to guide the search toward
    promising regions of the hyperparameter space.
    
    Note: This is a simplified implementation. For production use,
    consider using dedicated libraries like Optuna or scikit-optimize.
    
    Args:
        space: Hyperparameter search space
        n_trials: Number of trials
        n_initial: Number of random trials before using surrogate
        options: Search options
    """
    
    def __init__(
        self,
        space: HyperparamSpace,
        n_trials: int = 50,
        n_initial: int = 10,
        options: Optional[HypertuneOptions] = None
    ):
        super().__init__(space, options)
        self.n_trials = n_trials
        self.n_initial = n_initial
        self._history: List[Tuple[dict, float]] = []
    
    def _sample_config(self, exploit: bool = False) -> dict:
        """Sample configuration, optionally exploiting history."""
        if not exploit or len(self._history) < self.n_initial:
            # Random sampling
            return self._random_config()
        
        # Simple exploitation: perturb best config
        best_config, _ = max(self._history, key=lambda x: x[1])
        return self._perturb_config(best_config)
    
    def _random_config(self) -> dict:
        """Generate a random configuration."""
        lr_min, lr_max = self.space.learning_rate
        log_lr = random.uniform(math.log(lr_min), math.log(lr_max))
        
        n_hidden = random.randint(self.space.hidden_layers[0], self.space.hidden_layers[1])
        
        return {
            "learning_rate": math.exp(log_lr),
            "batch_size": random.choice(self.space.batch_sizes),
            "optimizer": random.choice(self.space.optimizers),
            "dropout_rate": random.choice(self.space.dropout_rates),
            "layer_sizes": [random.choice(self.space.layer_sizes) for _ in range(n_hidden)],
            "activations": [random.choice(self.space.activations) for _ in range(n_hidden)],
        }
    
    def _perturb_config(self, config: dict) -> dict:
        """Perturb a configuration slightly."""
        new_config = config.copy()
        new_config["layer_sizes"] = config["layer_sizes"].copy()
        new_config["activations"] = config["activations"].copy()
        
        # Perturb learning rate
        lr = config["learning_rate"]
        lr_min, lr_max = self.space.learning_rate
        factor = random.uniform(0.5, 2.0)
        new_lr = max(lr_min, min(lr_max, lr * factor))
        new_config["learning_rate"] = new_lr
        
        # Possibly change batch size
        if random.random() < 0.3:
            new_config["batch_size"] = random.choice(self.space.batch_sizes)
        
        # Possibly change a layer size
        if new_config["layer_sizes"] and random.random() < 0.3:
            idx = random.randrange(len(new_config["layer_sizes"]))
            new_config["layer_sizes"][idx] = random.choice(self.space.layer_sizes)
        
        return new_config
    
    def run(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_val: ArrayLike,
        y_val: ArrayLike,
        input_size: int,
        output_size: int,
        output_activation: Activation = Activation.SOFTMAX,
        loss: Loss = Loss.CATEGORICAL_CROSS_ENTROPY
    ) -> Tuple[HypertuneResult, List[HypertuneResult]]:
        """Run Bayesian optimization search."""
        X_train = to_array(X_train)
        y_train = to_array(y_train)
        X_val = to_array(X_val)
        y_val = to_array(y_val)
        
        for trial in range(self.n_trials):
            if self.options.verbose >= 1:
                print(f"Trial {trial + 1}/{self.n_trials}")
            
            # Alternate between exploration and exploitation
            exploit = trial >= self.n_initial and random.random() < 0.7
            config = self._sample_config(exploit=exploit)
            
            result = self._evaluate_config(
                config, X_train, y_train, X_val, y_val,
                input_size, output_size, output_activation, loss
            )
            
            self._history.append((config, result.accuracy))
            self._update_best(result)
            
            if self.options.verbose >= 2:
                print(f"  Accuracy: {result.accuracy:.4f}")
        
        if self.options.verbose >= 1:
            print(f"\nBest accuracy: {self.best_result.accuracy:.4f}")
        
        return self.best_result, self.results
