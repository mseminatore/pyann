"""Hyperparameter search algorithms."""

import itertools
import math
import random
import time
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass

from pyann.hypertune.space import HyperparamSpace, HypertuneResult, HypertuneOptions, topology_sizes
from pyann.callbacks import EarlyStopping
from pyann.core.network import Sequential
from pyann.core.layers import Dense
from pyann.activations import Activation
from pyann.optimizers import OptimizerType, parse_optimizer
from pyann.losses import Loss
from pyann.utils.compat import ArrayLike, to_array, has_numpy

if TYPE_CHECKING:
    import numpy as np


def _rank(score: float) -> float:
    """A score for comparisons: NaN (e.g. a trial whose training diverged) ranks last."""
    return -math.inf if math.isnan(score) else score


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
        self._topologies = self.space.topologies()
        
        if self.options.seed is not None:
            random.seed(self.options.seed)
    
    def _layer_sizes(self, topology: str, n_hidden: int, width_idx: List[int]) -> List[int]:
        """Hidden layer widths: one index into space.layer_sizes per layer for
        "any", or a single base-width index for a pattern."""
        sizes = self.space.layer_sizes
        if topology == "any":
            return [sizes[i] for i in width_idx]
        return topology_sizes(topology, sizes[width_idx[0]], n_hidden)
    
    def _random_structure(self) -> Tuple[int, int, List[int]]:
        """A random (topology index, hidden layer count, width indices)."""
        topo_idx = random.randrange(len(self._topologies))
        n_hidden = random.randint(self.space.hidden_layers[0], self.space.hidden_layers[1])
        n_widths = n_hidden if self._topologies[topo_idx] == "any" else 1
        width_idx = [random.randrange(len(self.space.layer_sizes)) for _ in range(n_widths)]
        return topo_idx, n_hidden, width_idx
    
    def _structure_config(self, topo_idx: int, n_hidden: int, width_idx: List[int]) -> dict:
        """The config entries describing the hidden layers' shape."""
        topology = self._topologies[topo_idx]
        return {
            "topology": topology,
            "width_idx": list(width_idx),
            "layer_sizes": self._layer_sizes(topology, n_hidden, width_idx),
        }
    
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
        
        # Train for up to space.epochs: early stopping, not libann's
        # convergence threshold, decides when a trial is done
        model.compile(convergence=0.0)
        callbacks = []
        if self.options.early_stopping:
            callbacks.append(EarlyStopping(patience=self.options.patience, restore_best=True))
        start = time.perf_counter()
        history = model.fit(
            X_train, y_train,
            epochs=self.space.epochs,
            batch_size=config["batch_size"],
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=callbacks
        )
        training_time = time.perf_counter() - start
        
        # Evaluate (the best epoch's weights with early stopping)
        accuracy = history["val_accuracy"]
        val_loss = history["val_loss"]
        if callable(self.options.scoring):
            score = float(self.options.scoring(model, X_val, y_val))
        elif self.options.scoring == "loss":
            score = -val_loss
        else:
            score = accuracy
        
        result = HypertuneResult(
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            optimizer=config["optimizer"],
            n_hidden_layers=len(config["layer_sizes"]),
            layer_sizes=config["layer_sizes"],
            activations=config["activations"],
            dropout_rate=config["dropout_rate"],
            accuracy=accuracy,
            loss=history["loss"],
            epochs=self.space.epochs,
            val_loss=val_loss,
            score=score,
            best_epoch=history.get("best_epoch"),
            training_time=training_time,
            topology=config.get("topology", "any")
        )
        
        return result
    
    def _update_best(self, result: HypertuneResult) -> None:
        """Update best result if this one is better."""
        self.results.append(result)
        
        best = self.best_result
        if best is None or _rank(result.score) > _rank(best.score):
            self.best_result = result
    
    def _describe_score(self, result: HypertuneResult) -> str:
        """The trial's score in words, for progress output."""
        if callable(self.options.scoring):
            return f"Score: {result.score:.4f}"
        if self.options.scoring == "loss":
            return f"Validation loss: {result.val_loss:.4f}"
        return f"Accuracy: {result.accuracy:.4f}"
    
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
        
        # Sample the hidden layers' topology, count and widths
        topo_idx, n_hidden, width_idx = self._random_structure()
        
        # Sample activations
        activations = [random.choice(self.space.activations) for _ in range(n_hidden)]
        
        return {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "optimizer": optimizer,
            "dropout_rate": dropout_rate,
            "activations": activations,
            **self._structure_config(topo_idx, n_hidden, width_idx),
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
                print(f"  {self._describe_score(result)}")
        
        if self.options.verbose >= 1:
            print(f"\nBest: {self._describe_score(self.best_result)}")
        
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
        if lr_steps < 1:
            raise ValueError(f"lr_steps must be at least 1, got {lr_steps}")
        self.lr_steps = lr_steps
    
    def _generate_configs(self) -> List[dict]:
        """Generate all configurations for grid search."""
        configs = []
        
        # Learning rate values (log scale; one step tries the minimum only)
        lr_min, lr_max = self.space.learning_rate
        if self.lr_steps == 1:
            lr_values = [lr_min]
        else:
            lr_values = [
                math.exp(math.log(lr_min) + i * (math.log(lr_max) - math.log(lr_min)) / (self.lr_steps - 1))
                for i in range(self.lr_steps)
            ]
        
        # Every hidden layer shape: for "any", each combination of widths
        # (len(layer_sizes) ** n_hidden of them); for a pattern, each base width
        structures = []
        n_sizes = len(self.space.layer_sizes)
        for topo_idx, topology in enumerate(self._topologies):
            for n_hidden in range(self.space.hidden_layers[0], self.space.hidden_layers[1] + 1):
                n_widths = n_hidden if topology == "any" else 1
                for width_idx in itertools.product(range(n_sizes), repeat=n_widths):
                    structures.append(self._structure_config(topo_idx, n_hidden, list(width_idx)))
        
        # Generate all combinations
        for lr in lr_values:
            for batch_size in self.space.batch_sizes:
                for optimizer in self.space.optimizers:
                    for dropout in self.space.dropout_rates:
                        for structure in structures:
                            n_hidden = len(structure["layer_sizes"])
                            for activation in self.space.activations:
                                configs.append({
                                    "learning_rate": lr,
                                    "batch_size": batch_size,
                                    "optimizer": optimizer,
                                    "dropout_rate": dropout,
                                    "activations": [activation] * n_hidden,
                                    **structure,
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
            print(f"\nBest: {self._describe_score(self.best_result)}")
        
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
        
        topo_idx, n_hidden, width_idx = self._random_structure()
        
        return {
            "learning_rate": math.exp(log_lr),
            "batch_size": random.choice(self.space.batch_sizes),
            "optimizer": random.choice(self.space.optimizers),
            "dropout_rate": random.choice(self.space.dropout_rates),
            "activations": [random.choice(self.space.activations) for _ in range(n_hidden)],
            **self._structure_config(topo_idx, n_hidden, width_idx),
        }
    
    def _perturb_config(self, config: dict) -> dict:
        """Perturb a configuration slightly."""
        new_config = config.copy()
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
        
        # Possibly change the topology, then possibly one width (a layer's
        # for "any", the base width for a pattern)
        topo_idx = self._topologies.index(config["topology"])
        n_hidden = len(config["layer_sizes"])
        width_idx = list(config["width_idx"])
        if len(self._topologies) > 1 and random.random() < 0.2:
            topo_idx = random.randrange(len(self._topologies))
            n_widths = n_hidden if self._topologies[topo_idx] == "any" else 1
            width_idx = [width_idx[0]] * n_widths
        if random.random() < 0.3:
            width_idx[random.randrange(len(width_idx))] = random.randrange(len(self.space.layer_sizes))
        new_config.update(self._structure_config(topo_idx, n_hidden, width_idx))
        
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
            
            self._history.append((config, _rank(result.score)))
            self._update_best(result)
            
            if self.options.verbose >= 2:
                print(f"  {self._describe_score(result)}")
        
        if self.options.verbose >= 1:
            print(f"\nBest: {self._describe_score(self.best_result)}")
        
        return self.best_result, self.results


def _scott_bandwidth(samples: List[float], factor: float = 1.0) -> float:
    """Scott's rule for KDE bandwidth: h = 1.06 * std * n^(-1/5)."""
    n = len(samples)
    if n < 2:
        return 0.1 * factor
    mean = sum(samples) / n
    var = sum((x - mean) ** 2 for x in samples) / (n - 1)
    std = max(math.sqrt(var), 0.01)
    return 1.06 * std * (n ** -0.2) * factor


def _kde_evaluate(x: float, samples: List[float], bandwidth: float) -> float:
    """Evaluate 1D Gaussian KDE at point x."""
    if not samples:
        return 1e-10
    sqrt_2pi = 2.5066282746
    total = sum(
        math.exp(-0.5 * ((x - xi) / bandwidth) ** 2) / (bandwidth * sqrt_2pi)
        for xi in samples
    )
    return total / len(samples) + 1e-10


def _kde_sample(samples: List[float], bandwidth: float) -> float:
    """Sample from 1D Gaussian KDE (pick random component + Gaussian noise)."""
    if not samples:
        return random.random()
    center = random.choice(samples)
    return random.gauss(center, bandwidth)


def _categorical_prob(
    value: int, samples: List[int], n_categories: int
) -> float:
    """Category probability with Laplace smoothing."""
    if not samples:
        return 1.0 / n_categories
    count = sum(1 for s in samples if s == value)
    return (count + 1.0) / (len(samples) + n_categories)


def _categorical_sample(samples: List[int], n_categories: int) -> int:
    """Sample from categorical distribution with Laplace smoothing."""
    if not samples:
        return random.randrange(n_categories)
    probs = [_categorical_prob(c, samples, n_categories) for c in range(n_categories)]
    return random.choices(range(n_categories), weights=probs, k=1)[0]


class TPESearch(BaseSearch):
    """Tree-structured Parzen Estimator (TPE) hyperparameter search.
    
    TPE is a Bayesian optimization method that models good and bad
    configurations separately using kernel density estimators, then
    samples configurations that maximize the expected improvement
    ratio l(x)/g(x).
    
    Args:
        space: Hyperparameter search space
        n_trials: Total number of trials (startup + TPE-guided)
        n_startup: Number of random trials before TPE kicks in
        gamma: Quantile for good/bad split (default 0.25 = top 25%)
        n_candidates: Candidates sampled from l(x) per TPE iteration
        bandwidth_factor: Multiplier for KDE bandwidth
        options: Search options
        
    Example:
        >>> space = HyperparamSpace(
        ...     learning_rate=(0.0001, 0.01),
        ...     batch_sizes=[32, 64],
        ...     hidden_layers=(1, 3)
        ... )
        >>> search = TPESearch(space, n_trials=50, n_startup=10)
        >>> best, results = search.run(X_train, y_train, X_val, y_val,
        ...                            input_size=784, output_size=10)
    """
    
    def __init__(
        self,
        space: HyperparamSpace,
        n_trials: int = 50,
        n_startup: int = 10,
        gamma: float = 0.25,
        n_candidates: int = 24,
        bandwidth_factor: float = 1.0,
        options: Optional[HypertuneOptions] = None
    ):
        super().__init__(space, options)
        self.n_trials = n_trials
        self.n_startup = n_startup
        self.gamma = gamma
        self.n_candidates = n_candidates
        self.bandwidth_factor = bandwidth_factor
        self._observations: List[Tuple[dict, float]] = []
    
    def _random_config_indexed(self) -> Tuple[dict, dict]:
        """Sample a random config, returning both the config and its indices."""
        lr_min, lr_max = self.space.learning_rate
        log_lr = random.uniform(math.log(lr_min), math.log(lr_max))
        
        batch_idx = random.randrange(len(self.space.batch_sizes))
        opt_idx = random.randrange(len(self.space.optimizers))
        dropout_idx = random.randrange(len(self.space.dropout_rates))
        act_idx = random.randrange(len(self.space.activations))
        topo_idx, n_hidden, width_idx = self._random_structure()
        
        indices = {
            "log_lr": log_lr,
            "batch_idx": batch_idx,
            "opt_idx": opt_idx,
            "dropout_idx": dropout_idx,
            "act_idx": act_idx,
            "layers_idx": n_hidden - self.space.hidden_layers[0],
            "topo_idx": topo_idx,
            "width_idx": width_idx,
        }
        return self._config_from_indices(indices), indices
    
    def _config_from_indices(self, indices: dict) -> dict:
        """The configuration an indices dict describes."""
        n_hidden = indices["layers_idx"] + self.space.hidden_layers[0]
        return {
            "learning_rate": math.exp(indices["log_lr"]),
            "batch_size": self.space.batch_sizes[indices["batch_idx"]],
            "optimizer": self.space.optimizers[indices["opt_idx"]],
            "dropout_rate": self.space.dropout_rates[indices["dropout_idx"]],
            "activations": [self.space.activations[indices["act_idx"]]] * n_hidden,
            **self._structure_config(indices["topo_idx"], n_hidden, indices["width_idx"]),
        }
    
    def _tpe_sample_config(self) -> Tuple[dict, dict]:
        """Sample a config using TPE (l(x)/g(x) maximization)."""
        n_obs = len(self._observations)
        n_good = max(1, int(self.gamma * n_obs))
        
        sorted_obs = sorted(self._observations, key=lambda x: x[1], reverse=True)
        good_obs = [idx for idx, _ in sorted_obs[:n_good]]
        bad_obs = [idx for idx, _ in sorted_obs[n_good:]]
        
        good_lr = [o["log_lr"] for o in good_obs]
        bad_lr = [o["log_lr"] for o in bad_obs]
        bw_good_lr = _scott_bandwidth(good_lr, self.bandwidth_factor)
        bw_bad_lr = _scott_bandwidth(bad_lr, self.bandwidth_factor)
        
        lr_min, lr_max = self.space.learning_rate
        log_min, log_max = math.log(lr_min), math.log(lr_max)
        
        # Categorical dimensions: name -> number of choices
        categories = {
            "batch_idx": len(self.space.batch_sizes),
            "opt_idx": len(self.space.optimizers),
            "dropout_idx": len(self.space.dropout_rates),
            "act_idx": len(self.space.activations),
            "layers_idx": self.space.hidden_layers[1] - self.space.hidden_layers[0] + 1,
            "topo_idx": len(self._topologies),
        }
        n_sizes = len(self.space.layer_sizes)
        
        def log_ratio(value: int, good: List[int], bad: List[int], n: int) -> float:
            return math.log(_categorical_prob(value, good, n) / _categorical_prob(value, bad, n))
        
        def widths_at(obs: List[dict], position: int) -> List[int]:
            return [o["width_idx"][position] for o in obs if len(o["width_idx"]) > position]
        
        best_ratio = float('-inf')
        best_indices = None
        
        for _ in range(self.n_candidates):
            lr = max(log_min, min(log_max, _kde_sample(good_lr, bw_good_lr)))
            # EI ratio in log space for numerical stability
            ratio = math.log(_kde_evaluate(lr, good_lr, bw_good_lr)
                             / _kde_evaluate(lr, bad_lr, bw_bad_lr))
            indices = {"log_lr": lr}
            for name, n in categories.items():
                good = [o[name] for o in good_obs]
                indices[name] = _categorical_sample(good, n)
                ratio += log_ratio(indices[name], good, [o[name] for o in bad_obs], n)
            
            # Widths: one per layer for "any", the base width for a pattern;
            # each position is modelled from the trials that had it
            n_hidden = indices["layers_idx"] + self.space.hidden_layers[0]
            n_widths = n_hidden if self._topologies[indices["topo_idx"]] == "any" else 1
            indices["width_idx"] = []
            for position in range(n_widths):
                good = widths_at(good_obs, position)
                width = _categorical_sample(good, n_sizes)
                indices["width_idx"].append(width)
                ratio += log_ratio(width, good, widths_at(bad_obs, position), n_sizes)
            
            if ratio > best_ratio:
                best_ratio = ratio
                best_indices = indices
        
        return self._config_from_indices(best_indices), best_indices
    
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
        """Run TPE hyperparameter search.
        
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
        
        if self.options.verbose >= 1:
            print(f"TPE Search: {self.n_startup} startup + "
                  f"{self.n_trials - self.n_startup} TPE iterations")
        
        for trial in range(self.n_trials):
            use_tpe = trial >= self.n_startup and len(self._observations) >= 2
            phase = "TPE" if use_tpe else "random"
            
            if self.options.verbose >= 1:
                print(f"Trial {trial + 1}/{self.n_trials} ({phase})")
            
            if use_tpe:
                config, indices = self._tpe_sample_config()
            else:
                config, indices = self._random_config_indexed()
            
            result = self._evaluate_config(
                config, X_train, y_train, X_val, y_val,
                input_size, output_size, output_activation, loss
            )
            
            self._observations.append((indices, _rank(result.score)))
            self._update_best(result)
            
            if self.options.verbose >= 2:
                print(f"  {self._describe_score(result)}")
        
        if self.options.verbose >= 1:
            print(f"\nBest: {self._describe_score(self.best_result)}")
        
        return self.best_result, self.results
