"""Sequential neural network model with Keras-like API."""

from typing import Optional, Union, List, Tuple, TYPE_CHECKING
from pathlib import Path

from pyann._bindings.ffi import ffi, lib
from pyann.core.layers import Layer, Dense, Input, parse_layer
from pyann.core.tensor import Tensor
from pyann.activations import Activation
from pyann.losses import Loss, parse_loss
from pyann.optimizers import Optimizer, parse_optimizer, OptimizerType
from pyann.callbacks import (
    LRScheduler, StepDecay, ExponentialDecay, CosineAnnealing,
    EarlyStopping
)
from pyann.utils.compat import ArrayLike, has_numpy, to_array, get_shape
from pyann.exceptions import (
    NetworkError, InvalidParameterError, AllocationError,
    raise_for_error_code
)

if TYPE_CHECKING:
    import numpy as np


class Sequential:
    """Sequential neural network model.
    
    A linear stack of layers, similar to Keras Sequential model.
    
    Example:
        >>> model = Sequential()
        >>> model.add(Dense(128, activation='relu', input_shape=(784,)))
        >>> model.add(Dense(64, activation='relu'))
        >>> model.add(Dense(10, activation='softmax'))
        >>> 
        >>> model.compile(optimizer='adam', loss='categorical_cross_entropy')
        >>> model.fit(X_train, y_train, epochs=10, batch_size=32)
        >>> predictions = model.predict(X_test)
    
    Args:
        layers: Optional list of layers to add
        optimizer: Optimizer instance or string name
        loss: Loss function enum or string name
        name: Optional model name
    """
    
    def __init__(
        self,
        layers: Optional[List[Layer]] = None,
        optimizer: Union[str, Optimizer, None] = None,
        loss: Union[str, Loss] = Loss.MSE,
        name: Optional[str] = None
    ):
        self._layers: List[Layer] = []
        self._optimizer = parse_optimizer(optimizer)
        self._loss = parse_loss(loss)
        self._name = name
        self._ptr: Optional[object] = None
        self._built = False
        self._input_shape: Optional[Tuple[int, ...]] = None
        self._weight_decay: Optional[float] = None
        self._l1_regularization: Optional[float] = None
        
        if layers:
            for layer in layers:
                self.add(layer)
    
    def add(self, layer: Union[Layer, int, Tuple[int, str]]) -> "Sequential":
        """Add a layer to the model.
        
        Args:
            layer: Layer instance, or shorthand (units or (units, activation))
            
        Returns:
            self for method chaining
        """
        layer = parse_layer(layer)
        
        # Handle input shape
        if isinstance(layer, Input):
            if self._input_shape is not None:
                raise InvalidParameterError("Input shape already set")
            self._input_shape = layer.shape
        elif isinstance(layer, Dense):
            if layer.input_shape is not None:
                if self._input_shape is not None and len(self._layers) > 0:
                    raise InvalidParameterError("Input shape already set")
                self._input_shape = layer.input_shape
            self._layers.append(layer)
        else:
            self._layers.append(layer)
        
        return self
    
    def compile(
        self,
        optimizer: Union[str, Optimizer, None] = None,
        loss: Union[str, Loss, None] = None,
        learning_rate: Optional[float] = None,
        weight_decay: Optional[float] = None,
        l1_regularization: Optional[float] = None
    ) -> "Sequential":
        """Configure the model for training.
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
            learning_rate: Override the optimizer's learning rate
            weight_decay: L2 regularization coefficient (0 = disabled)
            l1_regularization: L1 regularization coefficient (0 = disabled)
            
        Returns:
            self for method chaining
        """
        if optimizer is not None:
            self._optimizer = parse_optimizer(optimizer)
        if loss is not None:
            self._loss = parse_loss(loss)
        if learning_rate is not None:
            self._optimizer.learning_rate = learning_rate
        if weight_decay is not None:
            self._weight_decay = weight_decay
        if l1_regularization is not None:
            self._l1_regularization = l1_regularization
        
        return self
    
    def _build(self) -> None:
        """Build the C network from layer specifications."""
        if self._built:
            return
        
        if not self._layers:
            raise NetworkError("Cannot build empty model - add layers first")
        
        if self._input_shape is None:
            raise NetworkError(
                "Input shape not set. Provide input_shape to first Dense layer "
                "or add an Input layer."
            )
        
        # Create C network
        opt_type = self._optimizer._get_type()
        self._ptr = lib.ann_make_network(opt_type, self._loss)
        if self._ptr == ffi.NULL:
            raise AllocationError("Failed to create network")
        
        # Set learning rate
        if self._optimizer.learning_rate is not None:
            lib.ann_set_learning_rate(self._ptr, self._optimizer.learning_rate)
        
        # Add input layer
        input_units = self._input_shape[0] if len(self._input_shape) == 1 else (
            self._input_shape[0] * self._input_shape[1] if len(self._input_shape) == 2 
            else 1
        )
        for dim in self._input_shape:
            input_units = dim if len(self._input_shape) == 1 else input_units
        
        # Compute flat input size
        input_units = 1
        for d in self._input_shape:
            input_units *= d
        
        result = lib.ann_add_layer(self._ptr, input_units, lib.LAYER_INPUT, lib.ACTIVATION_NULL)
        raise_for_error_code(result, "Failed to add input layer")
        
        # Add hidden and output layers
        for i, layer in enumerate(self._layers):
            if isinstance(layer, Dense):
                is_output = (i == len(self._layers) - 1)
                layer_type = lib.LAYER_OUTPUT if is_output else lib.LAYER_HIDDEN
                
                result = lib.ann_add_layer(
                    self._ptr,
                    layer.units,
                    layer_type,
                    layer.activation_type
                )
                raise_for_error_code(result, f"Failed to add layer {i}")
                
                # Set dropout if specified
                if layer.dropout > 0:
                    lib.ann_set_layer_dropout(self._ptr, i + 1, layer.dropout)
        
        # Apply regularization settings
        if self._weight_decay is not None and self._weight_decay > 0:
            lib.ann_set_weight_decay(self._ptr, self._weight_decay)
        if self._l1_regularization is not None and self._l1_regularization > 0:
            lib.ann_set_l1_regularization(self._ptr, self._l1_regularization)
        
        self._built = True
    
    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        epochs: int = 1,
        batch_size: int = 32,
        validation_data: Optional[Tuple[ArrayLike, ArrayLike]] = None,
        verbose: int = 1,
        callbacks: Optional[List] = None,
        lr_scheduler: Optional[LRScheduler] = None
    ) -> dict:
        """Train the model on data.
        
        Args:
            x: Training input data
            y: Training target data  
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_data: Optional (x_val, y_val) tuple
            verbose: Verbosity level (0=silent, 1=progress)
            callbacks: List of callbacks (e.g., EarlyStopping)
            lr_scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training history
        """
        self._build()
        
        # Convert data to tensors
        x_arr = to_array(x)
        y_arr = to_array(y)
        
        if has_numpy():
            import numpy as np
            x_tensor = Tensor(x_arr)
            y_tensor = Tensor(y_arr)
            n_samples = x_arr.shape[0]
        else:
            x_tensor = Tensor(x_arr)
            y_tensor = Tensor(y_arr)
            n_samples = len(x_arr)
        
        # Configure training
        lib.ann_set_batch_size(self._ptr, batch_size)
        lib.ann_set_epoch_limit(self._ptr, epochs)
        
        # Set up LR scheduler if provided
        if lr_scheduler is not None:
            self._setup_lr_scheduler(lr_scheduler)
        
        # Train
        result = lib.ann_train_network(self._ptr, x_tensor._ptr, y_tensor._ptr, n_samples)
        raise_for_error_code(result, "Training failed")
        
        # Build history dict
        history = {"epochs": epochs}
        
        # Evaluate on validation data if provided
        if validation_data is not None:
            x_val, y_val = validation_data
            val_accuracy = self.evaluate(x_val, y_val)
            history["val_accuracy"] = val_accuracy
        
        return history
    
    def _setup_lr_scheduler(self, scheduler: LRScheduler) -> None:
        """Configure a learning rate scheduler."""
        if isinstance(scheduler, StepDecay):
            params = ffi.new("LRStepParams *")
            params.step_size = scheduler.step_size
            params.gamma = scheduler.gamma
            lib.ann_set_lr_scheduler(self._ptr, lib.ann_lr_scheduler_step, params)
        elif isinstance(scheduler, ExponentialDecay):
            params = ffi.new("LRExponentialParams *")
            params.gamma = scheduler.gamma
            lib.ann_set_lr_scheduler(self._ptr, lib.ann_lr_scheduler_exponential, params)
        elif isinstance(scheduler, CosineAnnealing):
            params = ffi.new("LRCosineParams *")
            params.T_max = scheduler.T_max
            params.min_lr = scheduler.min_lr
            lib.ann_set_lr_scheduler(self._ptr, lib.ann_lr_scheduler_cosine, params)
    
    def predict(self, x: ArrayLike) -> ArrayLike:
        """Generate predictions for input samples.
        
        Args:
            x: Input data
            
        Returns:
            Predictions (numpy array if available, else list)
        """
        if not self._built:
            raise NetworkError("Model must be trained before making predictions")
        
        x_arr = to_array(x)
        shape = get_shape(x_arr)
        
        # Handle single sample vs batch
        if len(shape) == 1:
            x_arr = to_array([x_arr.tolist() if has_numpy() else x_arr])
            shape = get_shape(x_arr)
        
        n_samples = shape[0]
        output_size = self._layers[-1].units
        
        if has_numpy():
            import numpy as np
            results = np.zeros((n_samples, output_size), dtype=np.float32)
            
            for i in range(n_samples):
                row = x_arr[i]
                inputs = ffi.cast("real *", row.ctypes.data)
                outputs = ffi.new("real[]", output_size)
                
                result = lib.ann_predict(self._ptr, inputs, outputs)
                raise_for_error_code(result, "Prediction failed")
                
                for j in range(output_size):
                    results[i, j] = outputs[j]
            
            return results
        else:
            results = []
            for i in range(n_samples):
                row = x_arr[i] if isinstance(x_arr[i], list) else list(x_arr[i])
                inputs = ffi.new("real[]", row)
                outputs = ffi.new("real[]", output_size)
                
                result = lib.ann_predict(self._ptr, inputs, outputs)
                raise_for_error_code(result, "Prediction failed")
                
                results.append([outputs[j] for j in range(output_size)])
            
            return results
    
    def evaluate(self, x: ArrayLike, y: ArrayLike) -> float:
        """Evaluate model accuracy on test data.
        
        Args:
            x: Test input data
            y: Test target data
            
        Returns:
            Accuracy as a float (0.0 to 1.0)
        """
        if not self._built:
            raise NetworkError("Model must be trained before evaluation")
        
        x_tensor = Tensor(to_array(x))
        y_tensor = Tensor(to_array(y))
        
        accuracy = lib.ann_evaluate_accuracy(self._ptr, x_tensor._ptr, y_tensor._ptr)
        return accuracy
    
    def save(self, filepath: Union[str, Path], format: str = "text") -> None:
        """Save the model to a file.
        
        Args:
            filepath: Output file path
            format: 'text' for human-readable, 'binary' for compact
        """
        if not self._built:
            raise NetworkError("Cannot save untrained model")
        
        filepath = str(filepath)
        
        if format == "binary":
            result = lib.ann_save_network_binary(self._ptr, filepath.encode())
        else:
            result = lib.ann_save_network(self._ptr, filepath.encode())
        
        raise_for_error_code(result, f"Failed to save model to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = "text") -> "Sequential":
        """Load a model from a file.
        
        Args:
            filepath: Path to saved model
            format: 'text' or 'binary'
            
        Returns:
            Loaded Sequential model
        """
        filepath = str(filepath)
        
        if format == "binary":
            ptr = lib.ann_load_network_binary(filepath.encode())
        else:
            ptr = lib.ann_load_network(filepath.encode())
        
        if ptr == ffi.NULL:
            raise NetworkError(f"Failed to load model from {filepath}")
        
        # Create model wrapper
        model = cls.__new__(cls)
        model._layers = []
        model._optimizer = None
        model._loss = Loss.MSE
        model._name = None
        model._ptr = ptr
        model._built = True
        model._input_shape = None
        
        # Reconstruct layer info from C network
        n_layers = lib.ann_get_layer_count(ptr)
        for i in range(n_layers):
            units = lib.ann_get_layer_nodes(ptr, i)
            activation = lib.ann_get_layer_activation(ptr, i)
            if i == 0:
                model._input_shape = (units,)
            else:
                model._layers.append(Dense(units, activation=Activation(activation)))
        
        return model
    
    def export_onnx(self, filepath: Union[str, Path]) -> None:
        """Export the model to ONNX JSON format.
        
        Args:
            filepath: Output file path
        """
        if not self._built:
            raise NetworkError("Cannot export untrained model")
        
        result = lib.ann_export_onnx(self._ptr, str(filepath).encode())
        raise_for_error_code(result, f"Failed to export ONNX to {filepath}")
    
    @classmethod
    def load_onnx(cls, filepath: Union[str, Path]) -> "Sequential":
        """Load a model from an ONNX JSON file.
        
        Args:
            filepath: Path to ONNX JSON file
            
        Returns:
            Loaded Sequential model
        """
        ptr = lib.ann_import_onnx(str(filepath).encode())
        
        if ptr == ffi.NULL:
            raise NetworkError(f"Failed to load ONNX model from {filepath}")
        
        # Create model wrapper
        model = cls.__new__(cls)
        model._layers = []
        model._optimizer = None
        model._loss = Loss.MSE
        model._name = None
        model._ptr = ptr
        model._built = True
        model._input_shape = None
        
        # Reconstruct layer info from C network
        n_layers = lib.ann_get_layer_count(ptr)
        for i in range(n_layers):
            units = lib.ann_get_layer_nodes(ptr, i)
            activation = lib.ann_get_layer_activation(ptr, i)
            if i == 0:
                model._input_shape = (units,)
            else:
                model._layers.append(Dense(units, activation=Activation(activation)))
        
        return model
    
    def export_learning_curve(self, filepath: Union[str, Path]) -> None:
        """Export training history as CSV for learning curve visualization.
        
        Writes epoch, loss, and learning rate data collected during training.
        
        Args:
            filepath: Output CSV file path
        """
        if not self._built:
            raise NetworkError("Cannot export learning curve from untrained model")
        
        result = lib.ann_export_learning_curve(self._ptr, str(filepath).encode())
        raise_for_error_code(result, f"Failed to export learning curve to {filepath}")
    
    def clear_history(self) -> None:
        """Clear training history to free memory or before retraining."""
        if self._built and self._ptr is not None:
            lib.ann_clear_history(self._ptr)
    
    def set_weight_decay(self, lambda_: float) -> "Sequential":
        """Set L2 regularization (weight decay) coefficient.
        
        L2 regularization penalizes large weights by adding lambda * ||W||^2 to the loss.
        Helps prevent overfitting by encouraging smaller, more distributed weights.
        
        Args:
            lambda_: L2 regularization strength (0 = disabled, typical: 1e-4 to 1e-2)
            
        Returns:
            self for method chaining
        """
        if not self._built:
            self._build()
        lib.ann_set_weight_decay(self._ptr, lambda_)
        return self
    
    def set_l1_regularization(self, lambda_: float) -> "Sequential":
        """Set L1 regularization (LASSO) coefficient.
        
        L1 regularization penalizes the absolute value of weights, encouraging sparsity.
        Pushes small weights toward exactly zero, useful for feature selection.
        
        Args:
            lambda_: L1 regularization strength (0 = disabled, typical: 1e-5 to 1e-3)
            
        Returns:
            self for method chaining
        """
        if not self._built:
            self._build()
        lib.ann_set_l1_regularization(self._ptr, lambda_)
        return self
    
    def confusion_matrix(
        self, 
        x: ArrayLike, 
        y: ArrayLike
    ) -> dict:
        """Compute binary confusion matrix and Matthews Correlation Coefficient.
        
        For binary classification problems (2 output classes).
        Class 0 = negative, Class 1 = positive.
        
        Args:
            x: Input data
            y: Expected outputs (one-hot encoded, shape [n_samples, 2])
            
        Returns:
            Dictionary with keys: 'tp', 'fp', 'tn', 'fn', 'mcc'
        """
        if not self._built:
            raise NetworkError("Model must be trained before computing confusion matrix")
        
        from pyann.core.tensor import Tensor
        x_tensor = Tensor(to_array(x))
        y_tensor = Tensor(to_array(y))
        
        tp = ffi.new("int *")
        fp = ffi.new("int *")
        tn = ffi.new("int *")
        fn = ffi.new("int *")
        
        mcc = lib.ann_confusion_matrix(
            self._ptr, x_tensor._ptr, y_tensor._ptr, tp, fp, tn, fn
        )
        
        return {
            'tp': tp[0],
            'fp': fp[0],
            'tn': tn[0],
            'fn': fn[0],
            'mcc': mcc
        }
    
    def summary(self) -> str:
        """Get a string summary of the model architecture."""
        lines = []
        lines.append(f"Sequential Model{f' ({self._name})' if self._name else ''}")
        lines.append("=" * 50)
        
        if self._input_shape:
            lines.append(f"Input shape: {self._input_shape}")
        
        total_params = 0
        prev_units = self._input_shape[0] if self._input_shape else 0
        
        for i, layer in enumerate(self._layers):
            if isinstance(layer, Dense):
                params = prev_units * layer.units + layer.units  # weights + biases
                total_params += params
                lines.append(
                    f"Dense({layer.units}, {layer.activation or 'linear'})"
                    f" - {params:,} params"
                )
                prev_units = layer.units
        
        lines.append("=" * 50)
        lines.append(f"Total parameters: {total_params:,}")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        n_layers = len(self._layers)
        return f"Sequential(layers={n_layers}, built={self._built})"
    
    def __del__(self):
        """Free the C network."""
        if hasattr(self, '_ptr') and self._ptr is not None:
            if self._ptr != ffi.NULL:
                lib.ann_free_network(self._ptr)
