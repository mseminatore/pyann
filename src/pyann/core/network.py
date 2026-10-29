"""Sequential neural network model with Keras-like API."""

from typing import Optional, Union, List, Tuple, TYPE_CHECKING
from pathlib import Path

from pyann._bindings.ffi import ffi, lib, gpu_enabled
from pyann.core.layers import (
    Layer, Dense, Input, Conv2D, MaxPool2D, Flatten, UpSample2D, Reshape,
    GlobalAveragePooling2D, BatchNormalization, DepthwiseConv2D, Conv1D, MaxPool1D, parse_layer,
    Embedding, GlobalMaxPooling2D,
)
from pyann.core.layers import _ConvBase, _PoolBase
from pyann.core.tensor import Tensor
from pyann.activations import Activation
from pyann.losses import Loss, parse_loss
from pyann.optimizers import Optimizer, parse_optimizer, OptimizerType
from pyann.weight_init import WeightInit, parse_weight_init
from pyann.callbacks import (
    LRScheduler, StepDecay, ExponentialDecay, CosineAnnealing,
    EarlyStopping
)
from pyann.utils.compat import ArrayLike, has_numpy, to_array, get_shape
from pyann.exceptions import (
    NetworkError, InvalidParameterError, AllocationError,
    raise_for_error_code, clear_c_error, c_error_message
)

if TYPE_CHECKING:
    import numpy as np
    from pyann.core.quantized import QuantizedModel


def _batch_buffer(data: ArrayLike, columns: int, name: str):
    """Convert one mini-batch to a C float buffer, checking its shape.
    
    Returns (keepalive, rows, pointer); keep the first element referenced for
    as long as the pointer is in use.
    """
    arr = to_array(data)
    
    if has_numpy():
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be 1-D (one sample) or 2-D (samples x features), "
                f"got {arr.ndim}-D"
            )
        rows, cols = arr.shape
        if rows == 0:
            raise ValueError(f"{name} has no samples")
        if cols != columns:
            raise ValueError(f"{name} has {cols} columns but the network expects {columns}")
        return arr, rows, ffi.cast("real *", arr.ctypes.data)
    
    rows = len(arr)
    if rows == 0 or any(len(row) == 0 for row in arr):
        raise ValueError(f"{name} has no samples")
    for row in arr:
        if len(row) != columns:
            raise ValueError(f"{name} has a row of {len(row)} values but the network expects {columns}")
    buffer = ffi.new("real[]", [v for row in arr for v in row])
    return buffer, rows, buffer


def as_rows(data: ArrayLike, input_shape: Optional[Tuple[int, ...]], name: str = "x") -> ArrayLike:
    """Flatten image samples to rows: (N, *input_shape) -> (N, features).
    
    A single sample of the input shape becomes one row. Data that is
    already one row per sample is returned unchanged.
    """
    arr = to_array(data)
    sample = tuple(input_shape or ())
    if not has_numpy() or len(sample) < 2 or arr.ndim <= 2 and arr.shape != sample:
        return arr
    if arr.shape == sample:
        return arr.reshape(1, -1)
    if tuple(arr.shape[1:]) == sample:
        return arr.reshape(arr.shape[0], -1)
    hint = ""
    if arr.ndim == 4 and len(sample) == 3 and (arr.shape[3], arr.shape[1], arr.shape[2]) == sample:
        hint = "; the data looks channels-last, use x.transpose(0, 3, 1, 2)"
    raise ValueError(f"{name} samples have shape {tuple(arr.shape[1:])} but the model expects "
                     f"{sample} (channels, height, width){hint}")


def _activation_name(layer) -> str:
    """A layer's activation for display: 'relu', 'softmax', ... or 'linear'."""
    act = layer.activation_type
    return "linear" if act == Activation.NONE else act.name.lower()


# names accepted by Sequential.set_input_noise()
_NOISE_TYPES = {"gaussian": "NOISE_GAUSSIAN", "masking": "NOISE_MASKING", "salt_pepper": "NOISE_SALT_PEPPER"}


def as_images(rows: ArrayLike, output_shape: Optional[Tuple[int, ...]]) -> ArrayLike:
    """Rows of a model's outputs as (N, channels, height, width) when the
    output is an image (output_shape has 3 dims), or (N, channels, steps)
    when it is a sequence (2 dims); otherwise unchanged."""
    if output_shape is None or len(output_shape) not in (2, 3):
        return rows
    if has_numpy():
        return rows.reshape((len(rows),) + tuple(output_shape))
    if len(output_shape) == 2:
        c, w = output_shape
        return [[row[k * w:(k + 1) * w] for k in range(c)] for row in rows]
    c, h, w = output_shape
    return [[[row[(k * h + i) * w:(k * h + i + 1) * w] for i in range(h)] for k in range(c)] for row in rows]


# layers that need an image (channels, height, width) input, and those that
# read a sequence (channels, steps): an image one row high
_IMAGE_LAYERS = (_ConvBase, _PoolBase, UpSample2D, GlobalAveragePooling2D, GlobalMaxPooling2D)
_SEQUENCE_LAYERS = (Conv1D, MaxPool1D)


def _prod(shape) -> int:
    result = 1
    for d in shape:
        result *= d
    return result


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
    
    # Class-level defaults so models created via __new__ (load/load_onnx) have them
    _gpu_uploaded = False
    _online_active = False
    _weight_decay: Optional[float] = None
    _l1_regularization: Optional[float] = None
    _gradient_clip: Optional[float] = None
    _weight_init: Optional[WeightInit] = None
    _convergence: Optional[float] = None
    _dropout: Optional[float] = None
    _noise: Optional[Tuple[str, float, float, float]] = None

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
        elif isinstance(layer, (Dense, Conv2D, Conv1D, Embedding)):
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
        l1_regularization: Optional[float] = None,
        gradient_clip: Optional[float] = None,
        weight_init: Union[str, WeightInit, None] = None,
        convergence: Optional[float] = None,
        dropout: Optional[float] = None
    ) -> "Sequential":
        """Configure the model for training.
        
        Args:
            optimizer: Optimizer to use
            loss: Loss function
            learning_rate: Override the optimizer's learning rate
            weight_decay: L2 regularization coefficient (0 = disabled)
            l1_regularization: L1 regularization coefficient (0 = disabled)
            gradient_clip: Clip gradients to [-value, value] (0 = disabled).
                Helps with exploding gradients; 1.0-5.0 is typical.
            weight_init: Weight initialization: 'uniform', 'xavier', 'he' or
                'auto' (default: chosen per layer from its activation)
            convergence: Stop training early once the average loss falls below
                this value (libann's default is 0.01)
            dropout: Default dropout rate for hidden layers that don't set
                their own via Dense(dropout=...), in [0, 1)
            
        Returns:
            self for method chaining
        """
        if gradient_clip is not None and gradient_clip < 0:
            raise ValueError(f"gradient_clip must be >= 0, got {gradient_clip}")
        if convergence is not None and convergence < 0:
            raise ValueError(f"convergence must be >= 0, got {convergence}")
        if dropout is not None and not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if weight_init is not None:
            weight_init = parse_weight_init(weight_init)

        if optimizer is not None:
            self._optimizer = parse_optimizer(optimizer)
        if loss is not None:
            self._loss = parse_loss(loss)
        if learning_rate is not None:
            # A loaded model has no Optimizer object; its rate lives in libann
            if self._optimizer is not None:
                self._optimizer.learning_rate = learning_rate
            # Model files don't store the rate, so a loaded model needs this
            # to keep training (fit() or train_on_batch()) at its old rate
            if self._built:
                lib.ann_set_learning_rate(self._ptr, learning_rate)
        if weight_decay is not None:
            self._weight_decay = weight_decay
        if l1_regularization is not None:
            self._l1_regularization = l1_regularization
        if gradient_clip is not None:
            self._gradient_clip = gradient_clip
        if weight_init is not None:
            self._weight_init = weight_init
        if convergence is not None:
            self._convergence = convergence
        if dropout is not None:
            self._dropout = dropout
        
        # Settings changed after the network exists take effect immediately
        if self._built:
            self._apply_training_settings()
        
        return self
    
    def _build(self) -> None:
        """Build the C network from layer specifications."""
        if self._built:
            return
        
        if not self._layers:
            raise NetworkError("Cannot build empty model - add layers first")
        
        if self._input_shape is None:
            raise NetworkError(
                "Input shape not set. Provide input_shape to the first layer "
                "or add an Input layer."
            )
        
        # Checks every layer fits its input before anything is allocated
        self._layer_shapes()
        
        # Create C network
        opt_type = self._optimizer._get_type()
        self._ptr = lib.ann_make_network(opt_type, self._loss)
        if self._ptr == ffi.NULL:
            raise AllocationError("Failed to create network")
        
        # Set learning rate
        if self._optimizer.learning_rate is not None:
            lib.ann_set_learning_rate(self._ptr, self._optimizer.learning_rate)
        
        try:
            # the input to each layer: the input layer's shape, then each layer's output
            in_shapes = [self._image_input_shape() or (_prod(self._input_shape),)]
            in_shapes += [shape for _, shape, _ in self._layer_shapes()]
            image = self._image_input_shape()
            if image is not None:
                result = lib.ann_add_input(self._ptr, *image)
            else:
                result = lib.ann_add_layer(self._ptr, _prod(self._input_shape), lib.LAYER_INPUT, lib.ACTIVATION_NULL)
            raise_for_error_code(result, "Failed to add input layer")
            
            # Add hidden and output layers; Python layer i is C layer i + 1
            for i, layer in enumerate(self._layers):
                if isinstance(layer, Dense):
                    is_output = (i == len(self._layers) - 1)
                    layer_type = lib.LAYER_OUTPUT if is_output else lib.LAYER_HIDDEN
                    result = lib.ann_add_layer(self._ptr, layer.units, layer_type, layer.activation_type)
                elif isinstance(layer, _ConvBase):
                    channels = in_shapes[i][0]
                    result = lib.ann_add_conv2d_ex(self._ptr, layer.output_filters(channels), *layer.window(),
                                                   layer.conv_groups(channels), layer.activation_type)
                elif isinstance(layer, _PoolBase):
                    result = lib.ann_add_maxpool2d_ex(self._ptr, *layer.window())
                elif isinstance(layer, Flatten):
                    result = lib.ann_add_flatten(self._ptr)
                elif isinstance(layer, UpSample2D):
                    result = lib.ann_add_upsample2d(self._ptr, layer.size)
                elif isinstance(layer, Reshape):
                    result = lib.ann_add_reshape(self._ptr, *layer.target_shape)
                elif isinstance(layer, GlobalAveragePooling2D):
                    result = lib.ann_add_global_avgpool2d(self._ptr)
                elif isinstance(layer, BatchNormalization):
                    result = lib.ann_add_batchnorm(self._ptr, layer.activation_type)
                elif isinstance(layer, Embedding):
                    add = lib.ann_add_embedding_masked if layer.mask_padding else lib.ann_add_embedding
                    padding = -1 if layer.padding_idx is None else layer.padding_idx
                    result = add(self._ptr, layer.input_dim, layer.output_dim, padding)
                elif isinstance(layer, GlobalMaxPooling2D):
                    result = lib.ann_add_global_maxpool2d(self._ptr)
                else:
                    raise InvalidParameterError(f"Unsupported layer type {type(layer).__name__}")
                raise_for_error_code(result, f"Failed to add layer {i} ({type(layer).__name__})")
                
                if isinstance(layer, Embedding):
                    if layer.weights is not None:
                        w = layer.weights
                        raise_for_error_code(lib.ann_set_layer_param(self._ptr, i + 1, 0, ffi.cast("real *", w.ctypes.data),
                                                                     w.size), "Failed to set the Embedding's weights")
                    if not layer.trainable:
                        lib.ann_set_layer_trainable(self._ptr, i + 1, 0)
                
                # Set dropout if specified
                if getattr(layer, "dropout", 0.0) > 0:
                    lib.ann_set_layer_dropout(self._ptr, i + 1, layer.dropout)
        except Exception:
            lib.ann_free_network(self._ptr)
            self._ptr = None
            raise
        
        self._apply_training_settings()
        
        self._built = True
    
    def _first_image_layer(self) -> Optional[Layer]:
        """The first layer that needs an image input, unless a Reshape or an
        Embedding makes an image (or sequence) of its own before it."""
        for layer in self._layers:
            if isinstance(layer, (Reshape, Embedding)):
                return None
            if isinstance(layer, _IMAGE_LAYERS):
                return layer
        return None

    def _uses_images(self) -> bool:
        """Whether the input must be an image: a Conv2D, MaxPool2D, Conv1D,
        UpSample2D or GlobalAveragePooling2D layer (and the like) reads it
        before any Reshape makes an image of its own."""
        return self._first_image_layer() is not None

    def _is_sequence(self) -> bool:
        """Whether the model works on sequences (channels, steps): a 2-D
        input shape read first by a Conv1D or MaxPool1D layer, or an
        Embedding, which makes a sequence of its ids."""
        if any(isinstance(layer, Embedding) for layer in self._layers):
            return True
        shape = self._input_shape
        return shape is not None and len(shape) == 2 and isinstance(self._first_image_layer(), _SEQUENCE_LAYERS)

    def _image_input_shape(self) -> Optional[Tuple[int, int, int]]:
        """(channels, height, width) of the input when it is an image, else None.
        
        A 3-D input shape is an image; a 2-D shape is a sequence (channels,
        steps), the image (channels, 1, steps), when a Conv1D or MaxPool1D
        layer reads it, or else a one-channel image (height, width) when
        convolution or pooling layers follow.
        """
        shape = self._input_shape
        if shape is not None and len(shape) == 3:
            return tuple(shape)
        if shape is not None and len(shape) == 2 and self._is_sequence() and \
                isinstance(self._first_image_layer(), _SEQUENCE_LAYERS):
            return (shape[0], 1, shape[1])
        if shape is not None and len(shape) == 2 and self._uses_images():
            return (1, shape[0], shape[1])
        return None

    def _public_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """A layer's shape as the user sees it: (channels, steps) for the
        images one row high of a sequence model, else unchanged."""
        if self._is_sequence() and len(shape) == 3 and shape[1] == 1:
            return (shape[0], shape[2])
        return shape

    def _layer_shapes(self) -> List[Tuple[Layer, Tuple[int, ...], int]]:
        """(layer, output shape, parameter count) for each layer.
        
        Raises:
            NetworkError: If a layer does not fit the output of the one before
        """
        image = self._image_input_shape()
        if self._uses_images() and image is None:
            raise NetworkError("Conv2D, MaxPool2D, UpSample2D and GlobalAveragePooling2D need an image "
                               "input_shape: (channels, height, width); Conv1D and MaxPool1D a sequence: "
                               "(channels, steps)")
        shape = image if image is not None else (_prod(self._input_shape),)
        
        result = []
        for i, layer in enumerate(self._layers):
            name = type(layer).__name__
            if isinstance(layer, Dense):
                new_shape = (layer.units,)
                params = _prod(shape) * layer.units + layer.units
            elif isinstance(layer, Reshape):
                if _prod(layer.target_shape) != _prod(shape):
                    raise NetworkError(f"Layer {i} (Reshape) to {layer.target_shape} needs {_prod(layer.target_shape)} "
                                       f"values, but the layer before outputs {_prod(shape)}")
                new_shape = layer.target_shape
                params = 0
            elif isinstance(layer, BatchNormalization):
                # gamma and beta, plus the running mean and variance, per channel
                new_shape = shape
                params = 4 * shape[0]
            elif isinstance(layer, Embedding):
                if len(shape) != 1:
                    raise NetworkError(f"Layer {i} (Embedding) needs a flat input of token ids, but the layer "
                                       f"before outputs {shape}")
                new_shape = layer.output_shape(shape)
                params = layer.param_count()
            elif isinstance(layer, _IMAGE_LAYERS):
                if len(shape) != 3:
                    raise NetworkError(f"Layer {i} ({name}) needs an image input, but the layer before outputs {shape}")
                if isinstance(layer, _ConvBase) and shape[0] % layer.conv_groups(shape[0]):
                    raise NetworkError(f"Layer {i} ({name}) has {layer.conv_groups(shape[0])} groups, which do "
                                       f"not divide its {shape[0]} input channels")
                new_shape = layer.output_shape(shape)
                params = layer.param_count(shape[0]) if isinstance(layer, _ConvBase) else 0
            else:
                new_shape = (_prod(shape),)
                params = 0
            if min(new_shape) < 1:
                raise NetworkError(f"Layer {i} ({name}) does not fit its input of shape {shape}")
            result.append((layer, new_shape, params))
            shape = new_shape
        return result

    def _as_rows(self, data: ArrayLike, name: str = "x") -> ArrayLike:
        """Flatten image samples to rows: (N, *input_shape) -> (N, features)."""
        return as_rows(data, self._input_shape, name)

    def _output_shape(self) -> Optional[Tuple[int, ...]]:
        """Shape of one output: (units,), (channels, height, width) for an
        image, or (channels, steps) for a sequence model's sequence."""
        if not self._layers or not self._input_shape:
            return None
        return self._public_shape(self._layer_shapes()[-1][1])

    def _targets_as_rows(self, data: ArrayLike, name: str = "y") -> ArrayLike:
        """Flatten image or sequence targets (e.g. an autoencoder's) to rows, like inputs."""
        shape = self._output_shape()
        return as_rows(data, shape if shape and len(shape) >= 2 else None, name)

    def _apply_training_settings(self) -> None:
        """Push the optional training settings from compile() to the C network."""
        if self._weight_decay is not None and self._weight_decay > 0:
            lib.ann_set_weight_decay(self._ptr, self._weight_decay)
        if self._l1_regularization is not None and self._l1_regularization > 0:
            lib.ann_set_l1_regularization(self._ptr, self._l1_regularization)
        if self._gradient_clip is not None:
            lib.ann_set_gradient_clip(self._ptr, self._gradient_clip)
        if self._weight_init is not None:
            lib.ann_set_weight_init(self._ptr, int(self._weight_init))
        if self._convergence is not None:
            lib.ann_set_convergence(self._ptr, self._convergence)
        # Applies only to hidden layers without their own rate, so it must run
        # after the layers (and their Dense(dropout=...) overrides) are in place
        if self._dropout is not None:
            lib.ann_set_dropout(self._ptr, self._dropout)
        if self._noise is not None:
            noise_type, amount, low, high = self._noise
            result = lib.ann_set_input_noise(self._ptr, getattr(lib, _NOISE_TYPES[noise_type]), amount, low, high)
            raise_for_error_code(result, "Failed to set input noise")
    
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
            validation_data: Optional (x_val, y_val) tuple; its loss is
                measured after every epoch and monitored by EarlyStopping
            verbose: 0 trains silently; 1 prints libann's progress (each
                epoch's loss and learning rate)
            callbacks: List of callbacks (e.g., EarlyStopping)
            lr_scheduler: Learning rate scheduler
            
        Returns:
            Dictionary with training history: ``epochs`` (the limit),
            ``epochs_trained`` (fewer when early stopping or the convergence
            threshold ended training) and ``loss`` (the last epoch's training
            loss); with validation data, ``val_loss`` and ``val_accuracy`` for
            the final weights; with EarlyStopping, ``best_epoch``. Per-epoch
            values are in :meth:`training_history`.
        """
        self._build()
        self.end_online_training()
        
        # Convert data to tensors
        x_arr = self._as_rows(x)
        y_arr = self._targets_as_rows(y)
        
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
        
        # Early stopping runs inside libann; every fit() sets it, so a
        # callback from an earlier fit() does not carry over
        stopping = next((c for c in (callbacks or []) if isinstance(c, EarlyStopping)), None)
        if stopping is not None:
            if stopping.patience < 1:
                raise InvalidParameterError("EarlyStopping patience must be at least 1")
            lib.ann_set_early_stopping(self._ptr, stopping.patience, stopping.min_delta,
                                       1 if stopping.restore_best else 0)
        else:
            lib.ann_set_early_stopping(self._ptr, 0, 0.0, 0)
        
        # libann borrows the validation tensors, so they stay referenced until
        # training ends and are detached before they are freed
        val_tensors = None
        if validation_data is not None:
            x_val, y_val = validation_data
            val_tensors = (Tensor(self._as_rows(x_val, "validation x")),
                           Tensor(self._targets_as_rows(y_val, "validation y")))
            raise_for_error_code(
                lib.ann_set_validation_data(self._ptr, val_tensors[0]._ptr, val_tensors[1]._ptr),
                "Invalid validation data",
            )
        
        # libann prints progress through the network's print function; a NULL
        # one silences it for this fit()
        print_func = lib.ann_get_print_function(self._ptr)
        if not verbose:
            lib.ann_set_print_function(self._ptr, ffi.NULL)
        
        # Train (on the GPU when pyann.enable_gpu() succeeded)
        try:
            self._gpu_upload()
            loss = lib.ann_train_network(self._ptr, x_tensor._ptr, y_tensor._ptr, n_samples)
            if loss == 0.0:
                raise NetworkError("Training failed")
            
            # Build history dict (loss: the last epoch's average training loss)
            history = {
                "epochs": epochs,
                "epochs_trained": lib.ann_get_epochs_trained(self._ptr),
                "loss": loss,
            }
            if stopping is not None:
                history["best_epoch"] = lib.ann_get_best_epoch(self._ptr)
            
            # Evaluate the final weights on the validation data
            if val_tensors is not None:
                history["val_loss"] = lib.ann_evaluate_loss(self._ptr, val_tensors[0]._ptr, val_tensors[1]._ptr)
                history["val_accuracy"] = lib.ann_evaluate_accuracy(self._ptr, val_tensors[0]._ptr, val_tensors[1]._ptr)
        finally:
            if val_tensors is not None:
                lib.ann_set_validation_data(self._ptr, ffi.NULL, ffi.NULL)
            lib.ann_set_print_function(self._ptr, print_func)
        
        return history
    
    def train_on_batch(self, x: ArrayLike, y: ArrayLike) -> float:
        """Run one training step on a single mini-batch (online training).
        
        Unlike fit(), this keeps the weights and the optimizer state (for
        example Adam's momentum) between calls, so it suits streaming data,
        fine-tuning a loaded model, and custom training loops. The learning
        rate scheduler and callbacks only apply to fit().
        
        The first call starts an online training session, which stays open
        across calls. predict() and evaluate() work during a session. Call
        end_online_training() when you are done (fit() does so itself).
        
        Args:
            x: Inputs, shape (samples, features); a 1-D array is one sample
            y: Targets, shape (samples, outputs); a 1-D array is one sample
            
        Returns:
            Loss for this mini-batch
            
        Raises:
            ValueError: If the shapes of x and y don't match the network
        """
        self._build()
        
        n_layers = lib.ann_get_layer_count(self._ptr)
        in_units = lib.ann_get_layer_nodes(self._ptr, 0)
        out_units = lib.ann_get_layer_nodes(self._ptr, n_layers - 1)
        
        if (self._loss == Loss.CATEGORICAL_CROSS_ENTROPY and
                lib.ann_get_layer_activation(self._ptr, n_layers - 1) not in
                (lib.ACTIVATION_SOFTMAX, lib.ACTIVATION_SIGMOID)):
            raise ValueError(
                "categorical cross-entropy needs a softmax or sigmoid output layer"
            )
        
        # libann copies batch_size * units values from these buffers without
        # checking their size, so the shapes must be verified here
        x_keep, x_rows, x_ptr = _batch_buffer(self._as_rows(x), in_units, "x")
        y_keep, y_rows, y_ptr = _batch_buffer(self._targets_as_rows(y), out_units, "y")
        if x_rows != y_rows:
            raise ValueError(f"x has {x_rows} samples but y has {y_rows}")
        
        if not self._online_active:
            result = lib.ann_train_begin(self._ptr)
            raise_for_error_code(result, "Failed to begin online training")
            self._online_active = True
        
        return float(lib.ann_train_step(self._ptr, x_ptr, y_ptr, x_rows))
    
    def end_online_training(self) -> "Sequential":
        """End the online training session started by train_on_batch().
        
        Turns dropout off and, when the GPU is enabled, re-uploads the updated
        weights. Does nothing if no session is open. A later train_on_batch()
        starts a new session and keeps the weights and optimizer state.
        
        Returns:
            self for method chaining
        """
        if self._online_active:
            lib.ann_train_end(self._ptr)
            self._online_active = False
        return self
    
    def _gpu_upload(self) -> None:
        """Upload weights to the GPU if a backend is active (else a no-op).

        Training and batch inference only use the GPU for uploaded networks.
        On upload failure the model keeps using the CPU path.
        """
        if self._gpu_uploaded or not gpu_enabled():
            return
        self._gpu_uploaded = lib.ann_gpu_upload_network(self._ptr) == 0

    def _setup_lr_scheduler(self, scheduler: LRScheduler) -> None:
        """Configure a learning rate scheduler."""
        if isinstance(scheduler, StepDecay):
            self._lr_params = ffi.new("LRStepParams *")
            self._lr_params.step_size = scheduler.step_size
            self._lr_params.gamma = scheduler.gamma
            lib.ann_set_lr_scheduler(self._ptr, lib.ann_lr_scheduler_step, self._lr_params)
        elif isinstance(scheduler, ExponentialDecay):
            self._lr_params = ffi.new("LRExponentialParams *")
            self._lr_params.gamma = scheduler.gamma
            lib.ann_set_lr_scheduler(self._ptr, lib.ann_lr_scheduler_exponential, self._lr_params)
        elif isinstance(scheduler, CosineAnnealing):
            self._lr_params = ffi.new("LRCosineParams *")
            self._lr_params.T_max = scheduler.T_max
            self._lr_params.min_lr = scheduler.min_lr
            lib.ann_set_lr_scheduler(self._ptr, lib.ann_lr_scheduler_cosine, self._lr_params)
    
    def predict(self, x: ArrayLike) -> ArrayLike:
        """Generate predictions for input samples.
        
        Args:
            x: Input data
            
        Returns:
            Predictions (numpy array if available, else nested lists):
            (samples, outputs), or (samples, channels, height, width) when the
            output layer produces an image (Conv2D, MaxPool2D, UpSample2D,
            Reshape, or a BatchNormalization of an image), e.g. an autoencoder's
        """
        if not self._built:
            raise NetworkError("Model must be trained before making predictions")
        
        x_arr = self._as_rows(x)
        shape = get_shape(x_arr)
        
        # Handle single sample vs batch
        if len(shape) == 1:
            x_arr = to_array([x_arr.tolist() if has_numpy() else x_arr])
            shape = get_shape(x_arr)
        
        n_samples = shape[0]
        output_size = lib.ann_get_layer_nodes(self._ptr, lib.ann_get_layer_count(self._ptr) - 1)
        
        self._gpu_upload()

        if has_numpy():
            import numpy as np
            x_f32 = np.ascontiguousarray(x_arr, dtype=np.float32)
            results = np.zeros((n_samples, output_size), dtype=np.float32)
            result = lib.ann_predict_batch(
                self._ptr,
                ffi.cast("real *", x_f32.ctypes.data),
                ffi.cast("real *", results.ctypes.data),
                n_samples,
            )
            raise_for_error_code(result, "Prediction failed")
            return as_images(results, self._output_shape())
        else:
            flat = [float(v) for row in x_arr for v in row]
            inputs = ffi.new("real[]", flat)
            outputs = ffi.new("real[]", n_samples * output_size)
            result = lib.ann_predict_batch(self._ptr, inputs, outputs, n_samples)
            raise_for_error_code(result, "Prediction failed")
            return as_images([
                [outputs[i * output_size + j] for j in range(output_size)]
                for i in range(n_samples)
            ], self._output_shape())

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
        
        x_tensor = Tensor(self._as_rows(x))
        y_tensor = Tensor(self._targets_as_rows(y))
        
        accuracy = lib.ann_evaluate_accuracy(self._ptr, x_tensor._ptr, y_tensor._ptr)
        return accuracy
    
    def evaluate_loss(self, x: ArrayLike, y: ArrayLike) -> float:
        """Compute the model's loss on data, averaged like the training loss.
        
        Args:
            x: Input data
            y: Target data
            
        Returns:
            Average loss (MSE, or cross-entropy, per the model's loss function)
        """
        if not self._built:
            raise NetworkError("Model must be trained before evaluation")
        
        x_tensor = Tensor(self._as_rows(x))
        y_tensor = Tensor(self._targets_as_rows(y))
        
        loss = lib.ann_evaluate_loss(self._ptr, x_tensor._ptr, y_tensor._ptr)
        if loss < 0.0:
            raise InvalidParameterError("Data shape does not match the model's input and output sizes")
        return loss
    
    @property
    def layers(self) -> List[Layer]:
        """The model's layers after the input, in order. Their positions (or
        the layers themselves) select a layer in get_layer_weights,
        set_layer_weights, set_trainable and is_trainable."""
        return list(self._layers)

    def _layer_index(self, layer: Union[int, Layer]) -> int:
        """libann's index of a layer, given as a position in ``layers`` (negative
        counts from the end) or as the layer itself."""
        if isinstance(layer, Layer):
            for i, own in enumerate(self._layers):
                if own is layer:
                    return i + 1
            raise InvalidParameterError(f"this {type(layer).__name__} is not a layer of the model")
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError(f"layer must be an int index or a Layer, got {type(layer).__name__}")
        n = len(self._layers)
        if not -n <= layer < n:
            raise IndexError(f"layer index {layer} is out of range for a model with {n} layers")
        return layer % n + 1

    def _param_shape(self, index: int, param: int, rows: int, cols: int) -> Tuple[int, ...]:
        """A parameter's array shape: libann keeps each as a [rows x cols] matrix."""
        layer = self._layers[index - 1]
        if isinstance(layer, (Dense, Embedding)) and param == 0:
            return (rows, cols)
        if isinstance(layer, _ConvBase) and param == 0:
            win = ffi.new("int[6]")
            lib.ann_get_layer_window(self._ptr, index, win, win + 1, win + 2, win + 3, win + 4, win + 5)
            kh, kw = win[0], win[1]
            if isinstance(layer, Conv1D):
                return (rows, cols // kw, kw)
            return (rows, cols // (kh * kw), kh, kw)
        return (cols,)

    def get_layer_weights(self, layer: Union[int, Layer]) -> List["np.ndarray"]:
        """A layer's parameters as float32 arrays (copies), in libann's order.

        Dense: weights (units, inputs) and bias (units,), the layout of
        PyTorch's nn.Linear (Keras stores the transpose). Conv2D and
        DepthwiseConv2D: weights (filters, channels / groups, kh, kw) and bias
        (filters,); Conv1D: weights (filters, channels / groups, k) and bias.
        BatchNormalization: gamma, beta, running mean and running variance,
        each (channels,). Embedding: the vectors (input_dim, output_dim).
        Other layers have none ([]).

        Before the first training (and without set_layer_weights or a load)
        the values are the ones the layer was created with: zero weights.

        Args:
            layer: Position in ``model.layers`` (negative counts from the
                end), or the layer itself
        """
        import numpy as np
        self._build()
        index = self._layer_index(layer)
        dims = ffi.new("int[2]")
        weights = []
        for p in range(lib.ann_get_layer_param_count(self._ptr, index)):
            raise_for_error_code(lib.ann_get_layer_param_shape(self._ptr, index, p, dims, dims + 1),
                                 "Failed to read a parameter's shape")
            values = np.empty(dims[0] * dims[1], dtype=np.float32)
            raise_for_error_code(lib.ann_get_layer_param(self._ptr, index, p, ffi.cast("real *", values.ctypes.data),
                                                         values.size), "Failed to read a parameter")
            weights.append(values.reshape(self._param_shape(index, p, dims[0], dims[1])))
        return weights

    def set_layer_weights(self, layer: Union[int, Layer], weights: List[ArrayLike]) -> "Sequential":
        """Set a layer's parameters, e.g. pre-trained embedding vectors or a
        layer of another model, as get_layer_weights returns them (same
        number of arrays and shapes).

        Training then starts from these values (other layers are still
        initialized randomly), and a model on the GPU gets them too. To keep
        them fixed while the rest trains, see set_trainable.

        Args:
            layer: Position in ``model.layers``, or the layer itself
            weights: One array per parameter

        Returns:
            self for method chaining
        """
        import numpy as np
        current = self.get_layer_weights(layer)
        index = self._layer_index(layer)
        name = type(self._layers[index - 1]).__name__
        if len(weights) != len(current):
            raise ValueError(f"{name} has {len(current)} parameter arrays, got {len(weights)}")
        arrays = []
        for p, (w, old) in enumerate(zip(weights, current)):
            w = np.ascontiguousarray(w, dtype=np.float32)
            if w.shape != old.shape:
                raise ValueError(f"{name} parameter {p} has shape {old.shape}, got {w.shape}")
            if not np.isfinite(w).all():
                raise ValueError(f"{name} parameter {p} has values that are NaN or infinite")
            arrays.append(w)
        for p, w in enumerate(arrays):
            raise_for_error_code(lib.ann_set_layer_param(self._ptr, index, p, ffi.cast("real *", w.ctypes.data),
                                                         w.size), f"Failed to set {name} parameter {p}")
        return self

    def get_weights(self) -> List["np.ndarray"]:
        """Every layer's parameters, one list in layer order (as Keras's
        get_weights; see get_layer_weights for the shapes)."""
        self._build()
        return [w for i in range(len(self._layers)) for w in self.get_layer_weights(i)]

    def set_weights(self, weights: List[ArrayLike]) -> "Sequential":
        """Set every layer's parameters from a list like get_weights returns,
        e.g. another model's with the same layers.

        Returns:
            self for method chaining
        """
        self._build()
        counts = [lib.ann_get_layer_param_count(self._ptr, i + 1) for i in range(len(self._layers))]
        if len(weights) != sum(counts):
            raise ValueError(f"the model has {sum(counts)} parameter arrays, got {len(weights)}")
        # check every shape before changing anything
        import numpy as np
        for p, (w, old) in enumerate(zip(weights, self.get_weights())):
            if np.shape(w) != old.shape:
                raise ValueError(f"weights[{p}] has shape {np.shape(w)}, expected {old.shape}")
        start = 0
        for i, n in enumerate(counts):
            if n:
                self.set_layer_weights(i, weights[start:start + n])
            start += n
        return self

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """Save every layer's parameters (get_weights) to a NumPy .npz file.

        Only the values: load them with load_weights into a model built with
        the same layers, e.g. to keep pre-trained embedding vectors apart from
        the model that learned them. ``save()`` keeps the whole model.
        The file holds ``layer<i>.<p>`` arrays (parameter p of model.layers[i])
        and ``layers``, the layer class names.
        """
        import numpy as np
        self._build()
        arrays = {"layers": np.array([type(layer).__name__ for layer in self._layers])}
        for i in range(len(self._layers)):
            for p, w in enumerate(self.get_layer_weights(i)):
                arrays[f"layer{i}.{p}"] = w
        with open(filepath, "wb") as f:     # an open file, so np.savez adds no .npz to the name
            np.savez(f, **arrays)

    def load_weights(self, filepath: Union[str, Path]) -> "Sequential":
        """Set every layer's parameters from a file save_weights wrote.

        The model must have the same layers (kinds and shapes); nothing
        changes when they differ.

        Returns:
            self for method chaining
        """
        import numpy as np
        self._build()
        keys = [f"layer{i}.{p}" for i in range(len(self._layers))
                for p in range(lib.ann_get_layer_param_count(self._ptr, i + 1))]
        with np.load(filepath, allow_pickle=False) as data:
            mine = [type(layer).__name__ for layer in self._layers]
            saved = [str(name) for name in data["layers"]] if "layers" in data.files else None
            if saved is not None and saved != mine:
                raise ValueError(f"{filepath} holds the weights of layers {saved}, but this model has {mine}")
            stored = set(data.files) - {"layers"}
            if stored != set(keys):
                raise ValueError(f"{filepath} has the weights of another model: expected arrays {keys[:4]}..., "
                                 f"found {sorted(stored)[:4]}...")
            weights = [data[k] for k in keys]
        return self.set_weights(weights)

    def set_trainable(self, layer: Union[int, Layer], trainable: bool = True) -> "Sequential":
        """Freeze a layer (trainable=False) or let it train again.

        Training (fit or train_on_batch, on the CPU or a GPU) leaves a frozen
        layer's parameters as they are, while gradients still pass through it
        to the layers below; a frozen BatchNormalization still updates its
        running statistics. Typical use: keep pre-trained embedding vectors
        fixed while the layers above learn, then unfreeze them and train a
        little more at a low learning rate. Layers below the lowest one that
        trains get no backward pass, so a frozen embedding also trains
        faster. ``save()`` keeps the setting (ONNX export does not).

        Args:
            layer: Position in ``model.layers``, or the layer itself
            trainable: False to freeze the layer

        Returns:
            self for method chaining
        """
        self._build()
        index = self._layer_index(layer)
        raise_for_error_code(lib.ann_set_layer_trainable(self._ptr, index, int(bool(trainable))),
                             "Failed to set whether the layer trains")
        if hasattr(self._layers[index - 1], "trainable"):     # Embedding(trainable=...) follows it
            self._layers[index - 1].trainable = bool(trainable)
        return self

    def is_trainable(self, layer: Union[int, Layer]) -> bool:
        """False if the layer is frozen (see set_trainable)."""
        self._build()
        return bool(lib.ann_get_layer_trainable(self._ptr, self._layer_index(layer)))

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
        
        clear_c_error()
        if format == "binary":
            ptr = lib.ann_load_network_binary(filepath.encode())
        else:
            ptr = lib.ann_load_network(filepath.encode())
        
        if ptr == ffi.NULL:
            raise NetworkError(c_error_message(f"Failed to load model from {filepath}"))
        
        return cls._from_ptr(ptr, filepath)
    
    @classmethod
    def _from_ptr(cls, ptr, filepath) -> "Sequential":
        """Wrap a network loaded by libann, rebuilding the Python layer list.
        
        A layer kind pyann has no class for (saved by a newer libann) is
        rejected rather than misread.
        """
        n_layers = lib.ann_get_layer_count(ptr)
        dims = ffi.new("int[3]")
        settings = ffi.new("int[3]")
        win = ffi.new("int[6]")
        layers, input_shape, in_dims = [], None, None
        
        for i in range(n_layers):
            kind = lib.ann_get_layer_kind(ptr, i)
            lib.ann_get_layer_shape(ptr, i, dims, dims + 1, dims + 2)
            lib.ann_get_layer_settings(ptr, i, settings, settings + 1, settings + 2)
            lib.ann_get_layer_window(ptr, i, win, win + 1, win + 2, win + 3, win + 4, win + 5)
            channels, height, width = dims[0], dims[1], dims[2]
            kh, kw, sh, sw, ph, pw = (win[k] for k in range(6))
            # a 1 x k window over an input one row high is a 1-D layer
            one_d = in_dims is not None and in_dims[1] == 1 and kh == 1 and sh == 1 and ph == 0
            activation = Activation(lib.ann_get_layer_activation(ptr, i))
            
            if i == 0:
                input_shape = (channels,) if height == width == 1 else (channels, height, width)
            elif kind == lib.LAYER_KIND_DENSE:
                layers.append(Dense(lib.ann_get_layer_nodes(ptr, i), activation=activation))
            elif kind == lib.LAYER_KIND_CONV2D:
                groups, in_channels = lib.ann_get_layer_groups(ptr, i), in_dims[0]
                if groups > 1 and groups == in_channels:
                    layers.append(DepthwiseConv2D((kh, kw), (sh, sw), (ph, pw), channels // in_channels,
                                                  activation=activation))
                elif one_d:
                    layers.append(Conv1D(channels, kw, sw, pw, activation=activation, groups=groups))
                else:
                    layers.append(Conv2D(channels, (kh, kw), (sh, sw), (ph, pw), activation=activation, groups=groups))
            elif kind == lib.LAYER_KIND_MAXPOOL2D:
                layers.append(MaxPool1D(kw, sw) if one_d else MaxPool2D((kh, kw), (sh, sw)))
            elif kind == lib.LAYER_KIND_FLATTEN:
                layers.append(Flatten())
            elif kind == lib.LAYER_KIND_UPSAMPLE2D:
                layers.append(UpSample2D(settings[0]))
            elif kind == lib.LAYER_KIND_RESHAPE:
                layers.append(Reshape((channels, height, width)))
            elif kind == lib.LAYER_KIND_GLOBAL_AVGPOOL2D:
                layers.append(GlobalAveragePooling2D())
            elif kind == lib.LAYER_KIND_BATCHNORM:
                layers.append(BatchNormalization(activation=activation))
            elif kind == lib.LAYER_KIND_EMBEDDING:
                # settings: the vocabulary and the padding id (-1 for none)
                padding = settings[2] if settings[2] >= 0 else None
                layers.append(Embedding(settings[0], channels, padding_idx=padding,
                                        mask_padding=bool(lib.ann_get_layer_masked(ptr, i)),
                                        trainable=bool(lib.ann_get_layer_trainable(ptr, i))))
            elif kind == lib.LAYER_KIND_GLOBAL_MAXPOOL2D:
                layers.append(GlobalMaxPooling2D())
            else:
                lib.ann_free_network(ptr)
                raise NetworkError(f"{filepath} contains layer types pyann does not support yet")
            in_dims = (channels, height, width)
        
        # Create model wrapper
        model = cls.__new__(cls)
        model._layers = layers
        model._optimizer = None
        model._loss = Loss.MSE
        model._name = None
        model._ptr = ptr
        model._built = True
        model._input_shape = input_shape
        # an input one row high that a 1-D layer reads first is a sequence (channels, steps)
        if len(input_shape) == 3 and input_shape[1] == 1:
            model._input_shape = (input_shape[0], input_shape[2])
            if not model._is_sequence():
                model._input_shape = input_shape
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
        clear_c_error()
        ptr = lib.ann_import_onnx(str(filepath).encode())
        
        if ptr == ffi.NULL:
            raise NetworkError(c_error_message(f"Failed to load ONNX model from {filepath}"))
        
        return cls._from_ptr(ptr, filepath)
    
    def quantize(self, x_calibration: ArrayLike) -> "QuantizedModel":
        """Make an int8 version of the trained model.

        Runs the model over the calibration samples to find each layer's
        value range; a few hundred to a few thousand representative training
        samples are usually enough. The model itself is not changed.

        A text model (an Embedding first) keeps its token ids as ids: the
        vectors become an int8 table, and padding masks work as in the float
        model. Its QuantizedModel and C headers take the ids directly.

        Args:
            x_calibration: Input samples, preprocessed like real inputs

        Returns:
            A QuantizedModel for checking int8 accuracy and exporting C headers
        """
        from pyann.core.quantized import QuantizedModel

        if not self._built:
            raise NetworkError("Model must be trained before quantizing")

        input_size = lib.ann_get_layer_nodes(self._ptr, 0)
        x_arr = self._as_rows(x_calibration, "x_calibration")
        shape = get_shape(x_arr)
        if len(shape) != 2 or shape[0] == 0 or shape[1] != input_size:
            raise ValueError(f"x_calibration must be samples x {input_size} features, got shape {shape}")

        x_tensor = Tensor(x_arr)    # keep it alive through the call
        clear_c_error()
        ptr = lib.ann_quantize(self._ptr, x_tensor._ptr)
        if ptr == ffi.NULL:
            raise NetworkError(c_error_message(
                "Quantization failed (softmax is supported only on the output layer, each "
                "BatchNormalization must follow a Dense or Conv2D layer without an activation, and "
                "an Embedding must be the first layer)"))
        output_size = lib.ann_get_layer_nodes(self._ptr, lib.ann_get_layer_count(self._ptr) - 1)
        takes_ids = bool(self._layers) and isinstance(self._layers[0], Embedding)
        return QuantizedModel(ptr, self._input_shape, input_size, output_size, self._output_shape(),
                              takes_ids=takes_ids)

    def export_c_header(self, filepath: Union[str, Path], prefix: Optional[str] = None,
                        runtime: bool = True) -> Path:
        """Export the model's float32 weights as a self-contained C header.

        For the int8 version, use ``model.quantize(x).export_c_header(...)``.
        The header defines ``<prefix>_predict(input, output, workspace)`` and
        ``<prefix>_argmax``, plus size macros. It needs ``ann_rt.h``, which is
        copied next to it unless ``runtime`` is False.

        Args:
            filepath: Output file, e.g. ``"model.h"``
            prefix: C identifier naming the header's symbols (default: from
                the file name)
            runtime: Also write ``ann_rt.h`` into the same directory

        Returns:
            The header's path
        """
        from pyann.core.quantized import export_c_header

        if not self._built:
            raise NetworkError("Cannot export untrained model")
        return export_c_header(self._ptr, ffi.NULL, filepath, prefix, runtime)

    def export_pikchr(self, filepath: Union[str, Path]) -> None:
        """Export network architecture as a PIKCHR diagram.
        
        Generates a PIKCHR text file that can be rendered to SVG using
        the pikchr tool: ``pikchr network.pikchr > network.svg``
        
        For small networks (≤10 nodes per layer), outputs a detailed diagram
        with individual node circles and connections. For larger networks,
        outputs a simplified box diagram with layer info.
        
        Args:
            filepath: Output file path (typically .pikchr extension)
        """
        if not self._built:
            raise NetworkError("Cannot export untrained model")
        
        result = lib.ann_export_pikchr(self._ptr, str(filepath).encode())
        raise_for_error_code(result, f"Failed to export PIKCHR to {filepath}")
    
    def export_learning_curve(self, filepath: Union[str, Path]) -> None:
        """Export training history as CSV for learning curve visualization.
        
        Writes epoch, loss, and learning rate data collected during training,
        plus a val_loss column when fit() was given validation_data.
        
        Args:
            filepath: Output CSV file path
        """
        if not self._built:
            raise NetworkError("Cannot export learning curve from untrained model")
        
        result = lib.ann_export_learning_curve(self._ptr, str(filepath).encode())
        raise_for_error_code(result, f"Failed to export learning curve to {filepath}")
    
    def training_history(self) -> dict:
        """The loss and learning rate of every epoch trained by fit().
        
        Epochs accumulate across fit() calls until :meth:`clear_history`
        (libann records at most 10,000). train_on_batch() steps are not
        recorded.
        
        Returns:
            Dictionary of equal-length lists, one entry per epoch: ``loss``
            (training loss), ``lr`` (learning rate) and ``val_loss``
            (validation loss, or None for epochs trained without
            validation_data)
        """
        count = lib.ann_get_history_count(self._ptr) if self._built else 0
        loss, lr, val_loss = (ffi.new("real[]", count) for _ in range(3))
        if count:
            count = lib.ann_get_history(self._ptr, loss, lr, val_loss, count)
        return {
            "loss": [float(loss[i]) for i in range(count)],
            "lr": [float(lr[i]) for i in range(count)],
            "val_loss": [float(val_loss[i]) if val_loss[i] >= 0.0 else None for i in range(count)],
        }
    
    def clear_history(self) -> None:
        """Clear training history to free memory or before retraining."""
        if self._built and self._ptr is not None:
            lib.ann_clear_history(self._ptr)
    
    def set_input_noise(self, noise: Optional[str], amount: float = 0.0,
                        clip: Optional[Tuple[float, float]] = None) -> "Sequential":
        """Corrupt the inputs of every training batch, e.g. for a denoising autoencoder.
        
        The network trains on noisy inputs and learns to output the clean
        targets: fit(x, x) with noise teaches it to remove that noise. Noise
        is drawn fresh for every batch (in fit() and train_on_batch());
        predict() and evaluate() use the inputs as given. It is not saved
        with the model.
        
        Args:
            noise: 'gaussian' (adds normal noise with standard deviation
                ``amount``), 'masking' (sets inputs to 0 with probability
                ``amount``), 'salt_pepper' (sets inputs to the clip range's low
                or high value with probability ``amount``), or None to turn
                noise off
            amount: Standard deviation, or probability in [0, 1]
            clip: (low, high) range to clamp noisy values to, e.g. (0, 1) for
                pixels; required for 'salt_pepper'
        
        Returns:
            self for method chaining
        """
        if noise is not None and noise not in _NOISE_TYPES:
            raise ValueError(f"noise must be one of {sorted(_NOISE_TYPES)} or None, got {noise!r}")
        if not amount >= 0.0 or (noise in ("masking", "salt_pepper") and amount > 1.0):
            raise ValueError(f"amount must be >= 0 (and at most 1 for {noise}), got {amount}")
        low, high = (0.0, 0.0) if clip is None else (float(clip[0]), float(clip[1]))
        if not low <= high or (noise == "salt_pepper" and not low < high):
            raise ValueError(f"clip must be (low, high) with low < high"
                             f"{' (required for salt_pepper)' if noise == 'salt_pepper' else ''}, got {clip}")
        self._noise = None if noise is None else (noise, float(amount), low, high)
        if self._built:
            if self._noise is None:
                lib.ann_set_input_noise(self._ptr, lib.NOISE_NONE, 0.0, 0.0, 0.0)
            else:
                self._apply_training_settings()
        return self

    @property
    def input_noise(self) -> Optional[Tuple[str, float, Optional[Tuple[float, float]]]]:
        """The input noise as (noise, amount, clip), or None when it is off."""
        if self._noise is None:
            return None
        noise, amount, low, high = self._noise
        return (noise, amount, (low, high) if low < high else None)

    def apply_input_noise(self, x: ArrayLike) -> ArrayLike:
        """A copy of x with the input noise applied, as training applies it.
        
        Use it to make noisy test inputs for evaluating a denoising
        autoencoder. Returns an unchanged copy when noise is off.
        
        Args:
            x: Inputs of any shape
        """
        self._build()
        if has_numpy():
            import numpy as np
            noisy = np.array(to_array(x), dtype=np.float32, order="C", copy=True)
            if noisy.size:
                result = lib.ann_apply_input_noise(self._ptr, ffi.cast("real *", noisy.ctypes.data), noisy.size)
                raise_for_error_code(result, "Failed to apply input noise")
            return noisy

        def flatten(v):
            return [w for item in v for w in flatten(item)] if isinstance(v, (list, tuple)) else [float(v)]

        def rebuild(v, it):
            return [rebuild(item, it) for item in v] if isinstance(v, (list, tuple)) else next(it)

        data = to_array(x)
        values = flatten(data)
        buffer = ffi.new("real[]", values)
        if values:
            raise_for_error_code(lib.ann_apply_input_noise(self._ptr, buffer, len(values)),
                                 "Failed to apply input noise")
        return rebuild(data, iter(buffer))

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
        x_tensor = Tensor(self._as_rows(x))
        y_tensor = Tensor(self._targets_as_rows(y))
        
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
        shapes = self._layer_shapes() if self._input_shape and self._layers else []
        in_channels = (self._image_input_shape() or (0,))[0]
        in_height = (self._image_input_shape() or (0, 0))[1]
        for i, (layer, internal_shape, params) in enumerate(shapes):
            shape = self._public_shape(internal_shape)
            total_params += params
            if isinstance(layer, Dense):
                text = f"Dense({layer.units}, {_activation_name(layer)})"
            elif isinstance(layer, Conv1D):
                groups = f", {layer.groups} groups" if layer.groups > 1 else ""
                text = (f"Conv1D({layer.filters}, {layer.kernel_size}, stride {layer.strides}, "
                        f"padding {layer.padding}{groups}, {_activation_name(layer)}) -> {shape}")
            elif isinstance(layer, _ConvBase):
                kh, kw = layer.window()[:2]
                groups = layer.conv_groups(in_channels)
                grouped = f", {groups} groups" if groups > 1 and isinstance(layer, Conv2D) else ""
                head = (f"DepthwiseConv2D(x{layer.depth_multiplier}" if isinstance(layer, DepthwiseConv2D)
                        else f"Conv2D({layer.filters}")
                text = (f"{head}, {kh}x{kw}, stride {layer.strides}, padding {layer.padding}{grouped}, "
                        f"{_activation_name(layer)}) -> {shape}")
            elif isinstance(layer, MaxPool1D):
                text = f"MaxPool1D({layer.pool_size}, stride {layer.strides}) -> {shape}"
            elif isinstance(layer, MaxPool2D):
                kh, kw = layer.window()[:2]
                text = f"MaxPool2D({kh}x{kw}, stride {layer.strides}) -> {shape}"
            elif isinstance(layer, UpSample2D):
                text = f"UpSample2D({layer.size}x) -> {shape}"
            elif isinstance(layer, Reshape):
                text = f"Reshape -> {shape}"
            elif isinstance(layer, (GlobalAveragePooling2D, GlobalMaxPooling2D)):
                # named for what it pools: a sequence's steps (1D) or an image (2D)
                kind = "Average" if isinstance(layer, GlobalAveragePooling2D) else "Max"
                dims = "1D" if self._is_sequence() and in_height == 1 else "2D"
                text = f"Global{kind}Pooling{dims} -> {shape}"
            elif isinstance(layer, Embedding):
                padding = "" if layer.padding_idx is None else f", padding {layer.padding_idx}"
                masked = ", masked" if layer.mask_padding else ""
                text = f"Embedding({layer.input_dim}, {layer.output_dim}{padding}{masked}) -> {shape}"
            elif isinstance(layer, BatchNormalization):
                text = f"BatchNormalization({_activation_name(layer)}) -> {shape}"
            else:
                text = f"Flatten -> {shape}"
            frozen = (not lib.ann_get_layer_trainable(self._ptr, i + 1) if self._built
                      else not getattr(layer, "trainable", True))
            if params and frozen:
                text += " (frozen)"
            lines.append(f"{text} - {params:,} params" if params else text)
            in_channels = internal_shape[0]
            in_height = internal_shape[1] if len(internal_shape) == 3 else 0
        
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
                if self._gpu_uploaded:
                    lib.ann_gpu_free_network(self._ptr)
                lib.ann_free_network(self._ptr)
