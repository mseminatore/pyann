"""Data loading and manipulation utilities."""

from typing import Tuple, Optional, List, Union, TYPE_CHECKING
from pyann.utils.compat import ArrayLike, to_array, has_numpy, HAS_NUMPY

if TYPE_CHECKING:
    import numpy as np


def load_csv(
    filename: str,
    has_header: bool = False,
    delimiter: str = ","
) -> Tuple[ArrayLike, int, int]:
    """Load data from a CSV file.
    
    This is a pure Python implementation. For large files, consider using
    the C library's ann_load_csv through the bindings.
    
    Args:
        filename: Path to CSV file
        has_header: If True, skip the first line
        delimiter: Column separator (default: comma)
        
    Returns:
        Tuple of (data, rows, cols) where data is array-like
    """
    rows = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    start = 1 if has_header else 0
    for line in lines[start:]:
        line = line.strip()
        if not line:
            continue
        values = [float(x) for x in line.split(delimiter)]
        rows.append(values)
    
    if not rows:
        return to_array([[]]), 0, 0
    
    n_rows = len(rows)
    n_cols = len(rows[0])
    
    return to_array(rows), n_rows, n_cols


def split_data(
    X: ArrayLike,
    y: ArrayLike,
    train_ratio: float = 0.8,
    shuffle: bool = True,
    seed: Optional[int] = None
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Split data into training and validation sets.
    
    Args:
        X: Input features
        y: Target values
        train_ratio: Fraction of data to use for training (default: 0.8)
        shuffle: Whether to shuffle before splitting (default: True)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, y_train, X_val, y_val)
    """
    if HAS_NUMPY:
        import numpy as np
        
        X = to_array(X)
        y = to_array(y)
        
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
    else:
        # Pure Python path
        import random
        
        X_list = list(X) if not isinstance(X, list) else X
        y_list = list(y) if not isinstance(y, list) else y
        
        n_samples = len(X_list)
        indices = list(range(n_samples))
        
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)
        
        split_idx = int(n_samples * train_ratio)
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
        
        X_train = [X_list[i] for i in train_idx]
        y_train = [y_list[i] for i in train_idx]
        X_val = [X_list[i] for i in val_idx]
        y_val = [y_list[i] for i in val_idx]
        
        return to_array(X_train), to_array(y_train), to_array(X_val), to_array(y_val)


def normalize(
    data: ArrayLike,
    axis: Optional[int] = None,
    mean: Optional[ArrayLike] = None,
    std: Optional[ArrayLike] = None
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Normalize data to zero mean and unit variance.
    
    Args:
        data: Input data to normalize
        axis: Axis along which to compute statistics (default: flatten)
        mean: Pre-computed mean (for applying to test data)
        std: Pre-computed std (for applying to test data)
        
    Returns:
        Tuple of (normalized_data, mean, std)
    """
    if not HAS_NUMPY:
        raise NotImplementedError("normalize() requires NumPy")
    
    import numpy as np
    data = to_array(data)
    
    if mean is None:
        mean = np.mean(data, axis=axis, keepdims=True)
    if std is None:
        std = np.std(data, axis=axis, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
    
    normalized = (data - mean) / std
    return normalized, mean, std


def one_hot_encode(
    labels: ArrayLike,
    num_classes: Optional[int] = None
) -> ArrayLike:
    """Convert integer labels to one-hot encoded vectors.
    
    Args:
        labels: 1D array of integer class labels
        num_classes: Number of classes (inferred from max label if not provided)
        
    Returns:
        2D array of shape (n_samples, num_classes)
    """
    if not HAS_NUMPY:
        raise NotImplementedError("one_hot_encode() requires NumPy")
    
    import numpy as np
    labels = to_array(labels).flatten().astype(int)
    
    if num_classes is None:
        num_classes = int(labels.max()) + 1
    
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0
    
    return one_hot
