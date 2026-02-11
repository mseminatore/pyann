"""NumPy and Python list compatibility layer."""

from typing import Any, List, Sequence, Tuple, Union, TYPE_CHECKING

# Check for NumPy availability
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

if TYPE_CHECKING:
    import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[List[float], List[List[float]], Sequence[float], "np.ndarray"]


def has_numpy() -> bool:
    """Check if NumPy is available."""
    return HAS_NUMPY


def is_numpy_array(obj: Any) -> bool:
    """Check if object is a NumPy array."""
    if not HAS_NUMPY:
        return False
    return isinstance(obj, np.ndarray)


def to_float32_contiguous(arr: "np.ndarray") -> "np.ndarray":
    """Ensure array is float32 and C-contiguous."""
    if not HAS_NUMPY:
        raise RuntimeError("NumPy is required for this operation")
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    if not arr.flags['C_CONTIGUOUS']:
        arr = np.ascontiguousarray(arr)
    return arr


def to_array(data: ArrayLike, copy: bool = False) -> Union["np.ndarray", List[List[float]]]:
    """Convert array-like data to internal representation.
    
    If NumPy is available, returns a float32 C-contiguous array.
    Otherwise, returns a nested list of floats.
    
    Args:
        data: Input data (list, nested list, or numpy array)
        copy: If True, always copy the data
        
    Returns:
        NumPy array if available, otherwise nested list
    """
    if HAS_NUMPY:
        if isinstance(data, np.ndarray):
            arr = data.copy() if copy else data
        else:
            arr = np.array(data, dtype=np.float32)
        return to_float32_contiguous(arr)
    else:
        # Pure Python path
        if isinstance(data, (list, tuple)):
            if len(data) == 0:
                return [[]]
            if isinstance(data[0], (list, tuple)):
                return [[float(x) for x in row] for row in data]
            else:
                return [[float(x) for x in data]]
        raise TypeError(f"Cannot convert {type(data)} to array without NumPy")


def from_array(data: Union["np.ndarray", List[List[float]]], shape: Tuple[int, ...]) -> ArrayLike:
    """Convert internal representation back to user-friendly format.
    
    Returns NumPy array if available, otherwise nested list.
    """
    if HAS_NUMPY:
        if isinstance(data, np.ndarray):
            return data.reshape(shape)
        return np.array(data, dtype=np.float32).reshape(shape)
    else:
        return data


def get_shape(data: ArrayLike) -> Tuple[int, ...]:
    """Get shape of array-like data."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        return data.shape
    
    # Pure Python path
    if isinstance(data, (list, tuple)):
        if len(data) == 0:
            return (0,)
        if isinstance(data[0], (list, tuple)):
            return (len(data), len(data[0]))
        return (len(data),)
    
    raise TypeError(f"Cannot get shape of {type(data)}")


def flatten(data: ArrayLike) -> List[float]:
    """Flatten array-like data to 1D list."""
    if HAS_NUMPY and isinstance(data, np.ndarray):
        return data.flatten().tolist()
    
    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            if isinstance(item, (list, tuple)):
                result.extend(item)
            else:
                result.append(float(item))
        return result
    
    raise TypeError(f"Cannot flatten {type(data)}")
