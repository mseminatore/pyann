"""Tensor wrapper class with NumPy interop."""

from typing import Optional, Tuple, Union, List, TYPE_CHECKING
from pyann._bindings.ffi import ffi, lib
from pyann.utils.compat import ArrayLike, has_numpy, is_numpy_array, to_float32_contiguous
from pyann.exceptions import AllocationError, InvalidParameterError

if TYPE_CHECKING:
    import numpy as np


class Tensor:
    """Python wrapper for libann Tensor.
    
    Provides NumPy-like interface to the C tensor library with automatic
    memory management and NumPy interoperability.
    
    Attributes:
        rows: Number of rows
        cols: Number of columns
        shape: Tuple of (rows, cols)
    """
    
    __slots__ = ('_ptr', '_owns_memory')
    
    def __init__(
        self,
        data: Optional[ArrayLike] = None,
        *,
        rows: Optional[int] = None,
        cols: Optional[int] = None,
        _ptr: Optional[object] = None,
        _owns_memory: bool = True
    ):
        """Create a new Tensor.
        
        Args:
            data: Optional initial data (list or numpy array)
            rows: Number of rows (required if data is None)
            cols: Number of columns (required if data is None)
            _ptr: Internal: existing C pointer to wrap
            _owns_memory: Internal: whether to free memory on deletion
        """
        self._owns_memory = _owns_memory
        
        if _ptr is not None:
            self._ptr = _ptr
            return
        
        if data is not None:
            self._ptr = self._from_data(data)
        elif rows is not None and cols is not None:
            self._ptr = lib.tensor_create(rows, cols)
            if self._ptr == ffi.NULL:
                raise AllocationError("Failed to allocate tensor")
        else:
            raise InvalidParameterError("Must provide data or (rows, cols)")
    
    def _from_data(self, data: ArrayLike) -> object:
        """Create tensor from array-like data."""
        if has_numpy() and is_numpy_array(data):
            import numpy as np
            arr = to_float32_contiguous(data)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            rows, cols = arr.shape
            ptr = lib.tensor_create_from_array(
                rows, cols, 
                ffi.cast("real *", arr.ctypes.data)
            )
        else:
            # Pure Python path
            if isinstance(data, (list, tuple)):
                if len(data) == 0:
                    ptr = lib.tensor_create(0, 0)
                elif isinstance(data[0], (list, tuple)):
                    rows = len(data)
                    cols = len(data[0])
                    flat = []
                    for row in data:
                        flat.extend(float(x) for x in row)
                    c_array = ffi.new("real[]", flat)
                    ptr = lib.tensor_create_from_array(rows, cols, c_array)
                else:
                    rows = 1
                    cols = len(data)
                    c_array = ffi.new("real[]", [float(x) for x in data])
                    ptr = lib.tensor_create_from_array(rows, cols, c_array)
            else:
                raise TypeError(f"Cannot create tensor from {type(data)}")
        
        if ptr == ffi.NULL:
            raise AllocationError("Failed to create tensor from data")
        return ptr
    
    def __del__(self):
        """Free the tensor memory."""
        if hasattr(self, '_ptr') and self._ptr is not None and self._owns_memory:
            if self._ptr != ffi.NULL:
                lib.tensor_free(self._ptr)
    
    @property
    def rows(self) -> int:
        """Number of rows."""
        return self._ptr.rows
    
    @property
    def cols(self) -> int:
        """Number of columns."""
        return self._ptr.cols
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape as (rows, cols)."""
        return (self._ptr.rows, self._ptr.cols)
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._ptr.rows * self._ptr.cols
    
    def __len__(self) -> int:
        """Return number of rows."""
        return self._ptr.rows
    
    def __getitem__(self, key: Union[int, Tuple[int, int]]) -> float:
        """Get element at index."""
        if isinstance(key, tuple):
            row, col = key
            return lib.tensor_get_element(self._ptr, row, col)
        else:
            # Single index - return entire row as list
            row = key
            return [lib.tensor_get_element(self._ptr, row, c) for c in range(self.cols)]
    
    def __setitem__(self, key: Union[int, Tuple[int, int]], value: float):
        """Set element at index."""
        if isinstance(key, tuple):
            row, col = key
            lib.tensor_set_element(self._ptr, row, col, value)
        else:
            raise TypeError("Use tensor[row, col] = value for setting elements")
    
    def __array__(self, dtype=None) -> "np.ndarray":
        """NumPy array protocol - enables np.asarray(tensor)."""
        if not has_numpy():
            raise RuntimeError("NumPy is required for __array__ protocol")
        import numpy as np
        
        # Copy data from C tensor to numpy array
        arr = np.zeros((self.rows, self.cols), dtype=np.float32)
        for i in range(self.rows):
            for j in range(self.cols):
                arr[i, j] = lib.tensor_get_element(self._ptr, i, j)
        
        if dtype is not None and dtype != np.float32:
            arr = arr.astype(dtype)
        return arr
    
    def numpy(self) -> "np.ndarray":
        """Convert to NumPy array."""
        return self.__array__()
    
    def tolist(self) -> List[List[float]]:
        """Convert to nested Python list."""
        result = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                row.append(lib.tensor_get_element(self._ptr, i, j))
            result.append(row)
        return result
    
    def copy(self) -> "Tensor":
        """Create a deep copy of this tensor."""
        ptr = lib.tensor_copy(self._ptr)
        if ptr == ffi.NULL:
            raise AllocationError("Failed to copy tensor")
        return Tensor(_ptr=ptr)
    
    def fill(self, value: float) -> "Tensor":
        """Fill tensor with a constant value (in-place)."""
        lib.tensor_fill(self._ptr, value)
        return self
    
    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Tensor":
        """Create a tensor filled with zeros."""
        ptr = lib.tensor_zeros(rows, cols)
        if ptr == ffi.NULL:
            raise AllocationError("Failed to create zero tensor")
        return cls(_ptr=ptr)
    
    @classmethod
    def ones(cls, rows: int, cols: int) -> "Tensor":
        """Create a tensor filled with ones."""
        ptr = lib.tensor_ones(rows, cols)
        if ptr == ffi.NULL:
            raise AllocationError("Failed to create ones tensor")
        return cls(_ptr=ptr)
    
    @classmethod
    def random_uniform(cls, rows: int, cols: int, low: float = 0.0, high: float = 1.0) -> "Tensor":
        """Create a tensor with random uniform values."""
        ptr = lib.tensor_create_random_uniform(rows, cols, low, high)
        if ptr == ffi.NULL:
            raise AllocationError("Failed to create random tensor")
        return cls(_ptr=ptr)
    
    def __add__(self, other: Union["Tensor", float]) -> "Tensor":
        """Add tensor or scalar."""
        result = self.copy()
        if isinstance(other, Tensor):
            lib.tensor_add(result._ptr, other._ptr)
        else:
            lib.tensor_add_scalar(result._ptr, float(other))
        return result
    
    def __mul__(self, other: Union["Tensor", float]) -> "Tensor":
        """Multiply by tensor or scalar."""
        result = self.copy()
        if isinstance(other, Tensor):
            lib.tensor_mul(result._ptr, other._ptr)
        else:
            lib.tensor_mul_scalar(result._ptr, float(other))
        return result
    
    def __sub__(self, other: "Tensor") -> "Tensor":
        """Subtract tensor."""
        result = self.copy()
        lib.tensor_sub(result._ptr, other._ptr)
        return result
    
    def sum(self) -> float:
        """Sum all elements."""
        return lib.tensor_sum(self._ptr)
    
    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape})"
    
    def __str__(self) -> str:
        if self.rows <= 10 and self.cols <= 10:
            return f"Tensor({self.tolist()})"
        return f"Tensor(shape={self.shape})"
