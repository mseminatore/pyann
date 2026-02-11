"""Custom exceptions for PyANN."""


class PyANNError(Exception):
    """Base exception for PyANN errors."""
    pass


class LibraryNotFoundError(PyANNError):
    """Raised when the libann shared library cannot be loaded."""
    pass


class AllocationError(PyANNError):
    """Raised when memory allocation fails in the C library."""
    pass


class InvalidParameterError(PyANNError):
    """Raised when an invalid parameter is passed to a function."""
    pass


class IOError(PyANNError):
    """Raised when a file I/O operation fails."""
    pass


class NetworkError(PyANNError):
    """Raised when a network operation fails."""
    pass


# Map C error codes to exceptions
ERROR_CODE_MAP = {
    -1: PyANNError,           # ERR_FAIL
    -2: InvalidParameterError, # ERR_NULL_PTR
    -3: AllocationError,       # ERR_ALLOC
    -4: InvalidParameterError, # ERR_INVALID
    -5: IOError,               # ERR_IO
}


def raise_for_error_code(code: int, message: str = "") -> None:
    """Raise an appropriate exception for a C error code.
    
    Args:
        code: Error code from C library (0 = success)
        message: Optional error message
    """
    if code == 0:
        return
    
    exc_class = ERROR_CODE_MAP.get(code, PyANNError)
    raise exc_class(message or f"Operation failed with error code {code}")
