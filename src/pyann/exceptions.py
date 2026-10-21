"""Custom exceptions for PyANN."""

import threading
from typing import Optional


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


# The last error libann reported on this thread, through the error log
# callback that pyann._bindings.ffi installs: (code, message, function name)
_c_error = threading.local()


def record_c_error(code: int, message: str, function: str) -> None:
    """Remember an error libann reported, for the exception raised next."""
    _c_error.last = (code, message, function)


def clear_c_error() -> None:
    """Forget any earlier libann error, before a call whose failure is checked."""
    _c_error.last = None


def c_error_message(message: str, code: Optional[int] = None) -> str:
    """message plus the last libann error, e.g. "... (ann_load_network: File I/O error)".

    The error is consumed. When code is given, it is added only if libann
    reported that same code, so a stale error from an earlier call is not.
    """
    last = getattr(_c_error, "last", None)
    _c_error.last = None
    if last is None or (code is not None and last[0] != code):
        return message
    return f"{message} ({last[2]}: {last[1]})"


def raise_for_error_code(code: int, message: str = "") -> None:
    """Raise an appropriate exception for a C error code.
    
    Args:
        code: Error code from C library (0 = success)
        message: Optional error message
    """
    if code == 0:
        clear_c_error()
        return
    
    exc_class = ERROR_CODE_MAP.get(code, PyANNError)
    raise exc_class(c_error_message(message or f"Operation failed with error code {code}", code))
