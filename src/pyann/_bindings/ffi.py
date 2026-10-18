"""CFFI FFI setup and library loading."""

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Optional

from cffi import FFI, FFIError

from pyann._bindings.tensor_cdef import TENSOR_CDEF
from pyann._bindings.ann_cdef import ANN_CDEF
from pyann.exceptions import LibraryNotFoundError

# Create FFI instance
ffi = FFI()

# Define all C types and functions
ffi.cdef(TENSOR_CDEF)
ffi.cdef(ANN_CDEF)

# Library handle (lazily loaded)
_lib: Optional[object] = None

# Detected backend info (populated on library load)
_backend_info: Optional[Dict[str, Optional[str]]] = None


def _find_library() -> str:
    """Find the libann shared library.
    
    Search order:
    1. PYANN_LIB_PATH environment variable
    2. Next to this Python package
    3. Common system locations
    
    Returns:
        Path to the shared library
        
    Raises:
        LibraryNotFoundError: If library cannot be found
    """
    # Platform-specific library names
    if sys.platform == "win32":
        lib_names = ["ann.dll", "libann.dll"]
    elif sys.platform == "darwin":
        lib_names = ["libann.dylib", "libann.so"]
    else:  # Linux and others
        lib_names = ["libann.so"]
    
    # Search paths
    search_paths = []
    
    # 1. Environment variable
    env_path = os.environ.get("PYANN_LIB_PATH")
    if env_path:
        search_paths.append(Path(env_path))
    
    # 2. Next to this package
    package_dir = Path(__file__).parent.parent
    search_paths.append(package_dir)
    search_paths.append(package_dir / "lib")
    search_paths.append(package_dir / "_lib")
    
    # 3. Next to package in site-packages (for installed packages)
    search_paths.append(package_dir.parent)
    
    # 4. Common build locations (for development)
    cwd = Path.cwd()
    search_paths.extend([
        cwd / "build",
        cwd / "build" / "Release",
        cwd / "build" / "Debug",
        cwd / "lib",
        cwd,
        # src layout: library at project root level
        package_dir.parent.parent / "build" / "Release",
        package_dir.parent.parent / "build",
        package_dir.parent.parent / "lib",
    ])
    
    # 5. System paths
    if sys.platform != "win32":
        search_paths.extend([
            Path("/usr/local/lib"),
            Path("/usr/lib"),
            Path("/opt/local/lib"),
        ])
    
    # Search for library
    for search_path in search_paths:
        if not search_path.exists():
            continue
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                return str(lib_path)
    
    # Library not found
    searched = "\n  ".join(str(p) for p in search_paths if p.exists())
    raise LibraryNotFoundError(
        f"Could not find libann shared library.\n"
        f"Searched for: {', '.join(lib_names)}\n"
        f"In directories:\n  {searched}\n\n"
        f"Set PYANN_LIB_PATH environment variable to the library location,\n"
        f"or build the library and place it in one of the search paths."
    )


def _detect_backends(loaded_lib: object, lib_path: str) -> Dict[str, Optional[str]]:
    """Detect which acceleration backends are compiled into the library."""
    info: Dict[str, Optional[str]] = {
        "blas": None,
        "gpu": None,
        "gpu_active": None,
        "cblas_config": None,
        "cblas_corename": None,
        "cblas_isa": None,
    }

    # Build-time BLAS name (absent in libraries built before pyann_info.c)
    try:
        name = loaded_lib.pyann_blas_backend()
        if name != ffi.NULL:
            info["blas"] = ffi.string(name).decode()
    except (FFIError, AttributeError):
        pass

    # Probe for the bundled CBLAS, which also reports its configuration
    try:
        loaded_lib.cblas_init(-1)
        info["blas"] = "cblas"
        try:
            info["cblas_config"] = ffi.string(loaded_lib.cblas_get_config()).decode()
        except (FFIError, AttributeError):
            pass
        try:
            info["cblas_corename"] = ffi.string(loaded_lib.cblas_get_corename()).decode()
        except (FFIError, AttributeError):
            pass
        try:
            info["cblas_isa"] = ffi.string(loaded_lib.cblas_get_isa_features()).decode()
        except (FFIError, AttributeError):
            pass
    except (FFIError, AttributeError):
        pass

    # Probe for GPU backends compiled into the library. Each backend exports
    # a vtable symbol; it is present whether or not a device is available.
    # cffi can't look up data symbols without a declaration, so use ctypes.
    probe = ctypes.CDLL(lib_path)
    for name in ("metal", "cuda"):
        if hasattr(probe, f"{name}_backend"):
            info["gpu"] = name
            break

    return info


def _load_library() -> object:
    """Load the libann shared library.

    Returns:
        CFFI library handle
    """
    global _lib, _backend_info
    if _lib is not None:
        return _lib

    lib_path = _find_library()
    _lib = ffi.dlopen(lib_path)
    _backend_info = _detect_backends(_lib, lib_path)

    return _lib


def get_backend_info() -> Dict[str, Optional[str]]:
    """Return detected acceleration backend information.

    Returns:
        Dict with keys: blas, gpu (backend compiled in), gpu_active
        (backend in use after enable_gpu(), else None), cblas_config, cblas_corename, cblas_isa.
        Values are None when the corresponding backend is not compiled in.
    """
    _load_library()
    return dict(_backend_info)


def enable_gpu() -> bool:
    """Initialize the GPU backend compiled into the library, if any.

    Safe to call repeatedly. When no GPU backend is compiled in or no device
    is available, returns False and libann keeps using its CPU path (CBLAS
    if the library was built with it).

    Returns:
        True if a GPU backend is now active.
    """
    _load_library()
    if _backend_info["gpu_active"] is None and _backend_info["gpu"] is not None:
        if _lib.ann_gpu_init():
            _backend_info["gpu_active"] = _backend_info["gpu"]
    return _backend_info["gpu_active"] is not None


def gpu_enabled() -> bool:
    """Return True if enable_gpu() has activated a GPU backend."""
    _load_library()
    return _backend_info["gpu_active"] is not None


@property
def lib():
    """Get the loaded library handle (lazy loading)."""
    return _load_library()


# For direct access, load on first attribute access
class _LibProxy:
    """Proxy object for lazy library loading."""
    
    _lib = None
    
    def __getattr__(self, name):
        if self._lib is None:
            self._lib = _load_library()
        return getattr(self._lib, name)


# Export the proxy
lib = _LibProxy()
