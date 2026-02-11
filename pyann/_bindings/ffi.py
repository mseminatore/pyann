"""CFFI FFI setup and library loading."""

import os
import sys
from pathlib import Path
from typing import Optional

from cffi import FFI

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


def _load_library() -> object:
    """Load the libann shared library.
    
    Returns:
        CFFI library handle
    """
    global _lib
    if _lib is not None:
        return _lib
    
    lib_path = _find_library()
    _lib = ffi.dlopen(lib_path)
    return _lib


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
