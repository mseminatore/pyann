"""CFFI FFI setup and library loading."""

import ctypes
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

from cffi import FFI, FFIError

from pyann._bindings.tensor_cdef import TENSOR_CDEF
from pyann._bindings.ann_cdef import ANN_CDEF
from pyann.exceptions import LibraryNotFoundError, record_c_error

# Create FFI instance
ffi = FFI()

# Define all C types and functions
ffi.cdef(TENSOR_CDEF)
ffi.cdef(ANN_CDEF)

# Library handle (lazily loaded)
_lib: Optional[object] = None

# Detected backend info (populated on library load)
_backend_info: Optional[Dict[str, Optional[str]]] = None

_logger = logging.getLogger("pyann")


def _on_c_error(code, message, function):
    """libann's error log callback: keep the error for the exception, and log it.

    Without a callback libann prints errors to stderr, which Python code
    can't catch or silence; the exception that follows carries them instead.
    """
    message = ffi.string(message).decode(errors="replace") if message != ffi.NULL else ""
    function = ffi.string(function).decode(errors="replace") if function != ffi.NULL else "libann"
    record_c_error(code, message, function)
    _logger.debug("libann error %d in %s: %s", code, function, message)


# Kept at module level: libann holds only a pointer to it
try:
    _error_callback = ffi.callback("ErrorLogCallback", _on_c_error)
except MemoryError:
    # No executable memory for callbacks (some hardened runtimes): libann
    # keeps its default of printing errors to stderr
    _error_callback = None


def _package_dirs() -> List[Path]:
    """The pyann package's directories, in the order Python searches them.
    
    A wheel install has one. An editable install (pip install -e .) lists the
    build it installed (with lib/) before the source tree, so the library it
    just built wins over an older one left in src/pyann/lib.
    """
    package = sys.modules.get("pyann")
    dirs = [Path(p) for p in getattr(package, "__path__", [])]
    here = Path(__file__).resolve().parent.parent
    if not any(d.resolve() == here for d in dirs):
        dirs.append(here)
    return dirs


def _find_library(package_dirs: Optional[List[Path]] = None) -> str:
    """Find the libann shared library.
    
    Search order:
    1. PYANN_LIB_PATH environment variable
    2. The pyann package's directories, in import order (see _package_dirs)
    3. Common build and system locations
    
    Warns when the package's directories hold a newer libann than the one
    found, which usually means a stale build is shadowing a fresh one.
    
    Args:
        package_dirs: The package's directories (default: _package_dirs())
    
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
    
    if package_dirs is None:
        package_dirs = _package_dirs()
    package_paths = [p for d in package_dirs for p in (d, d / "lib", d / "_lib")]
    
    # Search paths
    search_paths = []
    
    # 1. Environment variable
    env_path = os.environ.get("PYANN_LIB_PATH")
    if env_path:
        search_paths.append(Path(env_path))
    
    # 2. In the package
    search_paths.extend(package_paths)
    
    # 3. Next to package in site-packages (for installed packages)
    package_dir = Path(__file__).parent.parent
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
                if not env_path:
                    _warn_if_shadowed(lib_path, [p / n for p in package_paths for n in lib_names])
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


def _warn_if_shadowed(chosen: Path, candidates: List[Path]) -> None:
    """Warn when a newer libann among candidates loses to the chosen one."""
    chosen_real = chosen.resolve()
    chosen_time = chosen_real.stat().st_mtime
    for candidate in candidates:
        if not candidate.exists() or candidate.resolve() == chosen_real:
            continue
        if candidate.resolve().stat().st_mtime > chosen_time:
            warnings.warn(
                f"pyann is loading libann from {chosen}, but {candidate} is newer and is "
                f"ignored. Rebuild or delete the older one, or set PYANN_LIB_PATH to the "
                f"directory of the one you want.",
                RuntimeWarning, stacklevel=3,
            )
            return


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

    # GPU backend compiled into the library, whether or not a device is
    # available. pyann_info.c reports it from the build flags.
    try:
        name = loaded_lib.pyann_gpu_backend()
        if name != ffi.NULL:
            info["gpu"] = ffi.string(name).decode()
    except (FFIError, AttributeError):
        # Libraries built before pyann_gpu_backend() exported libann's
        # backend vtables; cffi can't look up data symbols, so use ctypes
        probe = ctypes.CDLL(lib_path)
        for name in ("metal", "cuda"):
            if hasattr(probe, f"{name}_backend"):
                info["gpu"] = name
                break

    return info


def _preload_windows_dependencies(lib_path: Path) -> None:
    """Load DLLs shipped next to libann (e.g. OpenBLAS) before libann itself.

    cffi loads libann with a plain LoadLibrary, which does not search the
    library's own directory for its dependencies. A DLL that is already loaded
    is matched by name, so loading the bundled ones first resolves them.
    """
    lib_dir = lib_path.parent
    os.add_dll_directory(str(lib_dir))
    for dll in sorted(lib_dir.glob("*.dll")):
        if dll != lib_path:
            ctypes.WinDLL(str(dll))


def _load_library() -> object:
    """Load the libann shared library.

    Returns:
        CFFI library handle
    """
    global _lib, _backend_info
    if _lib is not None:
        return _lib

    lib_path = _find_library()
    if sys.platform == "win32":
        _preload_windows_dependencies(Path(lib_path))
    _lib = ffi.dlopen(lib_path)
    _backend_info = _detect_backends(_lib, lib_path)
    if _error_callback is not None:
        _lib.ann_set_error_log_callback(_error_callback)

    return _lib


def get_backend_info() -> Dict[str, Optional[str]]:
    """Return detected acceleration backend information.

    Returns:
        Dict with keys: blas, gpu (backend compiled in), gpu_active
        (backend in use after enable_gpu(), else None), cblas_config, cblas_corename, cblas_isa
        (None when the corresponding backend is not compiled in), threads (CPU
        threads libann uses), and optimized (False for an unoptimized, e.g.
        Debug, build of libann, which trains much slower).
    """
    _load_library()
    info = dict(_backend_info)
    build = ffi.new("Backend_info *")
    if _lib.ann_get_backend_info(build) == 0:
        info["threads"] = build.threads
        info["optimized"] = bool(build.optimized)
        if info["blas"] is None and ffi.string(build.blas).decode() != "none":
            info["blas"] = ffi.string(build.blas).decode().lower()
    return info


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
            # A build with several backends starts the first that works (or
            # the one ANN_GPU names); libann reports which one is running
            build = ffi.new("Backend_info *")
            if _lib.ann_get_backend_info(build) == 0:
                _backend_info["gpu_active"] = ffi.string(build.gpu).decode().lower()
            else:
                _backend_info["gpu_active"] = _backend_info["gpu"]
    return _backend_info["gpu_active"] is not None


def set_num_threads(threads: int) -> None:
    """Set how many CPU threads libann uses for training and batched inference.

    libann splits its own work (convolution, pooling, activations, optimizer
    updates, and its matrix multiply when built without BLAS) across this many
    threads. A BLAS library keeps managing its own threads; the two never run
    at the same time. Results do not depend on the thread count.

    Args:
        threads: 1 for single-threaded; 0 restores the default (the
            ANN_NUM_THREADS environment variable if set, else the number of
            processors)

    Raises:
        ValueError: If threads is negative
    """
    if not isinstance(threads, int) or isinstance(threads, bool) or threads < 0:
        raise ValueError(f"threads must be an int >= 0, got {threads!r}")
    _load_library()
    _lib.ann_set_num_threads(threads)


def set_seed(seed: int) -> None:
    """Seed libann's random numbers, so training repeats exactly.

    Covers everything libann draws at random: weight initialization, the
    order samples are shuffled in each epoch, dropout masks, input noise and
    the hyperparameter searches. With the same seed, data and settings, a CPU
    run trains the same weights every time, with any thread count (a GPU or
    a threaded BLAS library may still round differently). Call it before
    building and training the model. NumPy's and Python's own random
    generators are separate (e.g. ``split_data(seed=...)``).

    Args:
        seed: An int from 0 to 2**32 - 1
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2 ** 32:
        raise ValueError(f"seed must be an int in [0, 2**32), got {seed!r}")
    _load_library()
    _lib.ann_set_seed(seed)


def get_num_threads() -> int:
    """Return the number of CPU threads libann uses (the calling thread included)."""
    _load_library()
    return _lib.ann_get_num_threads()


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
