#!/usr/bin/env python3
"""Build script for pyann native library."""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def find_cmake():
    """Find CMake executable."""
    cmake = shutil.which("cmake")
    if cmake is None:
        print("ERROR: CMake not found. Please install CMake and add it to PATH.")
        sys.exit(1)
    return cmake


def clone_ann_if_needed(project_dir: Path):
    """Clone ann repository if not present."""
    ann_dir = project_dir / "ann"
    if ann_dir.exists() and (ann_dir / "ann.c").exists():
        print(f"Found ann source at {ann_dir}")
        return
    
    print("Cloning ann repository...")
    subprocess.run(
        ["git", "clone", "https://github.com/mseminatore/ann.git", str(ann_dir)],
        check=True
    )


def build_library(
    project_dir: Path,
    build_type: str = "Release",
    use_blas: bool = False,
    use_cblas: bool = False,
    use_accelerate: bool = False,
    use_scipy_openblas: bool = False,
    use_metal: bool = False,
    use_cuda: bool = False,
    use_vulkan: bool = False,
):
    """Build the shared library."""
    cmake = find_cmake()
    build_dir = project_dir / "build"
    
    # Create build directory
    build_dir.mkdir(exist_ok=True)
    
    # CMake configure
    cmake_args = [
        cmake,
        "..",
        f"-DCMAKE_BUILD_TYPE={build_type}",
    ]
    
    if use_cblas:
        cmake_args.append("-DUSE_CBLAS=ON")
    elif use_accelerate:
        cmake_args.append("-DUSE_ACCELERATE=ON")
    elif use_scipy_openblas:
        cmake_args.append("-DUSE_SCIPY_OPENBLAS=ON")
        # CMake locates the scipy-openblas32 package with this interpreter
        cmake_args.append(f"-DPython_EXECUTABLE={sys.executable}")
    elif use_blas:
        cmake_args.append("-DUSE_BLAS=ON")

    if use_metal:
        cmake_args.append("-DUSE_METAL=ON")
    if use_cuda:
        cmake_args.append("-DUSE_CUDA=ON")
    if use_vulkan:
        cmake_args.append("-DUSE_VULKAN=ON")
    
    print(f"Configuring with: {' '.join(cmake_args)}")
    subprocess.run(cmake_args, cwd=build_dir, check=True)
    
    # CMake build
    build_args = [cmake, "--build", ".", "--config", build_type]
    print(f"Building with: {' '.join(build_args)}")
    subprocess.run(build_args, cwd=build_dir, check=True)
    
    print("\nBuild complete!")
    
    # Check output
    lib_dir = project_dir / "pyann" / "lib"
    if lib_dir.exists():
        libs = list(lib_dir.glob("*"))
        if libs:
            print(f"Library installed to: {lib_dir}")
            for lib in libs:
                print(f"  - {lib.name}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Build pyann native library")
    parser.add_argument("--debug", action="store_true", help="Build debug version")
    parser.add_argument("--blas", action="store_true", help="Enable OpenBLAS")
    parser.add_argument("--cblas", action="store_true", help="Enable CBLAS")
    parser.add_argument("--accelerate", action="store_true", help="Use Apple Accelerate for BLAS (macOS only)")
    parser.add_argument("--scipy-openblas", action="store_true", help="Use OpenBLAS from the scipy-openblas32 package (pip install scipy-openblas32)")
    parser.add_argument("--metal", action="store_true", help="Enable Apple Metal GPU acceleration (macOS only)")
    parser.add_argument("--cuda", action="store_true", help="Enable NVIDIA CUDA GPU acceleration")
    parser.add_argument("--vulkan", action="store_true", help="Enable Vulkan GPU acceleration (needs the Vulkan headers and glslc)")
    parser.add_argument("--clean", action="store_true", help="Clean build directory first")
    
    args = parser.parse_args()
    
    project_dir = Path(__file__).parent.absolute()
    build_dir = project_dir / "build"
    
    if args.clean and build_dir.exists():
        print(f"Cleaning {build_dir}...")
        shutil.rmtree(build_dir)
    
    clone_ann_if_needed(project_dir)
    
    build_library(
        project_dir,
        build_type="Debug" if args.debug else "Release",
        use_blas=args.blas,
        use_cblas=args.cblas,
        use_accelerate=args.accelerate,
        use_scipy_openblas=args.scipy_openblas,
        use_metal=args.metal,
        use_cuda=args.cuda,
        use_vulkan=args.vulkan,
    )


if __name__ == "__main__":
    main()
