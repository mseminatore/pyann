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
    use_cblas: bool = False
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
    elif use_blas:
        cmake_args.append("-DUSE_BLAS=ON")
    
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
        use_cblas=args.cblas
    )


if __name__ == "__main__":
    main()
