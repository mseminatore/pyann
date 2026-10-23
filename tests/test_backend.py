"""Check that the library was built with the BLAS and GPU backend it is supposed to use.

Under cibuildwheel (CIBUILDWHEEL is set) the expectation follows the platform,
so a wheel that quietly lost its BLAS or GPU support fails its tests.
PYANN_EXPECT_BLAS and PYANN_EXPECT_GPU ("none" for no GPU) set or override it
elsewhere; with neither set the test is skipped.
"""

import os
import sys

import pytest

import pyann


def _expected_blas():
    if "PYANN_EXPECT_BLAS" in os.environ:
        return os.environ["PYANN_EXPECT_BLAS"]
    if os.environ.get("CIBUILDWHEEL"):
        return "accelerate" if sys.platform == "darwin" else "openblas"
    return None


def _expected_gpu():
    if "PYANN_EXPECT_GPU" in os.environ:
        value = os.environ["PYANN_EXPECT_GPU"]
        return None if value == "none" else value
    if os.environ.get("CIBUILDWHEEL"):
        # Linux and Windows wheels build Vulkan from vulkan_prebuilt/
        return "metal" if sys.platform == "darwin" else "vulkan"
    return "unset"


EXPECTED = _expected_blas()
EXPECTED_GPU = _expected_gpu()


@pytest.mark.skipif(EXPECTED is None, reason="no expected BLAS configured")
class TestExpectedBackend:
    def test_blas_backend(self):
        assert pyann.backend_info()["blas"] == EXPECTED


@pytest.mark.skipif(EXPECTED_GPU == "unset", reason="no expected GPU backend configured")
def test_gpu_backend():
    assert pyann.backend_info()["gpu"] == EXPECTED_GPU
