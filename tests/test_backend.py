"""Check that the library was built with the BLAS the build is supposed to use.

Under cibuildwheel (CIBUILDWHEEL is set) the expectation follows the platform,
so a wheel that quietly lost its BLAS fails its tests. PYANN_EXPECT_BLAS sets
or overrides it elsewhere; with neither set the test is skipped.
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


EXPECTED = _expected_blas()


@pytest.mark.skipif(EXPECTED is None, reason="no expected BLAS configured")
class TestExpectedBackend:
    def test_blas_backend(self):
        assert pyann.backend_info()["blas"] == EXPECTED
