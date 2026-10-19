"""The package version must come from the installed metadata, not a hand-kept copy."""

from importlib.metadata import PackageNotFoundError, version

import pytest

import pyann


class TestVersion:
    def test_matches_installed_metadata(self):
        try:
            installed = version("pylibann")
        except PackageNotFoundError:
            pytest.skip("pylibann is not installed (running from a source tree)")
        assert pyann.__version__ == installed

    def test_is_a_string(self):
        assert isinstance(pyann.__version__, str) and pyann.__version__
