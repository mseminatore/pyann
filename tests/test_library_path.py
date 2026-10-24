"""Tests for how pyann finds the libann shared library."""

import os
import sys
import warnings
from pathlib import Path

import pytest

import pyann
from pyann._bindings.ffi import _find_library, _package_dirs

LIB = {"win32": "ann.dll", "darwin": "libann.dylib"}.get(sys.platform, "libann.so")


def fake_lib(package_dir: Path, mtime: float) -> Path:
    """An empty file where a build would put libann, with the given modification time."""
    path = package_dir / "lib" / LIB
    path.parent.mkdir(parents=True)
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))
    return path


@pytest.fixture(autouse=True)
def no_env_override(monkeypatch):
    monkeypatch.delenv("PYANN_LIB_PATH", raising=False)


class TestLibraryLookup:
    def test_package_dirs_follow_import_order(self):
        dirs = _package_dirs()
        assert dirs[:len(pyann.__path__)] == [Path(p) for p in pyann.__path__]
        ffi_module = sys.modules["pyann._bindings.ffi"]
        assert Path(ffi_module.__file__).resolve().parent.parent in [d.resolve() for d in dirs]

    def test_first_package_dir_wins(self, tmp_path):
        build = fake_lib(tmp_path / "site-packages" / "pyann", 2000)
        fake_lib(tmp_path / "src" / "pyann", 1000)
        assert _find_library([tmp_path / "site-packages" / "pyann", tmp_path / "src" / "pyann"]) == str(build)

    def test_warns_when_a_newer_build_is_shadowed(self, tmp_path):
        old = fake_lib(tmp_path / "a", 1000)
        new = fake_lib(tmp_path / "b", 2000)
        with pytest.warns(RuntimeWarning) as record:
            assert _find_library([tmp_path / "a", tmp_path / "b"]) == str(old)
        assert str(old) in str(record[0].message) and str(new) in str(record[0].message)

    def test_no_warning_when_the_newest_build_wins(self, tmp_path):
        new = fake_lib(tmp_path / "a", 2000)
        fake_lib(tmp_path / "b", 1000)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _find_library([tmp_path / "a", tmp_path / "b"]) == str(new)

    @pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
    def test_symlinks_to_the_same_build_are_not_a_conflict(self, tmp_path):
        real = fake_lib(tmp_path / "a", 1000)
        link_dir = tmp_path / "b" / "lib"
        link_dir.mkdir(parents=True)
        (link_dir / LIB).symlink_to(real)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _find_library([tmp_path / "a", tmp_path / "b"]) == str(real)

    def test_env_var_overrides_without_warning(self, tmp_path, monkeypatch):
        chosen = fake_lib(tmp_path / "chosen", 1000)
        fake_lib(tmp_path / "a", 2000)
        monkeypatch.setenv("PYANN_LIB_PATH", str(chosen.parent))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert _find_library([tmp_path / "a"]) == str(chosen)
