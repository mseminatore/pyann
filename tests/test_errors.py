"""Tests for libann errors: pyann's error log callback puts them in exceptions, not on stderr."""

import importlib
import logging

import pytest

from pyann import Dense, Sequential
from pyann.exceptions import (
    IOError as PyANNIOError, InvalidParameterError, NetworkError,
    c_error_message, clear_c_error, raise_for_error_code, record_c_error,
)

# pyann._bindings re-exports the FFI object as "ffi", hiding the module of that name
ffi_module = importlib.import_module("pyann._bindings.ffi")

pytestmark = pytest.mark.skipif(ffi_module._error_callback is None,
                                reason="cffi cannot make callbacks here")


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A trained model saved in the text and binary formats."""
    model = Sequential()
    model.add(Dense(4, activation="relu", input_shape=(2,)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse")
    model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [[0], [1], [1], [0]], epochs=2, verbose=0)
    directory = tmp_path_factory.mktemp("saved")
    model.save(directory / "m.nna")
    model.save(directory / "m.nnb", format="binary")
    return directory


class TestLoadErrors:
    def test_missing_file(self, tmp_path, capfd):
        with pytest.raises(NetworkError, match=r"\(ann_load_network: File I/O error"):
            Sequential.load(tmp_path / "missing.nna")
        assert capfd.readouterr().err == ""

    @pytest.mark.parametrize("name,fmt,function", [
        ("m.nna", "text", "ann_load_network"),
        ("m.nnb", "binary", "ann_load_network_binary"),
    ])
    def test_truncated_file(self, saved, tmp_path, capfd, name, fmt, function):
        data = (saved / name).read_bytes()
        (tmp_path / name).write_bytes(data[:len(data) // 2])
        with pytest.raises(NetworkError, match=rf"\({function}: Invalid parameter"):
            Sequential.load(tmp_path / name, format=fmt)
        assert capfd.readouterr().err == ""

    def test_good_load_is_silent(self, saved, capfd):
        Sequential.load(saved / "m.nna")
        Sequential.load(saved / "m.nnb", format="binary")
        captured = capfd.readouterr()
        assert captured.out == "" and captured.err == ""

    def test_missing_onnx_file(self, tmp_path, capfd):
        with pytest.raises(NetworkError, match=r"\(ann_import_onnx: File I/O error"):
            Sequential.load_onnx(tmp_path / "missing.onnx.json")
        assert capfd.readouterr().err == ""

    def test_logged_at_debug(self, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG, logger="pyann"):
            with pytest.raises(NetworkError):
                Sequential.load(tmp_path / "missing.nna")
        assert any("ann_load_network" in r.getMessage() for r in caplog.records)


class TestErrorCodes:
    def test_error_code_path_names_the_function(self, saved, tmp_path, capfd):
        model = Sequential.load(saved / "m.nna")
        with pytest.raises(PyANNIOError, match=r"\(ann_save_network: File I/O error"):
            model.save(tmp_path / "missing" / "m.nna")
        assert capfd.readouterr().err == ""

    def test_stale_error_is_not_attached(self):
        record_c_error(-5, "File I/O error", "ann_something")
        with pytest.raises(InvalidParameterError) as info:
            raise_for_error_code(-4, "Failed")
        assert str(info.value) == "Failed"

    def test_success_clears_the_error(self):
        record_c_error(-4, "Invalid parameter or state", "ann_something")
        raise_for_error_code(0)
        assert c_error_message("Failed") == "Failed"

    def test_error_is_consumed(self):
        clear_c_error()
        record_c_error(-4, "Invalid parameter or state", "ann_something")
        assert c_error_message("Failed") == "Failed (ann_something: Invalid parameter or state)"
        assert c_error_message("Failed") == "Failed"
