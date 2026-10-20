"""Tests for rebuilding a Sequential from a saved or imported libann network."""

import pytest

import pyann.core.network as network_module
from pyann import Dense, Sequential
from pyann.exceptions import NetworkError
from tests.helpers import LibSpy

np = pytest.importorskip("numpy")


def make_model():
    model = Sequential()
    model.add(Dense(6, activation="relu", input_shape=(3,)))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="cce", learning_rate=0.01)
    return model


def trained_model():
    rng = np.random.RandomState(0)
    x = rng.rand(16, 3).astype(np.float32)
    y = np.eye(2, dtype=np.float32)[(x.sum(axis=1) > 1.5).astype(int)]
    model = make_model()
    model.fit(x, y, epochs=2, batch_size=8, verbose=0)
    return model, x


class NonDenseLib(LibSpy):
    """Reports layer 1 as a layer kind pyann has no class for."""

    def __getattr__(self, name):
        if name == "ann_get_layer_kind":
            def kind(ptr, i):
                self.calls.append((name, (i,)))
                return 1 if i == 1 else self._real.LAYER_KIND_DENSE
            return kind
        return super().__getattr__(name)


@pytest.mark.parametrize("fmt,suffix", [("text", ".nna"), ("binary", ".nnb")])
def test_round_trip_rebuilds_dense_layers(tmp_path, fmt, suffix):
    model, x = trained_model()
    path = tmp_path / f"model{suffix}"
    model.save(path, format=fmt)

    loaded = Sequential.load(path, format=fmt)
    assert loaded._input_shape == (3,)
    assert [(type(l), l.units) for l in loaded._layers] == [(Dense, 6), (Dense, 2)]
    np.testing.assert_allclose(loaded.predict(x), model.predict(x), atol=1e-5)


def test_onnx_import_rebuilds_dense_layers(tmp_path):
    model, x = trained_model()
    path = tmp_path / "model.onnx.json"
    model.export_onnx(path)

    loaded = Sequential.load_onnx(path)
    assert [l.units for l in loaded._layers] == [6, 2]
    np.testing.assert_allclose(loaded.predict(x), model.predict(x), atol=1e-5)


def test_unsupported_layer_kind_is_rejected_and_freed(tmp_path, monkeypatch):
    model, _ = trained_model()
    path = tmp_path / "model.nna"
    model.save(path)

    lib = NonDenseLib(network_module.lib)
    monkeypatch.setattr(network_module, "lib", lib)
    with pytest.raises(NetworkError, match="layer types"):
        Sequential.load(path)
    assert "ann_free_network" in lib.names()
