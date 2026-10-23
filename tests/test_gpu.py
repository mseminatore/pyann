"""Tests for GPU backend detection and the CPU/CBLAS fallback."""

import pytest

import pyann
from pyann import Sequential, Dense

np = pytest.importorskip("numpy")


def _make_data(n=64):
    rng = np.random.RandomState(0)
    x = rng.rand(n, 4).astype(np.float32)
    y = (x.sum(axis=1, keepdims=True) / 4).astype(np.float32)
    return x, y


def _make_model():
    model = Sequential()
    model.add(Dense(8, activation="relu", input_shape=(4,)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse")
    return model


class TestBackendInfo:
    def test_info_keys(self):
        info = pyann.backend_info()
        for key in ("blas", "gpu", "gpu_active"):
            assert key in info

    def test_gpu_backend_name(self):
        assert pyann.backend_info()["gpu"] in (None, "metal", "cuda", "vulkan")


class TestEnableGpu:
    def test_enable_gpu_matches_build(self):
        compiled = pyann.backend_info()["gpu"]
        active = pyann.enable_gpu()
        assert isinstance(active, bool)
        if compiled is None:
            assert not active
        assert pyann.gpu_enabled() == active
        assert (pyann.backend_info()["gpu_active"] is not None) == active

    def test_enable_gpu_idempotent(self):
        assert pyann.enable_gpu() == pyann.enable_gpu()


class TestTrainPredict:
    """Must pass on CPU-only, CBLAS and GPU builds alike."""

    def test_fit_and_predict(self):
        pyann.enable_gpu()
        x, y = _make_data()
        model = _make_model()
        model.fit(x, y, epochs=5, batch_size=16, verbose=0)
        preds = model.predict(x)
        assert preds.shape == (64, 1)
        assert np.all(np.isfinite(preds))

    def test_batch_matches_single_sample(self):
        pyann.enable_gpu()
        x, y = _make_data(16)
        model = _make_model()
        model.fit(x, y, epochs=3, batch_size=8, verbose=0)
        batch = model.predict(x)
        single = np.vstack([model.predict(x[i]) for i in range(len(x))])
        np.testing.assert_allclose(batch, single, atol=1e-4)

    def test_save_load_predict(self, tmp_path):
        pyann.enable_gpu()
        x, y = _make_data(16)
        model = _make_model()
        model.fit(x, y, epochs=3, batch_size=8, verbose=0)
        path = tmp_path / "m.nna"
        model.save(path)
        loaded = Sequential.load(path)
        np.testing.assert_allclose(model.predict(x), loaded.predict(x), atol=1e-4)
