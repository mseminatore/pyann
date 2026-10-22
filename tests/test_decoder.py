"""Tests for decoder layers (UpSample2D, Reshape), image targets and input noise."""

import pytest

import pyann
import pyann.utils.compat as compat
from pyann import Conv2D, Dense, Input, MaxPool2D, Reshape, Sequential, UpSample2D, UpSampling2D
from pyann._bindings.ffi import ffi, lib
from pyann.exceptions import NetworkError

np = pytest.importorskip("numpy")


def bars(n=24, seed=0):
    """1x6x6 images of a horizontal or vertical bar, with a little noise."""
    rng = np.random.RandomState(seed)
    x = 0.05 * rng.rand(n, 1, 6, 6).astype(np.float32)
    for i in range(n):
        pos = (i // 2) % 6
        if i % 2:
            x[i, 0, :, pos] = 1.0
        else:
            x[i, 0, pos, :] = 1.0
    return x


def autoencoder():
    """conv -> pool -> upsample -> conv, reproducing 1x6x6 images."""
    model = Sequential()
    model.add(Input((1, 6, 6)))
    model.add(Conv2D(4, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2))
    model.add(UpSample2D(2))
    model.add(Conv2D(1, 3, padding="same", activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse", learning_rate=0.01, convergence=0.0)
    return model


def bottleneck():
    """A dense bottleneck turned back into an image: 8 -> dense 8 -> 2x2x2 -> 2x4x4 -> 1x2x2."""
    model = Sequential()
    model.add(Dense(8, activation="tanh", input_shape=(8,)))
    model.add(Reshape((2, 2, 2)))
    model.add(UpSample2D(2))
    model.add(Conv2D(1, 3, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse", learning_rate=0.01, convergence=0.0)
    return model


class TestLayers:
    def test_upsample_settings(self):
        assert UpSample2D().size == 2
        assert UpSample2D((3, 3)).size == 3
        assert UpSampling2D is UpSample2D
        assert UpSample2D(2).output_shape((4, 3, 5)) == (4, 6, 10)
        with pytest.raises(ValueError):
            UpSample2D(0)
        with pytest.raises(ValueError):
            UpSample2D((2, 3))

    def test_reshape_settings(self):
        assert Reshape([2, 3, 4]).target_shape == (2, 3, 4)
        for bad in [(2, 3), (2, 0, 4), (2, 3.0, 4)]:
            with pytest.raises(ValueError):
                Reshape(bad)

    def test_summary_shows_shapes(self):
        text = autoencoder().summary()
        assert "UpSample2D(2x) -> (4, 6, 6)" in text
        assert "Reshape -> (2, 2, 2)" in bottleneck().summary()

    def test_flat_input_before_reshape(self):
        model = bottleneck()
        model.fit(np.random.rand(16, 8).astype(np.float32), np.random.rand(16, 1, 2, 2).astype(np.float32),
                  epochs=2, batch_size=8, verbose=0)
        assert model.predict(np.zeros((3, 8), dtype=np.float32)).shape == (3, 1, 2, 2)

    def test_upsample_needs_an_image(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(8,)))
        model.add(UpSample2D(2))
        model.compile(loss="mse")
        with pytest.raises(NetworkError):
            model.fit(np.zeros((2, 8)), np.zeros((2, 8)), epochs=1, verbose=0)

    def test_reshape_size_must_match(self):
        model = Sequential()
        model.add(Dense(8, input_shape=(4,)))
        model.add(Reshape((2, 2, 3)))
        model.compile(loss="mse")
        with pytest.raises(NetworkError, match="Reshape"):
            model.fit(np.zeros((2, 4)), np.zeros((2, 12)), epochs=1, verbose=0)


class TestAutoencoder:
    def test_learns_with_image_targets(self):
        x = bars()
        model = autoencoder()
        model.fit(x, x, epochs=1, batch_size=8, verbose=0)
        before = model.evaluate_loss(x, x)
        model.fit(x, x, epochs=150, batch_size=8, verbose=0)
        assert model.evaluate_loss(x, x) < 0.5 * before
        out = model.predict(x)
        assert out.shape == x.shape     # an image output comes back as images
        assert np.abs(out - x).mean() < 0.2

    def test_train_on_batch_with_image_targets(self):
        x = bars(8)
        model = autoencoder()
        losses = [model.train_on_batch(x, x) for _ in range(30)]
        model.end_online_training()
        assert losses[-1] < losses[0]

    @pytest.mark.parametrize("fmt", ["text", "binary", "onnx"])
    def test_round_trip(self, tmp_path, fmt):
        x = bars(8)
        for model in (autoencoder(), bottleneck()):
            data = x if model._input_shape == (1, 6, 6) else x.reshape(8, -1)[:, :8]
            model.fit(data, _targets(model, data), epochs=2, verbose=0)
            path = tmp_path / ("m.onnx.json" if fmt == "onnx" else f"m.{fmt}")
            if fmt == "onnx":
                model.export_onnx(path)
                loaded = Sequential.load_onnx(path)
            else:
                model.save(path, format=fmt)
                loaded = Sequential.load(path, format=fmt)
            assert [type(layer) for layer in loaded._layers] == [type(layer) for layer in model._layers]
            assert loaded.summary().split("\n")[2:] == model.summary().split("\n")[2:]
            if fmt == "text":
                np.testing.assert_allclose(loaded.predict(data), model.predict(data), atol=1e-4)
            else:
                np.testing.assert_array_equal(loaded.predict(data), model.predict(data))


def _targets(model, data):
    """Targets of the model's output shape for data."""
    return np.zeros((len(data),) + model._layer_shapes()[-1][1], dtype=np.float32) + 0.5


class TestInputNoise:
    def test_off_by_default(self):
        assert autoencoder().input_noise is None

    def test_settings(self):
        model = autoencoder()
        assert model.set_input_noise("gaussian", 0.3, clip=(0, 1)) is model
        assert model.input_noise == ("gaussian", 0.3, (0.0, 1.0))
        model.set_input_noise("masking", 0.2)
        assert model.input_noise == ("masking", 0.2, None)
        model.set_input_noise(None)
        assert model.input_noise is None

    @pytest.mark.parametrize("args", [("blur", 0.1), ("gaussian", -0.1), ("masking", 1.5),
                                      ("salt_pepper", 0.1), ("gaussian", 0.1, (1, 0))])
    def test_invalid(self, args):
        with pytest.raises(ValueError):
            autoencoder().set_input_noise(*args)

    def test_reaches_the_network(self):
        x = bars(8)
        model = autoencoder().set_input_noise("salt_pepper", 0.25, clip=(0, 1))
        model.fit(x, x, epochs=1, verbose=0)
        noise, amount, low, high = ffi.new("Noise_type *"), ffi.new("real *"), ffi.new("real *"), ffi.new("real *")
        lib.ann_get_input_noise(model._ptr, noise, amount, low, high)
        assert (noise[0], amount[0], low[0], high[0]) == (lib.NOISE_SALT_PEPPER, 0.25, 0.0, 1.0)
        model.set_input_noise(None)
        lib.ann_get_input_noise(model._ptr, noise, amount, low, high)
        assert noise[0] == lib.NOISE_NONE

    def test_apply_input_noise(self):
        x = np.ones((50, 1, 6, 6), dtype=np.float32)
        model = autoencoder()
        assert np.array_equal(model.apply_input_noise(x), x)
        model.set_input_noise("masking", 0.3)
        noisy = model.apply_input_noise(x)
        assert noisy.shape == x.shape and np.all(x == 1.0)
        assert set(np.unique(noisy)) <= {0.0, 1.0}
        assert abs((noisy == 0).mean() - 0.3) < 0.05

    def test_apply_input_noise_without_numpy(self, monkeypatch):
        model = autoencoder().set_input_noise("gaussian", 1.0, clip=(0, 1))
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
        rows = [[0.5] * 36 for _ in range(4)]
        noisy = model.apply_input_noise(rows)
        assert len(noisy) == 4 and all(len(r) == 36 for r in noisy)
        assert all(0.0 <= v <= 1.0 for r in noisy for v in r)
        assert any(v != 0.5 for r in noisy for v in r) and rows[0][0] == 0.5


class TestBackendInfo:
    def test_threads_and_optimized(self):
        info = pyann.backend_info()
        assert isinstance(info["threads"], int) and info["threads"] >= 1
        assert info["optimized"] is True


class TestGpu:
    def test_autoencoder_trains_on_the_gpu(self):
        if not pyann.enable_gpu():
            pytest.skip("no GPU backend in this build")
        x = bars()
        model = autoencoder().set_input_noise("gaussian", 0.1, clip=(0, 1))
        model.fit(x, x, epochs=100, batch_size=8, verbose=0)
        # Metal runs every layer kind; CUDA is dense-only, so libann keeps this on the CPU
        assert model._gpu_uploaded == (pyann.backend_info()["gpu_active"] == "metal")
        assert np.abs(model.predict(x) - x).mean() < 0.2


class TestImageOutputs:
    def test_single_image_and_lists(self, monkeypatch):
        x = bars(3)
        model = autoencoder()
        model.fit(x, x, epochs=2, verbose=0)
        out = model.predict(x)
        assert model.predict(x[0]).shape == (1, 1, 6, 6)
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
        nested = model.predict(x.reshape(3, -1).tolist())   # without NumPy, inputs are rows
        assert len(nested) == 3 and len(nested[0]) == 1 and len(nested[0][0]) == 6 and len(nested[0][0][0]) == 6
        assert np.allclose(np.array(nested, dtype=np.float32), out)

    def test_dense_outputs_stay_rows(self):
        model = Sequential([Conv2D(2, 3, input_shape=(1, 6, 6), activation="relu"), Dense(3, activation="softmax")])
        model.compile(loss="cce", convergence=0.0)
        model.fit(bars(6), np.eye(3, dtype=np.float32)[np.arange(6) % 3], epochs=1, verbose=0)
        assert model.predict(bars(6)).shape == (6, 3)

    def test_quantized_model_matches(self):
        x = bars()
        model = autoencoder()
        model.fit(x, x, epochs=30, batch_size=8, verbose=0)
        q = model.quantize(x)
        assert q.predict(x).shape == x.shape
        assert np.abs(q.predict(x) - model.predict(x)).max() < 0.05
        assert q.predict_int8(q.quantize_input(x)).shape == (len(x), 36)   # int8 buffers stay flat, as in C
