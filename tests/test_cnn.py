"""Tests for the convolutional layers: Conv2D, MaxPool2D and Flatten."""

import pytest

import pyann
from pyann import Conv2D, Dense, Flatten, Input, MaxPool2D, MaxPooling2D, Sequential
from pyann.exceptions import NetworkError

np = pytest.importorskip("numpy")


def bars(n=24, seed=0):
    """1x6x6 images of a horizontal (class 0) or vertical (class 1) bar."""
    rng = np.random.RandomState(seed)
    x = 0.05 * rng.rand(n, 1, 6, 6).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        vertical, pos = i % 2, (i // 2) % 6
        if vertical:
            x[i, 0, :, pos] = 1.0
        else:
            x[i, 0, pos, :] = 1.0
        y[i, vertical] = 1.0
    return x, y


def make_cnn(**conv_kwargs):
    model = Sequential()
    model.add(Input((1, 6, 6)))
    model.add(Conv2D(4, 3, padding="same", activation="relu", **conv_kwargs))
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="cce", learning_rate=0.01, convergence=0.0)
    return model


def trained_cnn():
    x, y = bars()
    model = make_cnn()
    model.fit(x, y, epochs=60, batch_size=8, verbose=0)
    return model, x, y


class TestLayerSettings:
    def test_conv_defaults_and_pairs(self):
        layer = Conv2D(8, (5, 5), strides=(2, 2))
        assert (layer.kernel_size, layer.strides, layer.padding) == (5, 2, 0)

    def test_padding_names(self):
        assert Conv2D(4, 3, padding="valid").padding == 0
        assert Conv2D(4, 5, padding="same").padding == 2

    @pytest.mark.parametrize("kwargs", [
        dict(filters=0),
        dict(filters=4, kernel_size=(3, 5)),
        dict(filters=4, kernel_size=0),
        dict(filters=4, strides=0),
        dict(filters=4, padding=-1),
        dict(filters=4, padding="full"),
        dict(filters=4, kernel_size=2, padding="same"),
        dict(filters=4, strides=2, padding="same"),
        dict(filters=4, dropout=1.0),
        dict(filters=4, activation="softmax"),
    ])
    def test_invalid_conv(self, kwargs):
        with pytest.raises((ValueError, TypeError)):
            Conv2D(**kwargs)

    def test_pool_strides_default_to_pool_size(self):
        assert MaxPool2D(3).strides == 3
        assert MaxPool2D(3, strides=1).strides == 1

    def test_keras_alias(self):
        assert MaxPooling2D is MaxPool2D
        assert pyann.MaxPooling2D is MaxPool2D


class TestShapes:
    def test_summary_shows_shapes_and_params(self):
        text = make_cnn().summary()
        assert "Input shape: (1, 6, 6)" in text
        assert "Conv2D(4, 3x3, stride 1, padding 1, relu) -> (4, 6, 6) - 40 params" in text
        assert "MaxPool2D(2x2, stride 2) -> (4, 3, 3)" in text
        assert "Flatten -> (36,)" in text
        assert "Dense(2, softmax) - 74 params" in text
        assert "Total parameters: 114" in text

    def test_conv_needs_an_image_input(self):
        model = Sequential([Dense(8, input_shape=(36,)), Conv2D(2, 3), Dense(2)])
        with pytest.raises(NetworkError, match="image input"):
            model.summary()

    def test_window_that_does_not_fit(self):
        model = Sequential([Conv2D(2, 7, input_shape=(1, 6, 6)), Dense(2)])
        with pytest.raises(NetworkError, match="does not fit"):
            model.fit(np.zeros((2, 1, 6, 6)), np.zeros((2, 2)), epochs=1, verbose=0)
        assert model._ptr is None  # nothing was allocated

    def test_two_dimensional_input_is_one_channel(self):
        x, y = bars()
        model = Sequential([Conv2D(2, 3, input_shape=(6, 6), activation="relu"), Dense(2, activation="softmax")])
        model.compile(loss="cce", convergence=0.0)
        model.fit(x[:, 0], y, epochs=2, verbose=0)
        assert model.predict(x[:, 0]).shape == (24, 2)


class TestTraining:
    def test_learns_bars(self):
        model, x, y = trained_cnn()
        assert model.evaluate(x, y) == 1.0
        assert model.evaluate_loss(x, y) < 0.1

    def test_input_forms(self):
        model, x, _ = trained_cnn()
        batch = model.predict(x)
        assert batch.shape == (24, 2)
        np.testing.assert_allclose(model.predict(x[0]), batch[:1], atol=1e-6)        # one image
        np.testing.assert_allclose(model.predict(x.reshape(24, -1)), batch, atol=1e-6)  # flat rows

    def test_channels_last_data_is_rejected_with_a_hint(self):
        model = make_cnn()
        x = np.zeros((4, 6, 6, 1), dtype=np.float32)
        with pytest.raises(ValueError, match="transpose"):
            model.fit(x, np.zeros((4, 2)), epochs=1, verbose=0)

    def test_train_on_batch_with_images(self):
        x, y = bars()
        model = make_cnn()
        losses = [model.train_on_batch(x, y) for _ in range(40)]
        assert losses[-1] < losses[0]

    def test_validation_and_dropout(self):
        x, y = bars()
        model = make_cnn(dropout=0.2)
        history = model.fit(x, y, epochs=60, batch_size=8, verbose=0, validation_data=(x, y))
        assert history["val_accuracy"] >= 0.9


class TestPersistence:
    @pytest.mark.parametrize("fmt", ["text", "binary"])
    def test_save_and_load(self, tmp_path, fmt):
        model, x, _ = trained_cnn()
        path = tmp_path / ("model.nna" if fmt == "text" else "model.nnb")
        model.save(path, format=fmt)
        loaded = Sequential.load(path, format=fmt)
        assert loaded._input_shape == (1, 6, 6)
        assert [type(l) for l in loaded._layers] == [Conv2D, MaxPool2D, Flatten, Dense]
        conv = loaded._layers[0]
        assert (conv.filters, conv.kernel_size, conv.strides, conv.padding) == (4, 3, 1, 1)
        assert loaded.summary().splitlines()[2:] == model.summary().splitlines()[2:]
        np.testing.assert_allclose(loaded.predict(x), model.predict(x), atol=1e-4 if fmt == "text" else 0)

    def test_onnx_round_trip(self, tmp_path):
        model, x, _ = trained_cnn()
        path = tmp_path / "model.onnx.json"
        model.export_onnx(path)
        loaded = Sequential.load_onnx(path)
        assert [type(l) for l in loaded._layers] == [Conv2D, MaxPool2D, Flatten, Dense]
        np.testing.assert_array_equal(loaded.predict(x), model.predict(x))

    def test_loaded_model_keeps_training(self, tmp_path):
        model, x, y = trained_cnn()
        path = tmp_path / "model.nnb"
        model.save(path, format="binary")
        loaded = Sequential.load(path, format="binary")
        for _ in range(5):
            loaded.train_on_batch(x, y)
        assert loaded.evaluate(x, y) == 1.0


class TestGpu:
    def test_cnn_trains_when_the_gpu_is_enabled(self):
        if not pyann.enable_gpu():
            pytest.skip("no GPU backend in this build")
        model, x, y = trained_cnn()
        # Metal and Vulkan run CNNs; CUDA is dense-only, so libann keeps them on the CPU
        assert model._gpu_uploaded == (pyann.backend_info()["gpu_active"] in ("metal", "vulkan"))
        assert model.evaluate(x, y) == 1.0
