"""Tests for GlobalAveragePooling2D and BatchNormalization layers."""

import pytest

import pyann
from pyann import (BatchNorm, BatchNormalization, Conv2D, Dense, GlobalAveragePooling2D, GlobalAvgPool2D, Input,
                   MaxPool2D, Sequential)
from pyann._bindings.ffi import lib
from pyann.activations import Activation
from pyann.exceptions import NetworkError

np = pytest.importorskip("numpy")


def bars(n=48, seed=0):
    """1x6x6 images of a horizontal (class 0) or vertical (class 1) bar, and one-hot labels."""
    rng = np.random.RandomState(seed)
    x = 0.05 * rng.rand(n, 1, 6, 6).astype(np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        pos = (i // 2) % 6
        if i % 2:
            x[i, 0, :, pos] = 1.0
        else:
            x[i, 0, pos, :] = 1.0
        y[i, i % 2] = 1.0
    return x, y


def classifier():
    """conv -> batch norm (relu) -> global average pool -> dense batch norm -> softmax."""
    model = Sequential()
    model.add(Input((1, 6, 6)))
    model.add(Conv2D(4, 3, padding="same"))
    model.add(BatchNormalization(activation="relu"))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(6))
    model.add(BatchNormalization(activation="tanh"))
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="cce", learning_rate=0.02, convergence=0.0)
    return model


class TestLayers:
    def test_global_average_pooling(self):
        assert GlobalAvgPool2D is GlobalAveragePooling2D
        assert GlobalAveragePooling2D().output_shape((5, 3, 4)) == (5,)

    def test_batch_normalization(self):
        assert BatchNorm is BatchNormalization
        assert BatchNormalization().activation_type == Activation.NONE
        assert BatchNormalization(activation="relu").activation_type == Activation.RELU
        with pytest.raises(ValueError):
            BatchNormalization(activation="nope")

    def test_summary_shows_shapes_and_params(self):
        text = classifier().summary()
        assert "BatchNormalization(relu) -> (4, 6, 6) - 16 params" in text
        assert "GlobalAveragePooling2D -> (4,)" in text
        assert "BatchNormalization(tanh) -> (6,) - 24 params" in text
        # conv 4*(9+1) + bn 16 + dense 6*(4+1) + bn 24 + dense 2*(6+1)
        assert "Total parameters: 124" in text

    def test_global_average_pooling_needs_an_image(self):
        model = Sequential([Dense(8, input_shape=(8,)), GlobalAveragePooling2D(), Dense(2)])
        model.compile(loss="mse")
        with pytest.raises(NetworkError):
            model.fit(np.zeros((2, 8)), np.zeros((2, 2)), epochs=1, verbose=0)

    def test_c_layers(self):
        x, y = bars(8)
        model = classifier()
        model.fit(x, y, epochs=1, verbose=0)
        kinds = [lib.ann_get_layer_kind(model._ptr, i) for i in range(lib.ann_get_layer_count(model._ptr))]
        assert kinds == [lib.LAYER_KIND_DENSE, lib.LAYER_KIND_CONV2D, lib.LAYER_KIND_BATCHNORM,
                         lib.LAYER_KIND_GLOBAL_AVGPOOL2D, lib.LAYER_KIND_DENSE, lib.LAYER_KIND_BATCHNORM,
                         lib.LAYER_KIND_DENSE]
        assert lib.ann_get_layer_activation(model._ptr, 2) == lib.ACTIVATION_RELU
        assert lib.ann_get_layer_nodes(model._ptr, 3) == 4


class TestTraining:
    def test_learns_the_bars(self):
        x, y = bars()
        model = classifier()
        history = model.fit(x, y, epochs=60, batch_size=8, verbose=0)
        assert history["epochs_trained"] == 60
        assert model.evaluate(x, y) >= 0.95
        assert model.predict(x).shape == (len(x), 2)

    def test_prediction_uses_the_running_statistics(self):
        # a sample's prediction does not depend on the rest of the batch
        x, y = bars(16)
        model = classifier()
        model.fit(x, y, epochs=5, batch_size=8, verbose=0)
        np.testing.assert_allclose(model.predict(x[:1])[0], model.predict(x)[0], atol=1e-6)

    def test_online_training(self):
        x, y = bars(16)
        model = classifier()
        model.fit(x, y, epochs=1, verbose=0)
        losses = [model.train_on_batch(x, y) for _ in range(40)]
        model.end_online_training()
        assert losses[-1] < losses[0]

    def test_batch_norm_on_a_flat_input(self):
        rng = np.random.RandomState(1)
        x = rng.rand(64, 3).astype(np.float32) * [1.0, 100.0, 0.01]     # very different scales
        y = np.eye(2, dtype=np.float32)[(x[:, 1] > 50).astype(int)]
        model = Sequential([Input((3,)), BatchNormalization(), Dense(8, activation="relu"),
                            Dense(2, activation="softmax")])
        model.compile(optimizer="adam", loss="cce", learning_rate=0.05, convergence=0.0)
        model.fit(x, y, epochs=40, batch_size=16, verbose=0)
        assert model.evaluate(x, y) >= 0.9


class TestPersistence:
    @pytest.mark.parametrize("fmt", ["text", "binary", "onnx"])
    def test_round_trip(self, tmp_path, fmt):
        x, y = bars(16)
        model = classifier()
        model.fit(x, y, epochs=5, batch_size=8, verbose=0)
        path = tmp_path / ("m.onnx.json" if fmt == "onnx" else f"m.{fmt}")
        if fmt == "onnx":
            model.export_onnx(path)
            assert "BatchNormalization" in path.read_text() and "GlobalAveragePool" in path.read_text()
            loaded = Sequential.load_onnx(path)
        else:
            model.save(path, format=fmt)
            loaded = Sequential.load(path, format=fmt)
        assert [type(layer) for layer in loaded._layers] == [type(layer) for layer in model._layers]
        assert loaded._layers[1].activation_type == Activation.RELU
        assert loaded.summary().split("\n")[2:] == model.summary().split("\n")[2:]
        if fmt == "text":
            np.testing.assert_allclose(loaded.predict(x), model.predict(x), atol=1e-3)
        else:
            np.testing.assert_array_equal(loaded.predict(x), model.predict(x))


class TestQuantize:
    def test_batch_norm_folds_into_the_layer_before(self, tmp_path):
        x, y = bars()
        model = classifier()
        model.fit(x, y, epochs=30, batch_size=8, verbose=0)
        q = model.quantize(x)
        assert np.abs(q.predict(x) - model.predict(x)).max() < 0.1
        assert (q.predict(x).argmax(1) == model.predict(x).argmax(1)).mean() >= 0.95
        header = q.export_c_header(tmp_path / "bars_q.h", "bars_q", runtime=False)
        assert "bars_q_layers[4]" in header.read_text()     # conv, pool, dense, dense

    def test_batch_norm_that_cannot_fold_is_rejected(self):
        x, y = bars(8)
        model = Sequential([Input((1, 6, 6)), Conv2D(2, 3), MaxPool2D(2), BatchNormalization(),
                            Dense(2, activation="softmax")])
        model.compile(loss="cce", convergence=0.0)
        model.fit(x, y, epochs=1, verbose=0)
        with pytest.raises(NetworkError, match="BatchNormalization"):
            model.quantize(x)


class TestGpu:
    def test_trains_on_the_gpu(self):
        if not pyann.enable_gpu():
            pytest.skip("no GPU backend in this build")
        x, y = bars()
        model = classifier()
        model.fit(x, y, epochs=60, batch_size=8, verbose=0)
        # Metal and Vulkan run every layer kind; CUDA is dense-only, so libann keeps this on the CPU
        assert model._gpu_uploaded == (pyann.backend_info()["gpu_active"] in ("metal", "vulkan"))
        assert model.evaluate(x, y) >= 0.95
