"""Tests for non-square, grouped and depthwise convolutions, and 1-D layers for sequences."""

import pytest

import pyann
import pyann.utils.compat as compat
from pyann import (BatchNormalization, Conv1D, Conv2D, Dense, DepthwiseConv2D, GlobalAveragePooling2D, Input,
                   MaxPool1D, MaxPool2D, MaxPooling1D, Sequential)
from pyann._bindings.ffi import ffi, lib
from pyann.exceptions import NetworkError

np = pytest.importorskip("numpy")


def waves(n=48):
    """2-channel sequences of 24 steps: class 0 rises then falls, class 1 the
    reverse, at varying speeds and offsets; channel 1 is channel 0 squared."""
    x = np.zeros((n, 2, 24), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    t = np.arange(24)
    for i in range(n):
        cls = i % 2
        v = np.sin((0.15 + 0.03 * (i % 5)) * t + 0.3 * (i % 7)) * (-1.0 if cls else 1.0) + 0.05 * np.sin(7.1 * t + i)
        x[i, 0], x[i, 1] = v, v * v
        y[i, cls] = 1.0
    return x, y


def bars(n=48):
    """1x6x6 images of a horizontal (class 0) or vertical (class 1) bar, and one-hot labels."""
    rng = np.random.RandomState(0)
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


def sequence_model():
    """conv1d -> batch norm -> pool -> grouped conv1d -> global average pool -> softmax."""
    model = Sequential([
        Conv1D(8, 5, padding="same", input_shape=(2, 24)),     # -> (8, 24)
        BatchNormalization(activation="relu"),
        MaxPool1D(2),                                          # -> (8, 12)
        Conv1D(8, 3, padding=1, groups=2, activation="relu"),  # -> (8, 12)
        GlobalAveragePooling2D(),                              # -> (8,)
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce", learning_rate=0.02, convergence=0.0)
    return model


def mobilenet():
    """conv, then a depthwise separable block (depthwise, 1x1 conv), each with batch norm."""
    model = Sequential([
        Input((1, 6, 6)),
        Conv2D(4, 3, padding="same"),
        BatchNormalization(activation="relu"),
        DepthwiseConv2D(3, padding="same", depth_multiplier=2),    # -> (8, 6, 6)
        BatchNormalization(activation="relu"),
        Conv2D(8, 1),
        BatchNormalization(activation="relu"),
        GlobalAveragePooling2D(),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce", learning_rate=0.02, convergence=0.0)
    return model


def rect_model():
    """non-square conv and pool windows on 2x9x9 images."""
    model = Sequential([
        Input((2, 9, 9)),
        Conv2D(6, (2, 3), strides=(1, 2), padding=(1, 0), activation="relu"),   # -> (6, 10, 4)
        MaxPool2D((3, 1), strides=(3, 1)),                                     # -> (6, 3, 4)
        Conv2D(4, (1, 3), padding=(0, 1), groups=2, activation="tanh"),        # -> (4, 3, 4)
        Dense(3, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce", convergence=0.0)
    return model


def c_window(model, i):
    """C layer i's (kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w)."""
    w = ffi.new("int[6]")
    assert lib.ann_get_layer_window(model._ptr, i, w, w + 1, w + 2, w + 3, w + 4, w + 5) == 0
    return tuple(w[k] for k in range(6))


class TestLayerSettings:
    def test_non_square_conv(self):
        layer = Conv2D(4, (2, 3), strides=(1, 2), padding=(1, 0))
        assert (layer.kernel_size, layer.strides, layer.padding) == ((2, 3), (1, 2), (1, 0))
        assert layer.window() == (2, 3, 1, 2, 1, 0)
        assert layer.output_shape((2, 5, 9)) == (4, 6, 4)

    def test_square_settings_stay_ints(self):
        layer = Conv2D(4, (3, 3), strides=(2, 2), padding=(1, 1))
        assert (layer.kernel_size, layer.strides, layer.padding) == (3, 2, 1)

    def test_same_padding_per_axis(self):
        assert Conv2D(4, (3, 5), padding="same").padding == (1, 2)
        with pytest.raises(ValueError):
            Conv2D(4, (3, 4), padding="same")

    def test_groups(self):
        assert Conv2D(6, 3, groups=3).groups == 3
        for bad in (0, 4, True):
            with pytest.raises(ValueError):
                Conv2D(6, 3, groups=bad)

    def test_depthwise(self):
        layer = DepthwiseConv2D((3, 1), strides=(2, 1), padding="valid", depth_multiplier=2)
        assert layer.window() == (3, 1, 2, 1, 0, 0)
        assert layer.output_filters(5) == 10 and layer.conv_groups(5) == 5
        assert layer.output_shape((5, 7, 4)) == (10, 3, 4)
        with pytest.raises(ValueError):
            DepthwiseConv2D(depth_multiplier=0)
        with pytest.raises(ValueError):
            DepthwiseConv2D(activation="softmax")

    def test_conv1d(self):
        layer = Conv1D(8, 5, strides=2, padding=2)
        assert layer.window() == (1, 5, 1, 2, 0, 2)
        assert Conv1D(8, 5, padding="same").padding == 2
        assert layer.output_shape((3, 1, 20)) == (8, 1, 10)
        for kwargs in (dict(kernel_size=(1, 3)), dict(kernel_size=0), dict(padding=(1, 1)), dict(groups=3),
                       dict(kernel_size=4, padding="same")):
            with pytest.raises((ValueError, TypeError)):
                Conv1D(8, **kwargs)

    def test_pools(self):
        assert MaxPool2D((2, 3)).window() == (2, 3, 2, 3)
        assert MaxPool2D((2, 3), strides=1).strides == 1
        assert MaxPool2D((2, 3)).pool_size == (2, 3)
        assert MaxPool1D(3).window() == (1, 3, 1, 3)
        assert MaxPool1D(3, strides=1).window() == (1, 3, 1, 1)
        assert MaxPooling1D is MaxPool1D and pyann.MaxPooling1D is MaxPool1D
        with pytest.raises((ValueError, TypeError)):
            MaxPool1D((2, 2))


class TestShapes:
    def test_sequence_summary(self):
        text = sequence_model().summary()
        assert "Input shape: (2, 24)" in text
        assert "Conv1D(8, 5, stride 1, padding 2, linear) -> (8, 24) - 88 params" in text
        assert "MaxPool1D(2, stride 2) -> (8, 12)" in text
        assert "Conv1D(8, 3, stride 1, padding 1, 2 groups, relu) -> (8, 12) - 104 params" in text
        assert "GlobalAveragePooling2D -> (8,)" in text

    def test_mobilenet_summary(self):
        text = mobilenet().summary()
        assert "DepthwiseConv2D(x2, 3x3, stride 1, padding 1, linear) -> (8, 6, 6) - 80 params" in text
        assert "Conv2D(8, 1x1, stride 1, padding 0, linear) -> (8, 6, 6) - 72 params" in text

    def test_rect_summary(self):
        text = rect_model().summary()
        assert "Conv2D(6, 2x3, stride (1, 2), padding (1, 0), relu) -> (6, 10, 4) - 78 params" in text
        assert "MaxPool2D(3x1, stride (3, 1)) -> (6, 3, 4)" in text
        assert "Conv2D(4, 1x3, stride 1, padding (0, 1), 2 groups, tanh) -> (4, 3, 4) - 40 params" in text

    def test_groups_must_divide_the_channels(self):
        model = Sequential([Conv2D(4, 3, input_shape=(3, 6, 6), groups=2), Dense(2)])
        model.compile(loss="mse")
        with pytest.raises(NetworkError, match="groups"):
            model.fit(np.zeros((2, 3, 6, 6)), np.zeros((2, 2)), epochs=1, verbose=0)

    def test_c_layers(self):
        x, y = waves(8)
        model = sequence_model()
        model.fit(x, y, epochs=1, verbose=0)
        dims = ffi.new("int[3]")
        lib.ann_get_layer_shape(model._ptr, 0, dims, dims + 1, dims + 2)
        assert tuple(dims) == (2, 1, 24)
        assert c_window(model, 1) == (1, 5, 1, 1, 0, 2)
        assert c_window(model, 3) == (1, 2, 1, 2, 0, 0)
        assert lib.ann_get_layer_groups(model._ptr, 4) == 2
        bx, by = bars(8)
        mn = mobilenet()
        mn.fit(bx, by, epochs=1, verbose=0)
        assert lib.ann_get_layer_groups(mn._ptr, 3) == 4 and lib.ann_get_layer_nodes(mn._ptr, 3) == 8 * 36


class TestSequences:
    def test_learns_the_sequences(self):
        x, y = waves()
        model = sequence_model()
        model.fit(x, y, epochs=60, batch_size=8, verbose=0)
        assert model.evaluate(x, y) >= 0.95
        assert model.predict(x).shape == (len(x), 2)
        # flattened rows work too, and a single sequence
        np.testing.assert_allclose(model.predict(x.reshape(len(x), -1)), model.predict(x), atol=1e-6)
        assert model.predict(x[0]).shape == (1, 2)

    def test_sequence_outputs_come_back_as_sequences(self):
        x, _ = waves(16)
        model = Sequential([Conv1D(4, 3, padding="same", activation="tanh", input_shape=(2, 24)),
                            Conv1D(2, 3, padding="same")])
        model.compile(optimizer="adam", loss="mse", convergence=0.0)
        model.fit(x, x, epochs=3, batch_size=8, verbose=0)     # sequence targets
        assert model.predict(x).shape == (16, 2, 24)

    def test_sequence_outputs_without_numpy(self, monkeypatch):
        x, _ = waves(3)
        model = Sequential([Conv1D(2, 3, padding="same", input_shape=(2, 24))])
        model.compile(loss="mse", convergence=0.0)
        model.fit(x, x, epochs=1, verbose=0)
        out = model.predict(x)
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
        nested = model.predict(x.reshape(3, -1).tolist())
        assert len(nested) == 3 and len(nested[0]) == 2 and len(nested[0][0]) == 24
        assert np.allclose(np.array(nested, dtype=np.float32), out)


class TestMobileNet:
    def test_learns_the_bars(self):
        x, y = bars()
        model = mobilenet()
        model.fit(x, y, epochs=60, batch_size=8, verbose=0)
        assert model.evaluate(x, y) >= 0.95


class TestPersistence:
    @pytest.mark.parametrize("fmt", ["text", "binary", "onnx"])
    @pytest.mark.parametrize("which", ["sequence", "mobilenet", "rect"])
    def test_round_trip(self, tmp_path, fmt, which):
        make, (x, y) = {"sequence": (sequence_model, waves(16)), "mobilenet": (mobilenet, bars(16)),
                        "rect": (rect_model, (np.random.RandomState(2).rand(16, 2, 9, 9).astype(np.float32),
                                              np.eye(3, dtype=np.float32)[np.arange(16) % 3]))}[which]
        model = make()
        model.fit(x, y, epochs=3, batch_size=8, verbose=0)
        path = tmp_path / ("m.onnx.json" if fmt == "onnx" else f"m.{fmt}")
        if fmt == "onnx":
            model.export_onnx(path)
            loaded = Sequential.load_onnx(path)
        else:
            model.save(path, format=fmt)
            loaded = Sequential.load(path, format=fmt)
        assert [type(layer) for layer in loaded._layers] == [type(layer) for layer in model._layers]
        assert loaded._input_shape == model._input_shape
        assert loaded.summary().split("\n")[2:] == model.summary().split("\n")[2:]
        if fmt == "text":
            np.testing.assert_allclose(loaded.predict(x), model.predict(x), atol=1e-3)
        else:
            np.testing.assert_array_equal(loaded.predict(x), model.predict(x))


class TestQuantize:
    def test_sequence_model(self, tmp_path):
        x, y = waves()
        model = sequence_model()
        model.fit(x, y, epochs=30, batch_size=8, verbose=0)
        q = model.quantize(x)
        assert (q.predict(x).argmax(1) == model.predict(x).argmax(1)).mean() >= 0.95
        header = q.export_c_header(tmp_path / "waves_q.h", "waves_q", runtime=False)
        assert "ANN_RT_VERSION < 5" in header.read_text()

    def test_mobilenet(self):
        x, y = bars()
        model = mobilenet()
        model.fit(x, y, epochs=30, batch_size=8, verbose=0)
        q = model.quantize(x)
        assert (q.predict(x).argmax(1) == model.predict(x).argmax(1)).mean() >= 0.95


class TestGpu:
    @pytest.mark.parametrize("which", ["sequence", "mobilenet"])
    def test_trains_on_the_gpu(self, which):
        if not pyann.enable_gpu():
            pytest.skip("no GPU backend in this build")
        model, (x, y) = (sequence_model(), waves()) if which == "sequence" else (mobilenet(), bars())
        model.fit(x, y, epochs=60, batch_size=8, verbose=0)
        # Metal and Vulkan run every layer kind; CUDA is dense-only, so libann keeps these on the CPU
        assert model._gpu_uploaded == (pyann.backend_info()["gpu_active"] in ("metal", "vulkan"))
        assert model.evaluate(x, y) >= 0.95
