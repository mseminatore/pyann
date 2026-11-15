"""Tests for low-bit weights (Sequential.set_weight_format, WeightFormat)
and quantization-aware training."""

import pytest

import pyann
from pyann import Conv2D, Dense, Flatten, Input, MaxPool2D, Sequential, WeightFormat
from pyann.exceptions import InvalidParameterError

np = pytest.importorskip("numpy")


def classes(n=384, features=24, k=6, seed=3):
    """k classes around fixed prototypes plus noise; one-hot labels."""
    rng = np.random.RandomState(seed)
    protos = np.sin(1.3 * np.outer(np.arange(k), np.arange(features)) + np.arange(k)[:, None])
    labels = np.arange(n) % k
    x = (protos[labels] + 0.8 * (rng.rand(n, features) - 0.5)).astype(np.float32)
    y = np.eye(k, dtype=np.float32)[labels]
    return x, y


def mlp(features=24, hidden=24, k=6):
    model = Sequential()
    model.add(Dense(hidden, activation="relu", input_shape=(features,)))
    model.add(Dense(k, activation="softmax"))
    model.compile(optimizer="adam", loss="cce", learning_rate=0.005, convergence=0.0)
    return model


def rounded(w, fmt):
    """Weight_format's rules, row by row, written out independently of libann."""
    w = w.astype(np.float64)
    out = np.empty_like(w)
    for r, row in enumerate(w):
        a = np.abs(row)
        if fmt == "int4":
            s = a.max() / 7
            out[r] = s * np.clip(np.floor(row / s + 0.5), -7, 7) if s > 0 else 0
        elif fmt == "ternary":
            keep = a > 0.7 * a.mean()
            s = a[keep].mean() if keep.any() else 0.0
            out[r] = np.where(keep, np.sign(row) * s, 0.0)
        else:
            out[r] = np.where(row >= 0, 1.0, -1.0) * a.mean()
    return out.astype(np.float32)


def forward(weights, x):
    """NumPy forward pass of a relu -> softmax MLP from (W, b) pairs."""
    (w1, b1), (w2, b2) = weights
    h = np.maximum(x @ w1.T + b1, 0)
    z = h @ w2.T + b2
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def trained():
    x, y = classes()
    pyann.set_seed(12)
    model = mlp()
    model.fit(x, y, epochs=30, batch_size=16, verbose=0)
    return model, x, y


class TestWeightFormat:
    def test_values_match_libann(self):
        from pyann._bindings.ffi import lib
        assert WeightFormat.FLOAT == lib.WEIGHT_FORMAT_FLOAT
        assert WeightFormat.INT4 == lib.WEIGHT_FORMAT_INT4
        assert WeightFormat.TERNARY == lib.WEIGHT_FORMAT_TERNARY
        assert WeightFormat.BINARY == lib.WEIGHT_FORMAT_BINARY

    def test_names_and_bits(self):
        assert WeightFormat.from_string("Ternary") == WeightFormat.TERNARY
        assert [f.bits for f in WeightFormat] == [8, 4, 2, 1]
        with pytest.raises(ValueError, match="Valid options"):
            WeightFormat.from_string("int8")


class TestSetWeightFormat:
    def test_default_sets_hidden_dense_and_conv_layers(self):
        model = Sequential([Input((1, 6, 6)), Conv2D(4, 3, activation="relu"), MaxPool2D(2), Flatten(),
                            Dense(8, activation="relu"), Dense(2, activation="softmax")])
        model.compile(optimizer="adam", loss="cce")
        model.set_weight_format("binary")
        assert [model.get_weight_format(i) for i in range(5)] == [
            WeightFormat.BINARY, WeightFormat.FLOAT, WeightFormat.FLOAT, WeightFormat.BINARY, WeightFormat.FLOAT]
        assert "(binary weights)" in model.summary()

    def test_one_layer_by_index_or_object(self):
        model = mlp()
        model.set_weight_format(WeightFormat.INT4, layer=-1)
        model.set_weight_format("ternary", layer=model.layers[0])
        assert model.get_weight_format(0) == WeightFormat.TERNARY
        assert model.get_weight_format(1) == WeightFormat.INT4
        model.set_weight_format("float")
        assert model.get_weight_format(0) == WeightFormat.FLOAT
        assert model.get_weight_format(1) == WeightFormat.INT4

    def test_rejects_layers_without_weights_and_bad_formats(self):
        model = Sequential([Input((1, 6, 6)), Conv2D(2, 3), Flatten(), Dense(2)])
        model.compile(optimizer="adam", loss="mse")
        with pytest.raises(InvalidParameterError, match="Flatten"):
            model.set_weight_format("binary", layer=1)
        with pytest.raises(ValueError):
            model.set_weight_format("int2")
        with pytest.raises(TypeError):
            model.set_weight_format(3.0)


class TestRoundedWeights:
    @pytest.mark.parametrize("fmt", ["int4", "ternary", "binary"])
    def test_predict_uses_rounded_weights(self, trained, fmt):
        model, x, _ = trained
        latent = model.get_weights()
        model.set_weight_format(fmt, layer=0)
        model.set_weight_format(fmt, layer=1)
        try:
            # get_weights still reads the float weights underneath
            for a, b in zip(model.get_weights(), latent):
                np.testing.assert_array_equal(a, b)
            expected = forward([(rounded(latent[0], fmt), latent[1]), (rounded(latent[2], fmt), latent[3])], x)
            np.testing.assert_allclose(model.predict(x), expected, atol=1e-5)
        finally:
            model.set_weight_format("float", layer=0)
            model.set_weight_format("float", layer=1)
        np.testing.assert_allclose(model.predict(x), forward([(latent[0], latent[1]), (latent[2], latent[3])], x),
                                   atol=1e-5)


class TestQuantizationAwareTraining:
    def test_fine_tuning_recovers_accuracy(self, trained):
        model, x, y = trained
        before = model.get_weights()
        model.set_weight_format("ternary")
        rounded_acc = model.evaluate(x, y)
        try:
            for _ in range(15):
                for i in range(0, len(x), 16):
                    model.train_on_batch(x[i:i + 16], y[i:i + 16])
            model.end_online_training()
            tuned = model.evaluate(x, y)
            assert tuned >= 0.95 and tuned >= rounded_acc
            assert model.quantize(x[:200]).evaluate(x, y) >= tuned - 0.03
        finally:
            model.set_weight_format("float")
            model.set_weights(before)

    def test_training_from_the_start(self):
        x, y = classes()
        pyann.set_seed(11)
        model = mlp()
        model.set_weight_format("binary")
        model.fit(x, y, epochs=40, batch_size=16, verbose=0)
        assert model.evaluate(x, y) >= 0.95


class TestQuantizeAndExport:
    def test_packed_sizes(self, trained):
        model, x, _ = trained
        sizes = {}
        try:
            for fmt in ["float", "int4", "ternary", "binary"]:
                model.set_weight_format(fmt)
                sizes[fmt] = model.quantize(x[:100]).param_bytes
        finally:
            model.set_weight_format("float")
        # hidden layer: 24 rows of 24 weights; the rest (biases, multipliers, shifts, output layer) is shared
        rest = 24 * 9 + 6 * 24 + 6 * 9
        assert sizes == {"float": 24 * 24 + rest, "int4": 24 * 12 + rest, "ternary": 24 * 6 + rest,
                         "binary": 24 * 3 + rest}

    def test_save_and_load_keep_formats(self, trained, tmp_path):
        model, x, _ = trained
        model.set_weight_format("binary")
        try:
            for fmt in ["text", "binary"]:
                path = tmp_path / f"m.{fmt}"
                model.save(path, format=fmt)
                loaded = Sequential.load(path, format=fmt)
                assert loaded.get_weight_format(0) == WeightFormat.BINARY
                np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
        finally:
            model.set_weight_format("float")

    def test_c_and_js_export(self, trained, tmp_path):
        model, x, _ = trained
        model.set_weight_format("ternary")
        try:
            header = model.quantize(x[:100]).export_c_header(tmp_path / "lb_q.h")
            text = header.read_text()
            assert "2-bit weights, packed" in text
            assert ", 2 }," in text           # the hidden layer's w_bits
            model.export_js(tmp_path / "lb.js")
            assert (tmp_path / "lb.js").exists()
        finally:
            model.set_weight_format("float")
