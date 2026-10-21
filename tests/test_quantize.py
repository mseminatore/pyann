"""Tests for int8 quantization (Sequential.quantize, QuantizedModel) and C header export."""

import shutil
import subprocess

import pytest

import pyann
import pyann.utils.compat as compat
from pyann import Conv2D, Dense, Flatten, Input, MaxPool2D, QuantizedModel, Sequential
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


@pytest.fixture(scope="module")
def cnn():
    x, y = bars()
    model = Sequential()
    model.add(Input((1, 6, 6)))
    model.add(Conv2D(4, 3, padding="same", activation="relu"))
    model.add(MaxPool2D(2))
    model.add(Flatten())
    model.add(Dense(2, activation="softmax"))
    model.compile(optimizer="adam", loss="cce", learning_rate=0.01, convergence=0.0)
    model.fit(x, y, epochs=60, batch_size=8, verbose=0)
    return model, x, y


def blobs(n=200, seed=1):
    """Three Gaussian blobs in 4-D, one-hot labels."""
    rng = np.random.RandomState(seed)
    centers = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [1, 1, 0, 0]], dtype=np.float32)
    labels = rng.randint(0, 3, n)
    x = (centers[labels] + 0.15 * rng.randn(n, 4)).astype(np.float32)
    y = np.eye(3, dtype=np.float32)[labels]
    return x, y


@pytest.fixture(scope="module")
def mlp():
    x, y = blobs()
    model = Sequential()
    model.add(Dense(16, activation="tanh", input_shape=(4,)))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="cce", learning_rate=0.01, convergence=0.0)
    model.fit(x, y, epochs=40, batch_size=16, verbose=0)
    return model, x, y


class TestQuantize:
    def test_returns_quantized_model(self, mlp):
        model, x, _ = mlp
        q = model.quantize(x[:100])
        assert isinstance(q, QuantizedModel)
        assert (q.input_size, q.output_size) == (4, 3)
        assert q.input_scale > 0 and -128 <= q.input_zero_point <= 127
        assert q.output_scale > 0 and -128 <= q.output_zero_point <= 127

    def test_mlp_keeps_its_accuracy(self, mlp):
        model, x, y = mlp
        q = model.quantize(x)
        assert model.evaluate(x, y) > 0.95
        assert abs(q.evaluate(x, y) - model.evaluate(x, y)) <= 0.02

    def test_predictions_close_to_float(self, mlp):
        model, x, _ = mlp
        q = model.quantize(x)
        assert np.abs(q.predict(x) - model.predict(x)).max() < 0.05

    def test_cnn_takes_images(self, cnn):
        model, x, y = cnn
        q = model.quantize(x)
        assert q.evaluate(x, y) == 1.0
        assert q.predict(x[0]).shape == (1, 2)
        assert q.predict(x).shape == (len(x), 2)

    def test_model_unchanged_and_independent(self, mlp):
        model, x, _ = mlp
        before = model.predict(x)
        q = model.quantize(x)
        assert np.array_equal(model.predict(x), before)
        clone = Sequential()
        clone.add(Dense(8, activation="relu", input_shape=(4,)))
        clone.add(Dense(3))
        clone.compile(optimizer="adam", loss="mse")
        clone.fit(x, np.zeros((len(x), 3), dtype=np.float32), epochs=1, verbose=0)
        q2 = clone.quantize(x)
        out = q2.predict(x[:5])
        del clone
        assert np.array_equal(q2.predict(x[:5]), out)
        assert q.predict(x[:1]).shape == (1, 3)

    def test_untrained_model(self):
        model = Sequential()
        model.add(Dense(2, input_shape=(3,)))
        with pytest.raises(NetworkError):
            model.quantize([[0.0, 0.0, 0.0]])

    def test_wrong_calibration_shape(self, mlp):
        model, x, _ = mlp
        with pytest.raises(ValueError):
            model.quantize(x[:, :3])
        with pytest.raises(ValueError):
            model.quantize(np.zeros((0, 4), dtype=np.float32))

    def test_wrong_input_shape(self, mlp):
        model, x, y = mlp
        q = model.quantize(x)
        with pytest.raises(ValueError):
            q.predict(x[:, :3])
        with pytest.raises(ValueError):
            q.evaluate(x[:, :3], y)


class TestInt8:
    def test_int8_path_matches_float_path(self, mlp):
        """predict_int8(quantize_input(x)), dequantized and softmaxed, is predict(x)."""
        model, x, _ = mlp
        q = model.quantize(x)
        x_q = q.quantize_input(x)
        assert x_q.dtype == np.int8
        y_q = q.predict_int8(x_q)
        assert y_q.dtype == np.int8 and y_q.shape == (len(x), 3)
        logits = q.dequantize_output(y_q)
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        assert np.abs(probs - q.predict(x)).max() < 1e-5
        assert np.array_equal(y_q.argmax(axis=1), q.predict(x).argmax(axis=1))

    def test_quantize_input_rounds_like_c(self, mlp):
        """Values on a rounding half go away from zero, as in ann_rt.h."""
        model, x, _ = mlp
        q = model.quantize(x)
        halves = (np.arange(-3, 4, dtype=np.float32) + 0.5) * np.float32(q.input_scale)
        got = q.quantize_input(np.repeat(halves[:, None], 4, axis=1))[:, 0].astype(int)
        v = halves / np.float32(q.input_scale) + np.float32(q.input_zero_point)
        want = np.clip(np.where(v >= 0, np.floor(v + 0.5), -np.floor(0.5 - v)), -128, 127).astype(int)
        assert np.array_equal(got, want)

    def test_int8_range_checked(self, mlp):
        model, x, _ = mlp
        q = model.quantize(x)
        with pytest.raises(ValueError):
            q.predict_int8([[200, 0, 0, 0]])


@pytest.fixture(params=["as-installed", "numpy-disabled"])
def lists(request, monkeypatch):
    if request.param == "numpy-disabled":
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
    return request.param


class TestLists:
    def test_list_inputs(self, mlp, lists):
        model, x, y = mlp
        q = model.quantize(x.tolist())
        rows = x[:5].tolist()
        out = q.predict(rows)
        assert len(out) == 5 and len(out[0]) == 3
        x_q = q.quantize_input(rows)
        y_q = q.predict_int8(x_q)
        assert len(y_q) == 5 and all(-128 <= v <= 127 for row in y_q for v in row)
        assert q.evaluate(x.tolist(), y.tolist()) > 0.9

    def test_list_paths_agree_with_numpy(self, mlp, monkeypatch):
        model, x, _ = mlp
        q = model.quantize(x)
        want_q, want_y = q.quantize_input(x[:20]), q.predict(x[:20])
        want_i8 = q.predict_int8(want_q)
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
        rows = x[:20].tolist()
        assert q.quantize_input(rows) == want_q.tolist()
        assert q.predict_int8(want_q.tolist()) == want_i8.tolist()
        assert np.allclose(q.predict(rows), want_y, atol=0, rtol=0)


class TestExport:
    def test_int8_header_and_runtime(self, cnn, tmp_path):
        model, x, _ = cnn
        path = model.quantize(x).export_c_header(tmp_path / "bars.h")
        text = path.read_text()
        assert "#define BARS_INPUT_SIZE 36" in text
        assert "static const int8_t bars_w0[36]" in text
        assert "int bars_predict(const int8_t *input" in text
        assert (tmp_path / "ann_rt.h").read_text() == pyann.runtime_header_path().read_text()

    def test_float_header(self, cnn, tmp_path):
        model, _, _ = cnn
        model.export_c_header(tmp_path / "bars_f.h", prefix="bars_float", runtime=False)
        text = (tmp_path / "bars_f.h").read_text()
        assert "static const float bars_float_w0[36]" in text
        assert not (tmp_path / "ann_rt.h").exists()

    def test_prefix_from_file_name(self, mlp, tmp_path):
        model, x, _ = mlp
        model.quantize(x).export_c_header(tmp_path / "my-model.v2.h", runtime=False)
        assert "my_model_v2_predict" in (tmp_path / "my-model.v2.h").read_text()
        model.export_c_header(tmp_path / "9.h", runtime=False)
        assert "model_9_predict" in (tmp_path / "9.h").read_text()

    def test_bad_prefix(self, mlp, tmp_path):
        model, _, _ = mlp
        with pytest.raises(ValueError):
            model.export_c_header(tmp_path / "m.h", prefix="my-model")

    def test_unwritable_path(self, mlp, tmp_path):
        model, _, _ = mlp
        with pytest.raises(pyann.exceptions.PyANNError):
            model.export_c_header(tmp_path / "missing" / "m.h")

    def test_untrained_export(self, tmp_path):
        model = Sequential()
        model.add(Dense(2, input_shape=(3,)))
        with pytest.raises(NetworkError):
            model.export_c_header(tmp_path / "m.h")

    @pytest.mark.skipif(not shutil.which("cc"), reason="no C compiler")
    def test_header_compiles_and_matches(self, cnn, tmp_path):
        """The exported int8 header, built on its own, gives predict_int8's outputs."""
        model, x, _ = cnn
        q = model.quantize(x)
        q.export_c_header(tmp_path / "bars.h")
        x_q = q.quantize_input(x)
        rows = ",\n".join("{" + ",".join(str(v) for v in row) + "}" for row in x_q)
        (tmp_path / "main.c").write_text(
            '#include <stdio.h>\n#include "bars.h"\n'
            f"static const int8_t x[{len(x_q)}][BARS_INPUT_SIZE] = {{{rows}}};\n"
            "int main(void) {\n"
            "    static int8_t ws[BARS_WORKSPACE_SIZE];\n"
            "    int8_t y[BARS_OUTPUT_SIZE];\n"
            f"    for (int r = 0; r < {len(x_q)}; r++) {{\n"
            "        bars_predict(x[r], y, ws);\n"
            "        for (int k = 0; k < BARS_OUTPUT_SIZE; k++) printf(\"%d \", y[k]);\n"
            "        printf(\"\\n\");\n"
            "    }\n"
            "    return 0;\n"
            "}\n")
        exe = tmp_path / "main"
        subprocess.run(["cc", "-std=c99", "-Wall", "-Wextra", "-pedantic", "-Werror", "-DANN_RT_NO_FLOAT",
                        "-o", str(exe), str(tmp_path / "main.c")], check=True, cwd=tmp_path)
        out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
        got = np.array([[int(v) for v in line.split()] for line in out.splitlines()], dtype=np.int8)
        assert np.array_equal(got, q.predict_int8(x_q))
