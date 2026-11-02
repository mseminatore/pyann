"""Tests for Sequential.export_js: JavaScript modules run with libann's ann_rt.js.

The Node tests run each exported module and compare its outputs with
predict(); they are skipped where Node is not installed.
"""

import json
import shutil
import subprocess

import pytest

from pyann import (BatchNormalization, Conv1D, Conv2D, Dense, DepthwiseConv2D, Embedding, Flatten,
                   GlobalAveragePooling2D, GlobalMaxPooling1D, Input, MaxPool2D, Sequential, UpSample2D)
from pyann.core.quantized import runtime_path
from pyann.exceptions import NetworkError

np = pytest.importorskip("numpy")
NODE = shutil.which("node")

# imports the module named on the command line, predicts each row of the JSON
# inputs file and prints the outputs as JSON
RUNNER = """
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
const mod = await import(pathToFileURL(process.argv[2]).href);
const rows = JSON.parse(readFileSync(process.argv[3], "utf8"));
console.log(JSON.stringify(rows.map(r => Array.from(mod.predict(r)))));
"""


def run_in_node(module, x, tmp_path):
    (tmp_path / "runner.mjs").write_text(RUNNER)
    (tmp_path / "inputs.json").write_text(json.dumps(np.asarray(x, dtype=float).reshape(len(x), -1).tolist()))
    result = subprocess.run([NODE, str(tmp_path / "runner.mjs"), str(module), str(tmp_path / "inputs.json")],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return np.array(json.loads(result.stdout), dtype=np.float32)


def trained(model, x, y, epochs=3):
    model.fit(x, y, epochs=epochs, batch_size=16, verbose=0)
    return model


def one_hot(labels, classes):
    return np.eye(classes, dtype=np.float32)[labels]


class TestExportJs:
    def test_writes_module_and_runtime(self, tmp_path):
        rng = np.random.default_rng(0)
        model = trained(Sequential([Input((4,)), Dense(3, activation="softmax")], loss="cce"),
                        rng.random((32, 4)).astype(np.float32), one_hot(rng.integers(0, 3, 32), 3))
        path = model.export_js(tmp_path / "tiny.js")
        assert path == tmp_path / "tiny.js"
        text = path.read_text()
        assert 'from "./ann_rt.js"' in text and 'import { predict, argmax } from "./tiny.js"' in text
        assert "4 inputs, 3 outputs (softmax)" in text
        assert (tmp_path / "ann_rt.js").read_text() == runtime_path("ann_rt.js").read_text()

    def test_runtime_false_skips_the_runtime(self, tmp_path):
        model = Sequential([Input((2,)), Dense(1)])
        model.get_weights()                          # builds it
        model.export_js(tmp_path / "m.js", runtime=False)
        assert (tmp_path / "m.js").exists() and not (tmp_path / "ann_rt.js").exists()

    def test_unbuilt_model_is_rejected(self, tmp_path):
        with pytest.raises(NetworkError):
            Sequential([Input((2,)), Dense(1)]).export_js(tmp_path / "m.js")


@pytest.mark.skipif(NODE is None, reason="needs Node")
class TestExportJsInNode:
    """Exported modules give predict()'s outputs, to float32 rounding."""

    def check(self, model, x, tmp_path):
        out = run_in_node(model.export_js(tmp_path / "model.js"), x, tmp_path)
        np.testing.assert_allclose(out, model.predict(x).reshape(len(x), -1), rtol=1e-4, atol=1e-6)

    def test_dense(self, tmp_path):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(64, 6)).astype(np.float32)
        model = Sequential([Input((6,)), Dense(8, activation="leaky_relu"), Dense(5, activation="softsign"),
                            Dense(4, activation="gelu"), Dense(3, activation="softmax")], loss="cce")
        self.check(trained(model, x, one_hot(rng.integers(0, 3, 64), 3)), x[:8], tmp_path)

    def test_cnn(self, tmp_path):
        # batch norm (folded), depthwise and pointwise convs, pooling, upsampling, global pooling
        rng = np.random.default_rng(2)
        x = rng.random((48, 2, 8, 8)).astype(np.float32)
        model = Sequential([
            Input((2, 8, 8)),
            Conv2D(4, 3, padding="same"), BatchNormalization(activation="relu"),
            MaxPool2D(2),
            DepthwiseConv2D(3, padding="same", activation="tanh"), Conv2D(6, 1, activation="sigmoid"),
            UpSample2D(2),
            Conv2D(3, (3, 2), strides=(2, 1), activation="relu"),
            GlobalAveragePooling2D(),
            Dense(3, activation="softmax"),
        ], loss="cce")
        self.check(trained(model, x, one_hot(rng.integers(0, 3, 48), 3)), x[:6], tmp_path)

    def test_image_output(self, tmp_path):
        rng = np.random.default_rng(3)
        x = rng.random((32, 1, 6, 6)).astype(np.float32)
        model = Sequential([Input((1, 6, 6)), Conv2D(2, 3, padding="same", activation="relu"), Flatten(),
                            Dense(36, activation="sigmoid")])
        self.check(trained(model, x, x.reshape(32, -1)), x[:4], tmp_path)

    def test_masked_text_model(self, tmp_path):
        # token ids, padded with 0 at the end; the padding is masked out
        rng = np.random.default_rng(4)
        ids = np.zeros((64, 10), dtype=np.float32)
        for row, length in zip(ids, rng.integers(2, 11, 64)):
            row[:length] = rng.integers(1, 30, length)
        y = one_hot((ids == 3).any(axis=1).astype(int), 2)
        model = Sequential([Input((10,)), Embedding(30, 4, mask_zero=True),
                            Conv1D(5, 3, padding="same", activation="relu"), GlobalMaxPooling1D(),
                            Dense(2, activation="softmax")], loss="cce")
        self.check(trained(model, ids, y), ids[:8], tmp_path)
