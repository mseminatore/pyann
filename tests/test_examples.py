"""Smoke tests: every example in examples/ runs to completion.

The Fashion-MNIST examples get small synthetic CSV files in the Kaggle format,
so nothing is downloaded, and short settings keep each run to a few seconds.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
pytestmark = pytest.mark.skipif(not EXAMPLES.is_dir(), reason="examples/ not available")


@pytest.fixture(scope="module")
def fashion_csv(tmp_path_factory):
    """Tiny train and test CSVs: label first, 784 pixels, with a header row."""
    folder = tmp_path_factory.mktemp("fashion")
    rng = np.random.default_rng(0)
    paths = []
    for name, rows in (("train.csv", 2400), ("test.csv", 200)):
        labels = np.arange(rows) % 10
        pixels = rng.integers(0, 256, size=(rows, 784))
        data = np.column_stack([labels, pixels])
        path = folder / name
        header = ",".join(["label"] + [f"pixel{i}" for i in range(784)])
        np.savetxt(path, data, fmt="%d", delimiter=",", header=header, comments="")
        paths.append(str(path))
    return paths


def run_example(script, *args, cwd):
    result = subprocess.run(
        [sys.executable, str(EXAMPLES / script), *args],
        cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, f"{script} failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


def fashion_args(fashion_csv):
    return ["--train-csv", fashion_csv[0], "--test-csv", fashion_csv[1]]


class TestStandaloneExamples:
    def test_logic_gates(self, tmp_path):
        out = run_example("logic_gates.py", cwd=tmp_path)
        assert "XOR" in out
        assert (tmp_path / "xor.pikchr").exists()

    def test_digits_5x7(self, tmp_path):
        out = run_example("digits_5x7.py", "-e", "5", "--trials", "20", cwd=tmp_path)
        assert "noise-trained" in out

    def test_regression_activations(self, tmp_path):
        out = run_example("regression_activations.py", "-e", "20", cwd=tmp_path)
        assert "gelu" in out

    def test_online_learning(self, tmp_path):
        out = run_example("online_learning.py", "--batches", "40", cwd=tmp_path)
        assert "saved and reloaded" in out


class TestFashionExamples:
    def test_mnist_fashion(self, tmp_path, fashion_csv):
        out = run_example("mnist_fashion.py", *fashion_csv, "-e", "1", cwd=tmp_path)
        assert "Test accuracy" in out

    def test_fashion_cnn(self, tmp_path, fashion_csv):
        out = run_example("fashion_cnn.py", "-e", "1", *fashion_args(fashion_csv), cwd=tmp_path)
        assert "ONNX round trip predicts the same: yes" in out

    def test_fashion_denoise(self, tmp_path, fashion_csv):
        run_example("fashion_denoise.py", "-e", "1", "-n", "salt_pepper", "-a", "0.1",
                    *fashion_args(fashion_csv), cwd=tmp_path)
        assert (tmp_path / "fashion_denoise.png").read_bytes().startswith(b"\x89PNG")

    def test_fashion_anomaly(self, tmp_path, fashion_csv):
        out = run_example("fashion_anomaly.py", "-e", "1", *fashion_args(fashion_csv), cwd=tmp_path)
        assert "ROC AUC" in out

    def test_deploy_int8(self, tmp_path, fashion_csv):
        run_example("deploy_int8.py", "-e", "1", "--no-c", *fashion_args(fashion_csv), cwd=tmp_path)
        assert (tmp_path / "fashion_deploy" / "fashion_q.h").exists()
        assert (tmp_path / "fashion_deploy" / "ann_rt.h").exists()

    @pytest.mark.skipif(os.name == "nt" or not shutil.which("cc"), reason="needs a Unix C compiler")
    def test_deploy_int8_runs_the_header_in_c(self, tmp_path, fashion_csv):
        out = run_example("deploy_int8.py", "-e", "1", *fashion_args(fashion_csv), cwd=tmp_path)
        assert "bit for bit: yes" in out

    def test_hyperparameter_search(self, tmp_path, fashion_csv):
        out = run_example("hyperparameter_search.py", "--trials", "3", "--epochs", "1",
                          *fashion_args(fashion_csv), cwd=tmp_path)
        assert "Test accuracy" in out
