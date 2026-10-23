#!/usr/bin/env python3
"""
Deploying to a Microcontroller: int8 quantization and C header export

Trains a Fashion-MNIST classifier (an MLP, or the CNN from fashion_cnn.py
with --cnn), quantizes it to int8 using 1,000 training images to calibrate,
and compares float32 and int8 test accuracy. Then exports both as standalone
C headers (fashion_q.h, int8; fashion_f.h, float32) that need only ann_rt.h:
no libann, no malloc, integer-only math for int8.

If a C compiler is found (cc, gcc, clang, or $CC), the example also builds a
small C program from the int8 header, runs it on the test images, and checks
that its outputs match QuantizedModel.predict_int8() bit for bit. Mirrors
libann's examples/export_model.c and run_exported.c.

Usage:
    python deploy_int8.py [--cnn] [-e EPOCHS] [--gpu] [--out DIR] [--no-c]
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from pyann import Sequential, Input, Conv2D, MaxPool2D, Flatten, Dense
from example_data import as_images, load_fashion_mnist, one_hot, setup_backend

C_PROGRAM = r"""
/* Runs fashion_q.h on raw int8 inputs: inputs.bin -> outputs.bin */
#include <stdio.h>
#include "fashion_q.h"

static int8_t input[FASHION_Q_INPUT_SIZE];
static int8_t output[FASHION_Q_OUTPUT_SIZE];
static int8_t workspace[FASHION_Q_WORKSPACE_SIZE];

int main(void)
{
    FILE *in = fopen("inputs.bin", "rb"), *out = fopen("outputs.bin", "wb");
    long rows = 0;
    if (!in || !out)
        return 1;
    while (fread(input, 1, sizeof(input), in) == sizeof(input))
    {
        if (fashion_q_predict(input, output, workspace) != 0)
            return 2;
        fwrite(output, 1, sizeof(output), out);
        rows++;
    }
    fclose(in);
    fclose(out);
    printf("C program: ran %ld images through fashion_q.h\n", rows);
    return 0;
}
"""


def build_model(cnn: bool) -> Sequential:
    if cnn:
        layers = [Input((1, 28, 28)), Conv2D(8, 3, activation="relu"), MaxPool2D(2),
                  Conv2D(16, 3, activation="relu"), MaxPool2D(2), Flatten(),
                  Dense(64, activation="relu")]
    else:
        layers = [Input((784,)), Dense(128, activation="relu")]
    model = Sequential(layers + [Dense(10, activation="softmax")])
    model.compile(optimizer="adam", loss="cce", convergence=0.0)
    return model


def find_compiler():
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and shutil.which(name):
            return name
    return None


def run_in_c(q, x_test: np.ndarray, out_dir: Path):
    """Compile the int8 header into a C program, run it, and return its outputs."""
    compiler = find_compiler()
    if compiler is None:
        print("\nNo C compiler found (set CC to use one), skipping the C check")
        return None
    (out_dir / "run_model.c").write_text(C_PROGRAM)
    exe = (out_dir / ("run_model.exe" if os.name == "nt" else "run_model")).resolve()
    subprocess.run([compiler, "-O2", "-std=c99", "-o", exe.name, "run_model.c"], cwd=out_dir, check=True)
    (out_dir / "inputs.bin").write_bytes(q.quantize_input(x_test).tobytes())
    subprocess.run([str(exe)], cwd=out_dir, check=True)
    return np.frombuffer((out_dir / "outputs.bin").read_bytes(), dtype=np.int8).reshape(len(x_test), -1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--cnn", action="store_true", help="use the convolutional network instead of the MLP")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="training epochs (default: 5)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU (Metal or Vulkan) if available")
    parser.add_argument("--limit", type=int, help="use only the first N training images")
    parser.add_argument("--no-c", action="store_true", help="skip compiling and running the C program")
    parser.add_argument("--out", default="fashion_deploy", help="directory for the headers (default: fashion_deploy)")
    parser.add_argument("--train-csv", help="Kaggle-format training CSV instead of the download")
    parser.add_argument("--test-csv", help="Kaggle-format test CSV instead of the download")
    args = parser.parse_args()

    setup_backend(args.gpu)
    x_train, labels_train = load_fashion_mnist("train", args.train_csv, args.limit)
    x_test, labels_test = load_fashion_mnist("test", args.test_csv)
    if args.cnn:
        x_train, x_test = as_images(x_train), as_images(x_test)
    y_train, y_test = one_hot(labels_train), one_hot(labels_test)

    model = build_model(args.cnn)
    print(model.summary() + "\n", flush=True)
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=32, verbose=0)

    # calibrate the int8 activation ranges on representative inputs
    q = model.quantize(x_train[:1000])
    acc_f, acc_q = model.evaluate(x_test, y_test), q.evaluate(x_test, y_test)
    print(f"Test accuracy: float32 {acc_f:.2%}, int8 {acc_q:.2%} ({(acc_q - acc_f) * 100:+.2f} points)")
    print(f"int8 input: x_q = round(x / {q.input_scale:.6f}) + {q.input_zero_point}")

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    q.export_c_header(out_dir / "fashion_q.h")         # also copies ann_rt.h next to it
    model.export_c_header(out_dir / "fashion_f.h")
    print(f"\nWrote {out_dir / 'ann_rt.h'} and:")
    for name in ("fashion_q.h", "fashion_f.h"):
        # the header's opening comment describes the model and its parameter memory
        summary = (out_dir / name).read_text().splitlines()[2].lstrip(" *")
        print(f"  {out_dir / name}: {summary}")
    print(flush=True)

    outputs = None if args.no_c else run_in_c(q, x_test, out_dir)
    if outputs is not None:
        expected = q.predict_int8(q.quantize_input(x_test))
        c_acc = (outputs.argmax(axis=1) == labels_test).mean()
        print(f"C program accuracy: {c_acc:.2%}")
        print(f"Matches QuantizedModel.predict_int8() bit for bit: "
              f"{'yes' if np.array_equal(outputs, expected) else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
