#!/usr/bin/env python3
"""
Activity Recognition: a Conv1D network on accelerometer time series

A wrist sensor samples 3-axis acceleration at 32 Hz; each 2-second window of
64 steps is one sample, shaped (3 channels, 64 steps), channels-first as in
PyTorch's nn.Conv1d. Four activities make the signals: resting, walking,
running and jumping, each with its own rhythm and strength, at a random
phase, speed and wrist orientation, plus sensor noise. The data is
synthetic, so nothing is downloaded.

Two models learn the same data:

  - a dense MLP on the flattened window, which has to learn every phase of
    every rhythm separately
  - a 1-D CNN: Conv1D + BatchNormalization, MaxPool1D, Conv1D, then
    GlobalAveragePooling2D, so a pattern counts wherever it falls in the
    window, with far fewer weights

The CNN is then quantized to int8 for a microcontroller, exported as a C
header, and round-tripped through ONNX.

Usage:
    python activity_recognition.py [--epochs N] [--samples N] [--gpu]
"""

import argparse
import os
import tempfile

import numpy as np

from pyann import (Sequential, Conv1D, BatchNormalization, MaxPool1D, GlobalAveragePooling2D, Dense, Flatten,
                   Input)
from example_data import setup_backend

ACTIVITIES = ["resting", "walking", "running", "jumping"]
RATE, STEPS = 32.0, 64          # Hz, steps per window (2 s)


def make_windows(rng: np.random.Generator, per_class: int):
    """per_class windows of each activity: x (N, 3, 64) in g, y (N, 4) one-hot."""
    t = np.arange(STEPS) / RATE
    x, labels = [], []
    for label in range(len(ACTIVITIES)):
        for _ in range(per_class):
            phase = rng.uniform(0, 2 * np.pi)
            if label == 0:      # resting: gravity and a little tremor
                signal = 0.03 * np.sin(2 * np.pi * rng.uniform(4, 8) * t + phase)
                sway = 0.02 * np.sin(2 * np.pi * 0.3 * t + phase)
            elif label == 1:    # walking: about 2 steps a second, smooth
                f = rng.uniform(1.4, 2.6)
                signal = rng.uniform(0.2, 0.6) * np.sin(2 * np.pi * f * t + phase)
                sway = 0.15 * np.sin(np.pi * f * t + phase)
            elif label == 2:    # running: as fast or a little faster, with a sharp heel strike
                f = rng.uniform(2.0, 3.2)
                s = np.sin(2 * np.pi * f * t + phase)
                signal = rng.uniform(0.3, 0.7) * (s + 0.5 * np.sin(4 * np.pi * f * t + 2 * phase))
                sway = 0.3 * np.sin(np.pi * f * t + phase)
            else:               # jumping: about one landing a second, a spike then a lull
                f = rng.uniform(0.8, 1.3)
                s = np.sin(2 * np.pi * f * t + phase)
                signal = rng.uniform(1.0, 1.5) * np.maximum(s, 0.0) ** 6 - 0.2
                sway = 0.05 * s
            vertical = 1.0 + signal                     # gravity plus the motion
            axes = np.stack([sway, 0.5 * sway[::-1], vertical])
            # the wrist is never held the same way: a random small rotation mixes the axes
            a, b = rng.uniform(-0.5, 0.5, size=2)
            rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]) @ \
                  np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
            x.append(rot @ axes + rng.normal(0, 0.15, size=(3, STEPS)))
            labels.append(label)
    x = np.array(x, dtype=np.float32)
    y = np.eye(len(ACTIVITIES), dtype=np.float32)[labels]
    order = rng.permutation(len(x))
    return x[order], y[order]


def build_mlp() -> Sequential:
    model = Sequential([
        Input((3 * STEPS,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(len(ACTIVITIES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce", learning_rate=0.002, convergence=0.0)
    return model


def build_cnn() -> Sequential:
    model = Sequential([
        Conv1D(16, 7, padding="same", input_shape=(3, STEPS)),    # -> (16, 64)
        BatchNormalization(activation="relu"),
        MaxPool1D(2),                                             # -> (16, 32)
        Conv1D(24, 5, padding="same"),                            # -> (24, 32)
        BatchNormalization(activation="relu"),
        GlobalAveragePooling2D(),                                 # -> (24,): wherever a pattern fell
        Dense(len(ACTIVITIES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce", learning_rate=0.005, convergence=0.0)
    return model


def confusion(truth: np.ndarray, guess: np.ndarray) -> str:
    width = max(len(a) for a in ACTIVITIES)
    lines = [" " * (width + 2) + " ".join(f"{a[:7]:>7s}" for a in ACTIVITIES)]
    for i, name in enumerate(ACTIVITIES):
        counts = [(guess[truth == i] == j).sum() for j in range(len(ACTIVITIES))]
        lines.append(f"{name:>{width}s}  " + " ".join(f"{c:7d}" for c in counts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=30, help="training epochs (default: 30)")
    parser.add_argument("--samples", type=int, default=150, help="training windows per activity (default: 150)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU when available")
    args = parser.parse_args()
    setup_backend(args.gpu)

    rng = np.random.default_rng(7)
    x_train, y_train = make_windows(rng, args.samples)
    x_test, y_test = make_windows(rng, 250)
    truth = y_test.argmax(axis=1)
    print(f"{len(x_train)} training and {len(x_test)} test windows of shape {x_train.shape[1:]} "
          f"(channels, steps)\n")

    results = {}
    for name, build in (("dense MLP", build_mlp), ("1-D CNN", build_cnn)):
        model = build()
        data = x_train.reshape(len(x_train), -1) if name == "dense MLP" else x_train
        model.fit(data, y_train, epochs=args.epochs, batch_size=32, verbose=0)
        test = x_test.reshape(len(x_test), -1) if name == "dense MLP" else x_test
        params = int(model.summary().splitlines()[-1].split(":")[1].replace(",", ""))
        results[name] = (model, model.evaluate(test, y_test), params)
        print(f"{name:>9s}: {results[name][1]:6.1%} test accuracy with {params:,} parameters")

    cnn, accuracy, _ = results["1-D CNN"]
    print(f"\n{cnn.summary()}\n")
    print("1-D CNN confusion matrix (rows: true activity, columns: predicted):")
    print(confusion(truth, cnn.predict(x_test).argmax(axis=1)))

    # int8 for a microcontroller: batch norms fold into the convs before them
    q = cnn.quantize(x_train[:500])
    print(f"\nint8 test accuracy: {q.evaluate(x_test, y_test):.1%} (float32 {accuracy:.1%})")
    out_dir = "activity_deploy"
    os.makedirs(out_dir, exist_ok=True)
    header = q.export_c_header(os.path.join(out_dir, "activity_q.h"), "activity")
    print(f"C header: {header} ({os.path.getsize(header) / 1024:.1f} KB, with ann_rt.h beside it)")

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "activity.onnx.json")
        cnn.export_onnx(path)
        back = Sequential.load_onnx(path)
        same = np.array_equal(back.predict(x_test), cnn.predict(x_test))
    print(f"ONNX round trip predicts the same: {'yes' if same else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
