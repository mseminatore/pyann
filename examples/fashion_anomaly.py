#!/usr/bin/env python3
"""
Anomaly Detection with an Autoencoder: Dense bottleneck and Reshape

An autoencoder squeezes each 28x28 image (784 values) through a 32-value Dense
bottleneck, then Reshape turns a Dense layer back into a (8, 7, 7) image for a
convolutional decoder:

    Input(1, 28, 28) -> Conv2D 16@3x3 -> MaxPool2D 2 -> Conv2D 8@3x3 -> MaxPool2D 2
    -> Flatten -> Dense 32 (the code) -> Dense 392 -> Reshape (8, 7, 7)
    -> UpSample2D 2 -> Conv2D 16@3x3 -> UpSample2D 2 -> Conv2D 1@3x3 sigmoid

It is trained on every class except one (by default "Bag"). It learns to
rebuild the kinds of images it has seen, so images of the held-out class come
back worse: their reconstruction error flags them as anomalies, with no labels
needed at training time. Writes fashion_anomaly.png: original and
reconstructed images for each class.

Usage:
    python fashion_anomaly.py [-e EPOCHS] [--holdout CLASS] [--gpu] [--limit N]
"""

import argparse
import time

import numpy as np

from pyann import (Sequential, Input, Conv2D, MaxPool2D, Flatten, Dense,
                   Reshape, UpSample2D)
from example_data import CLASSES, as_images, load_fashion_mnist, setup_backend, write_png


def build_model(code_size: int) -> Sequential:
    model = Sequential([
        Input((1, 28, 28)),
        Conv2D(16, 3, padding="same", activation="relu"),      # -> (16, 28, 28)
        MaxPool2D(2),                                          # -> (16, 14, 14)
        Conv2D(8, 3, padding="same", activation="relu"),       # -> (8, 14, 14)
        MaxPool2D(2),                                          # -> (8, 7, 7)
        Flatten(),                                             # -> (392,)
        Dense(code_size, activation="relu"),                   # the bottleneck
        Dense(8 * 7 * 7, activation="relu"),
        Reshape((8, 7, 7)),                                    # back to an image
        UpSample2D(2),                                         # -> (8, 14, 14)
        Conv2D(16, 3, padding="same", activation="relu"),      # -> (16, 14, 14)
        UpSample2D(2),                                         # -> (16, 28, 28)
        Conv2D(1, 3, padding="same", activation="sigmoid"),    # -> (1, 28, 28)
    ])
    model.compile(optimizer="adam", loss="bce", learning_rate=0.002, convergence=0.0)
    return model


def auc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    """Chance that a random anomaly scores higher than a random normal image (ROC AUC)."""
    scores = np.concatenate([normal, anomalous])
    ranks = scores.argsort().argsort() + 1
    n, a = len(normal), len(anomalous)
    return float((ranks[n:].sum() - a * (a + 1) / 2) / (n * a))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=5, help="training epochs (default: 5)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="mini-batch size (default: 32)")
    parser.add_argument("--code", type=int, default=32, help="bottleneck size (default: 32)")
    parser.add_argument("--holdout", default="Bag", choices=CLASSES, help="class left out of training (default: Bag)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU (Metal or Vulkan) if available")
    parser.add_argument("--limit", type=int, help="use only the first N training images")
    parser.add_argument("--train-csv", help="Kaggle-format training CSV instead of the download")
    parser.add_argument("--test-csv", help="Kaggle-format test CSV instead of the download")
    parser.add_argument("-o", "--output", default="fashion_anomaly.png", help="picture to write")
    args = parser.parse_args()
    holdout = CLASSES.index(args.holdout)

    setup_backend(args.gpu)
    x_train, labels_train = load_fashion_mnist("train", args.train_csv, args.limit)
    x_test, labels_test = load_fashion_mnist("test", args.test_csv)
    x_train = as_images(x_train[labels_train != holdout])
    x_test = as_images(x_test)

    model = build_model(args.code)
    print(model.summary())
    print(f"\nEach image is compressed to {args.code} values ({784 / args.code:.0f}x smaller). "
          f"Training on {len(x_train)} images without any '{args.holdout}'\n", flush=True)

    start = time.time()
    model.fit(x_train, x_train, epochs=args.epochs, batch_size=args.batch_size)
    print(f"\nTrained in {time.time() - start:.1f} s")

    rebuilt = model.predict(x_test)
    errors = ((rebuilt - x_test) ** 2).reshape(len(x_test), -1).mean(axis=1)

    print("\nMean reconstruction error per test class:")
    per_class = [errors[labels_test == c].mean() for c in range(len(CLASSES))]
    scale = 40 / max(per_class)
    for c, name in enumerate(CLASSES):
        mark = "  <- never seen in training" if c == holdout else ""
        print(f"  {name:12s} {per_class[c]:.4f} {'#' * int(per_class[c] * scale)}{mark}")

    normal, anomalous = errors[labels_test != holdout], errors[labels_test == holdout]
    threshold = np.percentile(normal, 95)
    print(f"\nFlagging images whose error is above the 95th percentile of the known classes "
          f"({threshold:.4f}):")
    print(f"  {(anomalous > threshold).mean():.1%} of '{args.holdout}' images flagged, "
          f"{(normal > threshold).mean():.1%} of the others")
    print(f"  ROC AUC of reconstruction error as an anomaly score: {auc(normal, anomalous):.3f}")

    picks = [int(np.argmax(labels_test == c)) for c in range(len(CLASSES))]
    write_png(args.output, [[x_test[i] for i in picks], [rebuilt[i] for i in picks]])
    print(f"\nWrote {args.output} (rows: original, reconstructed; one image per class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
