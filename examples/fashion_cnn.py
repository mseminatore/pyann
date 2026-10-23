#!/usr/bin/env python3
"""
Fashion-MNIST CNN: convolutional image classification

Trains a small convolutional network on 28x28 grayscale clothing images:

    Input(1, 28, 28) -> Conv2D 8@3x3 relu -> MaxPool2D 2 -> Conv2D 16@3x3 relu
    -> MaxPool2D 2 -> Flatten -> Dense 64 relu -> Dense 10 softmax

It holds out 5,000 training images for validation, uses early stopping and an
exponentially decaying learning rate, and reports test accuracy per class and
the most common confusions. The trained model is saved (fashion_cnn.nna),
exported to ONNX JSON and reloaded to check it predicts the same, and drawn as
a PIKCHR diagram. 10 epochs reach about 89% test accuracy (the MLP in
mnist_fashion.py gets about 88%). Mirrors libann's examples/fashion_cnn.c.

Usage:
    python fashion_cnn.py [-e EPOCHS] [-b BATCH] [--gpu] [--limit N]
"""

import argparse
import time

import numpy as np

from pyann import Sequential, Input, Conv2D, MaxPool2D, Flatten, Dense, Adam
from pyann.callbacks import EarlyStopping, ExponentialDecay
from example_data import CLASSES, as_images, load_fashion_mnist, one_hot, setup_backend


def build_model() -> Sequential:
    model = Sequential([
        Input((1, 28, 28)),
        Conv2D(8, 3, activation="relu"),        # -> (8, 26, 26)
        MaxPool2D(2),                           # -> (8, 13, 13)
        Conv2D(16, 3, activation="relu"),       # -> (16, 11, 11)
        MaxPool2D(2),                           # -> (16, 5, 5)
        Flatten(),                              # -> (400,)
        Dense(64, activation="relu"),
        Dense(10, activation="softmax"),
    ], optimizer=Adam(lr=0.001), loss="cce")
    model.compile(convergence=0.0)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=10, help="maximum epochs (default: 10)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="mini-batch size (default: 32)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU (Metal or Vulkan) if available")
    parser.add_argument("--limit", type=int, help="use only the first N training images")
    parser.add_argument("--train-csv", help="Kaggle-format training CSV instead of the download")
    parser.add_argument("--test-csv", help="Kaggle-format test CSV instead of the download")
    args = parser.parse_args()

    setup_backend(args.gpu)
    x, labels = load_fashion_mnist("train", args.train_csv, args.limit)
    x_test, labels_test = load_fashion_mnist("test", args.test_csv)

    # images are channels-first: (N, 1, 28, 28)
    x, x_test = as_images(x), as_images(x_test)
    y, y_test = one_hot(labels), one_hot(labels_test)
    n_val = min(5000, len(x) // 10)
    x_train, y_train, x_val, y_val = x[n_val:], y[n_val:], x[:n_val], y[:n_val]
    print(f"Training on {len(x_train)} images, validating on {n_val}, testing on {len(x_test)}\n")

    model = build_model()
    print(model.summary() + "\n", flush=True)    # flush before libann prints progress

    start = time.time()
    history = model.fit(
        x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
        validation_data=(x_val, y_val),
        callbacks=[EarlyStopping(patience=3, restore_best=True)],
        lr_scheduler=ExponentialDecay(gamma=0.9),
    )
    print(f"\nTrained in {time.time() - start:.1f} s; best epoch {history['best_epoch']}, "
          f"validation accuracy {history['val_accuracy']:.2%}")

    predictions = model.predict(x_test)
    predicted = predictions.argmax(axis=1)
    print(f"\nTest accuracy: {model.evaluate(x_test, y_test):.2%}\n")

    print("Per-class accuracy:")
    for c, name in enumerate(CLASSES):
        mask = labels_test == c
        acc = (predicted[mask] == c).mean()
        print(f"  {name:12s} {acc:6.1%} {'#' * int(acc * 40)}")

    confusions = np.zeros((10, 10), dtype=int)
    np.add.at(confusions, (labels_test, predicted), 1)
    np.fill_diagonal(confusions, 0)
    print("\nMost common mistakes:")
    for flat in np.argsort(confusions, axis=None)[::-1][:3]:
        true, guess = divmod(int(flat), 10)
        print(f"  {confusions[true, guess]:4d} x {CLASSES[true]} taken for {CLASSES[guess]}")

    model.save("fashion_cnn.nna")
    model.export_onnx("fashion_cnn.onnx.json")
    model.export_pikchr("fashion_cnn.pikchr")
    reloaded = Sequential.load_onnx("fashion_cnn.onnx.json")
    same = np.allclose(reloaded.predict(x_test[:100]), predictions[:100], atol=1e-5)
    print("\nSaved fashion_cnn.nna, fashion_cnn.onnx.json and fashion_cnn.pikchr")
    print(f"ONNX round trip predicts the same: {'yes' if same else 'NO'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
