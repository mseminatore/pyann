#!/usr/bin/env python3
"""
Fashion-MNIST Denoising Autoencoder: Conv2D, MaxPool2D and UpSample2D

A convolutional autoencoder learns to output clean images from noisy ones:

    Input(1, 28, 28) -> Conv2D 16@3x3 relu -> MaxPool2D 2 -> Conv2D 8@3x3 relu
    -> MaxPool2D 2 (8, 7, 7) -> UpSample2D 2 -> Conv2D 16@3x3 relu
    -> UpSample2D 2 -> Conv2D 1@3x3 sigmoid (1, 28, 28)

set_input_noise() corrupts every training batch while the targets stay clean,
so fit(x, x) teaches the network to remove the noise. The test images are
corrupted the same way (apply_input_noise), and the example reports how close
the noisy and the denoised images are to the clean ones (MSE and PSNR), then
writes fashion_denoise.png: one test image per class, as clean, noisy and
denoised rows. Mirrors libann's examples/fashion_dae.c.

Usage:
    python fashion_denoise.py [-e EPOCHS] [-n gaussian|masking|salt_pepper]
                              [-a AMOUNT] [--gpu] [--limit N]
"""

import argparse
import time

import numpy as np

from pyann import Sequential, Input, Conv2D, MaxPool2D, UpSample2D
from example_data import CLASSES, as_images, load_fashion_mnist, setup_backend, write_png


def build_model() -> Sequential:
    model = Sequential([
        Input((1, 28, 28)),
        Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),    # -> (16, 28, 28)
        MaxPool2D(pool_size=2),                                                  # -> (16, 14, 14)
        Conv2D(filters=8, kernel_size=3, padding="same", activation="relu"),     # -> (8, 14, 14)
        MaxPool2D(pool_size=2),                                                  # -> (8, 7, 7)
        UpSample2D(size=2),                                                      # -> (8, 14, 14)
        Conv2D(filters=16, kernel_size=3, padding="same", activation="relu"),    # -> (16, 14, 14)
        UpSample2D(size=2),                                                      # -> (16, 28, 28)
        Conv2D(filters=1, kernel_size=3, padding="same", activation="sigmoid"),  # -> (1, 28, 28)
    ])
    model.compile(optimizer="adam", loss="bce", learning_rate=0.002, convergence=0.0)
    return model


def psnr(mse: float) -> float:
    """Peak signal-to-noise ratio in dB for pixels in [0, 1]."""
    return 10 * np.log10(1.0 / mse)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=5, help="training epochs (default: 5)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="mini-batch size (default: 32)")
    parser.add_argument("-n", "--noise", default="gaussian", choices=["gaussian", "masking", "salt_pepper"],
                        help="kind of noise (default: gaussian)")
    parser.add_argument("-a", "--amount", type=float, default=0.3,
                        help="noise standard deviation, or fraction of pixels changed (default: 0.3)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU (Metal or Vulkan) if available")
    parser.add_argument("--limit", type=int, help="use only the first N training images")
    parser.add_argument("--train-csv", help="Kaggle-format training CSV instead of the download")
    parser.add_argument("--test-csv", help="Kaggle-format test CSV instead of the download")
    parser.add_argument("-o", "--output", default="fashion_denoise.png", help="picture to write")
    args = parser.parse_args()

    setup_backend(args.gpu)
    x_train, _ = load_fashion_mnist("train", args.train_csv, args.limit)
    x_test, labels_test = load_fashion_mnist("test", args.test_csv)
    x_train, x_test = as_images(x_train), as_images(x_test)

    model = build_model()
    # noisy inputs, clamped to the pixel range; the targets stay clean
    model.set_input_noise(args.noise, args.amount, clip=(0, 1))
    print(model.summary())
    print(f"\nDenoising {args.noise} noise (amount {args.amount}), "
          f"training on {len(x_train)} images\n", flush=True)

    start = time.time()
    model.fit(x_train, x_train, epochs=args.epochs, batch_size=args.batch_size)
    print(f"\nTrained in {time.time() - start:.1f} s")

    noisy = model.apply_input_noise(x_test)      # corrupted like the training inputs
    denoised = model.predict(noisy)              # (N, 1, 28, 28): image outputs come back as images

    noisy_mse = float(np.mean((noisy - x_test) ** 2))
    denoised_mse = float(np.mean((denoised - x_test) ** 2))
    print(f"\nTest images ({len(x_test)}), compared with the clean ones:")
    print(f"  noisy:    MSE {noisy_mse:.5f}, PSNR {psnr(noisy_mse):.2f} dB")
    print(f"  denoised: MSE {denoised_mse:.5f}, PSNR {psnr(denoised_mse):.2f} dB")

    picks = [int(np.argmax(labels_test == c)) for c in range(len(CLASSES))]
    write_png(args.output, [[x_test[i] for i in picks], [noisy[i] for i in picks], [denoised[i] for i in picks]])
    print(f"\nWrote {args.output} (rows: clean, noisy, denoised; one image per class)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
