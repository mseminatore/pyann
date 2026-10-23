#!/usr/bin/env python3
"""
MNIST Fashion Classification Example

This example demonstrates training a neural network on the Fashion-MNIST dataset
using PyANN. It mirrors the functionality of mnist.c from the libann library.

Usage:
    python mnist_fashion.py [train_file] [test_file]

Dataset:
    Fashion-MNIST is downloaded into examples/data/ on first use. The Kaggle
    CSV files (https://www.kaggle.com/datasets/tk230147/fashion-mnist) work
    too: pass them as train_file and test_file.
"""

import argparse
import sys
import time

import numpy as np

from pyann import Sequential, Dense, Adam, Loss, Activation, backend_info
import pyann
from pyann.callbacks import ExponentialDecay
from example_data import CLASSES, load_fashion_mnist


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    n_samples = len(labels)
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1.0
    return one_hot


def print_ascii_art(image: np.ndarray, width: int = 28, height: int = 28):
    """Display a 28x28 image as ASCII art."""
    pixels = " `.^-':_,;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@"
    
    print("\nInput image:\n")
    for row in range(height):
        line = ""
        for col in range(width):
            idx = int(93 * image[row * width + col])
            idx = min(93, max(0, idx))
            line += pixels[idx]
        print(line)
    print()


def class_histogram(predictions: np.ndarray):
    """Print a histogram of predicted classes."""
    classes = predictions.argmax(axis=1)
    counts = np.bincount(classes, minlength=10)
    
    print("\nClass Histogram:")
    max_count = max(counts)
    
    for i, count in enumerate(counts):
        bar_len = int(40 * count / max_count) if max_count > 0 else 0
        print(f"{i:3d}|{'*' * bar_len} ({count})")
    
    print(f"   +{'-' * 40}")


def main():
    parser = argparse.ArgumentParser(description="Train on Fashion-MNIST dataset")
    parser.add_argument("train_file", nargs="?", help="Training data CSV file (default: download)")
    parser.add_argument("test_file", nargs="?", help="Test data CSV file (default: download)")
    parser.add_argument("-e", "--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("-b", "--batch-size", type=int, default=32,
                        help="Mini-batch size (default: 32)")
    parser.add_argument("--hidden", type=int, default=32,
                        help="Hidden layer size (default: 32)")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export trained model to ONNX format")
    parser.add_argument("--show-sample", action="store_true",
                        help="Show a sample image as ASCII art")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU backend if available")
    args = parser.parse_args()
    
    if args.use_gpu:
        if pyann.enable_gpu():          # True when a GPU backend is compiled in and a device is available
            print("Using", pyann.backend_info()["gpu_active"])   # 'metal', 'vulkan' or 'cuda'


    # Print backend configuration
    info = backend_info()
    blas = info["blas"] or "none (scalar)"
    gpu = info["gpu"] or "none"
    config = f"BLAS: {blas}, GPU: {gpu}"
    if info["cblas_isa"]:
        config += f", ISA: {info['cblas_isa']}"

    print(f"Backend: {config}")

    # Load data
    X_train, y_train_labels = load_fashion_mnist("train", args.train_file)
    X_test, y_test_labels = load_fashion_mnist("test", args.test_file)
    
    # One-hot encode labels
    y_train = one_hot_encode(y_train_labels)
    y_test = one_hot_encode(y_test_labels)
    
    print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Show a sample image
    if args.show_sample:
        sample_idx = np.random.randint(len(X_train))
        print(f"\nSample image (label: {CLASSES[y_train_labels[sample_idx]]})")
        print_ascii_art(X_train[sample_idx])
    
    # Create model (784 -> hidden -> 10)
    model = Sequential(optimizer=Adam(lr=0.001), loss=Loss.CATEGORICAL_CROSS_ENTROPY)
    model.add(Dense(args.hidden, activation='sigmoid', input_shape=(784,)))
    model.add(Dense(10, activation='softmax'))
    
    print(f"\n{model.summary()}\n")
    
    # Learning rate scheduler (5% decay per epoch)
    scheduler = ExponentialDecay(gamma=0.95)

    # Train
    start_time = time.time()
    model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_scheduler=scheduler
    )
    train_time = time.time() - start_time
    
    print(f"\nTraining time: {train_time:.2f} seconds")
    
    # Evaluate on test set
    test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.2%}")
    
    # Show prediction distribution
    predictions = model.predict(X_test)
    class_histogram(predictions)
    
    # Export to ONNX if requested
    if args.export_onnx:
        onnx_file = "fashion_mnist.onnx.json"
        model.export_onnx(onnx_file)
        print(f"\nExported model to: {onnx_file}")
    
    # Save the trained model
    model.save("fashion_mnist.nna")
    print("Saved model to: fashion_mnist.nna")
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(5):
        idx = np.random.randint(len(X_test))
        pred = predictions[idx]
        pred_class = pred.argmax()
        true_class = y_test_labels[idx]
        correct = "✓" if pred_class == true_class else "✗"
        print(f"  {correct} Predicted: {CLASSES[pred_class]:15s} (conf: {pred[pred_class]:.1%}) | True: {CLASSES[true_class]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
