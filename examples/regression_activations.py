#!/usr/bin/env python3
"""
Regression: fitting a curve and comparing activation functions

Networks with a linear output and MSE loss learn y = sin(3x) * exp(-x^2 / 2)
from noisy samples. The same 1 -> 32 -> 32 -> 1 network is trained with each
hidden activation libann offers, using a validation set, EarlyStopping with
restore_best, and cosine-annealed learning rates. Prints each one's test
error and plots the best fit as text.

Usage:
    python regression_activations.py [-e EPOCHS]
"""

import argparse

import numpy as np

from pyann import Sequential, Dense, Adam
from pyann.callbacks import CosineAnnealing, EarlyStopping

ACTIVATIONS = ["sigmoid", "tanh", "softsign", "relu", "leaky_relu", "gelu"]


def target(x: np.ndarray) -> np.ndarray:
    return np.sin(3 * x) * np.exp(-x ** 2 / 2)


def make_data(rng: np.random.Generator, n: int, noise: float):
    x = rng.uniform(-3, 3, size=(n, 1)).astype(np.float32)
    y = (target(x) + rng.normal(0, noise, size=x.shape)).astype(np.float32)
    return x, y


def train(activation: str, data, epochs: int):
    (x_train, y_train), validation = data
    model = Sequential(optimizer=Adam(lr=0.01), loss="mse")
    model.add(Dense(32, activation=activation, input_shape=(1,)))
    model.add(Dense(32, activation=activation))
    model.add(Dense(1, activation="linear"))
    model.compile(convergence=0.0)
    history = model.fit(
        x_train, y_train, epochs=epochs, batch_size=16, verbose=0,
        validation_data=validation,
        callbacks=[EarlyStopping(patience=40, min_delta=1e-5, restore_best=True)],
        lr_scheduler=CosineAnnealing(T_max=epochs, min_lr=1e-4),
    )
    return model, history


def plot(model: Sequential, width: int = 72, height: int = 17) -> str:
    """Text plot: '.' is the true curve, '*' the network's prediction."""
    xs = np.linspace(-3, 3, width, dtype=np.float32).reshape(-1, 1)
    grid = [[" "] * width for _ in range(height)]
    for series, mark in ((target(xs), "."), (model.predict(xs), "*")):
        for col, y in enumerate(np.asarray(series).ravel()):
            row = int(round((1.0 - (y + 1.0) / 2.0) * (height - 1)))
            if 0 <= row < height:
                grid[row][col] = mark
    return "\n".join("|" + "".join(r) for r in grid)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=400, help="maximum epochs (default: 400)")
    args = parser.parse_args()
    rng = np.random.default_rng(3)

    train_data = make_data(rng, 600, noise=0.05)
    val_data = make_data(rng, 200, noise=0.05)
    x_test = np.linspace(-3, 3, 500, dtype=np.float32).reshape(-1, 1)
    y_test = target(x_test)

    print(f"{'activation':12s} {'test MSE':>10s} {'best epoch':>11s}")
    results = {}
    for activation in ACTIVATIONS:
        model, history = train(activation, (train_data, val_data), args.epochs)
        mse = float(np.mean((model.predict(x_test) - y_test) ** 2))
        results[activation] = (mse, model)
        print(f"{activation:12s} {mse:10.5f} {history.get('best_epoch', history['epochs']):>11}")

    best = min(results, key=lambda a: results[a][0])
    print(f"\nBest: {best}  ('.' true curve, '*' prediction, x from -3 to 3)\n")
    print(plot(results[best][1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
