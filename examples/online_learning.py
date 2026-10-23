#!/usr/bin/env python3
"""
Online Learning: train_on_batch() on a drifting data stream

Points arrive in small batches, and the rule that labels them changes over
time (concept drift): the boundary between the two classes slowly rotates.
Two copies of the same network start from one warm-up fit():

  - frozen:  never trained again, like a model shipped and left alone
  - online:  scores each batch, then learns from it with train_on_batch(),
             which keeps the weights and Adam's state between calls

The online model tracks the drift while the frozen one falls behind. Every
batch is scored before the model learns from it, so the accuracy shown is on
data the online model has not seen. Halfway through, the online model is
saved and reloaded to show that a deployed model can keep learning (Adam's
moment estimates start over, and the learning rate, which model files don't
store, is set again with compile()).

Usage:
    python online_learning.py [--batches N] [--batch-size N]
"""

import argparse
import math

import numpy as np

from pyann import Sequential, Dense, Adam

LEARNING_RATE = 0.005


def make_batch(rng: np.random.Generator, n: int, angle: float):
    """Two classes split by a curved boundary rotated by ``angle`` radians."""
    x = rng.uniform(-1, 1, size=(n, 2)).astype(np.float32)
    u = x[:, 0] * math.cos(angle) + x[:, 1] * math.sin(angle)
    v = -x[:, 0] * math.sin(angle) + x[:, 1] * math.cos(angle)
    labels = (v > 0.5 * np.sin(3 * u)).astype(np.int64)
    return x, np.eye(2, dtype=np.float32)[labels]


def build_model() -> Sequential:
    model = Sequential(optimizer=Adam(lr=LEARNING_RATE), loss="cce")
    model.add(Dense(32, activation="tanh", input_shape=(2,)))
    model.add(Dense(32, activation="tanh"))
    model.add(Dense(2, activation="softmax"))
    model.compile(convergence=0.0)
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--batches", type=int, default=600, help="batches in the stream (default: 600)")
    parser.add_argument("--batch-size", type=int, default=32, help="points per batch (default: 32)")
    args = parser.parse_args()
    rng = np.random.default_rng(5)

    # warm up on the starting concept, then copy the model so both start equal
    x0, y0 = make_batch(rng, 4000, angle=0.0)
    online = build_model()
    online.fit(x0, y0, epochs=30, batch_size=32, verbose=0)
    online.save("online_start.nna")
    frozen = Sequential.load("online_start.nna")

    window = args.batches // 10
    drift = math.pi / 2                      # the boundary turns 90 degrees over the stream
    print(f"Stream of {args.batches} batches of {args.batch_size}; the boundary rotates "
          f"{math.degrees(drift):.0f} degrees\n")
    print(f"{'batches':>11s} {'angle':>6s}   {'frozen':>7s}   {'online':>7s}   {'online loss':>11s}")

    hits = np.zeros(2)
    losses = []
    for step in range(args.batches):
        angle = drift * step / args.batches
        x, y = make_batch(rng, args.batch_size, angle)
        truth = y.argmax(axis=1)
        for i, model in enumerate((frozen, online)):
            hits[i] += (model.predict(x).argmax(axis=1) == truth).sum()   # score first...
        losses.append(online.train_on_batch(x, y))                         # ...then learn

        if (step + 1) % window == 0:
            acc = hits / (window * args.batch_size)
            print(f"{step + 2 - window:5d}-{step + 1:<5d} {math.degrees(angle):5.0f}    "
                  f"{acc[0]:7.1%}   {acc[1]:7.1%}   {np.mean(losses):11.4f}")
            hits[:] = 0
            losses.clear()

        if step + 1 == args.batches // 2:
            # end the session (dropout off, GPU weights refreshed), save, and carry on
            online.end_online_training()
            online.save("online_mid.nna")
            online = Sequential.load("online_mid.nna")
            online.compile(learning_rate=LEARNING_RATE)    # model files don't store the rate
            print(f"{'':11s} saved and reloaded the online model; it keeps learning")

    online.end_online_training()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
