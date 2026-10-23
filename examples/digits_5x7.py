#!/usr/bin/env python3
"""
5x7 Digit Recognition: training with input noise for robustness

A small MLP (35 -> 48 -> 10) learns the ten digits of a 5x7 pixel font, then
reads digits with some of their pixels flipped. Two networks are trained on
the same ten glyphs: one on the clean glyphs, one with salt-and-pepper input
noise (set_input_noise), which corrupts every training batch afresh. Both
read lightly damaged digits; as more pixels flip, the noise-trained network
holds on to a few more points of accuracy.
Inspired by libann's examples/digit5x7.c.

Usage:
    python digits_5x7.py [-e EPOCHS] [--trials N]
"""

import argparse

import numpy as np

from pyann import Sequential, Dense, Adam
from example_data import ascii_image

GLYPHS = [
    [" ### ", "#   #", "#  ##", "# # #", "##  #", "#   #", " ### "],   # 0
    ["  #  ", " ##  ", "  #  ", "  #  ", "  #  ", "  #  ", " ### "],   # 1
    [" ### ", "#   #", "    #", "   # ", "  #  ", " #   ", "#####"],   # 2
    ["#####", "   # ", "  #  ", "   # ", "    #", "#   #", " ### "],   # 3
    ["   # ", "  ## ", " # # ", "#  # ", "#####", "   # ", "   # "],   # 4
    ["#####", "#    ", "#### ", "    #", "    #", "#   #", " ### "],   # 5
    ["  ## ", " #   ", "#    ", "#### ", "#   #", "#   #", " ### "],   # 6
    ["#####", "    #", "   # ", "  #  ", " #   ", " #   ", " #   "],   # 7
    [" ### ", "#   #", "#   #", " ### ", "#   #", "#   #", " ### "],   # 8
    [" ### ", "#   #", "#   #", " ####", "    #", "   # ", " ##  "],   # 9
]

X = np.array([[c == "#" for c in "".join(g)] for g in GLYPHS], dtype=np.float32)
Y = np.eye(10, dtype=np.float32)


def train(noise: float, epochs: int) -> Sequential:
    model = Sequential(optimizer=Adam(lr=0.01), loss="cce")
    model.add(Dense(48, activation="sigmoid", input_shape=(35,)))
    model.add(Dense(10, activation="softmax"))
    model.compile(convergence=0.0)
    if noise:
        # each pixel is set to 0 or 1 with this probability, fresh every batch
        model.set_input_noise("salt_pepper", noise, clip=(0, 1))
    # 50 copies of the glyphs per epoch, so each epoch sees many noisy versions
    model.fit(np.tile(X, (50, 1)), np.tile(Y, (50, 1)), epochs=epochs, batch_size=10, verbose=0)
    return model


def flip_pixels(rng: np.random.Generator, glyphs: np.ndarray, count: int) -> np.ndarray:
    noisy = glyphs.copy()
    for row in noisy:
        idx = rng.choice(35, size=count, replace=False)
        row[idx] = 1.0 - row[idx]
    return noisy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=30, help="training epochs (default: 30)")
    parser.add_argument("--trials", type=int, default=200, help="noisy copies of each digit per test (default: 200)")
    args = parser.parse_args()
    rng = np.random.default_rng(7)

    clean = train(noise=0.0, epochs=args.epochs)
    robust = train(noise=0.3, epochs=args.epochs)
    print(robust.summary())

    print("\nAccuracy on digits with k random pixels flipped "
          f"({args.trials} noisy copies of each digit):\n")
    print("   k   clean-trained   noise-trained")
    labels = np.tile(np.arange(10), args.trials)
    for k in range(0, 11, 2):
        noisy = flip_pixels(rng, np.tile(X, (args.trials, 1)), k)
        accs = [(m.predict(noisy).argmax(axis=1) == labels).mean() for m in (clean, robust)]
        print(f"  {k:2d}   {accs[0]:12.1%}   {accs[1]:13.1%}")

    print("\nA few noisy digits (6 pixels flipped):")
    for digit in rng.choice(10, size=3, replace=False):
        noisy = flip_pixels(rng, X[digit:digit + 1], 6)
        guesses = [int(m.predict(noisy)[0].argmax()) for m in (clean, robust)]
        print()
        print(ascii_image(noisy[0], 5, 7))
        print(f"true {digit}  clean-trained says {guesses[0]}  noise-trained says {guesses[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
