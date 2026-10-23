#!/usr/bin/env python3
"""
Logic Gates: the smallest networks, in plain Python

Learns AND, OR, NAND and XOR from their truth tables. The data is plain
Python lists: NumPy is optional for PyANN. A single neuron (a perceptron)
learns the linearly separable gates but cannot learn XOR; one hidden layer
fixes that. Mirrors libann's examples/logic.c.

Also writes a PIKCHR diagram of the XOR network (xor.pikchr; render it with
`pikchr --svg-only xor.pikchr > xor.svg`).

Usage:
    python logic_gates.py [-e EPOCHS]
"""

import argparse

from pyann import Sequential, Dense, SGD

INPUTS = [[0, 0], [0, 1], [1, 0], [1, 1]]
GATES = {
    "AND": [[0], [0], [0], [1]],
    "OR": [[0], [1], [1], [1]],
    "NAND": [[1], [1], [1], [0]],
    "XOR": [[0], [1], [1], [0]],
}


def train(targets, hidden: int, epochs: int) -> Sequential:
    """Train a 2-input network with an optional hidden layer."""
    model = Sequential(optimizer=SGD(lr=0.5), loss="mse")
    if hidden:
        model.add(Dense(hidden, activation="tanh", input_shape=(2,)))
        model.add(Dense(1, activation="sigmoid"))
    else:
        model.add(Dense(1, activation="sigmoid", input_shape=(2,)))
    model.compile(convergence=0.0)      # train every epoch, don't stop early
    model.fit(INPUTS, targets, epochs=epochs, batch_size=1, verbose=0)
    return model


def report(name: str, model: Sequential, targets) -> bool:
    outputs = [row[0] for row in model.predict(INPUTS)]
    correct = all(round(out) == t[0] for out, t in zip(outputs, targets))
    cells = "  ".join(f"{a}{b}->{out:.2f}" for (a, b), out in zip(INPUTS, outputs))
    print(f"  {name:5s} {cells}   {'learned' if correct else 'FAILED'}")
    return correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=2000, help="training epochs (default: 2000)")
    args = parser.parse_args()

    print("One neuron, no hidden layer (a perceptron):")
    for name, targets in GATES.items():
        report(name, train(targets, hidden=0, epochs=args.epochs), targets)
    print("  XOR is not linearly separable, so no single neuron can learn it.\n")

    print("One hidden layer of 4 tanh units:")
    xor = train(GATES["XOR"], hidden=4, epochs=args.epochs)
    learned = report("XOR", xor, GATES["XOR"])
    print()
    print(xor.summary())

    xor.export_pikchr("xor.pikchr")
    print("\nWrote xor.pikchr (render with: pikchr --svg-only xor.pikchr > xor.svg)")
    return 0 if learned else 1


if __name__ == "__main__":
    raise SystemExit(main())
