#!/usr/bin/env python3
"""
Hyperparameter Search: random search versus TPE

Searches learning rate, batch size, depth, layer widths, activations and
dropout for a Fashion-MNIST MLP, on a 10,000-image subset with a separate
validation set. Random search and TPE (Tree-structured Parzen Estimator, a
Bayesian method that samples more where good results were found) get the
same number of trials. The best configuration is then trained on the full
training set and scored on the test set. Inspired by libann's
examples/mnist_hypertune_*.c.

Usage:
    python hyperparameter_search.py [--trials N] [--epochs N] [--seed N]
"""

import argparse
import time

from pyann import Activation
from pyann.hypertune import HyperparamSpace, HypertuneOptions, RandomSearch, TPESearch
from example_data import load_fashion_mnist, one_hot


def describe(result) -> str:
    layers = " -> ".join(f"{n} {a.name.lower()}" for n, a in zip(result.layer_sizes, result.activations))
    return (f"lr {result.learning_rate:.4f}, batch {result.batch_size:3d}, "
            f"dropout {result.dropout_rate:.1f}, {layers}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--trials", type=int, default=20, help="trials per search (default: 20)")
    parser.add_argument("--epochs", type=int, default=5, help="epochs per trial (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed for the searches (default: 42)")
    parser.add_argument("--train-csv", help="Kaggle-format training CSV instead of the download")
    parser.add_argument("--test-csv", help="Kaggle-format test CSV instead of the download")
    args = parser.parse_args()

    x, labels = load_fashion_mnist("train", args.train_csv)
    x_test, labels_test = load_fashion_mnist("test", args.test_csv)
    y, y_test = one_hot(labels), one_hot(labels_test)
    x_search, y_search = x[:10000], y[:10000]
    x_val, y_val = x[-2000:], y[-2000:]

    space = HyperparamSpace(
        learning_rate=(0.0003, 0.03),           # sampled on a log scale
        batch_sizes=[16, 32, 64, 128],
        hidden_layers=(1, 3),
        layer_sizes=[32, 64, 128, 256],
        activations=[Activation.RELU, Activation.LEAKY_RELU, Activation.TANH, Activation.GELU],
        dropout_rates=[0.0, 0.1, 0.3],
        epochs=args.epochs,
    )
    options = HypertuneOptions(verbose=0, seed=args.seed)

    best = {}
    for name, search in (("Random search", RandomSearch(space, n_trials=args.trials, options=options)),
                         ("TPE", TPESearch(space, n_trials=args.trials, n_startup=args.trials // 3,
                                           options=options))):
        start = time.time()
        result, results = search.run(x_search, y_search, x_val, y_val, input_size=784, output_size=10)
        print(f"\n{name}: {len(results)} trials in {time.time() - start:.1f} s. Top three:")
        for r in sorted(results, key=lambda r: r.accuracy, reverse=True)[:3]:
            print(f"  {r.accuracy:6.2%}  {describe(r)}")
        running = [max(r.accuracy for r in results[:i + 1]) for i in range(len(results))]
        step = max(1, len(running) // 5)
        marks = ", ".join(f"{running[i - 1]:.1%} after {i}" for i in range(step, len(running) + 1, step))
        print(f"  best so far: {marks}")
        best[name] = (search, result)

    name, (search, result) = max(best.items(), key=lambda item: item[1][1].accuracy)
    print(f"\nBest overall ({name}): {result.accuracy:.2%} validation accuracy\n  {describe(result)}")

    model = search.create_model(result, input_size=784, output_size=10)
    print(f"\nTraining it on all {len(x)} training images for {args.epochs * 2} epochs ...")
    model.fit(x, y, epochs=args.epochs * 2, batch_size=result.batch_size, verbose=0)
    print(f"Test accuracy: {model.evaluate(x_test, y_test):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
