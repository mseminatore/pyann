"""Time pyann training and inference, to compare libann builds against different BLAS libraries.

Select the library under test with PYANN_LIB_PATH. Prints one JSON object per
workload so results from several builds can be merged.

    PYANN_LIB_PATH=build-openblas python benchmarks/bench_blas.py --variant openblas
"""

import argparse
import json
import os
import time

import numpy as np

import pyann
from pyann import Dense, Sequential

# (name, samples, input size, hidden layer sizes, output size, batch size, epochs)
WORKLOADS = [
    ("small-mlp", 20000, 784, [128], 10, 64, 3),
    ("medium-mlp", 20000, 256, [512, 512], 10, 128, 3),
    ("large-mlp", 10000, 1024, [1024, 1024], 10, 256, 2),
]


def run(name, n, n_in, hidden, n_out, batch, epochs, repeats):
    rng = np.random.RandomState(0)
    x = rng.rand(n, n_in).astype(np.float32)
    y = np.eye(n_out, dtype=np.float32)[rng.randint(0, n_out, n)]

    fit_times, predict_times = [], []
    for _ in range(repeats):
        model = Sequential()
        model.add(Dense(hidden[0], activation="relu", input_shape=(n_in,)))
        for units in hidden[1:]:
            model.add(Dense(units, activation="relu"))
        model.add(Dense(n_out, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_cross_entropy")

        t = time.perf_counter()
        model.fit(x, y, epochs=epochs, batch_size=batch, verbose=0)
        fit_times.append(time.perf_counter() - t)

        t = time.perf_counter()
        model.predict(x)
        predict_times.append(time.perf_counter() - t)

    # Best of N filters out noise from other processes on shared runners
    return min(fit_times), min(predict_times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, help="label for this build")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    info = pyann.backend_info()
    for name, n, n_in, hidden, n_out, batch, epochs in WORKLOADS:
        fit_s, predict_s = run(name, n, n_in, hidden, n_out, batch, epochs, args.repeats)
        print(json.dumps({
            "variant": args.variant,
            "blas": info["blas"],
            "cpus": os.cpu_count(),
            "workload": name,
            "fit_s": round(fit_s, 4),
            "predict_s": round(predict_s, 4),
        }), flush=True)


if __name__ == "__main__":
    main()
