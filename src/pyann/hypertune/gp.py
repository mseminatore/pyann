"""Gaussian process surrogate for Bayesian hyperparameter search.

A pure-Python port of the model in libann's ann_hypertune.c, so it needs no
NumPy: a squared-exponential kernel with one length scale per dimension
(continuous dimensions use the difference of values in [0, 1], categorical
ones 0 when equal and 1 otherwise), scores standardized before fitting, and
length scales and noise refit by maximizing the log marginal likelihood.
"""

import math
from typing import List, Sequence, Tuple

CONTINUOUS = 0
CATEGORICAL = 1

_SCALES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)   # per-dimension length scale multipliers tried
_NOISES = (1e-4, 1e-3, 1e-2, 1e-1)          # noise variances tried


class GaussianProcess:
    """A Gaussian process over points of mixed continuous and categorical dimensions.

    Args:
        dim_types: CONTINUOUS or CATEGORICAL for each dimension
        length_scale: Base kernel length scale
        noise_variance: Observation noise variance (of the standardized scores)
    """

    def __init__(self, dim_types: Sequence[int], length_scale: float = 0.3,
                 noise_variance: float = 0.01):
        self.dim_types = list(dim_types)
        self.length_scale = length_scale
        self.noise_variance = noise_variance
        self.ard_scales = [1.0] * len(self.dim_types)
        self.X: List[List[float]] = []
        self.y: List[float] = []
        self._fit = None

    def add(self, x: Sequence[float], y: float) -> None:
        """Add an observed point and its score."""
        if len(x) != len(self.dim_types):
            raise ValueError(f"point has {len(x)} values, the GP has {len(self.dim_types)} dimensions")
        self.X.append(list(x))
        self.y.append(float(y))
        self._fit = None

    def set_scores(self, y: Sequence[float]) -> None:
        """Replace the observed scores (one per point), e.g. after cleaning them."""
        if len(y) != len(self.X):
            raise ValueError(f"{len(y)} scores for {len(self.X)} points")
        self.y = [float(v) for v in y]
        self._fit = None

    @property
    def score_scale(self) -> float:
        """The standard deviation the scores were standardized by."""
        if self._fit is None:
            self.fit()
        return self._fit["scale"]

    def kernel(self, a: Sequence[float], b: Sequence[float]) -> float:
        total = 0.0
        for d, kind in enumerate(self.dim_types):
            if kind == CATEGORICAL:
                diff = 0.0 if a[d] == b[d] else 1.0
            else:
                diff = a[d] - b[d]
            scale = self.length_scale * self.ard_scales[d]
            total += diff * diff / (scale * scale)
        return math.exp(-0.5 * total)

    def fit(self) -> None:
        """Factor the kernel matrix for the current observations and hyperparameters."""
        n = len(self.y)
        mean = sum(self.y) / n if n else 0.0
        var = sum((v - mean) ** 2 for v in self.y) / n if n else 0.0
        scale = math.sqrt(var) if var > 1e-12 else 1.0
        yn = [(v - mean) / scale for v in self.y]

        # Cholesky factor of K + noise * I
        L = [[0.0] * n for _ in range(n)]
        log_det_half = 0.0
        for j in range(n):
            row_j = L[j]
            s = self.kernel(self.X[j], self.X[j]) + self.noise_variance - sum(v * v for v in row_j[:j])
            row_j[j] = math.sqrt(max(s, 1e-10))
            log_det_half += math.log(row_j[j])
            for i in range(j + 1, n):
                row_i = L[i]
                s = self.kernel(self.X[i], self.X[j]) - sum(row_i[k] * row_j[k] for k in range(j))
                row_i[j] = s / row_j[j]

        z = _forward(L, yn)
        alpha = _backward(L, z)
        self._fit = {
            "L": L, "alpha": alpha, "mean": mean, "scale": scale,
            "log_marginal": -0.5 * sum(a * b for a, b in zip(yn, alpha)) - log_det_half - 0.9189385332 * n,
        }

    def log_marginal_likelihood(self) -> float:
        if self._fit is None:
            self.fit()
        return self._fit["log_marginal"]

    def predict(self, x: Sequence[float]) -> Tuple[float, float]:
        """Posterior (mean, variance) of the score at x."""
        if not self.y:
            return 0.0, 1.0
        if self._fit is None:
            self.fit()
        fit = self._fit
        k_star = [self.kernel(x, xi) for xi in self.X]
        m = sum(k * a for k, a in zip(k_star, fit["alpha"]))
        v = _forward(fit["L"], k_star)
        var = max(self.kernel(x, x) - sum(t * t for t in v), 1e-10)
        return fit["mean"] + fit["scale"] * m, var * fit["scale"] ** 2

    def tune(self, passes: int = 2) -> None:
        """Refit the length scales and noise by coordinate search on the log
        marginal likelihood, like libann's gp_fit_hyperparams(). Dimensions
        that don't affect the score end up with long length scales."""
        if len(self.y) < 3:
            return
        best = self.log_marginal_likelihood()
        for _ in range(passes):
            for d in range(len(self.dim_types)):
                keep = self.ard_scales[d]
                for scale in _SCALES:
                    if scale == keep:
                        continue
                    self.ard_scales[d] = scale
                    self.fit()
                    if self._fit["log_marginal"] > best:
                        best, keep = self._fit["log_marginal"], scale
                self.ard_scales[d] = keep
            keep = self.noise_variance
            for noise in _NOISES:
                if noise == keep:
                    continue
                self.noise_variance = noise
                self.fit()
                if self._fit["log_marginal"] > best:
                    best, keep = self._fit["log_marginal"], noise
            self.noise_variance = keep
        self.fit()


def expected_improvement(mean: float, variance: float, best: float, xi: float = 0.01) -> float:
    """Expected amount by which a score with this posterior beats best + xi."""
    if variance <= 0.0:
        return 0.0
    std = math.sqrt(variance)
    gain = mean - best - xi
    z = gain / std
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return gain * cdf + std * pdf


def _forward(L: List[List[float]], b: Sequence[float]) -> List[float]:
    """Solve L z = b for lower-triangular L."""
    z = []
    for i, row in enumerate(L):
        z.append((b[i] - sum(row[j] * z[j] for j in range(i))) / row[i])
    return z


def _backward(L: List[List[float]], z: Sequence[float]) -> List[float]:
    """Solve L^T x = z for lower-triangular L."""
    n = len(z)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (z[i] - sum(L[j][i] * x[j] for j in range(i + 1, n))) / L[i][i]
    return x
