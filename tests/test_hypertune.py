"""Tests for hyperparameter tuning search algorithms."""

import math
import pytest

from pyann.hypertune.search import (
    TPESearch,
    _scott_bandwidth,
    _kde_evaluate,
    _kde_sample,
    _categorical_prob,
    _categorical_sample,
)
from pyann.hypertune.space import HyperparamSpace, HypertuneOptions


class TestScottBandwidth:
    """Tests for Scott's rule bandwidth estimation."""

    def test_single_sample(self):
        assert _scott_bandwidth([1.0]) == pytest.approx(0.1, abs=0.01)

    def test_empty_samples(self):
        assert _scott_bandwidth([]) == pytest.approx(0.1, abs=0.01)

    def test_identical_samples(self):
        bw = _scott_bandwidth([5.0, 5.0, 5.0, 5.0])
        assert bw > 0

    def test_varied_samples(self):
        samples = [1.0, 2.0, 3.0, 4.0, 5.0]
        bw = _scott_bandwidth(samples)
        assert bw > 0
        # Scott's rule: 1.06 * std * n^(-1/5)
        std = (sum((x - 3.0) ** 2 for x in samples) / 4) ** 0.5
        expected = 1.06 * std * (5 ** -0.2)
        assert bw == pytest.approx(expected, rel=0.01)

    def test_bandwidth_factor(self):
        samples = [1.0, 2.0, 3.0]
        bw1 = _scott_bandwidth(samples, factor=1.0)
        bw2 = _scott_bandwidth(samples, factor=2.0)
        assert bw2 == pytest.approx(bw1 * 2.0, rel=0.01)


class TestKDE:
    """Tests for KDE evaluation and sampling."""

    def test_evaluate_empty(self):
        assert _kde_evaluate(0.0, [], 1.0) == pytest.approx(1e-10, abs=1e-11)

    def test_evaluate_at_sample(self):
        """KDE should have higher density near samples."""
        samples = [0.0, 0.0, 0.0]
        bw = 1.0
        density_at_zero = _kde_evaluate(0.0, samples, bw)
        density_far = _kde_evaluate(10.0, samples, bw)
        assert density_at_zero > density_far

    def test_sample_returns_float(self):
        samples = [1.0, 2.0, 3.0]
        result = _kde_sample(samples, 0.5)
        assert isinstance(result, float)

    def test_sample_empty(self):
        result = _kde_sample([], 1.0)
        assert 0.0 <= result <= 1.0


class TestCategorical:
    """Tests for categorical probability and sampling."""

    def test_prob_empty(self):
        assert _categorical_prob(0, [], 3) == pytest.approx(1.0 / 3)

    def test_prob_laplace_smoothing(self):
        # 2 out of 3 are category 0, with 2 categories
        # (2+1)/(3+2) = 3/5 = 0.6
        assert _categorical_prob(0, [0, 0, 1], 2) == pytest.approx(0.6)
        # (1+1)/(3+2) = 2/5 = 0.4
        assert _categorical_prob(1, [0, 0, 1], 2) == pytest.approx(0.4)

    def test_prob_unseen_category(self):
        # Laplace smoothing gives nonzero prob to unseen categories
        prob = _categorical_prob(2, [0, 0, 1], 3)
        assert prob > 0

    def test_sample_returns_valid_index(self):
        for _ in range(20):
            idx = _categorical_sample([0, 1, 1, 2], 3)
            assert 0 <= idx < 3

    def test_sample_empty(self):
        for _ in range(20):
            idx = _categorical_sample([], 5)
            assert 0 <= idx < 5


class TestTPESearch:
    """Tests for TPESearch initialization and configuration."""

    def test_defaults(self):
        space = HyperparamSpace()
        search = TPESearch(space)
        assert search.n_trials == 50
        assert search.n_startup == 10
        assert search.gamma == 0.25
        assert search.n_candidates == 24
        assert search.bandwidth_factor == 1.0

    def test_custom_params(self):
        space = HyperparamSpace()
        search = TPESearch(
            space,
            n_trials=30,
            n_startup=5,
            gamma=0.3,
            n_candidates=16,
            bandwidth_factor=1.5,
        )
        assert search.n_trials == 30
        assert search.n_startup == 5
        assert search.gamma == 0.3
        assert search.n_candidates == 16
        assert search.bandwidth_factor == 1.5

    def test_random_config_indexed(self):
        space = HyperparamSpace(
            learning_rate=(0.001, 0.1),
            batch_sizes=[16, 32],
            hidden_layers=(1, 2),
        )
        search = TPESearch(space)
        config, indices = search._random_config_indexed()

        assert 0.001 <= config["learning_rate"] <= 0.1
        assert config["batch_size"] in [16, 32]
        assert "log_lr" in indices
        assert math.log(0.001) <= indices["log_lr"] <= math.log(0.1)
        assert 0 <= indices["batch_idx"] < 2
        assert 0 <= indices["layers_idx"] <= 1

    def test_invalid_space_raises(self):
        space = HyperparamSpace(learning_rate=(0.1, 0.001))  # min > max
        with pytest.raises(ValueError):
            TPESearch(space)
