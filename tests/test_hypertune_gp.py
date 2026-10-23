"""Tests for the Gaussian process and the Bayesian search built on it (pure Python)."""

import math

import pytest

from pyann import Activation
from pyann.hypertune import BayesianSearch, HyperparamSpace, HypertuneOptions, RandomSearch
from pyann.hypertune.gp import CATEGORICAL, CONTINUOUS, GaussianProcess, expected_improvement
from pyann.hypertune.search import BaseSearch
from pyann.hypertune.space import HypertuneResult


class TestGaussianProcess:
    def test_interpolates_observations(self):
        gp = GaussianProcess([CONTINUOUS], noise_variance=1e-4)
        for x, y in [(0.0, 1.0), (0.5, 3.0), (1.0, 2.0)]:
            gp.add([x], y)
        for x, y in [(0.0, 1.0), (0.5, 3.0), (1.0, 2.0)]:
            mean, variance = gp.predict([x])
            assert mean == pytest.approx(y, abs=0.01)
            assert variance < 0.01

    def test_uncertain_far_from_data(self):
        gp = GaussianProcess([CONTINUOUS])
        gp.add([0.0], 1.0)
        gp.add([0.1], 2.0)
        near, far = gp.predict([0.05])[1], gp.predict([5.0])[1]
        assert far > 10 * near
        assert gp.predict([5.0])[0] == pytest.approx(1.5, abs=0.01)   # back to the mean score

    def test_categorical_dimension_is_same_or_different(self):
        gp = GaussianProcess([CATEGORICAL])
        assert gp.kernel([1], [1]) == 1.0
        assert gp.kernel([1], [2]) == gp.kernel([1], [7])

    def test_tune_finds_the_relevant_dimension(self):
        gp = GaussianProcess([CONTINUOUS, CONTINUOUS])
        for i in range(12):
            x0, x1 = i / 11, (i * 7 % 12) / 11
            gp.add([x0, x1], math.sin(4 * x0))     # only x0 matters
        gp.tune()
        assert gp.ard_scales[1] > gp.ard_scales[0]

    def test_tuning_raises_the_likelihood(self):
        gp = GaussianProcess([CONTINUOUS, CONTINUOUS])
        for i in range(10):
            gp.add([i / 9, (i * 3 % 10) / 9], (i / 9) ** 2)
        before = gp.log_marginal_likelihood()
        gp.tune()
        assert gp.log_marginal_likelihood() >= before

    def test_shape_errors(self):
        gp = GaussianProcess([CONTINUOUS, CONTINUOUS])
        with pytest.raises(ValueError):
            gp.add([0.0], 1.0)
        gp.add([0.0, 0.0], 1.0)
        with pytest.raises(ValueError):
            gp.set_scores([1.0, 2.0])


class TestExpectedImprovement:
    def test_known_value(self):
        # gain 0.5, std 0.5: 0.5 * Phi(1) + 0.5 * phi(1)
        expected = 0.5 * 0.5 * (1 + math.erf(1 / math.sqrt(2))) + 0.5 * math.exp(-0.5) / math.sqrt(2 * math.pi)
        assert expected_improvement(1.0, 0.25, 0.5, 0.0) == pytest.approx(expected)

    def test_no_variance_no_improvement(self):
        assert expected_improvement(2.0, 0.0, 0.5) == 0.0

    def test_more_uncertainty_more_improvement(self):
        assert expected_improvement(0.0, 1.0, 0.5) > expected_improvement(0.0, 0.01, 0.5)


def objective(config):
    """A synthetic score, best (0) at lr 0.01, first layer 64 wide, batch 32 and tanh."""
    score = -(math.log10(config["learning_rate"]) + 2) ** 2
    score -= 0.3 * abs(math.log2(config["layer_sizes"][0]) - 6)
    score -= 0.2 * (config["batch_size"] != 32)
    score -= 0.3 * (config["activations"][0] != Activation.TANH)
    return score


@pytest.fixture
def synthetic(monkeypatch):
    """Score trials with objective() instead of training networks; returns the scored configs."""
    seen = []

    def evaluate(self, config, *args, **kwargs):
        seen.append(config)
        score = objective(config)
        return HypertuneResult(
            learning_rate=config["learning_rate"], batch_size=config["batch_size"],
            optimizer=config["optimizer"], n_hidden_layers=len(config["layer_sizes"]),
            layer_sizes=config["layer_sizes"], activations=config["activations"],
            dropout_rate=config["dropout_rate"], accuracy=score, loss=0.0, epochs=1,
            val_loss=-score, score=score, topology=config["topology"],
        )

    monkeypatch.setattr(BaseSearch, "_evaluate_config", evaluate)
    return seen


SPACE = dict(learning_rate=(1e-4, 1.0), batch_sizes=[8, 16, 32, 64, 128], hidden_layers=(1, 3),
             layer_sizes=[8, 16, 32, 64, 128, 256],
             activations=[Activation.RELU, Activation.TANH, Activation.SIGMOID], dropout_rates=[0.0, 0.2])
DATA = ([[0.0]], [[0.0]], [[0.0]], [[0.0]], 1, 1)


class TestBayesianSearch:
    def test_beats_random_search_on_the_same_budget(self, synthetic):
        bayes, random_ = [], []
        for seed in range(3):
            best, _ = BayesianSearch(HyperparamSpace(**SPACE), n_trials=25, n_initial=8,
                                     options=HypertuneOptions(verbose=0, seed=seed)).run(*DATA)
            bayes.append(best.score)
            best, _ = RandomSearch(HyperparamSpace(**SPACE), n_trials=25,
                                   options=HypertuneOptions(verbose=0, seed=seed)).run(*DATA)
            random_.append(best.score)
        assert sum(bayes) / 3 > sum(random_) / 3 + 0.1

    def test_model_guided_trials_score_better_than_random_ones(self, synthetic):
        BayesianSearch(HyperparamSpace(**SPACE), n_trials=30, n_initial=10,
                       options=HypertuneOptions(verbose=0, seed=0)).run(*DATA)
        scores = [objective(c) for c in synthetic]
        assert sum(scores[10:]) / 20 > sum(scores[:10]) / 10

    def test_encoding(self):
        search = BayesianSearch(HyperparamSpace(**SPACE), options=HypertuneOptions(verbose=0, seed=0))
        config = search._random_config()
        x = search._encode(config)
        assert len(x) == 6 + 2 * 3
        n = len(config["layer_sizes"])
        assert x[6 + n:9] == [-0.5] * (3 - n)          # widths of absent layers
        assert x[9 + n:] == [-1] * (3 - n)             # activations of absent layers
        assert all(0.0 <= v <= 1.0 for v in x[:2] + x[3:5] + x[6:6 + n])

    def test_failed_trials_do_not_break_the_model(self, monkeypatch, synthetic):
        def nan_for_large_lr(self, config, *args, **kwargs):
            result = evaluate(self, config)
            if config["learning_rate"] > 0.1:
                result.score = math.nan
            return result

        evaluate = BaseSearch._evaluate_config
        monkeypatch.setattr(BaseSearch, "_evaluate_config", nan_for_large_lr)
        search = BayesianSearch(HyperparamSpace(**SPACE), n_trials=15, n_initial=5,
                                options=HypertuneOptions(verbose=0, seed=2))
        best, results = search.run(*DATA)
        assert any(math.isnan(r.score) for r in results)
        assert math.isfinite(best.score)
        # the model saw failed trials as the worst score, not NaN
        assert all(math.isfinite(y) for y in search._gp.y)
        finite = [r.score for r in results if math.isfinite(r.score)]
        assert min(search._gp.y) == pytest.approx(min(finite))

    @pytest.mark.parametrize("kwargs", [{"n_initial": 0}, {"n_candidates": 0}])
    def test_invalid_settings(self, kwargs):
        with pytest.raises(ValueError):
            BayesianSearch(HyperparamSpace(**SPACE), **kwargs)
