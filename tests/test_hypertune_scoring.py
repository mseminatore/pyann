"""Tests for how hyperparameter searches train, score and rank their trials."""

import math

import pytest

from pyann.hypertune.search import BayesianSearch, GridSearch, RandomSearch, TPESearch, _rank
from pyann.hypertune.space import HyperparamSpace, HypertuneOptions, HypertuneResult
from pyann.optimizers import OptimizerType

np = pytest.importorskip("numpy")


def blobs(n=200, seed=0):
    """Two overlapping Gaussian blobs: (x_train, y_train, x_val, y_val), one-hot labels."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, size=n)
    x = (rng.normal(size=(n, 2)) + labels[:, None] * 1.5).astype(np.float32)
    y = np.eye(2, dtype=np.float32)[labels]
    half = n // 2
    return x[:half], y[:half], x[half:], y[half:]


def small_space(**kwargs):
    defaults = dict(learning_rate=(0.001, 0.1), batch_sizes=[16], hidden_layers=(1, 1),
                    layer_sizes=[4, 8], dropout_rates=[0.0], epochs=20)
    defaults.update(kwargs)
    return HyperparamSpace(**defaults)


def quiet(**kwargs):
    return HypertuneOptions(verbose=0, seed=1, **kwargs)


def fake_result(score, val_loss=0.5):
    return HypertuneResult(
        learning_rate=0.01, batch_size=16, optimizer=OptimizerType.ADAM, n_hidden_layers=1,
        layer_sizes=[4], activations=[], dropout_rate=0.0, accuracy=0.5, loss=0.5, epochs=1,
        val_loss=val_loss, score=score,
    )


class TestTrialResults:
    """What each trial records."""

    def test_losses_and_score_are_measured(self):
        best, results = RandomSearch(small_space(), n_trials=3, options=quiet()).run(
            *blobs(), input_size=2, output_size=2)
        for r in results:
            assert r.loss > 0                       # the last epoch's training loss, not a 0 placeholder
            assert math.isfinite(r.val_loss) and r.val_loss > 0
            assert r.score == r.accuracy            # default scoring
            assert r.training_time > 0
        assert best.score == max(r.score for r in results)

    def test_to_dict_includes_new_fields(self):
        d = fake_result(0.7).to_dict()
        assert d["score"] == 0.7 and d["val_loss"] == 0.5
        assert "best_epoch" in d and "training_time" in d


class TestScoring:
    """HypertuneOptions.scoring decides the ranking in every search."""

    @pytest.mark.parametrize("make_search", [
        lambda space, opts: RandomSearch(space, n_trials=6, options=opts),
        lambda space, opts: GridSearch(space, options=opts, lr_steps=3),
        lambda space, opts: BayesianSearch(space, n_trials=6, n_initial=2, options=opts),
        lambda space, opts: TPESearch(space, n_trials=6, n_startup=2, options=opts),
    ], ids=["random", "grid", "bayesian", "tpe"])
    def test_loss_scoring_picks_the_lowest_validation_loss(self, make_search):
        search = make_search(small_space(), quiet(scoring="loss"))
        best, results = search.run(*blobs(), input_size=2, output_size=2)
        assert all(r.score == pytest.approx(-r.val_loss) for r in results)
        assert best.val_loss == min(r.val_loss for r in results)

    def test_bayesian_learns_from_the_score(self):
        search = BayesianSearch(small_space(), n_trials=4, n_initial=2, options=quiet(scoring="loss"))
        _, results = search.run(*blobs(), input_size=2, output_size=2)
        assert [score for _, score in search._history] == [pytest.approx(-r.val_loss) for r in results]

    def test_tpe_learns_from_the_score(self):
        search = TPESearch(small_space(), n_trials=4, n_startup=2, options=quiet(scoring="loss"))
        _, results = search.run(*blobs(), input_size=2, output_size=2)
        assert [score for _, score in search._observations] == [pytest.approx(-r.val_loss) for r in results]

    def test_custom_scorer(self):
        calls = []

        def scorer(model, x_val, y_val):
            calls.append(model)
            return -model.evaluate_loss(x_val, y_val) * 2

        best, results = RandomSearch(small_space(), n_trials=3, options=quiet(scoring=scorer)).run(
            *blobs(), input_size=2, output_size=2)
        assert len(calls) == 3
        assert all(r.score == pytest.approx(-2 * r.val_loss, rel=1e-4) for r in results)
        assert best.score == max(r.score for r in results)

    def test_nan_score_ranks_last(self):
        search = RandomSearch(small_space(), options=quiet())
        search._update_best(fake_result(math.nan))
        search._update_best(fake_result(0.2))
        search._update_best(fake_result(math.nan))
        assert search.best_result.score == 0.2
        assert _rank(math.nan) < _rank(-1e9)

    def test_invalid_scoring(self):
        with pytest.raises(ValueError):
            HypertuneOptions(scoring="f1").validate()

    def test_invalid_patience(self):
        with pytest.raises(ValueError):
            HypertuneOptions(patience=0).validate()


class TestEarlyStopping:
    """HypertuneOptions.early_stopping and patience apply to every trial."""

    def test_trials_stop_early_and_report_the_best_epoch(self):
        space = small_space(epochs=500, learning_rate=(0.05, 0.1))
        _, results = RandomSearch(space, n_trials=2, options=quiet(patience=2)).run(
            *blobs(), input_size=2, output_size=2)
        for r in results:
            assert r.best_epoch is not None and r.best_epoch < 500

    def test_disabled(self):
        _, results = RandomSearch(small_space(epochs=5), n_trials=1, options=quiet(early_stopping=False)).run(
            *blobs(), input_size=2, output_size=2)
        assert results[0].best_epoch is None
