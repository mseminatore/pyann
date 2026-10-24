"""Tests for how hyperparameter searches train, score and rank their trials."""

import math

import pytest

from pyann.hypertune.search import BayesianSearch, GridSearch, RandomSearch, TPESearch, _rank
from pyann.hypertune.space import HyperparamSpace, HypertuneOptions, HypertuneResult, topology_sizes
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
            assert r.best_epoch <= r.epochs_trained < 500

    def test_disabled(self):
        _, results = RandomSearch(small_space(epochs=5), n_trials=1, options=quiet(early_stopping=False)).run(
            *blobs(), input_size=2, output_size=2)
        assert results[0].best_epoch is None
        assert results[0].epochs_trained == 5


class TestTopology:
    """HyperparamSpace.topology shapes the hidden layers in every search."""

    @pytest.mark.parametrize("topology, base, count, expected", [
        ("constant", 64, 3, [64, 64, 64]),
        ("pyramid", 128, 3, [128, 64, 32]),
        ("pyramid", 16, 3, [16, 8, 8]),                   # never narrower than 8
        ("inverse", 128, 3, [32, 64, 128]),
        ("funnel", 32, 3, [32, 64, 32]),
        ("funnel", 32, 5, [32, 64, 128, 64, 32]),
        ("funnel", 32, 7, [32, 64, 128, 128, 128, 64, 32]),  # widening capped at 4x
    ])
    def test_sizes_match_libann(self, topology, base, count, expected):
        assert topology_sizes(topology, base, count) == expected

    def test_names_and_aliases(self):
        assert HyperparamSpace().topologies() == ["any"]
        space = HyperparamSpace(topology=["uniform", "constant", "diamond"])
        assert space.topologies() == ["constant", "funnel"]

    @pytest.mark.parametrize("topology", ["triangle", []])
    def test_invalid(self, topology):
        with pytest.raises(ValueError):
            HyperparamSpace(topology=topology).validate()

    @pytest.mark.parametrize("make_search", [
        lambda space: RandomSearch(space, n_trials=4, options=quiet()),
        lambda space: BayesianSearch(space, n_trials=6, n_initial=2, options=quiet()),
        lambda space: TPESearch(space, n_trials=6, n_startup=2, options=quiet()),
    ], ids=["random", "bayesian", "tpe"])
    def test_pyramid_trials(self, make_search):
        space = small_space(topology="pyramid", hidden_layers=(2, 3), layer_sizes=[16, 32], epochs=3)
        _, results = make_search(space).run(*blobs(), input_size=2, output_size=2)
        for r in results:
            assert r.topology == "pyramid"
            assert r.layer_sizes == topology_sizes("pyramid", r.layer_sizes[0], len(r.layer_sizes))

    def test_any_picks_each_width_from_layer_sizes(self):
        space = small_space(hidden_layers=(3, 3), layer_sizes=[4, 8], epochs=2)
        _, results = RandomSearch(space, n_trials=4, options=quiet()).run(*blobs(), input_size=2, output_size=2)
        assert all(r.topology == "any" and set(r.layer_sizes) <= {4, 8} for r in results)

    def test_grid_covers_every_shape(self):
        patterns = GridSearch(small_space(topology=["constant", "pyramid"], hidden_layers=(1, 2),
                                          layer_sizes=[16, 32]), lr_steps=1)._generate_configs()
        shapes = {(c["topology"], tuple(c["layer_sizes"])) for c in patterns}
        assert shapes == {("constant", (16,)), ("constant", (32,)), ("constant", (16, 16)), ("constant", (32, 32)),
                          ("pyramid", (16,)), ("pyramid", (32,)), ("pyramid", (16, 8)), ("pyramid", (32, 16))}

        anything = GridSearch(small_space(hidden_layers=(2, 2), layer_sizes=[4, 8]), lr_steps=1)._generate_configs()
        assert sorted(c["layer_sizes"] for c in anything) == [[4, 4], [4, 8], [8, 4], [8, 8]]

    def test_grid_rejects_zero_lr_steps(self):
        with pytest.raises(ValueError):
            GridSearch(small_space(), lr_steps=0)

    def test_bayesian_perturbation_keeps_shapes_consistent(self):
        search = BayesianSearch(small_space(topology=["any", "pyramid", "inverse"], hidden_layers=(1, 3),
                                            layer_sizes=[16, 32, 64]), options=quiet())
        config = search._random_config()
        for _ in range(200):
            config = search._perturb_config(config)
            n = len(config["layer_sizes"])
            assert config["layer_sizes"] == search._layer_sizes(config["topology"], n, config["width_idx"])


class TestTPEModelsWidths:
    """TPE learns which widths and topologies score well, instead of drawing widths at random."""

    def observations(self, search, good_width, good_topo, n=30):
        for i in range(n):
            good = i % 3 == 0
            indices = {"log_lr": math.log(0.01), "batch_idx": 0, "opt_idx": 0, "dropout_idx": 0,
                       "act_idx": 0, "layers_idx": 1,
                       "topo_idx": good_topo if good else 1 - good_topo,
                       "width_idx": [good_width if good else 1 - good_width]}
            search._observations.append((indices, 1.0 if good else 0.0))

    def test_prefers_the_good_width_and_topology(self):
        space = small_space(topology=["constant", "pyramid"], hidden_layers=(1, 2), layer_sizes=[16, 64])
        search = TPESearch(space, gamma=0.3, n_candidates=24, options=quiet())
        self.observations(search, good_width=1, good_topo=1)
        picks = [search._tpe_sample_config() for _ in range(40)]
        assert sum(ind["width_idx"][0] == 1 for _, ind in picks) >= 36
        assert sum(cfg["topology"] == "pyramid" for cfg, _ in picks) >= 36
        assert all(cfg["layer_sizes"][0] == 64 for cfg, ind in picks if ind["width_idx"][0] == 1)
