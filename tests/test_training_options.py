"""Tests for the training options exposed by Sequential.compile()."""

import pytest

import pyann.core.network as network_module
from pyann import Dense, Sequential, WeightInit
from pyann._bindings.ffi import lib
from pyann.weight_init import parse_weight_init
from tests.helpers import LibSpy

np = pytest.importorskip("numpy")


@pytest.fixture
def spy(monkeypatch):
    spy = LibSpy(network_module.lib)
    monkeypatch.setattr(network_module, "lib", spy)
    return spy


def make_model(**dense_kwargs):
    model = Sequential()
    model.add(Dense(8, activation="relu", input_shape=(2,), **dense_kwargs))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model


class TestWeightInit:
    def test_values_match_c_enum(self):
        assert WeightInit.UNIFORM == lib.WEIGHT_INIT_UNIFORM
        assert WeightInit.XAVIER == lib.WEIGHT_INIT_XAVIER
        assert WeightInit.HE == lib.WEIGHT_INIT_HE
        assert WeightInit.AUTO == lib.WEIGHT_INIT_AUTO

    def test_from_string(self):
        assert WeightInit.from_string("he") == WeightInit.HE
        assert WeightInit.from_string("Xavier") == WeightInit.XAVIER
        assert WeightInit.from_string("glorot") == WeightInit.XAVIER

    def test_from_string_invalid(self):
        with pytest.raises(ValueError):
            WeightInit.from_string("orthogonal")

    def test_parse(self):
        assert parse_weight_init("auto") == WeightInit.AUTO
        assert parse_weight_init(WeightInit.HE) == WeightInit.HE
        with pytest.raises(TypeError):
            parse_weight_init(3.5)


class TestCompileValidation:
    def test_negative_gradient_clip(self):
        with pytest.raises(ValueError):
            Sequential().compile(gradient_clip=-1.0)

    def test_negative_convergence(self):
        with pytest.raises(ValueError):
            Sequential().compile(convergence=-0.1)

    @pytest.mark.parametrize("rate", [-0.1, 1.0, 1.5])
    def test_dropout_out_of_range(self, rate):
        with pytest.raises(ValueError):
            Sequential().compile(dropout=rate)

    def test_invalid_weight_init(self):
        with pytest.raises(ValueError):
            Sequential().compile(weight_init="bogus")

    def test_options_default_to_unset(self, spy):
        model = make_model()
        model._build()
        for name in ("ann_set_gradient_clip", "ann_set_weight_init",
                     "ann_set_convergence", "ann_set_dropout"):
            assert name not in spy.names()


class TestAppliedToNetwork:
    def test_options_reach_the_c_library(self, spy):
        model = make_model()
        model.compile(gradient_clip=2.0, weight_init="he", convergence=0.5, dropout=0.25)
        model._build()

        assert spy.args("ann_set_gradient_clip") == [(pytest.approx(2.0),)]
        assert spy.args("ann_set_weight_init") == [(int(WeightInit.HE),)]
        assert spy.args("ann_set_convergence") == [(pytest.approx(0.5),)]
        assert spy.args("ann_set_dropout") == [(pytest.approx(0.25),)]

    def test_default_dropout_is_set_after_the_layers_exist(self, spy):
        """ann_set_dropout only fills hidden layers that have no rate of their own."""
        model = make_model(dropout=0.5)
        model.compile(dropout=0.1)
        model._build()

        names = spy.names()
        assert names.index("ann_set_dropout") > max(
            i for i, n in enumerate(names) if n in ("ann_add_layer", "ann_set_layer_dropout")
        )
        assert spy.args("ann_set_layer_dropout") == [(1, pytest.approx(0.5))]

    def test_compile_after_build_applies_immediately(self, spy):
        model = make_model()
        model._build()
        assert "ann_set_gradient_clip" not in spy.names()

        model.compile(gradient_clip=3.0)
        assert spy.args("ann_set_gradient_clip") == [(pytest.approx(3.0),)]

    def test_zero_gradient_clip_is_passed_through(self, spy):
        """0 disables clipping in libann, so it must reach the library."""
        model = make_model()
        model.compile(gradient_clip=0.0)
        model._build()
        assert spy.args("ann_set_gradient_clip") == [(0.0,)]


class TestTrainingBehavior:
    @staticmethod
    def _train(convergence):
        rng = np.random.RandomState(0)
        x = rng.rand(64, 2).astype(np.float32)
        y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.float32).reshape(-1, 1)
        model = make_model()
        model.compile(optimizer="adam", loss="mse", learning_rate=0.05, convergence=convergence)
        model.fit(x, y, epochs=300, batch_size=16, verbose=0)
        return float(np.abs(model.predict(x) - y).mean())

    def test_large_convergence_stops_training_early(self):
        """With a huge threshold the first epoch already counts as converged."""
        stopped = self._train(convergence=1e6)
        trained = self._train(convergence=0.0)
        assert trained < stopped

    @pytest.mark.parametrize("init", ["uniform", "xavier", "he", "auto"])
    def test_every_weight_init_trains(self, init):
        rng = np.random.RandomState(1)
        x = rng.rand(32, 2).astype(np.float32)
        y = (x.sum(axis=1, keepdims=True) > 1.0).astype(np.float32)
        model = make_model()
        model.compile(weight_init=init, gradient_clip=5.0)
        model.fit(x, y, epochs=3, batch_size=16, verbose=0)
        assert np.all(np.isfinite(model.predict(x)))
