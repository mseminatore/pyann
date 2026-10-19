"""Tests for Sequential.train_on_batch() and end_online_training()."""

import pytest

import pyann
import pyann.core.network as network_module
from pyann import Dense, Sequential
from tests.helpers import LibSpy

np = pytest.importorskip("numpy")


@pytest.fixture
def spy(monkeypatch):
    spy = LibSpy(network_module.lib)
    monkeypatch.setattr(network_module, "lib", spy)
    return spy


def make_model(loss="mse", output_activation="sigmoid", **compile_kwargs):
    model = Sequential()
    model.add(Dense(8, activation="relu", input_shape=(2,)))
    model.add(Dense(1, activation=output_activation))
    model.compile(optimizer="adam", loss=loss, learning_rate=0.01, **compile_kwargs)
    return model


def make_data(n=32, seed=0):
    rng = np.random.RandomState(seed)
    x = rng.rand(n, 2).astype(np.float32)
    y = ((x[:, 0] + x[:, 1]) > 1.0).astype(np.float32).reshape(-1, 1)
    return x, y


class TestLearning:
    def test_returns_a_float_loss(self):
        model = make_model()
        loss = model.train_on_batch(*make_data())
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases_over_many_steps(self):
        model = make_model()
        x, y = make_data()
        first = model.train_on_batch(x, y)
        for _ in range(300):
            last = model.train_on_batch(x, y)
        assert last < 0.5 * first

    def test_predictions_change_after_training(self):
        model = make_model()
        x, y = make_data()
        model.train_on_batch(x, y)
        before = model.predict(x)
        for _ in range(50):
            model.train_on_batch(x, y)
        assert not np.allclose(before, model.predict(x))

    def test_ending_a_session_keeps_the_weights(self):
        """A new session continues from the trained weights instead of re-initializing."""
        model = make_model()
        x, y = make_data()
        first = model.train_on_batch(x, y)
        for _ in range(300):
            model.train_on_batch(x, y)
        model.end_online_training()
        resumed = model.train_on_batch(x, y)
        assert resumed < 0.5 * first

    def test_batch_size_may_change_between_calls(self):
        model = make_model()
        x, y = make_data(32)
        for rows in (8, 3, 32, 1):
            assert np.isfinite(model.train_on_batch(x[:rows], y[:rows]))

    def test_single_sample_as_1d_arrays(self):
        model = make_model()
        assert np.isfinite(model.train_on_batch(np.array([0.2, 0.9]), np.array([1.0])))

    def test_accepts_plain_lists(self):
        model = make_model()
        assert np.isfinite(model.train_on_batch([[0.1, 0.2], [0.8, 0.9]], [[0.0], [1.0]]))


class TestSession:
    def test_one_session_spans_many_calls(self, spy):
        model = make_model()
        x, y = make_data()
        for _ in range(3):
            model.train_on_batch(x, y)
        assert spy.names().count("ann_train_begin") == 1
        assert spy.names().count("ann_train_step") == 3
        assert "ann_train_end" not in spy.names()

    def test_end_online_training(self, spy):
        model = make_model()
        model.train_on_batch(*make_data())
        assert model.end_online_training() is model
        assert spy.names().count("ann_train_end") == 1

        model.end_online_training()  # no open session: nothing to end
        assert spy.names().count("ann_train_end") == 1

        model.train_on_batch(*make_data())  # starts a fresh session
        assert spy.names().count("ann_train_begin") == 2

    def test_end_without_any_training_is_a_noop(self, spy):
        make_model().end_online_training()
        assert "ann_train_end" not in spy.names()

    def test_fit_ends_the_session(self, spy):
        model = make_model()
        x, y = make_data()
        model.train_on_batch(x, y)
        model.fit(x, y, epochs=2, batch_size=16, verbose=0)
        assert "ann_train_end" in spy.names()
        assert spy.names().index("ann_train_end") < spy.names().index("ann_train_network")

    def test_predict_works_mid_session(self):
        model = make_model()
        x, y = make_data()
        model.train_on_batch(x, y)
        assert model.predict(x).shape == (32, 1)
        assert np.isfinite(model.train_on_batch(x, y))


class TestValidation:
    """libann copies batch_size * units values without checking, so bad shapes must never reach it."""

    @pytest.mark.parametrize("x,y", [
        (np.zeros((4, 3), np.float32), np.zeros((4, 1), np.float32)),   # wrong input width
        (np.zeros((4, 2), np.float32), np.zeros((4, 2), np.float32)),   # wrong target width
        (np.zeros((4, 2), np.float32), np.zeros((3, 1), np.float32)),   # row count mismatch
        (np.zeros((0, 2), np.float32), np.zeros((0, 1), np.float32)),   # empty batch
        (np.zeros((2, 2, 2), np.float32), np.zeros((2, 1), np.float32)),  # 3-D
        (np.float32(1.0), np.zeros((1, 1), np.float32)),                # scalar
    ])
    def test_bad_shapes_are_rejected_before_the_c_call(self, spy, x, y):
        model = make_model()
        with pytest.raises(ValueError):
            model.train_on_batch(x, y)
        assert "ann_train_step" not in spy.names()
        assert "ann_train_begin" not in spy.names()

    def test_cross_entropy_needs_softmax_or_sigmoid_output(self, spy):
        model = make_model(loss="categorical_cross_entropy", output_activation="relu")
        with pytest.raises(ValueError, match="softmax or sigmoid"):
            model.train_on_batch(*make_data())
        assert "ann_train_step" not in spy.names()

    def test_cross_entropy_with_softmax_trains(self):
        model = Sequential()
        model.add(Dense(8, activation="relu", input_shape=(2,)))
        model.add(Dense(2, activation="softmax"))
        model.compile(optimizer="adam", loss="categorical_cross_entropy")
        x, y = make_data()
        onehot = np.hstack([1 - y, y])
        assert np.isfinite(model.train_on_batch(x, onehot))

    def test_empty_model(self):
        with pytest.raises(pyann.exceptions.NetworkError):
            Sequential().train_on_batch([[0.0]], [[0.0]])


class TestLoadedModel:
    """Fine-tuning a loaded model is the main reason for online training."""

    def test_fine_tune_after_load(self, tmp_path):
        x, y = make_data()
        model = make_model()
        model.fit(x, y, epochs=5, batch_size=16, verbose=0)
        path = tmp_path / "model.nna"
        model.save(path)

        loaded = Sequential.load(path)
        before = loaded.predict(x)
        for _ in range(100):
            loss = loaded.train_on_batch(x, y)
        assert np.isfinite(loss)
        assert not np.allclose(before, loaded.predict(x))


class TestGpuInteraction:
    def test_predictions_use_the_updated_weights(self):
        """After GPU training the GPU copy is stale; predict() must not read it."""
        pyann.enable_gpu()  # a no-op on builds without a GPU backend
        model = make_model()
        x, y = make_data()
        model.fit(x, y, epochs=2, batch_size=16, verbose=0)  # uploads the weights
        before = model.predict(x)
        for _ in range(100):
            model.train_on_batch(x, y)
        assert not np.allclose(before, model.predict(x))
        model.end_online_training()
        assert np.all(np.isfinite(model.predict(x)))
