"""train_on_batch() with plain Python lists; these tests must pass with or without NumPy."""

import pytest

import pyann.utils.compat as compat
from pyann import Dense, Sequential

X = [[0.1, 0.2], [0.8, 0.9], [0.4, 0.3], [0.9, 0.7]]
Y = [[0.0], [1.0], [0.0], [1.0]]


def make_model():
    model = Sequential()
    model.add(Dense(8, activation="relu", input_shape=(2,)))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="mse", learning_rate=0.01)
    return model


@pytest.fixture(params=["as-installed", "numpy-disabled"])
def mode(request, monkeypatch):
    """Run every test as installed, and again with NumPy forced off."""
    if request.param == "numpy-disabled":
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
    return request.param


class TestLists:
    def test_returns_a_float_loss(self, mode):
        assert isinstance(make_model().train_on_batch(X, Y), float)

    def test_loss_decreases(self, mode):
        model = make_model()
        first = model.train_on_batch(X, Y)
        for _ in range(300):
            last = model.train_on_batch(X, Y)
        assert last < 0.5 * first

    def test_single_sample(self, mode):
        assert isinstance(make_model().train_on_batch([0.5, 0.5], [1.0]), float)

    def test_batch_size_may_change(self, mode):
        model = make_model()
        for rows in (4, 1, 3):
            assert isinstance(model.train_on_batch(X[:rows], Y[:rows]), float)

    def test_predict_between_steps(self, mode):
        model = make_model()
        model.train_on_batch(X, Y)
        predictions = model.predict(X)
        assert len(predictions) == 4

    @pytest.mark.parametrize("x,y", [
        ([[0.1, 0.2, 0.3]], [[0.0]]),             # wrong input width
        ([[0.1, 0.2]], [[0.0, 1.0]]),             # wrong target width
        ([[0.1, 0.2], [0.3, 0.4]], [[0.0]]),      # row count mismatch
        ([[0.1, 0.2], [0.8]], [[0.0], [1.0]]),    # ragged rows
        ([[]], [[]]),                              # empty rows
    ])
    def test_bad_shapes_raise_value_error(self, mode, x, y):
        with pytest.raises(ValueError):
            make_model().train_on_batch(x, y)
