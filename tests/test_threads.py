"""Tests for the CPU thread count setting."""

import pytest

import pyann
from pyann import Conv2D, Dense, Input, MaxPool2D, Sequential

np = pytest.importorskip("numpy")


@pytest.fixture(autouse=True)
def restore_default_threads():
    yield
    pyann.set_num_threads(0)


def test_set_and_get():
    pyann.set_num_threads(3)
    assert pyann.get_num_threads() == 3
    pyann.set_num_threads(1)
    assert pyann.get_num_threads() == 1


def test_zero_restores_the_default():
    pyann.set_num_threads(0)
    default = pyann.get_num_threads()
    assert default >= 1
    pyann.set_num_threads(2)
    pyann.set_num_threads(0)
    assert pyann.get_num_threads() == default


@pytest.mark.parametrize("bad", [-1, 2.5, "4", True])
def test_invalid_values(bad):
    with pytest.raises(ValueError):
        pyann.set_num_threads(bad)


def test_results_do_not_depend_on_the_thread_count(tmp_path):
    """The same starting model trained with 1 and 4 threads ends identical."""
    rng = np.random.RandomState(0)
    x = rng.rand(64, 2, 10, 10).astype(np.float32)
    y = np.eye(4, dtype=np.float32)[rng.randint(0, 4, 64)]

    model = Sequential([
        Input((2, 10, 10)),
        Conv2D(6, 3, padding="same", activation="relu"),
        MaxPool2D(2),
        Dense(32, activation="relu"),
        Dense(4, activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce")
    model.fit(x, y, epochs=1, batch_size=32, verbose=0)  # initializes the weights
    path = tmp_path / "start.nnb"
    model.save(path, format="binary")

    predictions = []
    for threads in (1, 4):
        pyann.set_num_threads(threads)
        trained = Sequential.load(path, format="binary")
        for _ in range(10):  # online steps: no random shuffling
            trained.train_on_batch(x, y)
        predictions.append(trained.predict(x))
    np.testing.assert_array_equal(predictions[0], predictions[1])
