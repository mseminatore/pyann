"""Tests for reading and setting layer weights, and freezing layers."""

import pytest

import pyann
from pyann import (BatchNormalization, Conv1D, Conv2D, Dense, DepthwiseConv2D, Embedding, Flatten,
                   GlobalMaxPooling1D, Input, MaxPool2D, Sequential, Tokenizer, pad_sequences)
from pyann.exceptions import InvalidParameterError

np = pytest.importorskip("numpy")


def cnn():
    return Sequential([
        Input((2, 6, 6)),
        Conv2D(4, 3, padding=1),
        BatchNormalization(activation="relu"),
        DepthwiseConv2D(3, padding=1, depth_multiplier=2, activation="relu"),
        MaxPool2D(2),
        Flatten(),
        Dense(5, activation="relu"),
        Dense(3, activation="softmax"),
    ]).compile(optimizer="adam", loss="cce", learning_rate=0.01, convergence=0.0)


def text_model(**embedding):
    return Sequential([
        Embedding(16, 4, mask_zero=True, input_length=8, **embedding),
        Conv1D(6, 3, padding="same", groups=2, activation="relu"),
        GlobalMaxPooling1D(),
        Dense(2, activation="softmax"),
    ]).compile(optimizer="adam", loss="cce", learning_rate=0.01, convergence=0.0)


@pytest.fixture(scope="module")
def images():
    rng = np.random.RandomState(0)
    x = rng.rand(64, 2, 6, 6).astype(np.float32)
    y = np.eye(3, dtype=np.float32)[(x[:, 0, :3].mean(axis=(1, 2)) * 3).astype(int) % 3]
    return x, y


@pytest.fixture(scope="module")
def texts():
    """Ids 1..15 padded with 0; label: whether id 3 occurs."""
    rng = np.random.RandomState(1)
    x = np.zeros((200, 8), dtype=np.float32)
    for row in x:
        n = rng.randint(2, 9)
        row[:n] = rng.randint(1, 16, n)
    y = np.eye(2, dtype=np.float32)[(x == 3).any(axis=1).astype(int)]
    return x, y


class TestShapes:
    def test_every_kind(self):
        model = cnn()
        shapes = [[w.shape for w in model.get_layer_weights(i)] for i in range(len(model.layers))]
        assert shapes == [
            [(4, 2, 3, 3), (4,)],                    # Conv2D: (filters, channels, kh, kw)
            [(4,), (4,), (4,), (4,)],                # BatchNormalization: gamma, beta, mean, variance
            [(8, 1, 3, 3), (8,)],                    # DepthwiseConv2D: (filters, channels / groups, kh, kw)
            [],                                      # MaxPool2D
            [],                                      # Flatten
            [(5, 72), (5,)],                         # Dense: (units, inputs)
            [(3, 5), (3,)],
        ]
        text = text_model()
        assert [w.shape for w in text.get_weights()] == [(16, 4), (6, 2, 3), (6,), (2, 6), (2,)]

    def test_all_float32(self):
        assert all(w.dtype == np.float32 for w in cnn().get_weights())

    def test_layer_by_object_or_negative_index(self):
        model = text_model()
        emb = model.layers[0]
        assert model.get_layer_weights(emb)[0].shape == (16, 4)
        assert model.get_layer_weights(-1)[1].shape == (2,)
        with pytest.raises(IndexError):
            model.get_layer_weights(4)
        with pytest.raises(InvalidParameterError):
            model.get_layer_weights(Dense(2))
        with pytest.raises(TypeError):
            model.get_layer_weights("0")

    def test_layers_is_a_copy(self):
        model = text_model()
        model.layers.append(Dense(3))
        assert len(model.layers) == 4


class TestLayouts:
    """The arrays mean what their documented layouts say: a NumPy forward
    pass with them gives the model's predictions."""

    def test_dense(self):
        model = Sequential([Input((4,)), Dense(3, activation="relu"), Dense(2)]).compile(loss="mse")
        x = np.random.RandomState(2).rand(5, 4).astype(np.float32)
        w1, b1, w2, b2 = model.get_weights()
        expected = np.maximum(x @ w1.T + b1, 0) @ w2.T + b2
        assert np.allclose(model.predict(x), expected, atol=1e-5)

    def test_conv2d(self):
        model = Sequential([Input((2, 5, 5)), Conv2D(3, (3, 2)), Flatten(), Dense(1)]).compile(loss="mse")
        x = np.random.RandomState(3).rand(1, 2, 5, 5).astype(np.float32)
        w, b, wd, bd = model.get_weights()
        out = np.zeros((3, 3, 4), dtype=np.float32)
        for f in range(3):
            for i in range(3):
                for j in range(4):
                    out[f, i, j] = (x[0, :, i:i + 3, j:j + 2] * w[f]).sum() + b[f]
        assert np.allclose(model.predict(x)[0], wd @ out.ravel() + bd, atol=1e-4)

    def test_embedding_and_grouped_conv1d(self):
        model = text_model()
        ids = np.array([[3, 1, 4, 1, 5, 9, 2, 6]], dtype=np.float32)
        emb, w, b, wd, bd = model.get_weights()
        seq = emb[ids[0].astype(int)].T                     # (4 channels, 8 steps)
        padded = np.pad(seq, ((0, 0), (1, 1)))
        out = np.zeros((6, 8), dtype=np.float32)
        for f in range(6):
            first = f // 3 * 2                              # 2 groups of 2 channels, 3 filters each
            for t in range(8):
                out[f, t] = (padded[first:first + 2, t:t + 3] * w[f]).sum() + b[f]
        z = wd @ np.maximum(out, 0).max(axis=1) + bd
        assert np.allclose(model.predict(ids)[0], np.exp(z) / np.exp(z).sum(), atol=1e-5)


class TestSetting:
    def test_round_trip(self):
        model = text_model()
        new = [np.full(w.shape, 0.25 * (i + 1), dtype=np.float32) for i, w in enumerate(model.get_layer_weights(1))]
        model.set_layer_weights(1, new)
        assert all(np.array_equal(a, b) for a, b in zip(model.get_layer_weights(1), new))

    def test_accepts_lists_and_float64(self):
        model = Sequential([Input((2,)), Dense(2)]).compile(loss="mse")
        model.set_layer_weights(0, [[[1, 2], [3, 4]], np.array([0.5, -0.5])])
        assert np.allclose(model.predict(np.array([[1.0, 1.0]], dtype=np.float32)), [[3.5, 6.5]])

    def test_errors_change_nothing(self):
        model = text_model()
        before = model.get_layer_weights(1)
        w, b = before
        with pytest.raises(ValueError, match="2 parameter arrays"):
            model.set_layer_weights(1, [w])
        with pytest.raises(ValueError, match="shape"):
            model.set_layer_weights(1, [w.reshape(6, 6), b])
        bad = w.copy()
        bad[0, 0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            model.set_layer_weights(1, [bad, b])
        with pytest.raises(ValueError, match="parameter arrays"):
            model.set_weights(model.get_weights()[:-1])
        with pytest.raises(ValueError, match="shape"):
            model.set_weights([np.zeros((2, 2))] * 5)
        assert all(np.array_equal(a, c) for a, c in zip(model.get_layer_weights(1), before))

    def test_weights_before_training_are_where_training_starts(self, texts):
        x, y = texts
        model = text_model()
        start = model.get_weights()
        assert np.abs(start[0]).sum() > 0                   # initialized, not zeros
        model.compile(learning_rate=0.0)
        model.fit(x, y, epochs=1, verbose=0)
        assert all(np.array_equal(a, b) for a, b in zip(start, model.get_weights()))

    def test_copying_a_trained_model(self, images, texts, tmp_path):
        x, y = images
        trained = cnn()
        trained.fit(x, y, epochs=3, batch_size=16, verbose=0)
        copy = cnn().set_weights(trained.get_weights())
        assert np.array_equal(copy.predict(x), trained.predict(x))

        xt, yt = texts
        trained = text_model()
        trained.fit(xt, yt, epochs=3, verbose=0)
        copy = text_model().set_weights(trained.get_weights())
        assert np.array_equal(copy.predict(xt), trained.predict(xt))
        copy.save(tmp_path / "copy.nnb", format="binary")
        back = Sequential.load(tmp_path / "copy.nnb", format="binary")
        assert np.array_equal(back.predict(xt), trained.predict(xt))

    def test_set_weights_keeps_training_from_them(self, texts):
        # pre-trained vectors only: training starts from them and still learns
        x, y = texts
        teacher = text_model()
        teacher.fit(x, y, epochs=20, verbose=0)
        student = text_model()
        student.set_layer_weights(0, teacher.get_layer_weights(0))
        student.compile(learning_rate=0.0)
        student.fit(x, y, epochs=1, verbose=0)
        assert np.array_equal(student.get_layer_weights(0)[0], teacher.get_layer_weights(0)[0])


class TestFreezing:
    def test_set_trainable(self, texts):
        x, y = texts
        model = text_model()
        assert model.is_trainable(0)
        model.set_trainable(0, False)
        assert not model.is_trainable(0) and model.is_trainable(1)
        assert "(frozen)" in model.summary().splitlines()[3]
        before = model.get_weights()
        model.fit(x, y, epochs=3, verbose=0)
        after = model.get_weights()
        assert np.array_equal(before[0], after[0])
        assert not any(np.array_equal(a, b) for a, b in zip(before[1:], after[1:]))

        model.set_trainable(model.layers[0], True)
        model.fit(x, y, epochs=1, verbose=0)
        assert not np.array_equal(after[0], model.get_layer_weights(0)[0])

    def test_frozen_batch_norm_keeps_gamma_and_beta(self, images):
        x, y = images
        model = cnn().set_trainable(1, False)
        gamma, beta, mean, var = model.get_layer_weights(1)
        model.fit(x, y, epochs=2, batch_size=16, verbose=0)
        g2, b2, m2, v2 = model.get_layer_weights(1)
        assert np.array_equal(gamma, g2) and np.array_equal(beta, b2)
        assert not np.array_equal(var, v2)                  # running statistics still update

    def test_embedding_with_pretrained_frozen_vectors(self, texts):
        x, y = texts
        teacher = text_model()
        teacher.fit(x, y, epochs=30, verbose=0)
        vectors = teacher.get_layer_weights(0)[0]

        model = text_model(weights=vectors, trainable=False)
        assert "(frozen)" in model.summary()
        model.fit(x, y, epochs=30, verbose=0)
        assert np.array_equal(model.get_layer_weights(0)[0], vectors)
        assert model.evaluate(x, y) >= 0.9

    def test_embedding_weights_are_checked(self):
        with pytest.raises(ValueError, match="shape"):
            Embedding(16, 4, weights=np.zeros((4, 16)))
        with pytest.raises(ValueError, match="finite"):
            Embedding(16, 4, weights=np.full((16, 4), np.inf))
        layer = Embedding(16, 4, weights=np.ones((16, 4)))
        assert layer.weights.dtype == np.float32
        assert layer == Embedding(16, 4)                    # weights don't take part in comparisons
        assert "weights" not in repr(layer)

    def test_on_the_gpu(self, texts):
        # frozen layers stay as they are on Metal and Vulkan too, and values
        # set on an uploaded model reach the GPU
        x, y = texts
        active = pyann.enable_gpu()
        model = text_model().set_trainable(0, False)
        before = model.get_layer_weights(0)[0]
        model.fit(x, y, epochs=3, verbose=0)
        if active and pyann.backend_info()["gpu_active"] in ("metal", "vulkan"):
            assert model._gpu_uploaded
        assert np.array_equal(model.get_layer_weights(0)[0], before)

        dense = [np.ones((2, 6), dtype=np.float32), np.array([1.0, -1.0], dtype=np.float32)]
        model.set_layer_weights(-1, dense)
        cpu = text_model().set_weights(model.get_weights())
        assert np.allclose(model.predict(x), cpu.predict(x), atol=1e-5)


def test_glove_style_workflow(tmp_path):
    """Tokenizer -> vector file -> frozen Embedding, end to end."""
    texts = ["good film", "bad film", "good plot", "bad plot"] * 10
    labels = np.eye(2, dtype=np.float32)[[1, 0, 1, 0] * 10]
    path = tmp_path / "vectors.txt"
    path.write_text("good 1 0 0\nbad -1 0 0\nfilm 0 1 0\nplot 0 0 1\nunused 9 9 9\n")
    tok = Tokenizer().fit_on_texts(texts)
    matrix, found = pyann.load_word_vectors(path, tok)
    assert found == 4
    x = pad_sequences(tok.texts_to_sequences(texts), maxlen=3)
    model = Sequential([
        Embedding(tok.vocab_size, 3, mask_zero=True, input_length=3, weights=matrix, trainable=False),
        GlobalMaxPooling1D(),
        Dense(2, activation="softmax"),
    ]).compile(optimizer="adam", loss="cce", learning_rate=0.05, convergence=0.0)
    model.fit(x, labels, epochs=40, batch_size=8, verbose=0)
    assert model.evaluate(x, labels) == 1.0
    assert np.array_equal(model.get_layer_weights(0)[0], matrix)
