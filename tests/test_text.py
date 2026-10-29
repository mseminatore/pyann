"""Tests for text models: Tokenizer, pad_sequences, Embedding (with padding
masks) and global max pooling."""

import gzip
import json
import warnings

import pytest

import pyann
import pyann.utils.compat as compat
from pyann import (BatchNormalization, Conv1D, Conv2D, Dense, Embedding, GlobalAveragePooling1D,
                   GlobalMaxPool2D, GlobalMaxPooling1D, GlobalMaxPooling2D, MaxPool1D, Sequential, Tokenizer,
                   load_word_vectors, pad_sequences, strip_html)
from pyann.exceptions import NetworkError

np = pytest.importorskip("numpy")


# --- Tokenizer ---------------------------------------------------------------

class TestTokenizer:
    def test_ids_by_frequency_then_first_seen(self):
        tok = Tokenizer().fit_on_texts(["b a c", "a c", "a d"])
        # a: 3, c: 2, b and d: 1 each, b seen first
        assert tok.word_index == {"a": 1, "c": 2, "b": 3, "d": 4}
        assert tok.index_word[2] == "c"
        assert tok.word_counts == {"b": 1, "a": 3, "c": 2, "d": 1}
        assert tok.vocab_size == 5

    def test_lowercases_and_filters_punctuation(self):
        tok = Tokenizer().fit_on_texts(["Hello, World! Don't stop."])
        assert set(tok.word_index) == {"hello", "world", "don't", "stop"}
        assert tok.texts_to_sequences(["WORLD hello"]) == [[tok.word_index["world"], tok.word_index["hello"]]]

    def test_unknown_words_dropped_without_oov_token(self):
        tok = Tokenizer().fit_on_texts(["a b"])
        assert tok.texts_to_sequences(["a z b"]) == [[1, 2]]

    def test_oov_token_is_id_1(self):
        tok = Tokenizer(oov_token="<unk>").fit_on_texts(["a b a"])
        assert tok.word_index == {"<unk>": 1, "a": 2, "b": 3}
        assert tok.texts_to_sequences(["a z b"]) == [[2, 1, 3]]

    def test_num_words_keeps_the_most_frequent(self):
        tok = Tokenizer(num_words=3).fit_on_texts(["a a a b b c"])
        assert tok.texts_to_sequences(["a b c"]) == [[1, 2]]
        assert tok.vocab_size == 3
        oov = Tokenizer(num_words=3, oov_token="?").fit_on_texts(["a a a b b c"])
        assert oov.texts_to_sequences(["a b c"]) == [[2, 1, 1]]
        assert oov.sequences_to_texts([[2, 1, 0]]) == ["a ?"]

    def test_char_level(self):
        tok = Tokenizer(char_level=True).fit_on_texts(["abca"])
        assert tok.texts_to_sequences(["ab"]) == [[1, 2]]
        assert tok.sequences_to_texts([[2, 1]]) == ["ba"]

    def test_fitting_again_adds_to_the_counts(self):
        tok = Tokenizer().fit_on_texts(["a b"]).fit_on_texts(["b c", "b"])
        assert tok.word_counts == {"a": 1, "b": 3, "c": 1}
        assert tok.word_index["b"] == 1

    def test_json_round_trip(self):
        tok = Tokenizer(num_words=50, oov_token="<unk>").fit_on_texts(["one two two three"])
        back = Tokenizer.from_json(tok.to_json())
        assert back.word_index == tok.word_index and back.num_words == 50
        assert back.texts_to_sequences(["two four"]) == tok.texts_to_sequences(["two four"])
        json.loads(tok.to_json())

    def test_errors(self):
        with pytest.raises(TypeError):
            Tokenizer().fit_on_texts("one string")
        with pytest.raises(ValueError):
            Tokenizer().texts_to_sequences(["not fitted"])
        with pytest.raises(ValueError):
            Tokenizer(num_words=0)
        with pytest.raises(TypeError):
            Tokenizer().fit_on_texts([1, 2])


class TestPadSequences:
    def test_post_padding_by_default(self):
        x = pad_sequences([[1, 2, 3], [4]], maxlen=4)
        assert x.dtype == np.float32 and x.tolist() == [[1, 2, 3, 0], [4, 0, 0, 0]]

    def test_pre_padding_and_truncating(self):
        assert pad_sequences([[1, 2, 3], [4]], maxlen=2, padding="pre", truncating="pre").tolist() == [[2, 3], [0, 4]]
        assert pad_sequences([[1, 2, 3]], maxlen=2).tolist() == [[1, 2]]

    def test_maxlen_defaults_to_the_longest(self):
        assert pad_sequences([[1], [2, 3, 4]], value=9).tolist() == [[1, 9, 9], [2, 3, 4]]

    def test_bad_arguments(self):
        with pytest.raises(ValueError):
            pad_sequences([[1]], padding="middle")
        with pytest.raises(ValueError):
            pad_sequences([[1]], maxlen=0)

    def test_without_numpy(self, monkeypatch):
        monkeypatch.setattr(compat, "HAS_NUMPY", False)
        assert pad_sequences([[1, 2]], maxlen=3) == [[1.0, 2.0, 0.0]]


# --- layers ------------------------------------------------------------------

class TestEmbeddingLayer:
    def test_mask_zero_pads_with_0(self):
        layer = Embedding(10, 4, mask_zero=True)
        assert layer.padding_idx == 0 and layer.mask_padding

    def test_bad_settings(self):
        for kwargs in ({"input_dim": 0, "output_dim": 4}, {"input_dim": 10, "output_dim": 0},
                       {"input_dim": 10, "output_dim": 4, "padding_idx": 10},
                       {"input_dim": 10, "output_dim": 4, "mask_padding": True},
                       {"input_dim": 10, "output_dim": 4, "mask_zero": True, "padding_idx": 3},
                       {"input_dim": 10, "output_dim": 4, "input_length": 0}):
            with pytest.raises(ValueError):
                Embedding(**kwargs)

    def test_aliases(self):
        assert GlobalMaxPooling1D is GlobalMaxPooling2D is GlobalMaxPool2D
        assert GlobalAveragePooling1D is pyann.GlobalAveragePooling2D

    def test_input_length_sets_the_input_shape(self):
        model = Sequential([Embedding(30, 4, input_length=7), GlobalMaxPooling1D(), Dense(2, activation="softmax")])
        assert model._input_shape == (7,)
        lines = model.summary().splitlines()
        assert "Embedding(30, 4) -> (4, 7) - 120 params" in lines
        assert "GlobalMaxPooling1D -> (4,)" in lines

    def test_needs_a_flat_input(self):
        model = Sequential([Conv2D(2, 3, input_shape=(1, 5, 5)), Embedding(10, 2), Dense(1)])
        with pytest.raises(NetworkError, match="flat input of token ids"):
            model.summary()

    def test_sequence_output_shape(self):
        # an embedding then a Conv1D output: predictions are (N, channels, steps)
        model = Sequential([Embedding(12, 3, padding_idx=0, input_length=5), Conv1D(2, 3, padding="same")])
        model.compile(optimizer="adam", loss="mse")
        x = np.array([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]], dtype=np.float32)
        model.fit(x, np.zeros((2, 2, 5), dtype=np.float32), epochs=1, batch_size=2, verbose=0)
        assert model.predict(x).shape == (2, 2, 5)


# --- training ----------------------------------------------------------------

POSITIVE = ["good", "great", "fun", "lovely", "fine"]
NEGATIVE = ["bad", "awful", "dull", "boring", "poor"]
FILLER = ["the", "film", "was", "plot", "and", "acting", "story", "really", "quite", "it"]


def reviews(n, seed):
    """Texts of 3 to 12 words: one sentiment word, often negated ("not good"
    is negative), among filler; label 1 for positive."""
    rng = np.random.RandomState(seed)
    texts, labels = [], []
    for i in range(n):
        words = list(rng.choice(FILLER, rng.randint(2, 10)))
        positive = i % 2 == 0
        negate = rng.rand() < 0.5
        word = rng.choice(POSITIVE if positive != negate else NEGATIVE)
        phrase = ["not", word] if negate else [word]
        at = rng.randint(0, len(words) + 1)
        texts.append(" ".join(words[:at] + phrase + words[at:]))
        labels.append(int(positive))
    return texts, np.eye(2, dtype=np.float32)[labels]


def text_model(length, masked=True, pool="max"):
    pooling = GlobalMaxPooling1D() if pool == "max" else GlobalAveragePooling1D()
    model = Sequential([
        Embedding(32, 8, padding_idx=0, mask_padding=masked, input_length=length),
        Conv1D(16, 3, padding="same", activation="relu"),
        pooling,
        Dense(2, activation="softmax"),
    ])
    return model.compile(optimizer="adam", loss="cce", learning_rate=0.01, convergence=0.0)


@pytest.fixture(scope="module")
def tokenized():
    texts, y = reviews(600, 1)
    test_texts, y_test = reviews(200, 2)
    tok = Tokenizer(oov_token="<unk>").fit_on_texts(texts)
    x = pad_sequences(tok.texts_to_sequences(texts), maxlen=14)
    x_test = pad_sequences(tok.texts_to_sequences(test_texts), maxlen=14)
    return tok, x, y, x_test, y_test


class TestTextClassifier:
    def test_learns_negation(self, tokenized):
        tok, x, y, x_test, y_test = tokenized
        assert tok.vocab_size <= 32
        model = text_model(14)
        model.fit(x, y, epochs=40, batch_size=32, verbose=0)
        assert model.evaluate(x_test, y_test) >= 0.9

    def test_masking_ignores_where_the_padding_goes(self, tokenized):
        # stride-1 layers and global max pooling: the same text padded after
        # or before gives the same answer when masked, not otherwise
        tok, x, y, x_test, _ = tokenized
        seqs = tok.texts_to_sequences(reviews(20, 3)[0])
        post, pre = pad_sequences(seqs, maxlen=14), pad_sequences(seqs, maxlen=14, padding="pre")
        masked, plain = text_model(14, True), text_model(14, False)
        for model in (masked, plain):
            model.fit(x, y, epochs=3, batch_size=32, verbose=0)
        assert np.allclose(masked.predict(post), masked.predict(pre), atol=1e-5)
        assert not np.allclose(plain.predict(post), plain.predict(pre), atol=1e-3)

    def test_save_load_and_onnx_keep_the_mask(self, tokenized, tmp_path):
        _, x, y, x_test, _ = tokenized
        model = Sequential([
            Embedding(32, 6, mask_zero=True, input_length=14),
            Conv1D(8, 3, padding="same"),
            BatchNormalization(activation="relu"),
            MaxPool1D(2),
            GlobalAveragePooling1D(),
            Dense(2, activation="softmax"),
        ]).compile(optimizer="adam", loss="cce", convergence=0.0)
        model.fit(x, y, epochs=2, batch_size=32, verbose=0)
        expected = model.predict(x_test)
        for back in (Sequential.load(_save(model, tmp_path / "m.nnb", "binary"), format="binary"),
                     Sequential.load_onnx(_onnx(model, tmp_path / "m.onnx.json"))):
            emb = back._layers[0]
            assert isinstance(emb, Embedding) and (emb.input_dim, emb.output_dim) == (32, 6)
            assert emb.padding_idx == 0 and emb.mask_padding
            assert back.summary().splitlines()[2:] == model.summary().splitlines()[2:]
            assert np.allclose(back.predict(x_test), expected, atol=1e-6)

    def test_unmasked_embedding_loads_unmasked(self, tokenized, tmp_path):
        _, x, y, _, _ = tokenized
        model = text_model(14, masked=False)
        model.fit(x[:64], y[:64], epochs=1, batch_size=32, verbose=0)
        back = Sequential.load(_save(model, tmp_path / "plain.nna", "text"))
        assert back._layers[0].padding_idx == 0 and not back._layers[0].mask_padding
        assert isinstance(back._layers[2], GlobalMaxPooling2D)

    def test_quantizes_to_int8(self, tokenized):
        # the ids stay ids; the int8 model keeps the float one's accuracy, masks included
        _, x, y, x_test, y_test = tokenized
        model = text_model(14)
        model.fit(x, y, epochs=40, batch_size=32, verbose=0)
        q = model.quantize(x[:300])
        assert q.takes_ids and (q.input_scale, q.input_zero_point) == (1.0, 0)
        assert q.evaluate(x_test, y_test) >= model.evaluate(x_test, y_test) - 0.03
        assert np.abs(q.predict(x_test) - model.predict(x_test)).max() < 0.1

        ids = q.quantize_input(x_test[:20])
        assert ids.dtype == np.int32 and np.array_equal(ids, x_test[:20])
        logits = q.predict_int8(ids)
        assert logits.dtype == np.int8 and logits.shape == (20, 2)
        assert np.array_equal(logits.argmax(1), q.predict(x_test[:20]).argmax(1))
        with pytest.raises(ValueError, match="int32"):
            q.predict_int8(x_test[:1] + 0.5)

    def test_int8_header_matches(self, tokenized, tmp_path):
        # compiled with only ann_rt.h, the header gives predict_int8's outputs bit for bit
        import shutil
        import subprocess
        compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if compiler is None:
            pytest.skip("no C compiler")
        _, x, y, x_test, _ = tokenized
        model = text_model(14)
        model.fit(x, y, epochs=5, batch_size=32, verbose=0)
        q = model.quantize(x[:200])
        q.export_c_header(tmp_path / "reviews_q.h", "reviews_q")
        ids = q.quantize_input(x_test[:10])
        rows = ",".join("{" + ",".join(str(v) for v in row) + "}" for row in ids)
        (tmp_path / "main.c").write_text(
            '#include <stdio.h>\n#include "reviews_q.h"\n'
            f"static const int32_t ids[10][REVIEWS_Q_INPUT_SIZE] = {{{rows}}};\n"
            "int main(void) { static int8_t ws[REVIEWS_Q_WORKSPACE_SIZE]; int8_t out[REVIEWS_Q_OUTPUT_SIZE];\n"
            "  for (int r = 0; r < 10; r++) { if (reviews_q_predict(ids[r], out, ws)) return 1;\n"
            '    for (int k = 0; k < REVIEWS_Q_OUTPUT_SIZE; k++) printf("%d ", out[k]); printf("\\n"); }\n'
            "  return 0; }\n")
        exe = tmp_path / "main"
        subprocess.run([compiler, "-std=c99", "-Wall", "-Werror", "-DANN_RT_NO_FLOAT", "-o", str(exe),
                        str(tmp_path / "main.c")], check=True, cwd=tmp_path)
        out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
        assert np.array_equal(np.array(out, dtype=np.int8).reshape(10, -1), q.predict_int8(ids))

    def test_trains_on_the_gpu(self, tokenized):
        # on Metal and Vulkan the embedding network is uploaded; elsewhere it stays on the CPU
        _, x, y, x_test, y_test = tokenized
        active = pyann.enable_gpu()
        model = text_model(14)
        model.fit(x, y, epochs=30, batch_size=32, verbose=0)
        if active and pyann.backend_info()["gpu_active"] in ("metal", "vulkan"):
            assert model._gpu_uploaded
        assert model.evaluate(x_test, y_test) >= 0.85


def _save(model, path, fmt):
    model.save(path, format=fmt)
    return path


def _onnx(model, path):
    model.export_onnx(path)
    return path


class TestGlobalMaxPoolingImages:
    def test_cnn_with_global_max_pooling(self, tmp_path):
        rng = np.random.RandomState(0)
        x = 0.05 * rng.rand(64, 1, 6, 6).astype(np.float32)
        y = np.zeros((64, 2), dtype=np.float32)
        for i in range(64):
            r, c = rng.randint(0, 5, 2)
            x[i, 0, r:r + 2, c:c + 2] = 1.0 if i % 2 else 0.0      # a bright square, or none
            y[i, i % 2] = 1.0
        model = Sequential([Conv2D(4, 2, activation="relu", input_shape=(1, 6, 6)), GlobalMaxPooling2D(),
                            Dense(2, activation="softmax")])
        model.compile(optimizer="adam", loss="cce", learning_rate=0.02, convergence=0.0)
        model.fit(x, y, epochs=60, batch_size=16, verbose=0)
        assert model.evaluate(x, y) >= 0.95
        back = Sequential.load_onnx(_onnx(model, tmp_path / "gmp.onnx.json"))
        assert isinstance(back._layers[1], GlobalMaxPooling2D)
        assert np.allclose(back.predict(x), model.predict(x), atol=1e-6)
        q = model.quantize(x)
        assert q.evaluate(x, y) >= 0.9


# --- Cleaning text -----------------------------------------------------------

class TestStripHtml:
    def test_tags_entities_scripts_and_comments(self):
        text = ('Great film!<br /><br />I <b class="x">loved</b> it &amp; the &quot;ending&quot;'
                '<script type="text/javascript">var a = 1 < 2;</script><STYLE>p {}</STYLE><!-- note -->.')
        assert strip_html(text).split() == ["Great", "film!", "I", "loved", "it", "&", "the", '"ending"', "."]

    def test_declarations(self):
        assert strip_html('<?xml version="1.0"?><!DOCTYPE html><html><p>hi</p></html>').split() == ["hi"]

    def test_plain_text_unchanged(self):
        assert strip_html("no tags, 3 < 4 and 5 > 2") == "no tags, 3 < 4 and 5 > 2"

    def test_rejects_non_strings(self):
        with pytest.raises(TypeError):
            strip_html(None)


class TestPreprocess:
    def test_tokenizer_applies_it_first(self):
        assert Tokenizer().tokenize("one<br />two &amp;") == ["one", "br", "two", "amp"]
        assert Tokenizer(preprocess=strip_html).tokenize("one<br />two &amp;") == ["one", "two"]
        upper = Tokenizer(preprocess=lambda t: t.replace("colour", "color")).fit_on_texts(["colour color"])
        assert upper.word_index == {"color": 1}

    def test_checks(self):
        with pytest.raises(TypeError):
            Tokenizer(preprocess="strip_html")
        with pytest.raises(TypeError, match="return a string"):
            Tokenizer(preprocess=lambda t: None).fit_on_texts(["x"])

    def test_json_keeps_strip_html_by_name(self):
        tok = Tokenizer(preprocess=strip_html).fit_on_texts(["a <i>b</i>"])
        assert json.loads(tok.to_json())["preprocess"] == "strip_html"
        back = Tokenizer.from_json(tok.to_json())
        assert back.preprocess is strip_html
        assert back.texts_to_sequences(["<p>b</p>"]) == tok.texts_to_sequences(["<p>b</p>"])

    def test_json_with_a_function_of_your_own(self):
        def clean(text):
            return text.replace("_", " ")
        tok = Tokenizer(preprocess=clean).fit_on_texts(["a_b"])
        assert json.loads(tok.to_json())["preprocess"] == "custom"
        with pytest.warns(UserWarning, match="preprocess"):
            plain = Tokenizer.from_json(tok.to_json())
        assert plain.preprocess is None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            back = Tokenizer.from_json(tok.to_json(), preprocess=clean)
        assert back.texts_to_sequences(["b_a"]) == [[2, 1]]

    def test_old_json_without_the_field(self):
        tok = Tokenizer().fit_on_texts(["a b"])
        data = json.loads(tok.to_json())
        del data["preprocess"]
        assert Tokenizer.from_json(json.dumps(data)).word_index == tok.word_index


# --- Pre-trained word vectors ------------------------------------------------

GLOVE = """the 0.1 0.2 0.3
good 1 2 3
. . . 7 7 7
film -1 -2 -3
bad 4 5 6
good 9 9 9
"""


class TestLoadWordVectors:
    def test_glove_rows_follow_the_ids(self, tmp_path):
        path = tmp_path / "glove.txt"
        path.write_text(GLOVE)
        tok = Tokenizer(oov_token="<unk>").fit_on_texts(["good film good", "bad movie"])
        matrix, found = load_word_vectors(path, tok)
        assert matrix.shape == (tok.vocab_size, 3) and matrix.dtype == np.float32
        assert found == 3                                   # good, film, bad; not movie
        assert np.array_equal(matrix[tok.word_index["good"]], [1, 2, 3])    # the first "good"
        assert np.array_equal(matrix[tok.word_index["film"]], [-1, -2, -3])
        assert not matrix[0].any() and not matrix[1].any()                  # padding, <unk>
        assert not matrix[tok.word_index["movie"]].any()

    def test_word2vec_header_and_gzip(self, tmp_path):
        path = tmp_path / "vectors.vec.gz"
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write("3 2\nb 1 1\na 2 2\nc 3 3\n")
        matrix, found = load_word_vectors(path, {"a": 1, "b": 2, "z": 3})
        assert found == 2 and matrix.shape == (4, 2)
        assert np.array_equal(matrix, [[0, 0], [2, 2], [1, 1], [0, 0]])

    def test_num_words_and_vocab_size(self, tmp_path):
        path = tmp_path / "glove.txt"
        path.write_text(GLOVE)
        tok = Tokenizer(num_words=3).fit_on_texts(["good good good film film bad"])
        matrix, found = load_word_vectors(path, tok)
        assert matrix.shape == (3, 3) and found == 2        # bad is past num_words
        matrix, _ = load_word_vectors(path, tok, vocab_size=10)
        assert matrix.shape == (10, 3)

    def test_random_vectors_for_missing_words(self, tmp_path):
        path = tmp_path / "glove.txt"
        path.write_text(GLOVE)
        index = {"good": 1, "film": 2, **{f"w{i}": i for i in range(3, 400)}}
        matrix, found = load_word_vectors(path, index, missing="random", seed=0)
        assert found == 2 and not matrix[0].any()
        assert np.array_equal(matrix[1], [1, 2, 3])
        found_values = np.array([1, 2, 3, -1, -2, -3])
        assert abs(matrix[3:].std() - found_values.std()) < 0.2
        again, _ = load_word_vectors(path, index, missing="random", seed=0)
        assert np.array_equal(matrix, again)

    def test_errors(self, tmp_path):
        path = tmp_path / "glove.txt"
        path.write_text(GLOVE)
        with pytest.raises(ValueError, match="expected 4"):
            load_word_vectors(path, {"good": 1}, dim=4)
        bad = tmp_path / "bad.txt"
        bad.write_text("good 1 2 3\nfilm 1 2\n")
        with pytest.raises(ValueError, match="line 2"):
            load_word_vectors(bad, {"good": 1, "film": 2})
        header = tmp_path / "header.vec"
        header.write_text("2 5\ngood 1 2 3 4 5\n")
        with pytest.raises(ValueError, match="5-dimensional"):
            load_word_vectors(header, {"good": 1}, dim=3)
        with pytest.raises(ValueError, match="missing"):
            load_word_vectors(path, {"good": 1}, missing="mean")
        with pytest.raises(ValueError, match="fit_on_texts"):
            load_word_vectors(path, Tokenizer())
        with pytest.raises(TypeError):
            load_word_vectors(path, ["good"])
        empty = tmp_path / "empty.txt"
        empty.write_text("")
        with pytest.raises(ValueError, match="no vectors"):
            load_word_vectors(empty, {"good": 1})
