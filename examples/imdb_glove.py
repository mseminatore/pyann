#!/usr/bin/env python3
"""
IMDB reviews: word vectors learned from scratch vs pre-trained GloVe vectors

50,000 movie reviews labelled positive or negative. Each review becomes a row
of word ids (pyann's Tokenizer, with strip_html for the reviews' <br /> tags),
with its contractions split as GloVe's text was ("don't" becomes "do n't",
so both pieces have vectors), and one network shape learns them three ways:

  1. Embedding learned from scratch, with the rest of the network
  2. Embedding started from GloVe vectors (trained on 6 billion words of
     Wikipedia and news) and frozen, so only the layers above it train,
     which is also faster
  3. model 2 fine-tuned: the GloVe vectors unfrozen and trained a little
     more at a low learning rate

The tokenizer is fitted on the training reviews only, so the test reviews'
words don't leak into the vocabulary. Words the training reviews never used
become <unk>.

On this task the vectors learned from scratch win, at every size tried (2,000
to 40,000 training reviews): GloVe learns from which words appear near which,
and "good" and "bad" appear in the same places, so its vectors for them are
about as close as those for "good" and "great". Sentiment needs exactly that
difference, which vectors learned from the labels get directly. Pre-trained
vectors suit tasks about topics and meaning better, and they still show the
API: load_word_vectors, Embedding(weights=..., trainable=False) and
set_trainable.

Then it shows each embedding's nearest words to a few probes and how close
it puts sentiment opposites (vectors read with get_layer_weights), keeps the
best model's weights in a .npz file (save_weights / load_weights), and
quantizes it to int8 for a C header: the 20,000-word table, most of the
model, becomes a quarter of its size.

The best model (by validation accuracy) is saved with its tokenizer in
--save-dir (examples/data/imdb by default) for imdb_review.py, which
predicts the sentiment of reviews you type.

Data, downloaded once into examples/data/:
  - the reviews: Stanford's aclImdb archive (84 MB; official 25k train and
    25k test split), or the Kaggle CSV "IMDB Dataset.csv" with --csv
    (review,sentiment), without its repeated reviews, split 80/10/10 before
    the tokenizer is fitted
  - GloVe: --glove PATH for a GloVe or word2vec text file you have, or
    --download-glove for glove.6B (862 MB zip, keeps the 100-dimension file).
    Without either, only model 1 is trained.

Usage:
    python imdb_glove.py [--csv "IMDB Dataset.csv"] [--glove FILE | --download-glove]
                         [-e EPOCHS] [--limit N] [--gpu] [--save-dir DIR]
"""

import argparse
import json
import os
import random
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from pyann import (Sequential, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Tokenizer, load_word_vectors,
                   pad_sequences, set_seed, split_contractions, strip_html)
from pyann.utils import load_text_csv, one_hot_encode
from example_data import DATA_DIR, setup_backend

IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_FILE = "glove.6B.100d.txt"
VOCAB = 20000       # most frequent training words kept
MAX_LEN = 200       # ids per review: longer ones keep their last 200 words, shorter ones are padded
DIM = 100           # vector length (GloVe's 100-dimension file)
PROBES = ["great", "awful", "boring", "funny"]


def download(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} ...", flush=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    return path


def split(texts, labels, fractions, seed=1):
    """Shuffle once, then cut into parts of the given fractions."""
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    parts, start = [], 0
    for f in fractions:
        end = start + int(round(f * len(order)))
        idx = order[start:end]
        parts.append(([texts[i] for i in idx], [labels[i] for i in idx]))
        start = end
    return parts


def load_reviews(csv: str = None):
    """(train, validation, test), each (texts, labels) with label 1 for positive."""
    if csv:
        data = load_text_csv(csv, columns=["review", "sentiment"])
        # the CSV repeats a few hundred reviews: keep one of each, so none is in two splits
        reviews = dict(zip(data["review"], data["sentiment"]))
        texts, labels = list(reviews), [int(s == "positive") for s in reviews.values()]
        # split before anything learns from the texts, the tokenizer included
        return split(texts, labels, (0.8, 0.1, 0.1))

    archive = download(IMDB_URL, DATA_DIR / "aclImdb_v1.tar.gz")
    print(f"Reading {archive} ...", flush=True)
    sets = {"train": ([], []), "test": ([], [])}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            parts = member.name.split("/")       # aclImdb/<train|test>/<pos|neg>/<id>_<stars>.txt
            if len(parts) == 4 and parts[1] in sets and parts[2] in ("pos", "neg") and member.isfile():
                text = tar.extractfile(member).read().decode("utf-8", errors="replace")
                sets[parts[1]][0].append(text)
                sets[parts[1]][1].append(int(parts[2] == "pos"))
    train, val = split(*sets["train"], (0.9, 0.1))
    return train, val, split(*sets["test"], (1.0,))[0]


def glove_path(args):
    if args.glove:
        return Path(args.glove)
    path = DATA_DIR / GLOVE_FILE
    if path.exists():
        return path
    if not args.download_glove:
        return None
    archive = download(GLOVE_URL, DATA_DIR / "glove.6B.zip")
    print(f"Extracting {GLOVE_FILE} ...", flush=True)
    with zipfile.ZipFile(archive) as z, z.open(GLOVE_FILE) as src, open(path, "wb") as dst:
        while chunk := src.read(1 << 20):
            dst.write(chunk)
    archive.unlink()                            # the other sizes aren't used
    return path


def build(vocab_size: int, weights=None, trainable=True) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, DIM, mask_zero=True, input_length=MAX_LEN, weights=weights, trainable=trainable),
        Conv1D(128, 5, padding="same", activation="relu", dropout=0.5),   # -> (128, 200): 5-word phrases
        GlobalMaxPooling1D(),                                             # -> (128,): wherever they occur
        Dense(2, activation="softmax"),
    ])
    return model.compile(optimizer="adam", loss="cce", learning_rate=0.001, convergence=0.0)


def train(name, model, data, epochs, results):
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = data
    start = time.time()
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_data=(x_val, y_val), verbose=0)
    accuracy = model.evaluate(x_test, y_test)
    # the vectors as they are now, and the validation accuracy that picks the best model
    results[name] = (accuracy, model.get_layer_weights(0)[0], history["val_accuracy"], model)
    print(f"{name:>28s}: {accuracy:6.1%} test accuracy ({history['val_accuracy']:.1%} validation), "
          f"{time.time() - start:.0f} s")


def similarity(vectors: np.ndarray, tok: Tokenizer, a: str, b: str) -> float:
    """Cosine similarity of two words' vectors (nan for a word not in the vocabulary)."""
    i, j = tok.word_index.get(a), tok.word_index.get(b)
    if i is None or j is None or max(i, j) >= len(vectors):
        return float("nan")
    u, v = vectors[i], vectors[j]
    return float(u @ v / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-12))


def nearest_words(vectors: np.ndarray, tok: Tokenizer, probe: str, count: int = 5):
    """The words whose vectors point most nearly the same way as the probe's."""
    i = tok.word_index.get(probe)
    if i is None or i >= len(vectors):
        return []
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
    scores = unit @ unit[i]
    scores[[0, 1, i]] = -2.0                    # padding, <unk> and the probe itself
    return [tok.index_word[j] for j in np.argsort(-scores)[:count]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--csv", help='the Kaggle "IMDB Dataset.csv" (review,sentiment) instead of the download')
    parser.add_argument("--glove", help="a GloVe or word2vec text file (e.g. glove.6B.100d.txt)")
    parser.add_argument("--download-glove", action="store_true", help="download glove.6B (862 MB) once")
    parser.add_argument("-e", "--epochs", type=int, default=3, help="training epochs per model (default: 3)")
    parser.add_argument("--limit", type=int, help="use at most this many training reviews")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU when available")
    parser.add_argument("--save-dir", default=str(DATA_DIR / "imdb"),
                        help="where to save the model for imdb_review.py (default: examples/data/imdb)")
    args = parser.parse_args()
    setup_backend(args.gpu)
    set_seed(1)         # the same starting weights, shuffles and dropout every run

    (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels) = load_reviews(args.csv)
    if args.limit:
        train_texts, train_labels = train_texts[:args.limit], train_labels[:args.limit]
    print(f"{len(train_texts)} training, {len(val_texts)} validation and {len(test_texts)} test reviews\n")

    # the vocabulary comes from the training reviews only; contractions are split as
    # GloVe's text was ("don't" -> "do n't"), so the pieces have vectors
    tok = Tokenizer(num_words=VOCAB, oov_token="<unk>",
                    preprocess=[strip_html, split_contractions]).fit_on_texts(train_texts)

    def rows(texts, labels):
        x = pad_sequences(tok.texts_to_sequences(texts), maxlen=MAX_LEN, truncating="pre")
        return x, one_hot_encode(np.array(labels), num_classes=2)

    data = (rows(train_texts, train_labels), rows(val_texts, val_labels), rows(test_texts, test_labels))
    unknown = np.mean(data[2][0][data[2][0] > 0] == 1)
    print(f"Vocabulary: {tok.vocab_size} ids; {unknown:.1%} of the test reviews' words are <unk>\n")

    results = {}
    scratch = build(tok.vocab_size)
    train("learned from scratch", scratch, data, args.epochs, results)

    path = glove_path(args)
    if path is None:
        print("\nFor GloVe vectors run with --download-glove (862 MB, once) or --glove FILE.")
    else:
        matrix, found = load_word_vectors(path, tok, dim=DIM, missing="random", seed=1)
        print(f"\nGloVe ({path.name}): {found} of {tok.vocab_size - 2} words have vectors; "
              f"the rest start random")
        glove = build(tok.vocab_size, weights=matrix, trainable=False)
        train("GloVe, frozen", glove, data, args.epochs, results)

        # unfreeze the vectors and nudge them, slowly, toward these reviews
        glove.set_trainable(0, True).compile(learning_rate=0.0001)
        train("GloVe, then fine-tuned", glove, data, 1, results)

    print("\nNearest words (cosine similarity of the embedding vectors):")
    for name, (_, vectors, _, _) in results.items():
        if name != "GloVe, then fine-tuned":       # barely moved from the frozen vectors
            for probe in PROBES:
                print(f"  {name:>20s}  {probe:>7s}: {', '.join(nearest_words(vectors, tok, probe))}")

    print("\nHow close are sentiment opposites? (cosine similarity, 1 = same direction)")
    for name, (_, vectors, _, _) in results.items():
        if name != "GloVe, then fine-tuned":
            print(f"  {name:>20s}: " + ", ".join(f"{a}/{b} {similarity(vectors, tok, a, b):.2f}"
                                                 for a, b in (("good", "bad"), ("good", "great"))))

    # the best model by validation accuracy goes on: its weights apart from it, int8, the demo
    best_name = max(results, key=lambda name: results[name][2])
    final = results[best_name][3]
    print(f"\nBest by validation accuracy: {best_name}")
    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "imdb_weights.npz")
        final.save_weights(weights)
        fresh = build(tok.vocab_size).load_weights(weights)
        same = np.array_equal(fresh.predict(data[2][0][:500]), final.predict(data[2][0][:500]))
        print(f"save_weights / load_weights into a new model: {'same predictions' if same else 'DIFFERENT predictions'}")

        # int8: the ids stay ids, the vectors become an int8 table
        (x_train, _), _, (x_test, y_test) = data
        q = final.quantize(x_train[:1000])
        header = Path(tmp) / "imdb_q.h"
        q.export_c_header(header, "imdb_q")
        params = sum(w.size for w in final.get_weights())
        print(f"int8: {q.evaluate(x_test, y_test):.1%} test accuracy (float32 {final.evaluate(x_test, y_test):.1%}); "
              f"parameters {params * 4 / 1e6:.1f} MB as float32, about {params / 1e6:.1f} MB as int8 "
              f"(C header imdb_q.h: imdb_q_predict(ids, out, ws))")

    save_for_review(Path(args.save_dir), final, tok, data[0][0], q)
    return 0


def save_for_review(folder: Path, model: Sequential, tok: Tokenizer, x_train: np.ndarray, q) -> None:
    """What imdb_review.py loads: the model, its tokenizer, how texts become
    rows, training rows to calibrate int8 on, and the int8 C header."""
    folder.mkdir(parents=True, exist_ok=True)
    model.save(folder / "model.nnb", format="binary")
    (folder / "tokenizer.json").write_text(tok.to_json())
    (folder / "settings.json").write_text(json.dumps({"maxlen": MAX_LEN, "truncating": "pre"}))
    np.save(folder / "calibration.npy", x_train[:1000])
    q.export_c_header(folder / "imdb_q.h", "imdb_q")    # and ann_rt.h beside it
    print(f"\nSaved the model and tokenizer in {folder}: try python imdb_review.py")


if __name__ == "__main__":
    sys.exit(main())
