#!/usr/bin/env python3
"""
Text Classification: an Embedding and Conv1D network on short reviews

Each synthetic review mentions three things, each with a sentiment word that
may be negated: "the plot was not good, the acting was dull and the music
was great". Its label is what most of them say, so "not" matters, but only
for the word right after it. The data is synthetic, so nothing is downloaded.

Two models learn the same reviews:

  - a dense MLP on word counts (bag of words), which sees that "not" occurs
    but not which word it negates
  - Embedding + Conv1D + global max pooling, whose filters read short
    phrases such as "not good" wherever they occur

The texts go through pyann's Tokenizer and pad_sequences; the Embedding
masks the padding (mask_zero), so it never counts. The CNN is then saved
with its tokenizer, reloaded, and round-tripped through ONNX.

Usage:
    python text_classification.py [--epochs N] [--samples N] [--gpu]
"""

import argparse
import os
import tempfile

import numpy as np

from pyann import (Sequential, Embedding, Conv1D, GlobalMaxPooling1D, Dense, Input, Tokenizer, pad_sequences)
from example_data import setup_backend

TOPICS = ["plot", "acting", "music", "story", "ending", "cast", "dialogue", "pacing"]
POSITIVE = ["good", "great", "fun", "lovely", "moving", "clever", "superb", "charming"]
NEGATIVE = ["bad", "dull", "awful", "boring", "weak", "clumsy", "silly", "tedious"]
FILLER = ["really", "quite", "so", "very", "just", "honestly"]
MAX_LEN = 32


def make_reviews(rng: np.random.Generator, count: int):
    """count reviews and their labels (1 positive, 0 negative)."""
    texts, labels = [], []
    for _ in range(count):
        votes, clauses = 0, []
        for topic in rng.choice(TOPICS, 3, replace=False):
            positive = rng.random() < 0.5
            negated = rng.random() < 0.4
            word = rng.choice(POSITIVE if positive != negated else NEGATIVE)
            # the vote is what the clause says: "not bad" is positive
            votes += 1 if positive else -1
            words = ["the", topic, "was"]
            if rng.random() < 0.4:
                words.append(rng.choice(FILLER))
            words += ["not", word] if negated else [word]
            clauses.append(" ".join(words))
        texts.append(", ".join(clauses[:2]) + " and " + clauses[2])
        labels.append(int(votes > 0))
    return texts, np.eye(2, dtype=np.float32)[labels]


def bag_of_words(sequences, vocab_size: int) -> np.ndarray:
    """Word counts per review (padding left out)."""
    counts = np.zeros((len(sequences), vocab_size), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for token in seq:
            counts[i, token] += 1
    counts[:, 0] = 0
    return counts


def build_mlp(vocab_size: int) -> Sequential:
    model = Sequential([
        Input((vocab_size,)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(2, activation="softmax"),
    ])
    return model.compile(optimizer="adam", loss="cce", learning_rate=0.003, convergence=0.0)


def build_cnn(vocab_size: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, 16, mask_zero=True, input_length=MAX_LEN),  # -> (16, 32): a vector per word
        Conv1D(32, 3, padding="same", activation="relu"),                 # -> (32, 32): 3-word phrases
        Conv1D(32, 3, padding="same", activation="relu"),                 # -> (32, 32): their neighbours
        GlobalMaxPooling1D(),                                             # -> (32,): wherever they occur
        Dense(2, activation="softmax"),
    ])
    return model.compile(optimizer="adam", loss="cce", learning_rate=0.005, convergence=0.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=15, help="training epochs (default: 15)")
    parser.add_argument("--samples", type=int, default=3000, help="training reviews (default: 3000)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU when available")
    args = parser.parse_args()
    setup_backend(args.gpu)

    rng = np.random.default_rng(3)
    train_texts, y_train = make_reviews(rng, args.samples)
    test_texts, y_test = make_reviews(rng, 1000)
    print(f'{len(train_texts)} training and {len(test_texts)} test reviews, e.g. "{train_texts[0]}" '
          f'({"positive" if y_train[0, 1] else "negative"})\n')

    # word ids: 0 pads, 1 stands for words the training texts never had
    tok = Tokenizer(oov_token="<unk>").fit_on_texts(train_texts)
    train_seqs, test_seqs = tok.texts_to_sequences(train_texts), tok.texts_to_sequences(test_texts)
    x_train, x_test = pad_sequences(train_seqs, maxlen=MAX_LEN), pad_sequences(test_seqs, maxlen=MAX_LEN)
    print(f"Vocabulary: {tok.vocab_size} ids; each review is {MAX_LEN} ids, padded with 0 at the end\n")

    results = {}
    for name, build, x_tr, x_te in (
            ("bag-of-words MLP", build_mlp, bag_of_words(train_seqs, tok.vocab_size),
             bag_of_words(test_seqs, tok.vocab_size)),
            ("Embedding + Conv1D", build_cnn, x_train, x_test)):
        model = build(tok.vocab_size)
        model.fit(x_tr, y_train, epochs=args.epochs, batch_size=32, verbose=0)
        params = int(model.summary().splitlines()[-1].split(":")[1].replace(",", ""))
        results[name] = (model, model.evaluate(x_te, y_test))
        print(f"{name:>18s}: {results[name][1]:6.1%} test accuracy with {params:,} parameters")

    cnn, accuracy = results["Embedding + Conv1D"]
    print(f"\n{cnn.summary()}\n")

    reviews = ["the plot was not bad, the acting was great and the music was not good",
               "the plot was not bad, the acting was not great and the music was not good",
               "the ending was clever, the cast was superb and the pacing was very tedious"]
    probs = cnn.predict(pad_sequences(tok.texts_to_sequences(reviews), maxlen=MAX_LEN))
    for text, p in zip(reviews, probs):
        print(f'{p[1]:5.1%} positive: "{text}"')

    # keep the tokenizer with the model, so new texts get the same ids
    with tempfile.TemporaryDirectory() as tmp:
        model_path, tok_path = os.path.join(tmp, "reviews.nnb"), os.path.join(tmp, "reviews_tokenizer.json")
        cnn.save(model_path, format="binary")
        with open(tok_path, "w") as f:
            f.write(tok.to_json())
        back = Sequential.load(model_path, format="binary")
        with open(tok_path) as f:
            tok_back = Tokenizer.from_json(f.read())
        x_back = pad_sequences(tok_back.texts_to_sequences(test_texts), maxlen=MAX_LEN)
        print(f"\nReloaded model and tokenizer: {back.evaluate(x_back, y_test):.1%} (was {accuracy:.1%})")

        onnx_path = os.path.join(tmp, "reviews.onnx.json")
        cnn.export_onnx(onnx_path)
        same = np.array_equal(Sequential.load_onnx(onnx_path).predict(x_test), cnn.predict(x_test))
    print(f"ONNX round trip predicts the same: {'yes' if same else 'no'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
