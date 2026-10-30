#!/usr/bin/env python3
"""
Review sentiment: type a movie review, get how positive it reads and which
words decided it

Loads the model imdb_glove.py trained (run it first) with its tokenizer, so
a new text gets the same word ids, padded the same way, and predicts it.
Then it predicts the review again once per word with that word left out:
the change in the positive probability is how much the word mattered, so
"not" shows up flipping "bad".

--int8 runs the int8 model (quantized here from the training rows saved
with the model), as the C header beside it would on a microcontroller.

Usage:
    python imdb_review.py                        # type reviews, an empty line quits
    python imdb_review.py --text "the plot was dull but the acting was superb"
    python imdb_review.py --int8 [--dir DIR] [--words N]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pyann import Sequential, Tokenizer, pad_sequences
from example_data import DATA_DIR


class Reviewer:
    """A saved model and tokenizer, with what turns a text into its row of ids."""

    def __init__(self, folder: Path, int8: bool = False):
        missing = [name for name in ("model.nnb", "tokenizer.json", "settings.json", "calibration.npy")
                   if not (folder / name).exists()]
        if missing:
            sys.exit(f"{folder} has no trained model ({', '.join(missing)} missing): run imdb_glove.py first")
        self.tok = Tokenizer.from_json((folder / "tokenizer.json").read_text())
        settings = json.loads((folder / "settings.json").read_text())
        self.maxlen, self.truncating = settings["maxlen"], settings["truncating"]
        model = Sequential.load(folder / "model.nnb", format="binary")
        self.model = model.quantize(np.load(folder / "calibration.npy")) if int8 else model

    def positive(self, sequences) -> np.ndarray:
        """The positive probability of each list of word ids."""
        x = pad_sequences(sequences, maxlen=self.maxlen, truncating=self.truncating)
        return np.asarray(self.model.predict(x))[:, 1]

    def review(self, text: str):
        """The text's positive probability, and each word with how much leaving it out lowers it."""
        words = self.tok.tokenize(text)
        ids = self.tok.texts_to_sequences([text])[0]
        if len(words) != len(ids):              # a tokenizer without <unk> drops unknown words
            words = self.tok.sequences_to_texts([[i] for i in ids])
        if len(ids) > self.maxlen:              # only the last maxlen words are read
            words, ids = words[-self.maxlen:], ids[-self.maxlen:]
        variants = [ids] + [ids[:i] + ids[i + 1:] for i in range(len(ids))]
        p = self.positive(variants)
        unknown = self.tok.word_index.get(self.tok.oov_token)
        effects = [(word + (" (unknown)" if i == unknown else ""), float(p[0] - without))
                   for word, i, without in zip(words, ids, p[1:])]
        return float(p[0]), effects


def show(reviewer: Reviewer, text: str, count: int) -> None:
    p, effects = reviewer.review(text)
    verdict = "positive" if p >= 0.5 else "negative"
    print(f"  {p:6.1%} positive: {verdict}")
    if not effects:
        print("  (no words it knows)")
        return
    for label, sign in (("pushed it up", 1), ("pulled it down", -1)):
        top = sorted((e for e in effects if sign * e[1] > 0.005), key=lambda e: -abs(e[1]))[:count]
        if top:
            print(f"  {label}: " + ", ".join(f"{word} ({delta:+.0%})" for word, delta in top))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dir", default=str(DATA_DIR / "imdb"), help="the model imdb_glove.py saved")
    parser.add_argument("--text", action="append", help="a review to predict (may repeat); else type them")
    parser.add_argument("--int8", action="store_true", help="use the int8 model")
    parser.add_argument("--words", type=int, default=5, help="words to show each way (default: 5)")
    args = parser.parse_args()

    reviewer = Reviewer(Path(args.dir), int8=args.int8)
    print(f"{'int8' if args.int8 else 'float32'} model, {reviewer.tok.vocab_size} word ids, "
          f"reviews of up to {reviewer.maxlen} words\n")

    if args.text:
        for text in args.text:
            print(f'"{text}"')
            show(reviewer, text, args.words)
        return 0

    print("Type a review (an empty line quits).")
    while True:
        try:
            text = input("review> ").strip()
        except EOFError:
            break
        if not text:
            break
        show(reviewer, text, args.words)
    return 0


if __name__ == "__main__":
    sys.exit(main())
