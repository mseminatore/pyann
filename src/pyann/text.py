"""Text preprocessing: turn texts into the padded rows of token ids an
Embedding layer reads.

>>> tok = Tokenizer(num_words=10000, oov_token="<unk>")
>>> tok.fit_on_texts(train_texts)
>>> x = pad_sequences(tok.texts_to_sequences(train_texts), maxlen=200)
>>> model = Sequential([Embedding(tok.vocab_size, 64, mask_zero=True, input_length=200), ...])

Id 0 is kept for padding, so a model's Embedding can use mask_zero=True;
with an oov_token, id 1 stands for words not in the vocabulary. Pure Python;
pad_sequences returns a NumPy array when NumPy is installed.
"""

import json
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Union

from pyann.utils.compat import has_numpy

# Keras's default: punctuation and tabs/newlines become spaces (apostrophes stay, so "don't" is one word)
DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


class Tokenizer:
    """Builds a vocabulary from texts and maps texts to lists of token ids.

    Words (or characters, with char_level) get ids by how often they occur
    in the texts given to ``fit_on_texts``, the most frequent first (ties in
    the order first seen), starting at 1, or at 2 after the oov_token. Id 0
    is never assigned: it pads texts (see ``pad_sequences``).

    Args:
        num_words: Keep only the ids below this (the num_words - 1 most
            frequent words, less one for the oov_token); None keeps all
        oov_token: Word standing for those not in the vocabulary, as id 1;
            None drops unknown words from the sequences
        lower: Lower-case the texts first
        filters: Characters replaced by spaces before splitting
        split: Word separator (None splits on any whitespace)
        char_level: Make each character a token

    Attributes:
        word_index: word -> id, for every word seen (also those past num_words)
        index_word: id -> word
        word_counts: word -> count in the fitted texts
    """

    def __init__(self, num_words: Optional[int] = None, oov_token: Optional[str] = None, lower: bool = True,
                 filters: str = DEFAULT_FILTERS, split: Optional[str] = None, char_level: bool = False):
        if num_words is not None and (not isinstance(num_words, int) or isinstance(num_words, bool) or num_words < 1):
            raise ValueError(f"num_words must be None or a positive int, got {num_words!r}")
        self.num_words = num_words
        self.oov_token = oov_token
        self.lower = lower
        self.filters = filters
        self.split = split
        self.char_level = char_level
        self.word_counts: Dict[str, int] = {}
        self.word_index: Dict[str, int] = {}
        self.index_word: Dict[int, str] = {}

    def tokenize(self, text: str) -> List[str]:
        """A text's tokens, before they are mapped to ids."""
        if not isinstance(text, str):
            raise TypeError(f"texts must be strings, got {type(text).__name__}")
        if self.lower:
            text = text.lower()
        if self.char_level:
            return list(text)
        if self.filters:
            text = text.translate(str.maketrans({c: " " for c in self.filters}))
        return [w for w in text.split(self.split) if w]

    def fit_on_texts(self, texts: Iterable[str]) -> "Tokenizer":
        """Count the words of texts and (re)build the vocabulary from all counts so far."""
        if isinstance(texts, str):
            raise TypeError("fit_on_texts takes a list of texts, not one string")
        counts = Counter(self.word_counts)
        order = {w: i for i, w in enumerate(self.word_counts)}
        for text in texts:
            for word in self.tokenize(text):
                counts[word] += 1
                order.setdefault(word, len(order))
        self.word_counts = {w: counts[w] for w in sorted(order, key=order.get)}

        words = sorted(self.word_counts, key=lambda w: (-self.word_counts[w], order[w]))
        if self.oov_token is not None:
            words = [self.oov_token] + [w for w in words if w != self.oov_token]
        self.word_index = {w: i + 1 for i, w in enumerate(words)}
        self.index_word = {i: w for w, i in self.word_index.items()}
        return self

    @property
    def vocab_size(self) -> int:
        """Ids the sequences can hold, padding included: the Embedding's input_dim."""
        size = len(self.word_index) + 1
        return size if self.num_words is None else min(size, self.num_words)

    def _id(self, word: str) -> Optional[int]:
        i = self.word_index.get(word)
        if i is not None and (self.num_words is None or i < self.num_words):
            return i
        return None if self.oov_token is None else self.word_index[self.oov_token]

    def texts_to_sequences(self, texts: Iterable[str]) -> List[List[int]]:
        """Each text as its list of token ids (unknown words become the
        oov_token's id, or are dropped without one)."""
        if isinstance(texts, str):
            raise TypeError("texts_to_sequences takes a list of texts, not one string")
        if not self.word_index:
            raise ValueError("the Tokenizer has no vocabulary yet: call fit_on_texts first")
        result = []
        for text in texts:
            ids = (self._id(w) for w in self.tokenize(text))
            result.append([i for i in ids if i is not None])
        return result

    def sequences_to_texts(self, sequences: Iterable[Sequence[int]]) -> List[str]:
        """Token ids back to texts (words joined by spaces; padding and
        unknown ids left out)."""
        sep = "" if self.char_level else " "
        return [sep.join(self.index_word[int(i)] for i in seq if int(i) in self.index_word and
                         (self.num_words is None or int(i) < self.num_words)) for seq in sequences]

    def to_json(self) -> str:
        """The settings and vocabulary as JSON, for ``Tokenizer.from_json``:
        keep it with a trained model, so new texts get the same ids."""
        return json.dumps({
            "num_words": self.num_words, "oov_token": self.oov_token, "lower": self.lower,
            "filters": self.filters, "split": self.split, "char_level": self.char_level,
            "word_counts": self.word_counts, "word_index": self.word_index,
        })

    @classmethod
    def from_json(cls, text: str) -> "Tokenizer":
        """A Tokenizer saved with ``to_json``."""
        data = json.loads(text)
        tok = cls(num_words=data["num_words"], oov_token=data["oov_token"], lower=data["lower"],
                  filters=data["filters"], split=data["split"], char_level=data["char_level"])
        tok.word_counts = dict(data["word_counts"])
        tok.word_index = {w: int(i) for w, i in data["word_index"].items()}
        tok.index_word = {i: w for w, i in tok.word_index.items()}
        return tok


def pad_sequences(sequences: Iterable[Sequence[int]], maxlen: Optional[int] = None, padding: str = "post",
                  truncating: str = "post", value: int = 0):
    """Pad (or cut) sequences of token ids to one length.

    Padding goes after the text by default (Keras pads before): with a
    masked Embedding, a text padded at the end gives exactly what it would
    unpadded, also through strided layers, while padding before does so
    only when every stride is 1.

    Args:
        sequences: Lists of token ids
        maxlen: The length; None uses the longest sequence's
        padding: 'post' (after the ids) or 'pre' (before them)
        truncating: Which end longer sequences lose: 'post' (the end) or 'pre' (the start)
        value: The padding id (0, as the Tokenizer leaves it free)

    Returns:
        A float32 array (samples, maxlen) with NumPy, else a list of lists
    """
    if padding not in ("post", "pre") or truncating not in ("post", "pre"):
        raise ValueError("padding and truncating must be 'post' or 'pre'")
    seqs = [list(s) for s in sequences]
    if maxlen is None:
        maxlen = max((len(s) for s in seqs), default=0)
    if not isinstance(maxlen, int) or isinstance(maxlen, bool) or maxlen < 1:
        raise ValueError(f"maxlen must be a positive int, got {maxlen!r}")

    rows = []
    for s in seqs:
        if len(s) > maxlen:
            s = s[:maxlen] if truncating == "post" else s[-maxlen:]
        pad = [value] * (maxlen - len(s))
        rows.append(s + pad if padding == "post" else pad + s)
    if has_numpy():
        import numpy as np
        return np.array(rows, dtype=np.float32).reshape(len(rows), maxlen)
    return [[float(v) for v in row] for row in rows]
