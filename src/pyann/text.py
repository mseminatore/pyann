"""Text preprocessing: turn texts into the padded rows of token ids an
Embedding layer reads.

>>> tok = Tokenizer(num_words=10000, oov_token="<unk>")
>>> tok.fit_on_texts(train_texts)
>>> x = pad_sequences(tok.texts_to_sequences(train_texts), maxlen=200)
>>> model = Sequential([Embedding(tok.vocab_size, 64, mask_zero=True, input_length=200), ...])

Id 0 is kept for padding, so a model's Embedding can use mask_zero=True;
with an oov_token, id 1 stands for words not in the vocabulary. Pure Python;
pad_sequences returns a NumPy array when NumPy is installed, and
load_word_vectors (pre-trained GloVe, word2vec or fastText vectors for an
Embedding) needs NumPy.

The Tokenizer lower-cases texts and turns ASCII punctuation into spaces; it
does not know HTML. For scraped text such as IMDB reviews, pass
``preprocess=strip_html``, which removes tags and decodes entities.
"""

import gzip
import html
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from pyann.utils.compat import has_numpy

# Keras's default: punctuation and tabs/newlines become spaces (apostrophes stay, so "don't" is one word)
DEFAULT_FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'

_SCRIPTS = re.compile(r"<(script|style)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
_TAGS = re.compile(r"<(?:/?[A-Za-z]|[!?][A-Za-z])[^>]*>")    # tags and <!DOCTYPE>, not "3 < 4 and 5 > 2"


def strip_html(text: str) -> str:
    """Text without its HTML: tags (``<br />``, ``<p class=...>``) become
    spaces, comments and the contents of ``<script>`` and ``<style>`` go, and
    entities are decoded (``&amp;`` to ``&``, ``&#39;`` to ``'``).

    For ``Tokenizer(preprocess=strip_html)``: without it ``<br />`` becomes
    the word "br" and ``&amp;`` the word "amp".
    """
    if not isinstance(text, str):
        raise TypeError(f"texts must be strings, got {type(text).__name__}")
    text = _COMMENTS.sub(" ", _SCRIPTS.sub(" ", text))
    return html.unescape(_TAGS.sub(" ", text))


# preprocess functions of this module, which to_json saves by name
_BUILTIN_PREPROCESS = {"strip_html": strip_html}


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
        preprocess: A function applied to each text first (str to str),
            e.g. ``strip_html`` or your own cleanup. ``to_json`` saves
            ``strip_html`` by name; another function must be given to
            ``from_json`` again.

    Attributes:
        word_index: word -> id, for every word seen (also those past num_words)
        index_word: id -> word
        word_counts: word -> count in the fitted texts
    """

    def __init__(self, num_words: Optional[int] = None, oov_token: Optional[str] = None, lower: bool = True,
                 filters: str = DEFAULT_FILTERS, split: Optional[str] = None, char_level: bool = False,
                 preprocess: Optional[Callable[[str], str]] = None):
        if num_words is not None and (not isinstance(num_words, int) or isinstance(num_words, bool) or num_words < 1):
            raise ValueError(f"num_words must be None or a positive int, got {num_words!r}")
        if preprocess is not None and not callable(preprocess):
            raise TypeError(f"preprocess must be a function or None, got {type(preprocess).__name__}")
        self.preprocess = preprocess
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
        if self.preprocess is not None:
            text = self.preprocess(text)
            if not isinstance(text, str):
                raise TypeError(f"preprocess must return a string, got {type(text).__name__}")
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
            "preprocess": self._preprocess_name(),
            "word_counts": self.word_counts, "word_index": self.word_index,
        })

    def _preprocess_name(self) -> Optional[str]:
        """What to_json records: None, a built-in's name, or "custom"."""
        if self.preprocess is None:
            return None
        for name, fn in _BUILTIN_PREPROCESS.items():
            if self.preprocess is fn:
                return name
        return "custom"

    @classmethod
    def from_json(cls, text: str, preprocess: Optional[Callable[[str], str]] = None) -> "Tokenizer":
        """A Tokenizer saved with ``to_json``.

        Args:
            text: The JSON
            preprocess: The preprocess function, when it was one of your own
                (``strip_html`` is restored by name); without it texts are
                not preprocessed, with a warning
        """
        data = json.loads(text)
        saved = data.get("preprocess")
        if preprocess is None and saved in _BUILTIN_PREPROCESS:
            preprocess = _BUILTIN_PREPROCESS[saved]
        elif preprocess is None and saved is not None:
            warnings.warn("this Tokenizer was saved with a preprocess function of your own: pass it to "
                          "from_json(text, preprocess=...) to clean texts as before", stacklevel=2)
        tok = cls(num_words=data["num_words"], oov_token=data["oov_token"], lower=data["lower"],
                  filters=data["filters"], split=data["split"], char_level=data["char_level"],
                  preprocess=preprocess)
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


def _open_text(path: Path):
    """A text file, gzip-compressed when it ends in .gz."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def load_word_vectors(path: Union[str, Path], word_index: Union["Tokenizer", Dict[str, int]],
                      vocab_size: Optional[int] = None, dim: Optional[int] = None, missing: str = "zeros",
                      seed: Optional[int] = None, padding_idx: Optional[int] = 0) -> Tuple["np.ndarray", int]:
    """An Embedding's starting vectors from a pre-trained word vector file.

    Reads the text formats of GloVe (a word and its numbers per line) and
    word2vec/fastText (the same after a "count dim" header line), plain or
    gzip-compressed (``.gz``). Row i of the matrix is the vector of the word
    with id i; only the words in the vocabulary are kept, so a large file
    needs little memory. The binary word2vec format is not read: convert it
    to text first (e.g. gensim's ``save_word2vec_format(..., binary=False)``).

    The words must match the file's: GloVe's are lower case, as the
    Tokenizer's are by default. The first match wins if a word occurs twice.

    >>> matrix, found = load_word_vectors("glove.6B.100d.txt", tok)
    >>> print(f"{found} of {tok.vocab_size - 1} words have vectors")
    >>> Embedding(tok.vocab_size, 100, mask_zero=True, weights=matrix, trainable=False)

    Args:
        path: The vector file
        word_index: A fitted Tokenizer (its ids below ``vocab_size``), or a
            dict of word -> id
        vocab_size: Rows of the matrix (the Embedding's input_dim): the
            Tokenizer's vocab_size, or for a dict the largest id + 1
        dim: Vector length; None takes the file's (given, it is checked)
        missing: Vectors for words not in the file: "zeros", or "random",
            normal with the found vectors' mean and standard deviation (lets
            them train apart when the Embedding is trainable)
        seed: Seed for the random vectors
        padding_idx: Id whose row stays zero (the padding id), or None

    Returns:
        (matrix, found): a float32 (vocab_size, dim) array, and how many of
        the vocabulary's words (padding and the OOV token aside) had vectors
    """
    from pyann.utils.compat import has_numpy
    if not has_numpy():
        raise RuntimeError("load_word_vectors needs NumPy")
    import numpy as np

    if missing not in ("zeros", "random"):
        raise ValueError(f"missing must be 'zeros' or 'random', got {missing!r}")
    oov = None
    if isinstance(word_index, Tokenizer):
        if not word_index.word_index:
            raise ValueError("the Tokenizer has no vocabulary yet: call fit_on_texts first")
        oov = word_index.oov_token
        size = word_index.vocab_size if vocab_size is None else vocab_size
        index = word_index.word_index
    elif isinstance(word_index, dict):
        index = word_index
        size = (max(index.values(), default=0) + 1) if vocab_size is None else vocab_size
    else:
        raise TypeError(f"word_index must be a Tokenizer or a dict, got {type(word_index).__name__}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError(f"vocab_size must be a positive int, got {size!r}")
    if dim is not None and (not isinstance(dim, int) or isinstance(dim, bool) or dim < 1):
        raise ValueError(f"dim must be None or a positive int, got {dim!r}")

    # the words to look for and their ids
    wanted = {w: i for w, i in index.items() if 0 <= i < size and i != padding_idx and w != oov}
    rows: Dict[int, "np.ndarray"] = {}
    with _open_text(Path(path)) as f:
        first = True
        for line_no, line in enumerate(f, 1):
            line = line.rstrip("\n").rstrip("\r")
            if first:
                first = False
                parts = line.split()
                # word2vec / fastText: "count dim"
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    if dim is not None and int(parts[1]) != dim:
                        raise ValueError(f"{path} has {parts[1]}-dimensional vectors, not {dim}")
                    dim = int(parts[1])
                    continue
            word, _, rest = line.partition(" ")
            if word not in wanted or wanted[word] in rows:
                continue
            values = rest.split()
            if dim is None:
                dim = len(values)
            if len(values) > dim:
                continue    # a "word" with spaces in it (glove.840B has some) that starts like ours
            if len(values) < dim:
                raise ValueError(f"{path} line {line_no}: {len(values)} numbers for {word!r}, expected {dim}")
            rows[wanted[word]] = np.asarray(values, dtype=np.float32)
    if dim is None:
        raise ValueError(f"{path} has no vectors")

    if missing == "random" and rows:
        found_vectors = np.stack(list(rows.values()))
        rng = np.random.default_rng(seed)
        matrix = rng.normal(found_vectors.mean(), found_vectors.std(), (size, dim)).astype(np.float32)
    else:
        matrix = np.zeros((size, dim), dtype=np.float32)
    for i, v in rows.items():
        matrix[i] = v
    if padding_idx is not None and 0 <= padding_idx < size:
        matrix[padding_idx] = 0.0
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path} has vectors with NaN or infinite values")
    return matrix, len(rows)
