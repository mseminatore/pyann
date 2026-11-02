#!/usr/bin/env python3
"""
Keyword Spotting: recognizing spoken commands with a tiny CNN

Recognizes eight spoken words (down, go, left, no, right, stop, up, yes) in
one-second clips, or none of them, small enough for a microcontroller:

    1 s of 16 kHz audio -> log-mel spectrogram (1, 49 frames, 40 bands)
    -> Conv2D 32@3x3 -> BatchNorm relu -> MaxPool2D 2
    -> DepthwiseConv2D 3x3 -> BatchNorm relu -> Conv2D 48@1x1 -> BatchNorm relu -> MaxPool2D 2
    -> DepthwiseConv2D 3x3 -> BatchNorm relu -> Conv2D 64@1x1 -> BatchNorm relu
    -> GlobalAveragePooling2D -> Dense 9 softmax

The spectrogram (computed with NumPy: 30 ms frames every 20 ms, 40 mel bands,
log energy) turns sound into a small image, and the depthwise-separable
convolutions (a 3x3 per channel, then a 1x1 across channels) keep the model to
a few thousand parameters. Training clips are shifted in time and mixed with
noise, so the model hears words wherever they fall in the second.

The data is Google's mini Speech Commands (8,000 clips, 182 MB, CC BY 4.0),
downloaded once into examples/data/ and read straight from the zip. Clips are
split by speaker, so the test set holds voices the model never heard. The
"(none)" class is generated noise and clips with the word mostly shifted out
of the second, so a window that catches only part of a word says so. --synthetic uses made-up "words" (tone
glides) instead, with no download.

After training, the model listens to a stream of test words separated by
noise, a second at a time every 100 ms, averages each window's probabilities
with its neighbors' and reports the words it is sure of. Then it is
quantized to int8 and exported as a C header (a device also needs the log-mel
front end, an FFT and the 40 mel filters, to make its inputs). The float model
is exported as a JavaScript module with a browser demo that listens to your
microphone: serve kws_deploy/ (python -m http.server --directory kws_deploy)
and open http://localhost:8000.
Classify your own recording with --wav FILE.wav (mono or stereo, any rate).

Usage:
    python keyword_spotting.py [-e EPOCHS] [--synthetic] [--wav FILE.wav] [--gpu]
"""

import argparse
import io
import json
import os
import shutil
import time
import urllib.request
import wave
import zipfile
import zlib
from pathlib import Path

import numpy as np

from pyann import (Sequential, Input, Conv2D, DepthwiseConv2D, MaxPool2D, BatchNormalization,
                   GlobalAveragePooling2D, Dense, set_seed)
from pyann.callbacks import EarlyStopping
from example_data import DATA_DIR, setup_backend

URL = "http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip"
WORDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
CLASSES = WORDS + ["(none)"]
RATE = 16000                    # samples per second; every clip is one second
FRAME, HOP, FFT, MELS = 480, 320, 512, 40   # 30 ms frames every 20 ms, 40 mel bands
FRAMES = 1 + (RATE - FRAME) // HOP           # 49


# --- audio ---

def read_wav(data: bytes) -> np.ndarray:
    """16-bit PCM WAV bytes to float32 samples in [-1, 1], mixed to mono and resampled to 16 kHz."""
    with wave.open(io.BytesIO(data)) as w:
        if w.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV files are supported")
        rate, channels = w.getframerate(), w.getnchannels()
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != RATE:                    # linear interpolation is enough for speech at these rates
        t = np.arange(int(len(samples) * RATE / rate)) * rate / RATE
        samples = np.interp(t, np.arange(len(samples)), samples).astype(np.float32)
    return samples


def fit_clip(samples: np.ndarray) -> np.ndarray:
    """Pad (centered) or crop (around the loudest part) to exactly one second."""
    if len(samples) <= RATE:
        pad = RATE - len(samples)
        return np.pad(samples, (pad // 2, pad - pad // 2))
    energy = np.cumsum(samples.astype(np.float64) ** 2)
    energy = energy[RATE // 10:] - energy[:-RATE // 10]          # each 100 ms window's energy
    center = int(np.argmax(energy)) + RATE // 20
    start = int(np.clip(center - RATE // 2, 0, len(samples) - RATE))
    return samples[start:start + RATE]


def download(data_dir: Path = DATA_DIR) -> Path:
    path = data_dir / "mini_speech_commands.zip"
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} (182 MB) ...", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(path)
    return path


def load_speech_commands(path: Path):
    """Every clip as (samples, word index, speaker id), read from the zip."""
    clips = []
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            parts = name.split("/")
            if name.startswith("__MACOSX") or not name.endswith(".wav") or parts[-2] not in WORDS:
                continue
            speaker = parts[-1].split("_")[0]
            clips.append((fit_clip(read_wav(z.read(name))), WORDS.index(parts[-2]), speaker))
    return clips


def synthetic_word(rng, word: int) -> np.ndarray:
    """A made-up "word": two or three tone glides, distinct per word, spoken by a
    random "speaker" (pitch and speed) at a random time, with some noise."""
    plans = [  # (start Hz, end Hz) of each syllable
        [(300, 200), (600, 500)], [(200, 400)], [(400, 400), (300, 600)], [(500, 250)],
        [(250, 500), (500, 250)], [(700, 700), (300, 300)], [(300, 700)], [(200, 300), (400, 800), (600, 300)],
    ]
    pitch, speed = rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.2)
    out = np.zeros(RATE, np.float32)
    pos = int(rng.uniform(0.1, 0.35) * RATE)
    for f0, f1 in plans[word]:
        n = int(0.18 * RATE * speed)
        if pos + n > RATE:
            break
        freq = pitch * np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(freq) / RATE
        tone = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)    # a few harmonics
        out[pos:pos + n] += (np.hanning(n) * tone * 0.3).astype(np.float32)
        pos += n + int(0.04 * RATE * speed)
    return out + rng.normal(0, 0.01, RATE).astype(np.float32)


def load_synthetic(rng, count: int):
    return [(synthetic_word(rng, i % len(WORDS)), i % len(WORDS), f"s{i // len(WORDS) % 200}")
            for i in range(count)]


def noise(rng, n: int = RATE) -> np.ndarray:
    """Background noise: white, pink-ish or brown-ish, at a random level."""
    white = rng.normal(0, 1, n)
    color = rng.integers(3)
    if color == 1:
        white = np.convolve(white, np.ones(8) / 8, mode="same") * 2
    elif color == 2:
        white = np.cumsum(white) / 60
        white -= np.convolve(white, np.ones(400) / 400, mode="same")
    return (white * 10 ** rng.uniform(-3.5, -1.5)).astype(np.float32)


def shift(clip: np.ndarray, samples: int) -> np.ndarray:
    """Move the audio later (or earlier, when negative), filling with zeros."""
    out = np.roll(clip, samples)
    if samples > 0:
        out[:samples] = 0
    elif samples < 0:
        out[samples:] = 0
    return out


def augment(rng, clip: np.ndarray) -> np.ndarray:
    """Shift by up to 250 ms, which keeps a centered half-second word inside the
    clip (a sliding window sees it at every offset), and add noise."""
    return shift(clip, int(rng.integers(-RATE // 4, RATE // 4 + 1))) + noise(rng)


def partial(rng, clip: np.ndarray) -> np.ndarray:
    """The word shifted 0.6 to 0.85 s out of the clip, either way, so at most its edge shows."""
    return shift(clip, int(rng.choice([-1, 1]) * rng.uniform(0.6, 0.85) * RATE)) + noise(rng)


# --- log-mel features ---

def mel_filters() -> np.ndarray:
    """(MELS, FFT // 2 + 1) triangular filters evenly spaced on the mel scale, 20 Hz to 7.6 kHz."""
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    edges = hz(np.linspace(mel(20), mel(7600), MELS + 2))
    bins = np.fft.rfftfreq(FFT, 1 / RATE)
    lower, center, upper = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    return np.maximum(0, np.minimum((bins - lower) / (center - lower), (upper - bins) / (upper - center)))


MEL = mel_filters()
WINDOW = np.hanning(FRAME)


def log_mel(clips: np.ndarray) -> np.ndarray:
    """(N, RATE) samples to (N, 1, FRAMES, MELS) log mel energies."""
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(FRAMES)[:, None]
    frames = clips[:, idx] * WINDOW                               # (N, FRAMES, FRAME)
    power = np.abs(np.fft.rfft(frames, FFT)) ** 2
    return np.log(power @ MEL.T + 1e-6).astype(np.float32)[:, None]


def write_frontend(path, mean: np.ndarray, std: np.ndarray) -> None:
    """The front end's constants for the browser demo (web/keyword_spotting/frontend.js):
    each mel filter as its first FFT bin and weights, and the per-band normalization."""
    filters = []
    for row in MEL:
        nonzero = np.flatnonzero(row)
        start, stop = (int(nonzero[0]), int(nonzero[-1]) + 1) if len(nonzero) else (0, 0)
        filters.append({"start": start, "weights": [round(float(v), 7) for v in row[start:stop]]})
    data = {"rate": RATE, "frame": FRAME, "hop": HOP, "fft": FFT, "mels": MELS, "frames": FRAMES,
            "classes": CLASSES, "filters": filters,
            "mean": [float(v) for v in mean], "std": [float(v) for v in std]}
    Path(path).write_text(json.dumps(data))


# --- the model ---

def build_model() -> Sequential:
    model = Sequential([
        Input((1, FRAMES, MELS)),
        Conv2D(32, 3, padding="same"),                       # -> (32, 49, 40)
        BatchNormalization(activation="relu"),
        MaxPool2D(2),                                        # -> (32, 24, 20)
        DepthwiseConv2D(3, padding="same"),                  # a 3x3 per channel
        BatchNormalization(activation="relu"),
        Conv2D(48, 1),                                       # mixes the channels -> (48, 24, 20)
        BatchNormalization(activation="relu"),
        MaxPool2D(2),                                        # -> (48, 12, 10)
        DepthwiseConv2D(3, padding="same"),
        BatchNormalization(activation="relu"),
        Conv2D(64, 1),                                       # -> (64, 12, 10)
        BatchNormalization(activation="relu"),
        GlobalAveragePooling2D(),                            # -> (64,)
        Dense(len(CLASSES), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="cce", learning_rate=0.002, convergence=0.0)
    return model


def one_hot(labels: np.ndarray) -> np.ndarray:
    return np.eye(len(CLASSES), dtype=np.float32)[labels]


def split_by_speaker(clips):
    """About 80% of the speakers for training, 10% for validation and 10% for testing."""
    parts = ([], [], [])
    for clip in clips:
        bucket = zlib.crc32(clip[2].encode()) % 10
        parts[2 if bucket == 0 else 1 if bucket == 1 else 0].append(clip)
    return parts


def make_set(rng, clips, copies: int, normalize):
    """Features and one-hot labels: copies augmented versions of each clip, plus
    "(none)" twice as often as an average word: noise alone, and clips with the
    word shifted mostly out."""
    audio, labels = [], []
    for _ in range(copies):
        for samples, word, _ in clips:
            audio.append(augment(rng, samples) if copies > 1 else samples + noise(rng))
            labels.append(word)
    for i in range(2 * len(audio) // len(WORDS)):
        audio.append(noise(rng) * rng.uniform(1, 5) if i % 2 else partial(rng, clips[rng.integers(len(clips))][0]))
        labels.append(len(WORDS))
    return normalize(log_mel(np.array(audio))), one_hot(np.array(labels))


def make_stream(rng, clips, words: int):
    """Test words one after another, each in its own second, separated by half
    a second of noise; with each word's (second it starts, word)."""
    gap = RATE // 2
    stream, said = [noise(rng, gap)], []
    for i in rng.choice(len(clips), words, replace=False):
        said.append((sum(map(len, stream)) / RATE, WORDS[clips[i][1]]))
        stream += [clips[i][0] + noise(rng), noise(rng, gap)]
    return np.concatenate(stream), said


def listen(predict, stream: np.ndarray, normalize, threshold: float):
    """Slide a one-second window along the stream every 100 ms and return the
    words heard as (second where the word's window starts, word, probability)."""
    starts = np.arange(0, len(stream) - RATE + 1, RATE // 10)
    probs = predict(normalize(log_mel(np.stack([stream[s:s + RATE] for s in starts]))))
    # average each window with its neighbors: a word is in the window for
    # several steps, a passing confusion usually for one
    probs = np.stack([np.convolve(probs[:, c], np.ones(3) / 3, mode="same") for c in range(len(CLASSES))], axis=1)
    # each run of windows where a word stays above the threshold is one
    # utterance; its peak is the window with the word in the middle
    heard = []
    for c, word in enumerate(WORDS):
        above = np.concatenate([[False], probs[:, c] >= threshold, [False]])
        edges = np.flatnonzero(above[1:] != above[:-1])
        for first, last in zip(edges[::2], edges[1::2]):
            peak = first + int(np.argmax(probs[first:last, c]))
            heard.append((starts[peak] / RATE, word, float(probs[peak, c])))
    # a dip below the threshold can split one utterance into runs: keep the
    # stronger of two peaks of the same word less than 0.6 s apart
    merged = []
    for t, word, p in sorted(heard):
        same = next((i for i, m in enumerate(merged) if m[1] == word and t - m[0] < 0.6), None)
        if same is None:
            merged.append((t, word, p))
        elif p > merged[same][2]:
            merged[same] = (t, word, p)
    return sorted(merged)


def score(said, heard):
    """(hits, misses, false alarms): a word heard within 0.5 s of where it was said is a hit."""
    unmatched = list(said)
    for t, word, _ in heard:
        match = next((x for x in unmatched if x[1] == word and abs(x[0] - t) <= 0.5), None)
        if match:
            unmatched.remove(match)
    hits = len(said) - len(unmatched)
    return hits, len(unmatched), len(heard) - hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=20, help="most training epochs (default: 20)")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="mini-batch size (default: 32)")
    parser.add_argument("--copies", type=int, default=2, help="augmented copies of each training clip (default: 2)")
    parser.add_argument("--synthetic", action="store_true", help="made-up tone words instead of the download")
    parser.add_argument("--samples", type=int, default=2400, help="clips to make with --synthetic (default: 2400)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="smoothed probability needed to report a word in the stream (default: 0.7)")
    parser.add_argument("--stream-words", type=int, default=40, help="words in the test stream (default: 40)")
    parser.add_argument("--wav", action="append", default=[], help="also classify this WAV file (repeatable)")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU (Metal or Vulkan) if available")
    args = parser.parse_args()

    setup_backend(args.gpu)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    start = time.time()
    clips = load_synthetic(rng, args.samples) if args.synthetic else load_speech_commands(download())
    train, val, test = split_by_speaker(clips)
    print(f"{len(clips)} clips ({'synthetic' if args.synthetic else 'mini Speech Commands'}): "
          f"{len(train)} training, {len(val)} validation, {len(test)} test, split by speaker "
          f"({time.time() - start:.1f} s)")

    # each mel band scaled to zero mean and unit variance over the training clips
    raw = log_mel(np.array([c[0] for c in train[:2000]]))
    mean, std = raw.mean(axis=(0, 1, 2)), raw.std(axis=(0, 1, 2)) + 1e-6
    normalize = lambda feats: ((feats - mean) / std).astype(np.float32)

    x_train, y_train = make_set(rng, train, args.copies, normalize)
    x_val, y_val = make_set(rng, val, 1, normalize)
    x_test, y_test = make_set(rng, test, 1, normalize)

    model = build_model()
    print(model.summary())
    print(f"\nTraining on {len(x_train)} spectrograms ({args.copies} augmented copies of each clip, plus none)\n",
          flush=True)
    start = time.time()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_data=(x_val, y_val),
              callbacks=[EarlyStopping(patience=4, restore_best=True)])
    print(f"\nTrained in {time.time() - start:.1f} s")

    accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy (speakers it never heard): {accuracy:.1%}")
    predicted, actual = np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1)
    print("  " + "  ".join(f"{name} {np.mean(predicted[actual == c] == c):.0%}"
                           for c, name in enumerate(CLASSES) if np.any(actual == c)))
    confusion = np.zeros((len(CLASSES),) * 2, int)
    np.add.at(confusion, (actual, predicted), 1)
    np.fill_diagonal(confusion, 0)
    worst = np.dstack(np.unravel_index(np.argsort(confusion, axis=None)[::-1][:3], confusion.shape))[0]
    print("  most confused: " + ", ".join(f"{CLASSES[a]} heard as {CLASSES[b]} ({confusion[a, b]})"
                                          for a, b in worst if confusion[a, b]))

    stream, said = make_stream(rng, test, args.stream_words)
    heard = listen(model.predict, stream, normalize, args.threshold)
    hits, misses, false_alarms = score(said, heard)
    print(f"\nListening to {len(stream) / RATE:.0f} s of {len(said)} test words separated by noise, "
          f"a second every 100 ms (threshold {args.threshold:.0%}):")
    print(f"  heard {hits} of {len(said)} words, {false_alarms} false alarms")
    shown = said[8][0] - 0.6 if len(said) > 8 else float("inf")      # the first eight words
    print("  said:  " + ", ".join(f"{word} @{t:.1f}s" for t, word in said[:8]) + ", ...")
    early = [f"{word} @{t:.1f}s ({p:.0%})" for t, word, p in heard if t < shown]
    print("  heard: " + (", ".join(early) + ", ..." if early else "(nothing)"))

    for path in args.wav:
        clip = fit_clip(read_wav(Path(path).read_bytes()))
        p = model.predict(normalize(log_mel(clip[None])))[0]
        best = np.argsort(p)[::-1][:3]
        print(f"\n{path}: " + ", ".join(f"{CLASSES[c]} {p[c]:.0%}" for c in best))

    # int8 for a microcontroller
    q = model.quantize(x_train[:500])
    print(f"\nint8 test accuracy: {q.evaluate(x_test, y_test):.1%} (float32 {accuracy:.1%})")
    out_dir = "kws_deploy"
    os.makedirs(out_dir, exist_ok=True)
    header = q.export_c_header(os.path.join(out_dir, "kws_q.h"), "kws")
    params = sum(w.size for w in model.get_weights())
    print(f"C header: {header} ({params} parameters, about {params / 1024:.1f} KB as int8; "
          f"ann_rt.h beside it)")
    module = model.export_js(os.path.join(out_dir, "kws.js"))
    write_frontend(os.path.join(out_dir, "kws_frontend.json"), mean, std)
    for name in ("index.html", "frontend.js"):
        shutil.copyfile(Path(__file__).resolve().parent / "web" / "keyword_spotting" / name, os.path.join(out_dir, name))
    print(f"JavaScript: {module} ({os.path.getsize(module) / 1024:.0f} KB; ann_rt.js beside it) and a browser demo:\n"
          f"  python -m http.server --directory {out_dir}, then open http://localhost:8000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
