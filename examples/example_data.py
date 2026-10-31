"""Shared helpers for the PyANN examples: Fashion-MNIST, images and backend setup.

Fashion-MNIST (https://github.com/zalandoresearch/fashion-mnist) is downloaded
once into ``examples/data/`` (about 30 MB). The Kaggle CSV files work too: pass
them with ``--train-csv`` / ``--test-csv`` or put them in the data directory.
"""

import gzip
import struct
import sys
import urllib.request
import zlib
from pathlib import Path

import numpy as np

import pyann

DATA_DIR = Path(__file__).resolve().parent / "data"
FASHION_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
FASHION_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", "fashion-mnist_train.csv"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", "fashion-mnist_test.csv"),
}

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def _download(name: str, data_dir: Path) -> Path:
    path = data_dir / name
    if not path.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {FASHION_URL + name} ...", flush=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(FASHION_URL + name, tmp)
        tmp.replace(path)
    return path


def _read_idx(path: Path) -> np.ndarray:
    data = gzip.decompress(path.read_bytes())
    ndim = data[3]
    shape = struct.unpack(f">{ndim}I", data[4:4 + 4 * ndim])
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def load_fashion_mnist(split: str = "train", csv: str = None, limit: int = None,
                       data_dir: Path = DATA_DIR):
    """Load a Fashion-MNIST split as (x, labels).

    x is float32 of shape (N, 784) scaled to [0, 1]; labels are int class ids.
    Uses ``csv`` (Kaggle format: label first, with a header row) when given or
    present in ``data_dir``; otherwise the original IDX files, downloaded once.
    """
    images, labels, csv_name = FASHION_FILES[split]
    csv_path = Path(csv) if csv else data_dir / csv_name
    if csv_path.exists():
        print(f"Loading {csv_path} ...", flush=True)
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float32, max_rows=limit)
        y, x = data[:, 0].astype(np.int64), data[:, 1:]
    else:
        if csv:
            sys.exit(f"Error: {csv} not found")
        x = _read_idx(_download(images, data_dir)).reshape(-1, 784).astype(np.float32)
        y = _read_idx(_download(labels, data_dir)).astype(np.int64)
        x, y = x[:limit], y[:limit]
    return x / np.float32(255.0), y


def one_hot(labels: np.ndarray, classes: int = 10) -> np.ndarray:
    return np.eye(classes, dtype=np.float32)[labels]


def as_images(x: np.ndarray) -> np.ndarray:
    """(N, 784) rows to (N, 1, 28, 28) channels-first images."""
    return x.reshape(-1, 1, 28, 28)


def setup_backend(use_gpu: bool) -> None:
    """Enable the GPU when asked and available, and print what is in use."""
    if use_gpu and not pyann.enable_gpu():
        print("No GPU backend or device available, using the CPU")
    info = pyann.backend_info()
    print(f"Backend: BLAS {info['blas'] or 'none'}, GPU {info['gpu_active'] or 'off'}, "
          f"{info['threads']} threads")


def ascii_image(pixels, width: int, height: int) -> str:
    """Render values in [0, 1] as text, darkest to brightest."""
    ramp = " .:-=+*#%@"
    rows = []
    for r in range(height):
        row = pixels[r * width:(r + 1) * width]
        rows.append("".join(ramp[min(len(ramp) - 1, max(0, int(v * len(ramp))))] for v in row))
    return "\n".join(rows)


def write_png(path, rows, zoom: int = 2, gap: int = 4) -> None:
    """Write a grid of 28x28 images as a grayscale PNG.

    ``rows`` is a list of image lists; each image holds 784 values in [0, 1].
    """
    side = 28 * zoom
    n_rows, n_cols = len(rows), max(len(r) for r in rows)
    width, height = n_cols * (side + gap) + gap, n_rows * (side + gap) + gap
    picture = np.full((height, width), 128, dtype=np.uint8)
    for i, images in enumerate(rows):
        for j, image in enumerate(images):
            pixels = np.clip(np.asarray(image, dtype=np.float32).reshape(28, 28), 0, 1)
            big = np.kron(pixels, np.ones((zoom, zoom), dtype=np.float32))
            y0, x0 = gap + i * (side + gap), gap + j * (side + gap)
            picture[y0:y0 + side, x0:x0 + side] = (big * 255 + 0.5).astype(np.uint8)
    _write_png_bytes(path, picture)


def write_rgb_png(path, picture) -> None:
    """Write an RGB picture with values in [0, 1], (3, H, W) or (H, W, 3), as a PNG."""
    picture = np.asarray(picture, dtype=np.float32)
    if picture.shape[0] == 3 and picture.shape[-1] != 3:
        picture = picture.transpose(1, 2, 0)
    _write_png_bytes(path, (np.clip(picture, 0, 1) * 255 + 0.5).astype(np.uint8))


def _write_png_bytes(path, picture: np.ndarray) -> None:
    """Write uint8 pixels, (H, W) grayscale or (H, W, 3) RGB, as a PNG."""
    height, width = picture.shape[:2]
    color_type = 2 if picture.ndim == 3 else 0

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    raw = b"".join(b"\x00" + picture[y].tobytes() for y in range(height))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 9))
           + chunk(b"IEND", b""))
    Path(path).write_bytes(png)


def read_png(path) -> np.ndarray:
    """Read an 8-bit PNG (grayscale, RGB, with or without alpha) as float32 RGB (3, H, W) in [0, 1].

    Handles what most texture files use; palette, 16-bit and interlaced PNGs
    are rejected (convert them to 8-bit RGB first).
    """
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"{path} is not a PNG file")
    pos, idat, header = 8, [], None
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        pos += 12 + length
    width, height, depth, color_type, _, _, interlace = header
    channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color_type)
    if depth != 8 or channels is None or interlace:
        raise ValueError(f"{path}: only 8-bit, non-interlaced grayscale or RGB(A) PNGs are supported")

    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    pixels = np.zeros((height, stride), dtype=np.uint8)
    prior = bytearray(stride)
    for y in range(height):
        kind, row = raw[y * (stride + 1)], bytearray(raw[y * (stride + 1) + 1:(y + 1) * (stride + 1)])
        if kind == 2:                                           # Up
            row = bytearray((a + b) & 255 for a, b in zip(row, prior))
        elif kind in (1, 3, 4):                                 # Sub, Average, Paeth
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                if kind == 1:
                    row[i] = (row[i] + left) & 255
                elif kind == 3:
                    row[i] = (row[i] + ((left + prior[i]) >> 1)) & 255
                else:
                    up_left = prior[i - channels] if i >= channels else 0
                    p = left + prior[i] - up_left
                    pa, pb, pc = abs(p - left), abs(p - prior[i]), abs(p - up_left)
                    best = left if pa <= pb and pa <= pc else prior[i] if pb <= pc else up_left
                    row[i] = (row[i] + best) & 255
        pixels[y] = np.frombuffer(bytes(row), dtype=np.uint8)
        prior = row
    image = pixels.reshape(height, width, channels)[:, :, :3 if channels >= 3 else 1]
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    return image.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
