#!/usr/bin/env python3
"""
Texture Super-Resolution: upscaling game textures 2x with Conv2D and UpSample2D

Ship a texture at half resolution (a quarter of the memory) and rebuild the
full-size one at load time. A small convolutional network learns to add the
detail that plain filtering loses:

    Input(3, 16, 16) -> UpSample2D 2 -> 3 x Conv2D 32@3x3 relu
    -> Conv2D 3@3x3 linear (3, 32, 32)

The convolutions all work at the full size, so each output pixel gets its own
answer. Upsampling at the end instead (convolutions at half size, then one
convolution after UpSample2D) is cheaper per layer, but that last convolution
has to sharpen edges from blocky features, and it paints a faint checkerboard
over flat areas. The network predicts only the correction to
the bilinear upscale (what GPU texture filtering gives), which trains faster
than predicting the pixels, and it sees the pixels centered on zero (minus
0.5), which convolutions without batch normalization learn from much faster.

The training textures are procedural (marble, wood, bricks, stones, tiles,
fabric), so nothing is downloaded. The half-size textures are the 2x2 averages
of the full-size ones, like the next mipmap level. The model trains on 16x16
patches; for whole textures the weights are copied with get_weights /
set_weights into a model built for the texture's size, since convolution
weights don't depend on it. The example reports PSNR against the original for
nearest, bilinear and bicubic upscaling and the network, on textures it did not
train on, writes texture_upscale.png (rows: half size, bilinear, network,
original), then quantizes the network to int8 and exports a C header.

Try your own texture with --image FILE.png (8-bit RGB): it is halved, upscaled
and compared with the original, and FILE_2x.png is the original upscaled 2x.

Usage:
    python texture_upscale.py [-e EPOCHS] [--patches N] [--image FILE.png] [--gpu]
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from pyann import Sequential, Input, Conv2D, UpSample2D, set_seed
from pyann.callbacks import CosineAnnealing
from example_data import read_png, setup_backend, write_rgb_png

SIZE = 128          # procedural textures are SIZE x SIZE
PATCH = 16          # low-resolution training patches; the network outputs 2 * PATCH
MARGIN = 4          # low-resolution pixels of context around each output (more than the network sees)
CENTER = 0.5        # subtracted from the pixels the network sees
KINDS = ["marble", "wood", "bricks", "stones", "tiles", "fabric"]


# --- procedural textures, (3, SIZE, SIZE) in [0, 1] ---

def smooth_noise(rng, cells: int, size: int = SIZE) -> np.ndarray:
    """Value noise: random values on a cells x cells grid, smoothly interpolated."""
    grid = rng.random((cells + 1, cells + 1))
    t = np.arange(size) * cells / size
    i, f = t.astype(int), t % 1
    f = f * f * (3 - 2 * f)
    rows = grid[i] * (1 - f)[:, None] + grid[i + 1] * f[:, None]
    return rows[:, i] * (1 - f) + rows[:, i + 1] * f


def fbm(rng, base: int = 4, octaves: int = 5, size: int = SIZE) -> np.ndarray:
    """Fractal noise in [0, 1]: octaves of value noise, each twice as fine and half as strong."""
    total = sum(0.5 ** o * smooth_noise(rng, base * 2 ** o, size) for o in range(octaves))
    return total / (2 - 0.5 ** (octaves - 1))


def colorize(rng, v: np.ndarray, dark, light) -> np.ndarray:
    """Map values in [0, 1] between two colors, each jittered a little."""
    dark = np.clip(np.array(dark) + rng.normal(0, 0.04, 3), 0, 1)
    light = np.clip(np.array(light) + rng.normal(0, 0.04, 3), 0, 1)
    return dark[:, None, None] + (light - dark)[:, None, None] * np.clip(v, 0, 1)[None]


def make_texture(kind: str, rng) -> np.ndarray:
    y, x = np.mgrid[0:SIZE, 0:SIZE] / SIZE
    if kind == "marble":
        turbulence = fbm(rng, 3)
        v = 0.5 + 0.5 * np.sin((x * rng.uniform(2, 5) + y * rng.uniform(-2, 2) + 4 * turbulence) * np.pi)
        tex = colorize(rng, v ** 0.6, (0.35, 0.33, 0.36), (0.93, 0.91, 0.88))
    elif kind == "wood":
        cx, cy = rng.uniform(-0.5, 1.5, 2)
        r = np.hypot((x - cx) * rng.uniform(0.3, 1.0), y - cy)
        v = (r * rng.uniform(12, 24) + 1.5 * fbm(rng, 4)) % 1
        tex = colorize(rng, v ** 2 * 0.8 + 0.2 * fbm(rng, 16, 3), (0.35, 0.2, 0.09), (0.75, 0.52, 0.3))
    elif kind in ("bricks", "tiles"):
        if kind == "bricks":
            bw, bh = rng.integers(20, 33), rng.integers(9, 17)
        else:
            bw = bh = rng.integers(14, 33)
        row = (np.arange(SIZE) // bh)[:, None]
        px = np.arange(SIZE)[None, :] + (row % 2) * (bw // 2 if kind == "bricks" else 0)
        col = px // bw
        mortar = rng.integers(1, 4)
        joint = ((np.arange(SIZE)[:, None] % bh) < mortar) | ((px % bw) < mortar)
        shade = rng.random((SIZE // bh + 2, SIZE // bw + 3))[row, col]
        v = 0.45 + 0.35 * shade + 0.25 * (fbm(rng, 16, 3) - 0.5)
        colors = ((0.45, 0.16, 0.1), (0.8, 0.42, 0.3)) if kind == "bricks" else ((0.2, 0.3, 0.45), (0.65, 0.75, 0.85))
        tex = colorize(rng, v, *colors)
        tex = np.where(joint[None], np.array([0.7, 0.68, 0.64])[:, None, None] * (0.8 + 0.2 * fbm(rng, 32, 2)), tex)
    elif kind == "stones":
        points = rng.random((rng.integers(10, 25), 2))
        d = np.sort(np.hypot(x[..., None] - points[:, 0], y[..., None] - points[:, 1]), axis=-1)
        nearest = np.argmin(np.hypot(x[..., None] - points[:, 0], y[..., None] - points[:, 1]), axis=-1)
        shade = rng.random(len(points))[nearest]
        v = 0.3 + 0.5 * shade + 0.3 * (fbm(rng, 8, 4) - 0.5) - 0.3 * d[..., 0] / d[..., 0].max()
        v = np.where(d[..., 1] - d[..., 0] < rng.uniform(0.01, 0.025), 0.05, v)
        tex = colorize(rng, v, (0.2, 0.19, 0.17), (0.72, 0.68, 0.6))
    elif kind == "fabric":
        period = rng.integers(6, 13)               # finer weaves vanish when halved
        a, b = 2 * np.pi * np.arange(SIZE) / period, (np.arange(SIZE) // period)
        over = ((b[:, None] + b[None, :]) % 2).astype(float)
        warp = 0.5 + 0.5 * np.cos(a)[None, :]
        weft = 0.5 + 0.5 * np.cos(a)[:, None]
        v = over * warp + (1 - over) * weft
        hue = rng.random(3) * 0.6
        tex = colorize(rng, 0.25 + 0.6 * v + 0.15 * (fbm(rng, 8, 3) - 0.5), hue * 0.4, hue + 0.35)
    else:
        raise ValueError(kind)
    return np.clip(tex, 0, 1).astype(np.float32)


# --- resampling: (..., 3, H, W) images ---

def downscale(hr: np.ndarray) -> np.ndarray:
    """Halve the size by averaging each 2x2 block, like building the next mipmap level."""
    h, w = hr.shape[-2] // 2 * 2, hr.shape[-1] // 2 * 2
    hr = hr[..., :h, :w]
    return 0.25 * (hr[..., 0::2, 0::2] + hr[..., 1::2, 0::2] + hr[..., 0::2, 1::2] + hr[..., 1::2, 1::2])


def upscale_nearest(lr: np.ndarray) -> np.ndarray:
    return lr.repeat(2, axis=-2).repeat(2, axis=-1)


def _upscale_axis(lr: np.ndarray, axis: int, taps) -> np.ndarray:
    """Upscale one axis 2x with a filter: even outputs use taps, odd outputs the taps reversed."""
    lr = np.moveaxis(lr, axis, -1)
    k = len(taps)
    pad = np.pad(lr, [(0, 0)] * (lr.ndim - 1) + [(k, k)], mode="edge")
    n = lr.shape[-1]
    # output 2i sits a quarter pixel left of input i, output 2i + 1 a quarter right
    even = sum(t * pad[..., k + o:k + o + n] for o, t in zip(range(-(k // 2), k - k // 2), taps))
    odd = sum(t * pad[..., k + o:k + o + n] for o, t in zip(range(-(k // 2) + 1, k - k // 2 + 1), taps[::-1]))
    out = np.stack([even, odd], axis=-1).reshape(*lr.shape[:-1], 2 * n)
    return np.moveaxis(out, -1, axis)


def _keys(t: float, a: float = -0.5) -> float:
    """The Keys cubic convolution kernel (what image editors call bicubic)."""
    t = abs(t)
    if t <= 1:
        return (a + 2) * t ** 3 - (a + 3) * t ** 2 + 1
    return a * t ** 3 - 5 * a * t ** 2 + 8 * a * t - 4 * a if t < 2 else 0.0


BILINEAR = [0.25, 0.75]
BICUBIC = [_keys(t) for t in (1.75, 0.75, 0.25, 1.25)]


def upscale_bilinear(lr: np.ndarray) -> np.ndarray:
    return _upscale_axis(_upscale_axis(lr, -2, BILINEAR), -1, BILINEAR).astype(np.float32)


def upscale_bicubic(lr: np.ndarray) -> np.ndarray:
    return np.clip(_upscale_axis(_upscale_axis(lr, -2, BICUBIC), -1, BICUBIC), 0, 1).astype(np.float32)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB for pixels in [0, 1]; higher is closer."""
    return float(10 * np.log10(1.0 / np.mean((np.clip(a, 0, 1) - b) ** 2)))


# --- the network ---

def build_model(lr_height: int = PATCH, lr_width: int = PATCH) -> Sequential:
    model = Sequential([
        Input((3, lr_height, lr_width)),
        UpSample2D(size=2),                                                    # -> (3, 2h, 2w), blocky
        Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),  # -> (32, 2h, 2w)
        Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        Conv2D(filters=32, kernel_size=3, padding="same", activation="relu"),
        Conv2D(filters=3, kernel_size=3, padding="same", activation="linear"), # -> (3, 2h, 2w): the correction
    ])
    # xavier starts the linear output near zero, where the small corrections are;
    # the default init gives it outputs of several units, which take many epochs to undo
    model.compile(optimizer="adam", loss="mse", learning_rate=0.002, convergence=0.0, weight_init="xavier")
    return model


def make_patches(rng, count: int):
    """Random half-size patches, centered for the network, and the correction
    from their bilinear upscale to the original."""
    lr, residual = [], []
    hr_patch = 2 * PATCH
    textures_needed = max(len(KINDS), count // 40)
    for t in range(textures_needed):
        hr = make_texture(KINDS[t % len(KINDS)], rng)
        for _ in range(count // textures_needed + (t < count % textures_needed)):
            r, c = rng.integers(0, SIZE - hr_patch + 1, 2) // 2 * 2       # even, so the 2x2 blocks line up
            patch = hr[:, r:r + hr_patch, c:c + hr_patch]
            patch = np.rot90(patch, rng.integers(4), axes=(1, 2))           # rotations are free extra data
            small = downscale(patch)
            lr.append(small - CENTER)
            residual.append(patch - upscale_bilinear(small))
    return np.array(lr, dtype=np.float32), np.array(residual, dtype=np.float32)


def upscale_network(model: Sequential, lr: np.ndarray) -> np.ndarray:
    """Upscale whole (3, h, w) images with the trained patch model's weights.

    The image is padded by mirroring its edges, so the zero padding of the
    convolutions doesn't darken the borders, then cropped back.
    """
    h, w = lr.shape[-2:]
    full = build_model(h + 2 * MARGIN, w + 2 * MARGIN)
    full.set_weights(model.get_weights())                # same layers, any image size
    padded = np.pad(lr, [(0, 0)] * (lr.ndim - 2) + [(MARGIN, MARGIN)] * 2, mode="reflect")
    correction = full.predict(padded.reshape(-1, 3, h + 2 * MARGIN, w + 2 * MARGIN) - CENTER)
    correction = correction[..., 2 * MARGIN:-2 * MARGIN, 2 * MARGIN:-2 * MARGIN].reshape(*lr.shape[:-2], 2 * h, 2 * w)
    return np.clip(upscale_bilinear(lr) + correction, 0, 1)


def report(name: str, hr: np.ndarray, lr: np.ndarray, network: np.ndarray) -> None:
    results = [("nearest", upscale_nearest(lr)), ("bilinear", upscale_bilinear(lr)),
               ("bicubic", upscale_bicubic(lr)), ("network", network)]
    print(f"  {name:<10}" + "".join(f"{psnr(up, hr):>10.2f}" for _, up in results))


def picture_grid(rows, gap: int = 4, zoom: int = 2) -> np.ndarray:
    """Lay out (3, H, W) images in rows on a grey background, each zoomed so the pixels show."""
    images = [[im.repeat(zoom, axis=-2).repeat(zoom, axis=-1) for im in row] for row in rows]
    h = max(im.shape[-2] for row in images for im in row)
    w = max(im.shape[-1] for row in images for im in row)
    grid = np.full((3, len(images) * (h + gap) + gap, max(map(len, images)) * (w + gap) + gap), 0.5, np.float32)
    for i, row in enumerate(images):
        for j, im in enumerate(row):
            y0, x0 = gap + i * (h + gap), gap + j * (w + gap)
            grid[:, y0:y0 + im.shape[-2], x0:x0 + im.shape[-1]] = im
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("-e", "--epochs", type=int, default=12, help="training epochs (default: 12)")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="mini-batch size (default: 8)")
    parser.add_argument("--patches", type=int, default=6000, help="training patches (default: 6000)")
    parser.add_argument("--image", help="also upscale this 8-bit RGB PNG texture")
    parser.add_argument("--seed", type=int, default=1, help="random seed (default: 1)")
    parser.add_argument("--gpu", action="store_true", help="train on the GPU (Metal or Vulkan) if available")
    parser.add_argument("-o", "--output", default="texture_upscale.png", help="picture to write")
    args = parser.parse_args()

    setup_backend(args.gpu)
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    x_train, y_train = make_patches(rng, args.patches)

    model = build_model()
    print(model.summary())
    print(f"\nTraining on {len(x_train)} patches of {PATCH}x{PATCH} -> {2 * PATCH}x{2 * PATCH}\n", flush=True)
    start = time.time()
    model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
              lr_scheduler=CosineAnnealing(T_max=args.epochs, min_lr=1e-5))
    print(f"\nTrained in {time.time() - start:.1f} s")

    # new textures the model has never seen, two of each kind
    test_rng = np.random.default_rng(args.seed + 1000)
    hr_test = np.array([make_texture(kind, test_rng) for kind in KINDS for _ in range(2)])
    lr_test = downscale(hr_test)
    net_test = upscale_network(model, lr_test)

    print(f"\nPSNR in dB against the original {SIZE}x{SIZE} textures (higher is better):")
    print(f"  {'':<10}" + "".join(f"{name:>10}" for name in ("nearest", "bilinear", "bicubic", "network")))
    for k, kind in enumerate(KINDS):
        sl = slice(2 * k, 2 * k + 2)
        report(kind, hr_test[sl], lr_test[sl], net_test[sl])
    report("all", hr_test, lr_test, net_test)

    picks = range(0, len(hr_test), 2)
    rows = [[upscale_nearest(lr_test[i]) for i in picks], [upscale_bilinear(lr_test[i]) for i in picks],
            [net_test[i] for i in picks], [hr_test[i] for i in picks]]
    write_rgb_png(args.output, picture_grid(rows))
    print(f"\nWrote {args.output} (rows: half size, bilinear, network, original; one texture per kind)")

    if args.image:
        image = read_png(args.image)
        image = image[:, :image.shape[1] // 2 * 2, :image.shape[2] // 2 * 2]
        half = downscale(image)
        print(f"\n{args.image} ({image.shape[2]}x{image.shape[1]}), halved and upscaled again:")
        print(f"  {'':<10}" + "".join(f"{name:>10}" for name in ("nearest", "bilinear", "bicubic", "network")))
        report("PSNR", image, half, upscale_network(model, half))
        out = Path(args.image).with_name(Path(args.image).stem + "_2x.png")
        write_rgb_png(out, upscale_network(model, image))
        print(f"Wrote {out} ({2 * image.shape[2]}x{2 * image.shape[1]}, the original upscaled 2x)")

    # int8 for a game engine or a small device: the header runs the network in integer math
    q = model.quantize(x_train[:200])
    corner = lr_test[:, :, :PATCH, :PATCH]
    q_test = np.clip(upscale_bilinear(corner) + q.predict(corner - CENTER), 0, 1)
    f_test = np.clip(upscale_bilinear(corner) + model.predict(corner - CENTER), 0, 1)
    hr_corner = hr_test[:, :, :2 * PATCH, :2 * PATCH]
    print(f"\nint8 vs float32 on {PATCH}x{PATCH} patches: PSNR {psnr(q_test, hr_corner):.2f} dB vs "
          f"{psnr(f_test, hr_corner):.2f} dB")
    out_dir = "texture_deploy"
    os.makedirs(out_dir, exist_ok=True)
    header = q.export_c_header(os.path.join(out_dir, "texture_q.h"), "texture")
    params = sum(w.size for w in model.get_weights())
    print(f"C header: {header} ({params} parameters, {os.path.getsize(header) / 1024:.1f} KB of text, "
          f"with ann_rt.h beside it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
