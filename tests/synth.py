"""Synthetic evidence with known ground truth: a simulated camera pipeline and
forgeries built from it."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def scene(h: int = 768, w: int = 1024, seed: int = 0) -> np.ndarray:
    """Natural-statistics test scene: 1/f colour noise plus textured polygons."""
    rng = np.random.default_rng(seed)
    fy, fx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    falloff = 1 / np.maximum(np.hypot(fy, fx), 1 / max(h, w)) ** 1.3

    def pink():
        field = np.fft.ifft2(np.fft.fft2(rng.standard_normal((h, w))) * falloff).real
        return (field - field.min()) / (np.ptp(field) + 1e-9)

    base = pink()
    img = np.stack([0.6 * base + 0.4 * pink() for _ in range(3)], axis=2)
    for _ in range(6):
        pts = rng.integers(0, [w, h], size=(rng.integers(3, 6), 2)).astype(np.int32)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        tint = rng.uniform(0.2, 0.9, 3)
        img[mask == 1] = 0.5 * img[mask == 1] + 0.5 * tint
    return np.clip(img * 235 + 10, 0, 255).astype(np.float32)


def camera(img: np.ndarray, noise: float = 2.0, seed: int = 1) -> np.ndarray:
    """Bayer-sample (RGGB), add sensor noise, demosaic: leaves CFA traces and uniform noise."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    mosaic = np.zeros((h, w), np.float32)
    mosaic[0::2, 0::2] = img[0::2, 0::2, 2]
    mosaic[0::2, 1::2] = img[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = img[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = img[1::2, 1::2, 0]
    mosaic += rng.normal(0, noise, mosaic.shape) * np.sqrt(mosaic / 128 + 0.2)
    raw = np.clip(mosaic, 0, 255).astype(np.uint8)
    return cv2.cvtColor(raw, cv2.COLOR_BayerBG2BGR)


def save(path: Path, bgr: np.ndarray, quality: int | None = 92) -> Path:
    if quality is None:
        cv2.imwrite(str(path), bgr)
    else:
        cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return path


def jpeg_roundtrip(bgr: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


REGION = (slice(250, 506), slice(403, 659))       # y, x of the forged area (not 8-aligned on purpose)


def build(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    original = camera(scene(seed=0))
    donor = camera(scene(seed=5), seed=6)
    out = {"original": save(folder / "original.jpg", original, 92)}
    out["original_png"] = save(folder / "original.png", original, None)

    # splice: donor content that was JPEG-compressed at low quality, pasted off-grid, saved at high quality
    spliced = original.copy()
    spliced[REGION] = jpeg_roundtrip(donor, 60)[100:356, 203:459]
    out["splice"] = save(folder / "splice.jpg", spliced, 95)

    # copy-move
    cloned = original.copy()
    cloned[REGION] = original[60:316, 40:296]
    out["clone"] = save(folder / "clone.jpg", cloned, 92)

    # locally resampled insert (scaled 1.3x, rotated 9 degrees), kept lossless
    ys, xs = REGION
    m = cv2.getRotationMatrix2D((128, 128), 9, 1.3)
    patch = cv2.warpAffine(donor[300:556, 500:756], m, (256, 256), flags=cv2.INTER_LINEAR)
    resampled = original.copy()
    resampled[ys, xs] = patch
    out["resampled"] = save(folder / "resampled.png", resampled, None)

    # local smoothing (airbrush / inpaint-like): removes noise and CFA traces
    smooth = original.copy()
    smooth[ys, xs] = cv2.GaussianBlur(original, (0, 0), 2.5)[ys, xs]
    out["smoothed"] = save(folder / "smoothed.png", smooth, None)

    # double compression and whole-image resize
    out["double"] = save(folder / "double.jpg", jpeg_roundtrip(original, 65), 90)
    out["resized"] = save(folder / "resized.png", cv2.resize(original, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC), None)
    out["cropped"] = save(folder / "cropped.png", jpeg_roundtrip(original, 80)[3:, 5:], None)

    # generator-like periodic artifact (transposed-convolution checkerboard)
    yy, xx = np.indices(original.shape[:2])
    lattice = 1.5 * np.cos(np.pi * yy / 2) * np.cos(np.pi * xx / 4)
    out["periodic"] = save(folder / "periodic.png", np.clip(original + lattice[..., None], 0, 255).astype(np.uint8), None)
    return out


def embed_watermark(bgr: np.ndarray, bits: np.ndarray, scale: int = 36) -> np.ndarray:
    """Reference embedding of the 'invisible-watermark' library (method dwtDct, scales [0, 36, 36]): in every 4x4
    block of the Haar approximation of U the largest coefficient is quantized to carry one bit."""
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    h, w = bgr.shape[0] // 4 * 4, bgr.shape[1] // 4 * 4
    u = yuv[:h, :w, 1].astype(np.float64)
    approx = (u[0::2, 0::2] + u[0::2, 1::2] + u[1::2, 0::2] + u[1::2, 1::2]) / 2
    new, num = approx.copy(), 0
    for i in range(approx.shape[0] // 4):
        for j in range(approx.shape[1] // 4):
            block = new[i * 4:i * 4 + 4, j * 4:j * 4 + 4]
            pos = int(np.argmax(np.abs(block.flatten()[1:]))) + 1
            y, x = divmod(pos, 4)
            block[y, x] = (abs(block[y, x]) // scale + 0.25 + 0.5 * bits[num % len(bits)]) * scale
            num += 1
    yuv[:h, :w, 1] = np.clip(u + np.kron(new - approx, np.ones((2, 2))) / 2, 0, 255)
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
