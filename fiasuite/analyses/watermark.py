"""Decoder for the invisible watermark that Stability AI's reference code embeds in generated images
('invisible-watermark' library, method dwtDct, MIT licence; re-implemented here with NumPy only).

Embedding: the U channel (OpenCV YUV) is Haar-transformed once; in every 4x4 block of the approximation band the
largest coefficient (the first one excluded) is quantized to (n + 0.25) * 36 for a 0 bit and (n + 0.75) * 36 for a
1 bit; the message repeats block after block. Decoding reads those bits back and takes a majority vote per bit.

Known messages: Stable Diffusion 1.x scripts 'StableDiffusionV1', Stable Diffusion 2.x 'SDV2', SDXL a fixed
48-bit number. Most community front-ends (AUTOMATIC1111, ComfyUI) do not embed it, and resizing or cropping
destroys it: only a positive result means something.
"""
from __future__ import annotations

from math import comb

import cv2
import numpy as np

from ..core import ALERT, INFO, Evidence, Finding, Result

SCALE, BLOCK = 36, 4


def _bits(text: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(text, np.uint8))


MESSAGES = {
    "Stable Diffusion 1.x reference scripts ('StableDiffusionV1')": _bits(b"StableDiffusionV1"),
    "Stable Diffusion 2.x reference scripts ('SDV2')": _bits(b"SDV2"),
    "Stable Diffusion XL (Stability AI generative-models)": np.array(
        [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]], np.uint8),
}


def votes(bgr: np.ndarray) -> np.ndarray:
    """Per 4x4 block of the Haar approximation of U: the embedded bit (0/1), in embedding order."""
    u = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)[:, :, 1].astype(np.float64)
    h, w = u.shape[0] // 4 * 4, u.shape[1] // 4 * 4
    u = u[:h, :w]
    approx = (u[0::2, 0::2] + u[0::2, 1::2] + u[1::2, 0::2] + u[1::2, 1::2]) / 2
    rows, cols = approx.shape[0] // BLOCK, approx.shape[1] // BLOCK
    blocks = approx[:rows * BLOCK, :cols * BLOCK].reshape(rows, BLOCK, cols, BLOCK).transpose(0, 2, 1, 3).reshape(-1, 16)
    pos = np.abs(blocks[:, 1:]).argmax(axis=1) + 1
    value = np.abs(blocks[np.arange(len(blocks)), pos])
    return (value % SCALE > 0.5 * SCALE).astype(np.uint8)


def decode(block_bits: np.ndarray, length: int) -> np.ndarray:
    index = np.arange(len(block_bits)) % length
    mean = np.bincount(index, weights=block_bits, minlength=length) / np.maximum(np.bincount(index, minlength=length), 1)
    return (mean * 255 > 127).astype(np.uint8)


def _p_value(matches: int, length: int) -> float:
    """Probability of at least this many matching bits by chance."""
    return sum(comb(length, k) for k in range(matches, length + 1)) / 2 ** length


def detect(ev: Evidence) -> Result:
    res = Result()
    if ev.width * ev.height < 256 * 256:
        res.findings.append(Finding(INFO, "Image too small to carry this watermark"))
        return res
    block_bits = votes(ev.bgr)
    found = None
    for name, message in MESSAGES.items():
        got = decode(block_bits, len(message))
        matches = int((got == message).sum())
        p = _p_value(matches, len(message))
        res.table.append((name, f"{matches} of {len(message)} bits match   (chance: {p:.1e})"))
        if p < 1e-6 and (found is None or p < found[1]):
            found = (name, p, matches, len(message))
    if found:
        name, p, matches, length = found
        res.score = 0.97
        res.findings.append(Finding(ALERT, "Stable Diffusion watermark found",
            f"{matches} of {length} bits of the message embedded by {name} were decoded from the pixels; the "
            f"probability of that happening by chance is {p:.0e}. The image, or the copy it derives from, was "
            "written by that software."))
    else:
        res.score = 0.05
        res.findings.append(Finding(INFO, "No known invisible watermark",
            "Means nothing by itself: most Stable Diffusion front-ends never embed this watermark, other generators "
            "use different marks (SynthID, used by Google, OpenAI, NVIDIA and Kakao, is checked under 'Watermark check' "
            "in the toolbar), and "
            "resizing or cropping destroys it."))
    return res
