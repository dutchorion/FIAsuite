"""Enhancement views for visual inspection (no automatic verdict)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core import INFO, Evidence, Finding, Result

ENHANCEMENTS = ["None", "Auto contrast", "Auto contrast by channel", "Histogram equalization", "CLAHE"]


def enhance_image(bgr: np.ndarray, mode: str) -> np.ndarray:
    if mode == "Auto contrast":
        lo, hi = np.percentile(bgr, (0.5, 99.5))
        return np.clip((bgr.astype(np.float32) - lo) * 255 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    if mode == "Auto contrast by channel":
        return cv2.merge([enhance_image(c, "Auto contrast") for c in cv2.split(bgr)])
    if mode in ("Histogram equalization", "CLAHE"):
        ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        op = cv2.equalizeHist if mode == "Histogram equalization" else cv2.createCLAHE(3.0, (8, 8)).apply
        ycc[:, :, 0] = op(np.ascontiguousarray(ycc[:, :, 0]))
        return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return bgr


def enhance(ev: Evidence, mode: str = "CLAHE") -> Result:
    return Result(views={"Enhanced": enhance_image(ev.bgr, mode)})


def luminance_gradient(ev: Evidence) -> Result:
    gx = cv2.Sobel(ev.luma, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ev.luma, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy) + 1e-6
    img = np.stack([np.clip(mag * 2, 0, 255), (gy / mag * 0.5 + 0.5) * 255, (gx / mag * 0.5 + 0.5) * 255], axis=2)
    res = Result(views={"Luminance gradient": img.astype(np.uint8)})
    res.findings.append(Finding(INFO, "Visual tool",
        "Colour encodes the direction of brightness change, i.e. the direction light comes from. Objects lit from a "
        "different side than their surroundings, and the soft edges of pasted or generated regions, stand out."))
    return res


def pca(ev: Evidence, component: int = 1, invert: bool = False, enhancement: str = "Auto contrast") -> Result:
    pixels = ev.bgr.reshape(-1, 3).astype(np.float32)
    sample = pixels[:: max(1, len(pixels) // 200_000)]
    mean, vectors, values = cv2.PCACompute2(sample, mean=None)
    proj = (pixels - mean) @ vectors[int(component) - 1]
    lo, hi = np.percentile(proj, (0.2, 99.8))
    img = np.clip((proj - lo) / max(hi - lo, 1e-6), 0, 1).reshape(ev.height, ev.width)
    if invert:
        img = 1 - img
    out = enhance_image(cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), enhancement)
    res = Result(views={f"Principal component {component}": out})
    total = float(values.sum()) or 1.0
    res.table = [(f"Variance in component {i + 1}", f"{float(v) / total:.1%}") for i, v in enumerate(values.ravel())]
    res.findings.append(Finding(INFO, "Visual tool",
        "Components 2 and 3 discard most of the scene and keep subtle colour variation, where compression "
        "differences, cloning seams and retouching often become visible."))
    return res


def level_sweep(ev: Evidence, sweep: int = 128, width: int = 32) -> Result:
    lo = sweep - width / 2
    img = np.clip((ev.bgr.astype(np.float32) - lo) / max(width, 1), 0, 1) * 255
    res = Result(views={"Level sweep": img.astype(np.uint8)})
    res.findings.append(Finding(INFO, "Visual tool",
        "Stretches a narrow brightness band to full contrast. Sweep through the range and watch for sharp outlines "
        "around pasted objects, especially in near-white and near-black areas."))
    return res


def bit_plane(ev: Evidence, plane: int = 0, channel: str = "All") -> Result:
    bits = ((ev.bgr >> int(plane)) & 1) * 255
    if channel != "All":
        bits = bits[:, :, "BGR".index(channel[0])][..., None].repeat(3, 2)
    res = Result(views={f"Bit plane {plane}": bits.astype(np.uint8)})
    res.findings.append(Finding(INFO, "Visual tool",
        "The lowest bit planes of a photo look like uniform random noise. Flat, patterned or blank areas mark "
        "synthetic fills, heavy denoising, or embedded data."))
    return res


_SPACES = {"RGB": (None, "BGR"), "HSV": (cv2.COLOR_BGR2HSV, "HSV"), "Lab": (cv2.COLOR_BGR2LAB, "Lab"),
           "YCbCr": (cv2.COLOR_BGR2YCrCb, "YRB")}
CHANNELS = ["RGB: R", "RGB: G", "RGB: B", "HSV: H", "HSV: S", "HSV: V", "Lab: L", "Lab: a", "Lab: b",
            "YCbCr: Y", "YCbCr: Cb", "YCbCr: Cr"]


def channel(ev: Evidence, which: str = "YCbCr: Cb", enhancement: str = "Auto contrast") -> Result:
    space, name = which.split(": ")
    code, _ = _SPACES[space]
    conv = ev.bgr if code is None else cv2.cvtColor(ev.bgr, code)
    index = {"R": 2, "G": 1, "B": 0, "H": 0, "S": 1, "V": 2, "L": 0, "a": 1, "b": 2, "Y": 0, "Cr": 1, "Cb": 2}[name]
    img = cv2.cvtColor(np.ascontiguousarray(conv[:, :, index]), cv2.COLOR_GRAY2BGR)
    res = Result(views={which: enhance_image(img, enhancement)})
    res.findings.append(Finding(INFO, "Visual tool",
        "Chroma channels (Cb, Cr, a, b) are compressed harder and edited less carefully than luminance: splice "
        "borders and colour retouching often show here first."))
    return res
