"""Evidence loading and the result types shared by every analysis.

Evidence is always opened read-only and decoded WITHOUT applying EXIF
orientation, so every map is aligned with the pixel grid stored in the file
(the JPEG 8x8 grid, the CFA lattice, ...).
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".webp", ".tif", ".tiff",
              ".bmp", ".heic", ".heif", ".avif"}

# severity levels, ordered
INFO, NOTE, WARN, ALERT = "info", "note", "warn", "alert"
SEVERITY_ORDER = {INFO: 0, NOTE: 1, WARN: 2, ALERT: 3}


def app_dir() -> Path:
    """Folder holding the exe (frozen) or the project root (source run)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


@dataclass
class Finding:
    severity: str
    title: str
    detail: str = ""


@dataclass
class Result:
    """Output of one analysis.

    views    ordered name -> BGR uint8 image. A view with the same height/width
             as the evidence can be blended over it in the viewer.
    score    0..1 suspicion for automated triage, None for visual-only tools.
    table    key/value rows shown in the details panel and the report.
    metrics  named raw measurements; like the score they are uncalibrated until
             measured on a corpus with the calibration tool.
    """
    views: dict[str, np.ndarray] = field(default_factory=dict)
    score: float | None = None
    findings: list[Finding] = field(default_factory=list)
    table: list[tuple[str, str]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)    # raw measurements, for calibration

    def worst(self) -> str:
        return max((f.severity for f in self.findings), key=SEVERITY_ORDER.get, default=INFO)

    def severity_score(self) -> float:
        """Score for rule-based tools whose output is a list of findings."""
        return {INFO: 0.05, NOTE: 0.25, WARN: 0.6, ALERT: 0.95}[self.worst()]


class Evidence:
    def __init__(self, path: str | Path, data: bytes | None = None):
        """data: in-memory file content (laundered copies); the path then only names the evidence."""
        self.path = Path(path)
        self.data = self.path.read_bytes() if data is None else data
        self.bgr = _decode(self.data, self.path.suffix.lower())
        self.height, self.width = self.bgr.shape[:2]
        self.is_jpeg = self.data[:2] == b"\xff\xd8"
        self.is_png = self.data[:8] == b"\x89PNG\r\n\x1a\n"
        self.cache: dict = {}  # per-evidence memo shared between analyses

    @cached_property
    def luma(self) -> np.ndarray:
        """float32 luminance 0..255 (BT.601, same as JPEG Y)."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    @cached_property
    def hashes(self) -> dict[str, str]:
        return {name: hashlib.new(name, self.data).hexdigest() for name in ("md5", "sha1", "sha256")}

    @cached_property
    def phash(self) -> int:
        """64-bit DCT perceptual hash."""
        small = cv2.resize(self.luma, (32, 32), interpolation=cv2.INTER_AREA)
        low = cv2.dct(small)[:8, :8].flatten()
        bits = low > np.median(low[1:])
        return int("".join("1" if b else "0" for b in bits), 2)


def _decode(data: bytes, suffix: str) -> np.ndarray:
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None and suffix in {".heic", ".heif", ".avif"}:
        img = _decode_heif(data)
    if img is None:
        raise ValueError("Unsupported or corrupt image file")
    return img


def _decode_heif(data: bytes) -> np.ndarray | None:
    try:
        import io
        import pillow_heif
        from PIL import Image
        pillow_heif.register_heif_opener()
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ---------------------------------------------------------------- helpers

def heatmap(values: np.ndarray, size: tuple[int, int] | None = None,
            vmin: float | None = None, vmax: float | None = None,
            cmap: int = cv2.COLORMAP_INFERNO, smooth: bool = True) -> np.ndarray:
    """Colour-map a float array; optionally upscale to size=(height, width)."""
    v = np.nan_to_num(values.astype(np.float32))
    lo = float(v.min()) if vmin is None else vmin
    hi = float(v.max()) if vmax is None else vmax
    v = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
    if size is not None and v.shape != tuple(size):
        interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
        v = cv2.resize(v, (size[1], size[0]), interpolation=interp)
    return cv2.applyColorMap((v * 255).astype(np.uint8), cmap)


def block_reduce(a: np.ndarray, b: int, fn=np.mean) -> np.ndarray:
    """Reduce a 2-D array over non-overlapping b x b blocks (edges cropped)."""
    h, w = (a.shape[0] // b) * b, (a.shape[1] // b) * b
    return fn(a[:h, :w].reshape(h // b, b, w // b, b), axis=(1, 3))


def coherent_fraction(mask: np.ndarray, k: int = 3) -> float:
    """Fraction of True cells that survive a k x k opening: isolated outliers are
    ignored, spatially coherent regions are kept."""
    if mask.size == 0:
        return 0.0
    m = mask.astype(np.uint8)
    opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    return float(opened.sum()) / mask.size


def plot(series: list[tuple[str, np.ndarray, tuple[int, int, int]]], title: str = "",
         size: tuple[int, int] = (900, 360), bars: bool = False,
         xlabels: tuple[str, str] | None = None) -> np.ndarray:
    """Minimal chart renderer (avoids shipping matplotlib). series = (label, y, bgr)."""
    w, h = size
    img = np.full((h, w, 3), 24, np.uint8)
    left, right, top, bottom = 50, 15, 30, 30
    pw, ph = w - left - right, h - top - bottom
    ymax = max(float(np.max(s[1])) for s in series) or 1.0
    ymin = min(0.0, min(float(np.min(s[1])) for s in series))
    cv2.rectangle(img, (left, top), (left + pw, top + ph), (70, 70, 70), 1)
    for label_i, (label, y, color) in enumerate(series):
        n = len(y)
        xs = left + (np.arange(n) + 0.5) * pw / n
        ys = top + ph - (np.asarray(y, np.float64) - ymin) / (ymax - ymin + 1e-12) * ph
        if bars:
            bw = max(1, int(pw / n) - 1)
            for x, yy in zip(xs, ys):
                cv2.rectangle(img, (int(x - bw / 2), int(yy)), (int(x + bw / 2), top + ph), color, -1)
        else:
            pts = np.stack([xs, ys], 1).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], False, color, 1, cv2.LINE_AA)
        cv2.putText(img, label, (left + 8 + 170 * label_i, top + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, title, (left, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"{ymax:.3g}", (4, top + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    if xlabels:
        cv2.putText(img, xlabels[0], (left, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(xlabels[1], cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(img, xlabels[1], (left + pw - tw, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)
    return img
