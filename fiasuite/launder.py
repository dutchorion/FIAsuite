"""Laundering: what an image goes through before an examiner sees it (platform
resizing, recompression). A detector that only works on pristine files is of
little use, so the calibration tool scores every detector under these chains.

Chain syntax (from imtriage, MIT licence):  none | jpeg85 | resize0.5 | resize0.5+jpeg75
"""
from __future__ import annotations

import cv2
import numpy as np

from .core import Evidence


def parse_chain(spec: str) -> list[tuple[str, float]]:
    spec = spec.strip().lower()
    if spec in ("", "none"):
        return []
    ops = []
    for token in spec.split("+"):
        token = token.strip()
        name = token.rstrip("0123456789.")
        if name not in ("jpeg", "resize") or name == token:
            raise ValueError(f"bad laundering step '{token}': use jpegNN or resizeF, joined with +")
        value = float(token[len(name):])
        low, high = (1, 100) if name == "jpeg" else (0.05, 4)
        if not low <= value <= high:
            raise ValueError(f"'{token}': {name} takes a value between {low} and {high}")
        ops.append((name, value))
    return ops


def apply(ev: Evidence, spec: str) -> Evidence:
    ops = parse_chain(spec)
    if not ops:
        return ev
    img, data = ev.bgr, None
    for name, value in ops:
        if name == "resize":
            interp = cv2.INTER_AREA if value < 1 else cv2.INTER_LANCZOS4
            img = cv2.resize(img, (max(1, round(img.shape[1] * value)), max(1, round(img.shape[0] * value))), interpolation=interp)
            data = None
        else:
            data = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(value)])[1].tobytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if data is None:
        data = cv2.imencode(".png", img)[1].tobytes()
    return Evidence(ev.path, data)
