"""Sensor pattern noise (PRNU): camera identification and fingerprint-based tamper map.

Every sensor has a fixed multiplicative noise pattern K (photo-response non-uniformity). A photo is roughly
I = I0 * (1 + K) + other noise, so the noise residual W = I - denoise(I) correlates with I * K for photos from that
sensor and with nothing else (Lukas, Fridrich, Goljan 2006). The fingerprint is the maximum-likelihood estimate
K = sum(W_i * I_i) / sum(I_i^2) over reference photos (Chen, Fridrich, Goljan, Lukas 2008); the match statistic is
the peak-to-correlation energy PCE (Goljan, Fridrich, Filler 2009). Regions of a matching photo where the
correlation vanishes were not recorded by that sensor.

Works on the luminance at native resolution. Resizing, strong compression, digital stabilisation, HDR stacking and
lens-distortion correction break the pixel alignment the method relies on.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from ..core import ALERT, INFO, NOTE, WARN, Evidence, Finding, Result, app_dir, block_reduce, coherent_fraction, heatmap

NONE = "(none - build one under 'Camera fingerprint ...')"
PCE_MATCH = 60.0                                   # Goljan et al.: false-alarm probability around 1e-5


def store() -> Path:
    return app_dir() / "fingerprints"


def stored() -> list[str]:
    return [NONE] + sorted(p.stem for p in store().glob("*.npz")) if store().exists() else [NONE]


# ------------------------------------------------------------------ residual

def residual(luma: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Noise residual with the scene suppressed: Mihcak-style adaptive Wiener filtering of the high-pass image
    (local variance = smallest estimate over 3..9 px windows), then zero-mean rows and columns, which removes
    the periodic traces of demosaicing and JPEG blocks that all cameras of one model share."""
    high = luma - cv2.GaussianBlur(luma, (0, 0), 1.5)
    square, variance = high * high, None
    for k in (3, 5, 7, 9):
        local = np.maximum(cv2.blur(square, (k, k)) - sigma * sigma, 0)
        variance = local if variance is None else np.minimum(variance, local)
    w = high * (sigma * sigma / (variance + sigma * sigma))
    w -= w.mean(axis=0, keepdims=True)
    w -= w.mean(axis=1, keepdims=True)
    return _suppress_peaks(w)


def _suppress_peaks(w: np.ndarray) -> np.ndarray:
    """Wiener filtering in the Fourier domain: flattens periodic (non-unique) components of the residual."""
    spec = np.fft.fft2(w)
    mag = np.abs(spec)
    noise = float(w.var())
    local = np.maximum(cv2.blur((mag * mag / w.size).astype(np.float32), (3, 3)) - noise, 0)
    return np.fft.ifft2(spec * (noise / (local + noise))).real.astype(np.float32)


def _luma(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


# ------------------------------------------------------------------ fingerprint

def build_fingerprint(paths: list[str], progress=lambda done, total: None, cancelled=lambda: False) -> dict:
    """Maximum-likelihood fingerprint from reference photos of ONE camera. Photos must share their pixel
    dimensions (portrait shots are turned to landscape); others are skipped and counted."""
    num = den = None
    used, skipped, shape = 0, [], None
    for i, path in enumerate(paths):
        if cancelled():
            break
        progress(i, len(paths))
        try:
            luma = _luma(Evidence(path).bgr)
        except Exception:
            skipped.append(Path(path).name)
            continue
        if shape is not None and luma.shape == shape[::-1]:
            luma = np.rot90(luma)                  # which way a portrait shot was turned is unknown; one try only
        if shape is None:
            shape = luma.shape
            num, den = np.zeros(shape, np.float64), np.zeros(shape, np.float64)
        if luma.shape != shape:
            skipped.append(Path(path).name)
            continue
        valid = (luma > 8) & (luma < 250)          # clipped pixels carry no pattern noise
        num += residual(luma) * luma * valid
        den += luma * luma * valid
        used += 1
    if used == 0:
        raise ValueError("No usable reference photos.")
    k = (num / np.maximum(den, 1.0)).astype(np.float32)
    k -= k.mean(axis=0, keepdims=True)
    k -= k.mean(axis=1, keepdims=True)
    return {"k": _suppress_peaks(k), "count": used, "skipped": skipped}


def save_fingerprint(name: str, fp: dict, source: str) -> Path:
    store().mkdir(exist_ok=True)
    path = store() / f"{name}.npz"
    np.savez_compressed(path, k=fp["k"].astype(np.float16) if np.abs(fp["k"]).max() < 60000 else fp["k"],
                        info=json.dumps({"count": fp["count"], "source": source, "skipped": len(fp["skipped"])}))
    return path


def load_fingerprint(name: str) -> tuple[np.ndarray, dict]:
    with np.load(store() / f"{name}.npz") as data:
        return data["k"].astype(np.float32), json.loads(str(data["info"]))


# ------------------------------------------------------------------ matching

def pce(w: np.ndarray, expected: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Peak-to-correlation energy of the cross-correlation and the (dy, dx) shift of its peak. `w` may be smaller
    than `expected` (a crop): it is zero-padded and every shift is searched."""
    h, wd = expected.shape
    a = np.zeros((h, wd), np.float32)
    a[:w.shape[0], :w.shape[1]] = w - w.mean()
    b = expected - expected.mean()
    xc = np.fft.ifft2(np.conj(np.fft.fft2(a)) * np.fft.fft2(b)).real
    peak = np.unravel_index(int(np.argmax(xc)), xc.shape)
    value = float(xc[peak])
    ys, xs = np.arange(-5, 6) + peak[0], np.arange(-5, 6) + peak[1]
    energy = float((xc ** 2).sum() - (xc[np.ix_(ys % h, xs % wd)] ** 2).sum()) / (xc.size - 121)
    return float(np.sign(value) * value * value / max(energy, 1e-12)), (int(peak[0]), int(peak[1]))


def _orientations(luma: np.ndarray, shape: tuple[int, int]):
    """The image as stored and, where the dimensions allow, turned by 90 / 180 / 270 degrees."""
    for turns in range(4):
        turned = np.rot90(luma, turns)
        if turned.shape[0] <= shape[0] and turned.shape[1] <= shape[1]:
            yield turns, np.ascontiguousarray(turned)


def identify(ev: Evidence, fingerprint: str = NONE, block: str = "Auto") -> Result:
    res = Result()
    if fingerprint == NONE or not (store() / f"{fingerprint}.npz").exists():
        res.findings.append(Finding(INFO, "No camera fingerprint selected",
            "Build a fingerprint from 20 or more photos of the suspected camera with 'Camera fingerprint ...' in the "
            "toolbar, then select it here."))
        return res
    k, info = load_fingerprint(fingerprint)
    best = None
    for turns, luma in _orientations(ev.luma, k.shape):
        w = residual(luma)
        value, shift = pce(w, k * luma if luma.shape == k.shape else k)   # a crop is located against K itself
        if best is None or value > best[0]:
            best = (value, shift, turns, luma, w)
    if best is None:
        res.findings.append(Finding(INFO, "Image is larger than the fingerprint",
            f"The fingerprint covers {k.shape[1]} x {k.shape[0]} px. An image with more pixels was resized or comes "
            "from another camera; PRNU cannot be compared."))
        return res
    value, (dy, dx), turns, luma, w = best
    same_size = luma.shape == k.shape
    res.metrics = {"pce": round(value, 1)}
    res.table = [("Fingerprint", f"{fingerprint}  ({info['count']} reference photos, {k.shape[1]} x {k.shape[0]} px)"),
                 ("PCE", f"{value:.1f}  (match above {PCE_MATCH:.0f})"),
                 ("Orientation that fits best", f"turned {turns * 90} degrees" if turns else "as stored"),
                 ("Position in the sensor frame", "full frame" if same_size else f"crop at x={dx}, y={dy}")]
    if value < PCE_MATCH:
        res.findings.append(Finding(INFO, "No match with this camera fingerprint",
            "Either the photo was not taken with this camera, or the pattern was lost: resizing, heavy compression, "
            "digital stabilisation, HDR / night modes and lens-distortion correction all destroy the pixel alignment. "
            "A missing match does not prove another camera."))
        return res
    res.findings.append(Finding(NOTE, f"Sensor pattern matches '{fingerprint}'",
        f"PCE {value:.0f}: the noise residual carries this camera's sensor pattern. Chance matches above "
        f"{PCE_MATCH:.0f} are of the order of one in 100,000 for unrelated sensors."))

    # tamper map: local correlation between the residual and the expected pattern
    region = k[dy:dy + luma.shape[0], dx:dx + luma.shape[1]] if not same_size else k
    if region.shape != luma.shape:                 # crop wraps around the frame edge: no map
        return res
    expected = region * luma
    # a correlation measured on b x b pixels scatters by about 1/b; the window must be large enough for the pattern
    # to stand clear of that, which depends on how strong this fingerprint is in this image
    overall = float(np.corrcoef(w.ravel(), expected.ravel())[0, 1])
    sizes = [64, 128, 256, 512] if block == "Auto" else [int(block)]
    b = next((n for n in sizes if overall > 4.0 / n), sizes[-1])
    b = min(b, min(luma.shape) // 2)
    n = lambda a: a - cv2.blur(a, (b, b))
    wz, ez = n(w), n(expected)
    rho = cv2.blur(wz * ez, (b, b)) / np.sqrt(np.maximum(cv2.blur(wz * wz, (b, b)) * cv2.blur(ez * ez, (b, b)), 1e-12))
    rho = block_reduce(rho, b // 4)
    # how much correlation to expect depends on the content: dark, clipped and busy blocks correlate less
    bright = block_reduce(luma, b // 4)
    busy = block_reduce(np.abs(cv2.Laplacian(luma, cv2.CV_32F)), b // 4)
    usable = (bright > 25) & (bright < 240) & (busy < np.percentile(busy, 85))
    if usable.sum() < 30:
        return res
    design = np.stack([np.ones(usable.sum()), bright[usable] / 255, np.log1p(busy[usable])], axis=1)
    coef, *_ = np.linalg.lstsq(design, rho[usable], rcond=None)
    predicted = coef[0] + coef[1] * bright / 255 + coef[2] * np.log1p(busy)
    spread = 1.0 / b
    testable = usable & (predicted > 3.5 * spread)   # elsewhere absence of the pattern cannot be told from noise
    z = np.where(testable, (rho - predicted) / spread, 0)
    res.table += [("Analysis window", f"{b} px" + ("  (chosen automatically)" if block == "Auto" else "")),
                  ("Area where the pattern is strong enough to test", f"{testable.mean():.0%}")]
    shown = np.rot90(rho, -turns) if turns else rho
    res.views["Local correlation with the fingerprint"] = heatmap(shown, (ev.height, ev.width), 0,
                                                                  float(np.percentile(rho[usable], 95)), cv2.COLORMAP_VIRIDIS)
    if testable.mean() < 0.2:
        res.findings.append(Finding(INFO, "Fingerprint too weak for a tamper map",
            "The pattern is strong enough to identify the camera but not to test regions of this image. A "
            "fingerprint from more (and flatter, brighter) reference photos gives a usable map."))
    missing = (z < -3) & testable
    missing = np.rot90(missing, -turns) if turns else missing
    res.views["Fingerprint missing"] = heatmap(missing.astype(np.float32), (ev.height, ev.width), 0, 1, smooth=False)
    frac = coherent_fraction(missing, 3)
    res.table.append(("Area without the sensor pattern", f"{frac:.1%}"))
    res.metrics["prnu_missing_area"] = round(frac, 4)
    if frac > 0.01:
        res.score = float(np.clip(0.55 + frac * 4, 0, 0.95))
        res.findings.append(Finding(ALERT if frac > 0.03 else WARN, "Region without this camera's sensor pattern",
            f"{frac:.1%} of the image forms a connected region where the pattern is absent although the content "
            "there should carry it: that area was not recorded by this sensor (paste, inpainting, generative fill)."))
    else:
        res.score = 0.05
    return res
