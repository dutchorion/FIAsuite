"""Frequency-domain check for generator artifacts.

Up-sampling layers in GANs and the latent decoders of diffusion models leave
periodic traces that appear as isolated peaks in the spectrum of the noise
residual (Frank et al. 2020, Corvi et al. 2023, Synthbuster 2023). The power
spectrum is averaged over tiles (Welch) so scene content cancels and periodic
traces add up; it is then flattened by its per-radius median and searched for
narrow peaks, which are classified by origin.

Residual options, radial-median flattening, peak classification and the angular
profile follow the residual_spectrum detector of imtriage 0.2.0 (MIT licence).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..core import INFO, NOTE, WARN, Evidence, Finding, Result, heatmap, plot

RESIDUALS = ["Wiener (adaptive)", "Cross-difference", "Gaussian", "Median"]
CHANNELS = ["Luminance", "Red", "Green", "Blue"]
EPS = 1e-9
# BGR marker colours per peak class
CLASS_COLORS = {"other": (40, 180, 255), "jpeg_lattice": (255, 150, 60), "axis": (150, 150, 150),
                "cfa": (110, 220, 110), "low_frequency": (90, 90, 90)}


def _plane(ev: Evidence, channel: str) -> np.ndarray:
    if channel == "Luminance":
        return ev.luma
    return ev.bgr[:, :, {"Blue": 0, "Green": 1, "Red": 2}[channel]].astype(np.float32)


def _residual(plane: np.ndarray, method: str):
    """Returns tile(y, x, n) -> residual of that tile."""
    if method == "Cross-difference":               # Synthbuster; blind to purely horizontal/vertical structure
        full = plane[:-1, :-1] + plane[1:, 1:] - plane[1:, :-1] - plane[:-1, 1:]
        return lambda y, x, n: full[y:y + n, x:x + n]
    if method == "Gaussian":
        full = plane - cv2.GaussianBlur(plane, (0, 0), 1.2, borderType=cv2.BORDER_REPLICATE)
        return lambda y, x, n: full[y:y + n, x:x + n]
    if method == "Median":
        full = plane - cv2.medianBlur(plane, 3)
        return lambda y, x, n: full[y:y + n, x:x + n]
    # Lee / adaptive Wiener with a Gaussian window: its transfer function is isotropic,
    # so the filter adds no cross or lobes of its own to the spectrum
    mu = cv2.GaussianBlur(plane, (0, 0), 1.2, borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(cv2.GaussianBlur(plane * plane, (0, 0), 1.2, borderType=cv2.BORDER_REPLICATE) - mu * mu, 0)

    def tile(y, x, n):
        v, m, p = var[y:y + n, x:x + n], mu[y:y + n, x:x + n], plane[y:y + n, x:x + n]
        noise = float(v.mean())
        gain = np.where(v > noise, v - noise, 0) / np.maximum(np.maximum(v, noise), EPS)
        return p - (m + gain * (p - m))
    return tile


def averaged_spectrum(plane: np.ndarray, n: int, method: str, overlap: bool = False, limit: int = 300):
    """(shifted amplitude, tiles used, tiles skipped, residual of the first used tile)"""
    h, w = plane.shape[0] - 1, plane.shape[1] - 1
    step = n // 2 if overlap else n
    origins = [(y, x) for y in range(0, h - n + 1, step) for x in range(0, w - n + 1, step)]
    if len(origins) > limit:
        origins = [origins[int((k + 0.5) * len(origins) / limit)] for k in range(limit)]
    tile = _residual(plane, method)
    hann = np.outer(np.hanning(n), np.hanning(n))
    acc, used, skipped, keep = np.zeros((n, n)), 0, 0, None
    for y, x in origins:
        raw = plane[y:y + n, x:x + n]
        if np.mean((raw <= 1) | (raw >= 254)) > 0.5:
            skipped += 1
            continue
        res = tile(y, x, n)
        res = res - res.mean()
        if res.var() < 1e-4:
            skipped += 1
            continue
        acc += np.abs(np.fft.fft2(res * hann)) ** 2
        used += 1
        keep = res if keep is None else keep
    if not used:
        return None, 0, skipped, None
    return np.fft.fftshift(np.sqrt(acc / used) / n), used, skipped, keep


def flatten(amp: np.ndarray):
    """Divide by the per-radius median (median: strong peaks must not lift their own baseline)."""
    n = amp.shape[0]
    v, u = np.mgrid[0:n, 0:n] - n // 2
    rr = np.hypot(u, v)
    labels = np.minimum(np.rint(rr), n // 2).astype(np.int32)   # the sparse corners share one baseline
    med = np.asarray(ndimage.median(amp, labels=labels, index=np.arange(labels.max() + 1)))
    return np.log10(amp / np.maximum(med[labels], EPS) + EPS), med[: n // 2], rr, u, v


def find_peaks(log_flat, rr, u, v, n: int, jpeg: bool, z_min: float = 8.0, ratio_min: float = 1.5) -> list[dict]:
    """Narrow peaks, classified by origin. Only class 'other' is suspicious."""
    detr = log_flat - ndimage.median_filter(log_flat, size=9, mode="nearest")
    valid = rr >= max(4.0, 0.02 * n)
    sigma = max(1e-6, 1.4826 * float(np.median(np.abs(detr[valid] - np.median(detr[valid])))))
    z = detr / sigma
    # generator traces are isolated dots; edges in the scene make streaks and blobs
    crowd = ndimage.uniform_filter((z > 4).astype(np.float32), size=11) * 121
    upper = (v < 0) | ((v == 0) & (u > 0))         # the spectrum is point-symmetric: keep one copy
    # with few tiles the log-spectrum is speckle and z stays small: a peak far above background still counts
    strong = (z >= z_min) | ((z >= 0.6 * z_min) & (detr >= 1.0))
    local_max = detr == ndimage.maximum_filter(detr, size=5, mode="nearest")
    hit = valid & upper & local_max & strong & (detr >= np.log10(ratio_min))
    step, peaks = n / 8, []
    for y, x in zip(*np.nonzero(hit)):
        ub, vb = int(u[y, x]), int(-v[y, x])
        on_lattice = abs(ub - round(ub / step) * step) <= 1.5 and abs(vb - round(vb / step) * step) <= 1.5
        if abs(abs(ub) - n / 2) <= 4 and abs(abs(vb) - n / 2) <= 4:
            cls = "cfa"                            # Bayer demosaicing peaks at the Nyquist corner of camera originals
        elif abs(ub) <= 2 or abs(vb) <= 2:
            cls = "axis"
        elif on_lattice and jpeg:
            cls = "jpeg_lattice"                   # JPEG block corners
        elif rr[y, x] < 0.1 * n or crowd[y, x] > 14:
            cls = "low_frequency"                  # scene structure, not a periodic trace
        else:
            cls = "other"
        f = float(np.hypot(ub, vb)) / n
        peaks.append({"x": int(x), "y": int(y), "class": cls, "period_px": round(1 / f, 2),
                      "orientation_deg": round(float(np.degrees(np.arctan2(vb, ub)) % 180), 1),
                      "ratio": round(float(10 ** detr[y, x]), 2), "z": round(float(z[y, x]), 1)})
    return sorted(peaks, key=lambda p: -p["z"])


def annotated(log_flat: np.ndarray, peaks: list[dict]) -> np.ndarray:
    n = log_flat.shape[0]
    view = heatmap(log_flat, None, np.percentile(log_flat, 2), np.percentile(log_flat, 99.95), cv2.COLORMAP_BONE)
    for p in peaks:
        for x, y in ((p["x"], p["y"]), (n - p["x"], n - p["y"])):
            cv2.circle(view, (x, y), 7, CLASS_COLORS[p["class"]], 1, cv2.LINE_AA)
    return cv2.resize(view, (768, 768), interpolation=cv2.INTER_NEAREST) if n < 768 else view


def spectrum(ev: Evidence, tile: int = 512, residual: str = "Wiener (adaptive)", channel: str = "Luminance",
             overlap: bool = False) -> Result:
    n = int(tile)
    while n > 128 and ((ev.height - 1) // n) * ((ev.width - 1) // n) < 8:   # averaging needs several tiles
        n //= 2
    res = Result()
    amp, used, skipped, first = averaged_spectrum(_plane(ev, channel), n, residual, bool(overlap))
    if amp is None:
        res.findings.append(Finding(INFO, "Image too small, flat or clipped for spectral analysis"))
        return res
    log_flat, radial, rr, u, v = flatten(amp)
    peaks = find_peaks(log_flat, rr, u, v, n, ev.is_jpeg)
    other = [p for p in peaks if p["class"] == "other"]
    count = len(other)

    # angular profile of the flattened spectrum: directional energy (motion blur, resampling, sharpening)
    ring = (rr >= max(4, round(0.08 * n))) & (rr <= n // 2)
    bins = np.minimum(179, (np.degrees(np.arctan2(-v[ring], u[ring])) % 180).astype(int))
    angular = np.bincount(bins, weights=(10 ** log_flat)[ring], minlength=180) / np.maximum(np.bincount(bins, minlength=180), 1)
    anisotropy = float(angular.max() / max(np.median(angular), EPS))

    r = np.arange(1, n // 2)
    mid = (r > n * 0.02) & (r < n * 0.25)
    lum, _, _, _ = averaged_spectrum(ev.luma, n, "Gaussian")
    lum_radial = flatten(lum)[1][1:] if lum is not None else radial[1:]
    slope, intercept = np.polyfit(np.log(r[mid]), np.log(lum_radial[mid] + EPS), 1)

    res.views["Flattened spectrum (peaks marked)"] = annotated(log_flat, peaks)
    res.views["Spectrum"] = cv2.resize(heatmap(np.log10(amp + EPS), None), (768, 768), interpolation=cv2.INTER_NEAREST)
    res.views["Radial profile"] = plot([("log10 amplitude, per-radius median", np.log10(radial[1:] + EPS), (120, 220, 255))],
                                       "Radial profile of the residual spectrum", xlabels=("low frequency", "Nyquist"))
    res.views["Angular profile"] = plot([("relative energy by orientation", angular, (120, 255, 160))],
                                        f"Angular profile   anisotropy {anisotropy:.2f}", xlabels=("0 deg", "180 deg"))
    shown = np.clip(first / (4 * first.std() + EPS) * 127 + 128, 0, 255).astype(np.uint8)
    res.views["Residual tile"] = cv2.cvtColor(shown, cv2.COLOR_GRAY2BGR)

    by_class = {c: sum(p["class"] == c for p in peaks) for c in CLASS_COLORS}
    res.metrics = {"n_peaks_other": count, "max_peak_ratio_other": max((p["ratio"] for p in other), default=1.0),
                   "angular_anisotropy": round(anisotropy, 4)}
    res.table = [("Tiles averaged", f"{used} x {n} px  ({skipped} flat or clipped tiles skipped)"),
                 ("Unexplained peaks (amber)", str(count)),
                 ("JPEG 8x8 lattice peaks (blue)", f"{by_class['jpeg_lattice']}  - compression, ignored"),
                 ("On-axis peaks (grey)", f"{by_class['axis']}  - ignored"),
                 ("Nyquist-corner peaks (green)", f"{by_class['cfa']}  - Bayer demosaicing of a camera original"),
                 ("Low-frequency / streak peaks", f"{by_class['low_frequency']}  - scene structure, ignored"),
                 ("Angular anisotropy", f"{anisotropy:.2f}  at {int(angular.argmax())} deg"),
                 ("Spectral slope of the image", f"{slope:.2f}  (natural photos: about -1 to -1.6)")]
    res.table += [(f"  peak {i + 1}", f"period {p['period_px']} px at {p['orientation_deg']} deg, "
                   f"{p['ratio']}x background, z = {p['z']}") for i, p in enumerate(other[:12])]
    if used < 8:
        res.findings.append(Finding(INFO, "Few tiles averaged", "The spectrum is noisy and weak peaks will be missed."))
    if count >= 2:                                 # each peak stands for a point-symmetric pair
        res.score = float(np.clip(0.5 + count / 20, 0, 0.9))
        res.findings.append(Finding(WARN, f"{count} unexplained periodic peaks in the noise spectrum",
            "A regular lattice of isolated peaks (amber circles) is the fingerprint of learned up-sampling in image "
            "generators. Resampling, some upscalers, halftone prints and screens photographed at an angle create "
            "peaks too. In non-JPEG files a period-8 trace cannot be told from an earlier JPEG compression."))
    else:
        res.score = 0.1
        res.findings.append(Finding(INFO, "No generator-type spectral peaks",
            "A clean spectrum proves nothing: recompression and resizing remove most generator traces, and current "
            "diffusion models often leave none."))
    if by_class["cfa"]:
        res.findings.append(Finding(INFO, "Demosaicing peak present",
                                    "Energy at the Nyquist corner: the Bayer interpolation trace of a camera original."))
    if anisotropy > 3:
        res.findings.append(Finding(NOTE, f"Strongly directional residual ({anisotropy:.1f}x at {int(angular.argmax())} deg)",
            "Directional energy in the noise: motion blur, one-directional resampling or sharpening. Uncalibrated."))
    return res
