"""Enhancement views for visual inspection (no automatic verdict)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core import INFO, NOTE, Evidence, Finding, Result, plot

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


# ------------------------------------------------------------------ histogram (contrast-enhancement traces)

def _comb(hist: np.ndarray) -> tuple[float, float]:
    """(share of empty bins inside the occupied range, mean |log| ratio of a bin to its neighbours).
    A camera histogram is smooth; remapping 8-bit values (levels, curves, contrast, gamma) leaves gaps where
    the range was stretched and spikes where it was compressed (Stamm & Liu 2010)."""
    used = np.nonzero(hist)[0]
    if used.size < 16:
        return 0.0, 0.0
    h = hist[used[0]:used[-1] + 1].astype(np.float64)
    mid, nb = h[1:-1], 0.5 * (h[:-2] + h[2:])
    ok = (mid + nb) > 0.0005 * h.sum()             # ignore the sparse tails
    ripple = float(np.mean(np.abs(np.log((mid[ok] + 1) / (nb[ok] + 1))))) if ok.sum() >= 8 else 0.0
    return float((h[1:-1] == 0).mean()), ripple


def histogram(ev: Evidence, channel: str = "All", log_scale: bool = False) -> Result:
    res = Result()
    planes = {"Red": ev.bgr[:, :, 2], "Green": ev.bgr[:, :, 1], "Blue": ev.bgr[:, :, 0],
              "Luminance": cv2.cvtColor(ev.bgr, cv2.COLOR_BGR2GRAY)}
    colors = {"Red": (90, 90, 255), "Green": (110, 220, 110), "Blue": (255, 170, 80), "Luminance": (220, 220, 220)}
    hists = {name: np.bincount(plane.ravel(), minlength=256) for name, plane in planes.items()}
    shown = list(planes) if channel == "All" else [channel]
    scale = (lambda h: np.log10(h + 1.0)) if log_scale else (lambda h: h.astype(np.float64))
    res.views["Histogram"] = plot([(name, scale(hists[name]), colors[name]) for name in shown],
                                  "Pixel-value histogram" + ("  (log scale)" if log_scale else ""), xlabels=("0", "255"))
    clipped = np.zeros(ev.bgr.shape[:2], np.uint8)
    clipped[(ev.bgr == 0).all(axis=2)] = 1
    clipped[(ev.bgr == 255).all(axis=2)] = 2
    view = (ev.bgr * 0.4).astype(np.uint8)
    view[clipped == 1] = (255, 120, 40)
    view[clipped == 2] = (60, 60, 255)
    res.views["Clipped pixels (blue = black, red = white)"] = view

    gaps, ripple = zip(*(_comb(hists[name]) for name in ("Red", "Green", "Blue")))
    gap, rip = float(np.median(gaps)), float(np.median(ripple))
    res.metrics = {"histogram_gaps": round(gap, 4), "histogram_ripple": round(rip, 4)}
    res.table = [("Empty bins inside the used range", f"{gap:.1%}  (median of R, G, B)"),
                 ("Bin-to-bin ripple", f"{rip:.2f}  (photographs measured: up to about 0.13)"),
                 ("Pure black / pure white pixels", f"{(clipped == 1).mean():.2%} / {(clipped == 2).mean():.2%}")]
    res.table += [(f"{name} range", f"{np.nonzero(h)[0][0]}..{np.nonzero(h)[0][-1]}") for name, h in hists.items()]
    if gap > 0.08 or rip > 0.2:
        res.findings.append(Finding(NOTE, "Comb-shaped histogram: tonal adjustment",
            "Regular gaps or spikes between neighbouring brightness values: the tones were remapped after the image "
            "became 8-bit (levels, curves, contrast, brightness, gamma or a filter). A global adjustment is ordinary "
            "editing, not content manipulation. Graphics, screenshots and images with few colours look the same."))
    else:
        res.findings.append(Finding(INFO, "Smooth histogram", "No trace of tonal remapping. JPEG compression after the "
                                    "adjustment, resizing and noise reduction smooth the comb away."))
    return res


# ------------------------------------------------------------------ transparency (hidden pixels)

def hidden_pixels(ev: Evidence) -> Result:
    """Colour stored under fully transparent pixels. Editors that 'erase' only set alpha to 0; the colour stays
    in the file (FotoForensics calls these hidden pixels)."""
    res = Result()
    raw = cv2.imdecode(np.frombuffer(ev.data, np.uint8), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
    if raw is None or raw.ndim != 3 or raw.shape[2] != 4:
        res.findings.append(Finding(INFO, "No transparency channel", "Applies to PNG, WebP and TIFF files with alpha."))
        return res
    alpha = raw[:, :, 3]
    alpha = (alpha / 257).astype(np.uint8) if alpha.dtype == np.uint16 else alpha
    hidden = alpha == 0
    res.views["Colour with transparency ignored"] = ev.bgr
    res.views["Alpha channel"] = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
    only = ev.bgr.copy()
    only[~hidden] = (ev.bgr[~hidden] * 0.15).astype(np.uint8)
    res.views["Hidden pixels only"] = only
    share = float(hidden.mean())
    res.table = [("Fully transparent pixels", f"{share:.1%}"), ("Partly transparent pixels", f"{((alpha > 0) & (alpha < 255)).mean():.1%}")]
    spread = float(ev.bgr[hidden].std()) if hidden.any() else 0.0
    res.table.append(("Colour variation under transparency", f"{spread:.1f}"))
    if share > 0.001 and spread > 8:
        res.findings.append(Finding(NOTE, "Picture content hidden under transparent pixels",
            f"{share:.1%} of the image is fully transparent but still stores varied colour: content that was erased "
            "or masked in an editor without being removed from the file. See 'Hidden pixels only'."))
    elif share > 0:
        res.findings.append(Finding(INFO, "Transparent areas carry no picture content"))
    else:
        res.findings.append(Finding(INFO, "Alpha channel present, nothing fully transparent"))
    return res
