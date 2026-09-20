"""Noise forensics: median-filter residual (Forensically style) and a local
noise-level inconsistency map (Immerkaer estimator, brightness-compensated)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core import INFO, WARN, Evidence, Finding, Result, block_reduce, coherent_fraction, heatmap

_IMMERKAER = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


DENOISERS = ["Median 3x3", "Median 5x5", "Gaussian", "Non-local means (slow)"]


def _denoise(bgr: np.ndarray, denoiser: str) -> np.ndarray:
    if denoiser == "Gaussian":
        return cv2.GaussianBlur(bgr, (0, 0), 1.2)
    if denoiser.startswith("Non-local"):           # keeps edges out of the residual; minutes on very large images
        return cv2.fastNlMeansDenoisingColored(bgr, None, 5, 5, 5, 15)
    return cv2.medianBlur(bgr, 5 if "5x5" in denoiser else 3)


def residual(ev: Evidence, amplitude: int = 25, equalize: bool = False, denoiser: str = "Median 3x3",
             luminance: bool = False) -> Result:
    diff = cv2.absdiff(ev.bgr, _denoise(ev.bgr, denoiser)).astype(np.float32) * amplitude
    if luminance:                                  # grain strength without its colour
        diff = diff.mean(axis=2, keepdims=True).repeat(3, 2)
    img = np.clip(diff, 0, 255).astype(np.uint8)
    if equalize:
        img = cv2.merge([cv2.equalizeHist(c) for c in cv2.split(img)])
    res = Result(views={"Noise residual": img})
    res.findings.append(Finding(INFO, "Visual tool",
        "Sensor noise covers a genuine photo like a uniform film. Look for regions where the grain is missing "
        "(airbrushing, inpainting, AI fill), coarser, or of a different colour than on comparable surfaces."))
    return res


def noise_map(ev: Evidence, block: int = 32, sensitivity: float = 2.5) -> Result:
    b = int(block)
    lap = np.abs(cv2.filter2D(ev.luma, -1, _IMMERKAER, borderType=cv2.BORDER_REFLECT))
    sigma = np.sqrt(np.pi / 2) / 6 * block_reduce(lap, b)
    mean = block_reduce(ev.luma, b)
    gx, gy = cv2.Sobel(ev.luma, cv2.CV_32F, 1, 0), cv2.Sobel(ev.luma, cv2.CV_32F, 0, 1)
    texture = block_reduce(cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 2), b)
    valid = (texture < np.percentile(texture, 60)) & (mean > 10) & (mean < 245) & (sigma > 0.05)

    res = Result()
    if valid.size < 30:
        res.findings.append(Finding(INFO, "Image too small for this block size"))
        return res
    shown = np.where(valid, sigma, 0)
    res.views["Noise level"] = heatmap(shown, (ev.height, ev.width), 0, np.percentile(sigma[valid], 98) if valid.any() else 1,
                                       smooth=False)
    if valid.sum() < 30:
        res.findings.append(Finding(INFO, "Too few smooth regions to compare noise levels"))
        return res
    # sensor noise grows with brightness: compare each block with the trend for its brightness
    logs = np.log(sigma[valid])
    coef = np.polyfit(mean[valid], logs, 2)
    dev = np.zeros_like(sigma)
    dev[valid] = logs - np.polyval(coef, mean[valid])
    mad = 1.4826 * np.median(np.abs(dev[valid] - np.median(dev[valid]))) + 1e-6
    z = np.where(valid, dev / mad, 0)
    res.views["Deviation from trend"] = heatmap(z, (ev.height, ev.width), -5, 5, cv2.COLORMAP_TWILIGHT_SHIFTED, smooth=False)
    # pool over 3x3 blocks: a manipulated region is a neighbourhood of moderate outliers, not single extreme blocks
    pooled = cv2.blur(z, (3, 3)) / np.maximum(cv2.blur(valid.astype(np.float32), (3, 3)), 1e-3)
    frac = coherent_fraction((np.abs(pooled) > float(sensitivity)) & valid)
    res.score = float(np.clip(frac * 8, 0, 0.85))
    res.table = [("Median noise sigma", f"{np.median(sigma[valid]):.2f}"), ("Blocks analysed", f"{int(valid.sum())} of {valid.size}"),
                 ("Coherent outlier area", f"{frac:.1%}")]
    if frac > 0.03:
        res.findings.append(Finding(WARN, "Inconsistent noise level",
            f"{frac:.1%} of the image forms connected regions whose noise level departs from the rest at equal "
            "brightness. Occurs with splices from another source, local denoising/blur, inpainting. Phone HDR and "
            "portrait modes can also cause this."))
    else:
        res.findings.append(Finding(INFO, "Noise level is spatially consistent"))
    return res
