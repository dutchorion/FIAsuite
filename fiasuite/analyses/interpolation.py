"""Interpolation traces: resampling detection (Popescu & Farid's p-map in
Kirchner's fast fixed-predictor form) and CFA demosaicing consistency
(Popescu & Farid 2005, Ferrara et al. 2012)."""
from __future__ import annotations

import cv2
import numpy as np

from ..core import INFO, NOTE, WARN, Evidence, Finding, Result, block_reduce, coherent_fraction, heatmap

_PREDICT = np.array([[-0.25, 0.5, -0.25], [0.5, 0, 0.5], [-0.25, 0.5, -0.25]], np.float32)


def _spectrum_mask(n: int) -> np.ndarray:
    """Frequencies kept for peak search: drop low frequencies, the axes (natural
    image energy) and the n/8 lattice (JPEG blocking)."""
    f = np.fft.fftshift(np.fft.fftfreq(n)) * n
    fy, fx = np.meshgrid(f, f, indexing="ij")
    keep = np.hypot(fy, fx) > n * 0.15
    keep &= (np.abs(fy) > 2) & (np.abs(fx) > 2)
    lattice = n / 8
    near = lambda a: np.abs(a / lattice - np.rint(a / lattice)) * lattice <= 1.5
    keep &= ~(near(fy) & near(fx))
    return keep


def resampling(ev: Evidence, window: int = 128) -> Result:
    n = int(window)
    # Interpolation makes the variance of the second derivative periodic (Gallagher 2005,
    # Mahdian & Saic 2008); its magnitude separates better than the p-map on textured content.
    pmap = np.abs(cv2.Laplacian(ev.luma, cv2.CV_32F))
    res = Result()
    err = ev.luma - cv2.filter2D(ev.luma, -1, _PREDICT, borderType=cv2.BORDER_REFLECT)
    res.views["Probability map"] = (np.exp(-(err ** 2)) * 255).astype(np.uint8)[..., None].repeat(3, 2)
    hop = n // 2
    ys = range(0, ev.height - n + 1, hop)
    xs = range(0, ev.width - n + 1, hop)
    if not ys or not xs:
        res.findings.append(Finding(INFO, "Image too small for the selected window"))
        return res
    hann = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    keep = _spectrum_mask(n)
    ratio = np.zeros((len(ys), len(xs)), np.float32)
    valid = np.zeros_like(ratio, bool)
    mean_spec = np.zeros((n, n), np.float64)
    for i, y in enumerate(ys):
        rows = np.stack([pmap[y:y + n, x:x + n] for x in xs])
        tex = np.stack([ev.luma[y:y + n, x:x + n].std() for x in xs])
        rows = (rows - rows.mean(axis=(1, 2), keepdims=True)) * hann
        spec = np.fft.fftshift(np.abs(np.fft.fft2(rows)), axes=(1, 2)).astype(np.float32)
        for j, s in enumerate(spec):              # peak height over the local spectral background
            ratio[i, j] = (s / (cv2.medianBlur(s, 5) + 1e-9))[keep].max()
        valid[i] = tex > 3
        mean_spec += spec[valid[i]].sum(axis=0)
    ratio = np.where(valid, ratio, 0)
    res.views["Peak strength"] = heatmap(ratio, (ev.height, ev.width), 3, 12)
    avg = np.log1p(mean_spec / max(valid.sum(), 1))
    res.views["Mean p-map spectrum"] = cv2.resize(heatmap(avg, None, np.percentile(avg, 50), avg.max()), (512, 512),
                                                  interpolation=cv2.INTER_NEAREST)
    hits = ratio > 6
    share = float(hits.sum() / max(valid.sum(), 1))
    local = coherent_fraction(hits, 2)
    res.table = [("Windows with spectral peaks", f"{share:.1%}"), ("Median peak ratio", f"{np.median(ratio[valid]):.1f}" if valid.any() else "-")]
    if share > 0.5:
        res.score = 0.3
        res.findings.append(Finding(NOTE, "Whole image has been resampled",
            "Periodic interpolation correlations everywhere: the image was resized or rotated (usual for web and "
            "social-media copies)."))
    elif local > 0.01:
        res.score = float(np.clip(0.5 + local * 5, 0, 0.95))
        res.findings.append(Finding(WARN, "Locally resampled region",
            f"About {local:.1%} of the image shows interpolation peaks that the rest lacks: an object was scaled or "
            "rotated before being inserted. Regular textures (fabric, fences, screens) also trigger this."))
    else:
        res.score = 0.05
        res.findings.append(Finding(INFO, "No resampling traces",
            "Traces do not survive strong JPEG compression applied after the resize."))
    return res


def cfa(ev: Evidence, block: int = 32) -> Result:
    b = int(block) // 2 * 2
    green = ev.bgr[:, :, 1].astype(np.float32)
    kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], np.float32) / 4
    err2 = (green - cv2.filter2D(green, -1, kernel, borderType=cv2.BORDER_REFLECT)) ** 2
    yy, xx = np.indices(green.shape)
    lattice = ((yy + xx) % 2 == 0)
    var_a = block_reduce(err2 * lattice, b) * 2
    var_b = block_reduce(err2 * ~lattice, b) * 2
    mean = block_reduce(green, b)
    valid = (var_a + var_b > 0.5) & (mean > 8) & (mean < 247)
    ratio = np.where(valid, np.log((var_a + 1e-3) / (var_b + 1e-3)), 0)

    res = Result()
    if valid.sum() < 30:
        res.findings.append(Finding(INFO, "Too little texture for CFA analysis"))
        return res
    med = float(np.median(ratio[valid]))
    sign = np.sign(med) or 1.0
    agree = float((np.sign(ratio[valid]) == sign).mean())
    res.views["CFA trace strength"] = heatmap(ratio * sign, (ev.height, ev.width), -abs(med) * 2 - 0.05, abs(med) * 2 + 0.05,
                                              cv2.COLORMAP_TWILIGHT_SHIFTED, smooth=False)
    res.table = [("Median lattice log-ratio", f"{med:+.3f}"), ("Blocks agreeing in sign", f"{agree:.0%}")]
    if abs(med) < 0.04 or agree < 0.7:
        res.score = 0.15
        res.findings.append(Finding(INFO, "No demosaicing (Bayer) traces",
            "Expected for resized or strongly compressed images, computational phone photos, screenshots and "
            "AI-generated images. Only a camera original at native resolution keeps these traces."))
        return res
    missing = valid & (ratio * sign < 0.25 * abs(med))
    frac = coherent_fraction(missing)
    res.table.append(("Area lacking CFA traces", f"{frac:.1%}"))
    res.findings.append(Finding(INFO, "Bayer demosaicing traces present",
                                "The image still carries its sensor interpolation pattern: consistent with a camera original."))
    res.score = float(np.clip(frac * 6, 0.02, 0.9))
    if frac > 0.03:
        res.findings.append(Finding(WARN, "Region without demosaicing traces",
            f"{frac:.1%} of the image lacks the Bayer pattern found elsewhere: content that did not come through this "
            "camera's sensor pipeline (paste, inpainting, heavy retouching)."))
    return res
