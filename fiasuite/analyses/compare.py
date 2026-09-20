"""Two-image comparison: align a reference onto the questioned image, map the
differences, diff the metadata."""
from __future__ import annotations

import cv2
import numpy as np

from ..core import INFO, NOTE, WARN, Evidence, Finding, Result, coherent_fraction
from . import spectral
from .metadata import read_metadata

_SKIP_GROUPS = ("System", "File", "ExifTool", "Composite")


def _align(ref: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    """Warp ref into target's frame using ORB matches; fall back to plain resizing.
    Returns (aligned reference, mask of target pixels the reference covers, description)."""
    h, w = target.shape[:2]
    everywhere = np.ones((h, w), bool)
    if ref.shape[:2] == (h, w):
        return ref, everywhere, "same dimensions, compared pixel by pixel"
    orb = cv2.ORB_create(5000)
    ka, da = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), None)
    kb, db = orb.detectAndCompute(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY), None)
    if da is not None and db is not None:
        matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(da, db), key=lambda m: m.distance)[:800]
        if len(matches) >= 12:
            src = np.float32([ka[m.queryIdx].pt for m in matches])
            dst = np.float32([kb[m.trainIdx].pt for m in matches])
            H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
            if H is not None and inliers.sum() >= 12:
                covered = cv2.warpPerspective(np.full(ref.shape[:2], 255, np.uint8), H, (w, h), flags=cv2.INTER_NEAREST)
                return cv2.warpPerspective(ref, H, (w, h), flags=cv2.INTER_LINEAR), covered == 255, \
                    f"registered with {int(inliers.sum())} feature matches (crop / scale / perspective compensated)"
    return cv2.resize(ref, (w, h), interpolation=cv2.INTER_AREA), everywhere, \
        "no registration possible: reference simply resized"


def compare(ev: Evidence, ref: Evidence, amplify: int = 8) -> Result:
    aligned, covered, how = _align(ref.bgr, ev.bgr)
    diff = cv2.absdiff(cv2.GaussianBlur(ev.bgr, (0, 0), 1.0), cv2.GaussianBlur(aligned, (0, 0), 1.0)).max(axis=2)
    changed = (diff > 24) & covered                            # outside the warped reference nothing can be compared
    # 8x8 cells in which at least a quarter of the pixels changed: keeps small edits, drops single-pixel speckle
    cells = cv2.resize(changed.astype(np.float32), None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA) >= 0.25
    frac = coherent_fraction(cells, 2)
    res = Result()
    res.views["Difference"] = cv2.applyColorMap(np.clip(diff.astype(np.int32) * amplify, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    res.views["Reference (aligned)"] = aligned
    mask = cv2.dilate(changed.astype(np.uint8), np.ones((9, 9), np.uint8))
    marked = ev.bgr.copy()
    marked[mask > 0] = (0.5 * marked[mask > 0] + 0.5 * np.array([60, 60, 255])).astype(np.uint8)
    res.views["Changed regions"] = marked
    hamming = bin(ev.phash ^ ref.phash).count("1")
    res.table = [("Alignment", how), ("Questioned image", f"{ev.width} x {ev.height}, {len(ev.data):,} bytes"),
                 ("Reference image", f"{ref.width} x {ref.height}, {len(ref.data):,} bytes"),
                 ("Identical files", "yes" if ev.hashes["sha256"] == ref.hashes["sha256"] else "no"),
                 ("Perceptual-hash distance", f"{hamming} of 64 bits"), ("Mean difference", f"{diff[covered].mean():.2f}"),
                 ("Changed area", f"{frac:.2%}")]
    if ev.hashes["sha256"] == ref.hashes["sha256"]:
        res.findings.append(Finding(INFO, "The two files are bit-for-bit identical"))
    elif hamming > 24:
        res.findings.append(Finding(NOTE, "The images show different content", "The perceptual hashes are far apart."))
    elif frac > 0.002:
        res.findings.append(Finding(WARN, "Localised differences between the two versions",
            f"{frac:.1%} of the picture differs in connected regions while the rest matches: content was added, removed "
            "or retouched in one version. See the Changed regions view."))
    else:
        res.findings.append(Finding(INFO, "Same picture; differences are global only",
            "Only compression, resizing or tonal differences - no localised change."))

    # noise-residual spectra side by side: does the questioned image carry the same sensor / processing
    # signature as a known image from the claimed camera?
    n = 256 if min(ev.height, ev.width, ref.height, ref.width) < 1200 else 512
    for label, item in (("questioned", ev), ("reference", ref)):
        amp = spectral.averaged_spectrum(item.luma, n, "Wiener (adaptive)")[0]
        if amp is not None:
            log_flat, _, rr, u, v = spectral.flatten(amp)
            peaks = spectral.find_peaks(log_flat, rr, u, v, n, item.is_jpeg)
            res.views[f"Spectrum: {label}"] = spectral.annotated(log_flat, peaks)
            res.table.append((f"Unexplained spectral peaks, {label}", str(sum(p["class"] == "other" for p in peaks))))

    a, b = read_metadata(ev), read_metadata(ref)
    for key in sorted(set(a) | set(b)):
        if key.split(":")[0] in _SKIP_GROUPS or key == "SourceFile":
            continue
        va, vb = str(a.get(key, "(absent)")), str(b.get(key, "(absent)"))
        if va != vb:
            res.table.append((key, f"{va[:120]}   |   reference: {vb[:120]}"))
    return res
