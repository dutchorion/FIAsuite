"""Compression-history forensics: ELA, JPEG ghosts (Farid 2009), double
quantization via DCT coefficient histograms (Lukas/Fridrich, Popescu/Farid),
and blocking-artifact grid analysis (Li et al. 2009)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy.fft import dctn

from ..core import ALERT, INFO, NOTE, WARN, Evidence, Finding, Result, block_reduce, coherent_fraction, heatmap, plot
from .jpegstruct import STD_LUMA, ZIGZAG, estimate_quality, ijg_table, luma_qtable


def _recompress(bgr: np.ndarray, quality: int) -> np.ndarray:
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


# ------------------------------------------------------------------ ELA

ELA_MODES = ["Colour", "Brightest channel"]


def ela(ev: Evidence, quality: int = 90, scale: int = 20, mode: str = "Colour", auto_scale: bool = True) -> Result:
    diff = cv2.absdiff(ev.bgr, _recompress(ev.bgr, quality)).astype(np.float32)
    res = Result()
    shown = diff if mode == "Colour" else diff.max(axis=2, keepdims=True).repeat(3, 2)   # grey: no colour fringes to misread
    if auto_scale:                                 # high-quality files leave errors of 1-2 levels: invisible at a fixed scale
        scale = float(np.clip(200 / max(np.percentile(diff, 99.5), 1e-3), 1, 100))
    res.views["Error level"] = np.clip(shown * scale, 0, 255).astype(np.uint8)
    blocks = block_reduce(diff.mean(axis=2), 8)
    res.views["Block average"] = heatmap(blocks, (ev.height, ev.width), 0, np.percentile(blocks, 99.5) + 1e-6, smooth=False)
    res.table = [("Mean error", f"{diff.mean():.2f}"), ("99th percentile", f"{np.percentile(diff, 99):.1f}"),
                 ("Error scale used", f"{scale:.0f}x" + ("  (automatic)" if auto_scale else ""))]
    res.findings.append(Finding(INFO, "Visual tool",
        "Compare surfaces of similar texture: they should show similar error levels. A region that is clearly "
        "brighter or darker than comparable surfaces was compressed a different number of times."))
    return res


# ------------------------------------------------------------------ JPEG ghosts

def _ghost_curves(gray: np.ndarray, qualities: list[int], dy: int, dx: int, block: int) -> np.ndarray:
    """(Q, h, w) block-averaged squared error between the luminance and its
    recompression at each quality, on the 8x8 grid starting at (dy, dx)."""
    sub = gray[dy:, dx:]
    hop = max(block // 2, 1)
    while (gray.shape[0] // hop) * (gray.shape[1] // hop) > 120_000:
        hop *= 2
    out = []
    for q in qualities:
        _, enc = cv2.imencode(".jpg", sub, [cv2.IMWRITE_JPEG_QUALITY, q])
        d = (sub.astype(np.float32) - cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)) ** 2
        d = cv2.copyMakeBorder(d, dy, 0, dx, 0, cv2.BORDER_REPLICATE)
        out.append(cv2.blur(d, (block, block))[::hop, ::hop])
    return np.stack(out)


def _dips(log_curve: np.ndarray, reach: int = 6) -> np.ndarray:
    """Depth of the valley at each point of a curve (axis 0 = quality), after 3-tap smoothing: how far the point
    lies below BOTH the highest value within `reach` steps to its left and the highest within `reach` steps to its
    right. A ghost is such a valley (the error falls towards the earlier quality and rises again above it); a
    monotone curve, however steep, scores zero."""
    rest = [(0, 0)] * (log_curve.ndim - 1)
    pad = np.pad(log_curve, [(1, 1)] + rest, mode="edge")
    smooth = (pad[:-2] + pad[1:-1] + pad[2:]) / 3
    n = len(smooth)
    pad = np.pad(smooth, [(reach, reach)] + rest, mode="edge")
    left = np.max(np.stack([pad[reach - o:reach - o + n] for o in range(2, reach + 1)]), axis=0)
    right = np.max(np.stack([pad[reach + o:reach + o + n] for o in range(2, reach + 1)]), axis=0)
    return np.minimum(left, right) - smooth


def ghosts(ev: Evidence, qmin: int = 40, qmax: int = 96, step: int = 2, block: int = 16,
           all_shifts: bool = False) -> Result:
    """Farid 2009. Content that was once saved at JPEG quality Q1 shows a minimum of
    its recompression error at Q1. Each block's error curve is compared with the
    image-wide curve, which cancels the file's own compression history."""
    gray = cv2.cvtColor(ev.bgr, cv2.COLOR_BGR2GRAY)
    qualities = list(range(int(qmin), int(qmax) + 1, int(step)))
    mean = cv2.blur(ev.luma, (block, block))
    texture = np.sqrt(np.maximum(cv2.blur(ev.luma ** 2, (block, block)) - mean ** 2, 0))
    # A ghost only appears when the recompression grid matches the grid of the earlier
    # compression, so test the file's own grid plus any foreign grid found in the image.
    if all_shifts:
        shifts = [(y, x) for y in range(8) for x in range(8)]
    else:
        shifts = list(dict.fromkeys([(0, 0), *foreign_grid(ev)[1], *grid_hints(ev)]))

    best, global_dips = None, []
    reach = max(2, round(12 / int(step)))               # the valley of a ghost spans about 12 quality units a side
    for dy, dx in shifts:
        d = _ghost_curves(gray, qualities, dy, dx, int(block))
        # Smooth blocks are useless: their error is pure DC rounding and vanishes at arbitrary qualities.
        valid = cv2.resize(texture, d.shape[:0:-1], interpolation=cv2.INTER_AREA) > 3
        if valid.sum() < 50:
            continue
        level = d.mean(0)
        overall = np.median(d[:, valid], axis=1)
        usable = overall > 0.1 * overall.max()    # near the file's own quality the error vanishes
        if (dy, dx) == (0, 0):
            own = _dips(np.log(overall + 0.02 * overall.max() + 1e-6), reach)
            global_dips = [q for q, s in zip(qualities, own) if s > 0.4]
        dip = _dips(np.log(d + 0.05 * level + 1e-3) - np.log(overall + 0.05 * overall.mean())[:, None, None], reach)
        if global_dips and (dy, dx) == (0, 0):
            # The whole image was compressed twice on this grid. Its blocks then show minima at the earlier quality
            # AND at harmonics of it (Q70 then Q90 gives dips near Q80), more or less clearly depending on texture:
            # 15 of 40 clean re-saved photographs were flagged before this rule. The double-quantization map of the
            # DCT tool is the right instrument for that case; ghosts on a FOREIGN grid remain valid.
            continue
        dip[~usable] = 0
        dip[:, ~valid] = 0
        for i in np.nonzero(usable)[0]:
            # an insert is a neighbourhood of moderate dips; scene texture gives isolated strong ones
            frac = coherent_fraction(cv2.blur(np.clip(dip[i], 0, 1), (3, 3)) > 0.3)
            if best is None or frac > best[0]:
                best = (frac, qualities[i], (dy, dx), np.clip(dip[i], 0, 2), d, valid)
    if best is None:
        res = Result()
        if global_dips:
            res.score = 0.05
            res.table = [("Image-wide error minima", ", ".join(f"Q{g}" for g in global_dips))]
            res.findings.append(Finding(NOTE, f"Whole image was previously saved at about Q{global_dips[0]}",
                "After a recompression of the entire picture a local ghost on the file's own grid cannot be told "
                "from the image-wide one, so only misaligned grids were searched (none found). Use the "
                "double-quantization map of the DCT tool to localise content added between the two saves."))
        else:
            res.findings.append(Finding(INFO, "Image too flat for ghost analysis"))
        return res

    frac, q, (dy, dx), strength, d, valid = best
    lo, hi = d.min(0), d.max(0)
    norm = (d - lo) / np.maximum(hi - lo, 1e-6)
    res = Result()
    res.views["Ghost strength"] = heatmap(strength, (ev.height, ev.width), 0, 1.5)
    res.views["All qualities"] = _montage(norm[::2], valid, qualities[::2])
    found = 0.003 < frac < 0.7
    res.score = float(np.clip(0.55 + frac * 4, 0, 1)) if found else 0.05
    res.table = [("Qualities tested", f"{qualities[0]}..{qualities[-1]} step {step}"),
                 ("Grid alignments tested", ", ".join(f"dx={x} dy={y}" for y, x in shifts) if len(shifts) < 9 else "all 64"),
                 ("Ghost", f"Q{q} on grid dx={dx} dy={dy}, {frac:.1%} of the image" if found else "none"),
                 ("Image-wide error minima", ", ".join(f"Q{g}" for g in global_dips) or "none")]
    table = luma_qtable(ev)
    current = estimate_quality(table, STD_LUMA)[0] if table is not None else 101
    earlier = [g for g in global_dips if g < current - 6]
    if earlier:
        res.findings.append(Finding(NOTE, f"Whole image was previously saved at about Q{earlier[0]}",
            "The recompression error of the entire image has a minimum at this quality in addition to the file's "
            "current quality: the picture was re-saved (double compression)."))
    if found:
        res.findings.append(Finding(ALERT if frac > 0.02 else WARN, f"JPEG ghost at quality {q}",
            f"{frac:.1%} of the image forms a connected region whose recompression error has a minimum at "
            f"quality {q} (grid offset dx={dx}, dy={dy}) that the rest of the image lacks: that content was "
            "previously saved as a lower-quality JPEG, i.e. it was spliced in from another file."))
    else:
        res.findings.append(Finding(INFO, "No JPEG ghost",
            "Only detects inserted content whose earlier JPEG quality was LOWER than the final save. Enable "
            "'Test all 64 grid alignments' for an exhaustive (slow) search."))
    return res


def _montage(norm: np.ndarray, valid: np.ndarray, qualities: list[int], cols: int = 4) -> np.ndarray:
    n, h, w = norm.shape
    scale = 320 / w
    tw, th = 320, max(1, int(h * scale))
    rows = -(-n // cols)
    canvas = np.full((rows * (th + 22), cols * tw, 3), 24, np.uint8)
    for i, q in enumerate(qualities):
        tile = heatmap(np.where(valid, 1 - norm[i], 0), (th, tw), 0, 1)
        y, x = (i // cols) * (th + 22), (i % cols) * tw
        canvas[y + 22:y + 22 + th, x:x + tw] = tile
        cv2.putText(canvas, f"Q{q}", (x + 6, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return canvas


# ------------------------------------------------------------------ blocking-artifact grid

def _grid_signal(luma: np.ndarray, axis: int) -> np.ndarray:
    """Excess of each pixel-to-pixel step over its two neighbouring steps. Entry i
    describes the boundary between line i+1 and i+2 along `axis`."""
    d = np.clip(np.abs(np.diff(luma, axis=axis)), 0, 24)
    if axis == 0:
        g = d[1:-1] - 0.5 * (d[:-2] + d[2:])
    else:
        g = d[:, 1:-1] - 0.5 * (d[:, :-2] + d[:, 2:])
    return np.clip(g, -4, 4)


def _phase_stats(profile: np.ndarray) -> np.ndarray:
    """t-statistic of each of the 8 grid phases of a 1-D boundary profile."""
    phases = (np.arange(profile.size) + 2) % 8
    centred = profile - profile.mean()
    sd = centred.std() + 1e-9
    return np.array([centred[phases == p].mean() / (sd / np.sqrt(max((phases == p).sum(), 1))) for p in range(8)])


def grid_offset(ev: Evidence) -> tuple[int, int, float]:
    """Dominant 8x8 grid origin (dy, dx) and its significance."""
    with ev.lock:
        if "grid" not in ev.cache:
            tv = _phase_stats(_grid_signal(ev.luma, 0).mean(axis=1))
            th = _phase_stats(_grid_signal(ev.luma, 1).mean(axis=0))
            ev.cache["grid"] = (int(tv.argmax()), int(th.argmax()), float(min(tv.max(), th.max())), tv, th)
    return ev.cache["grid"][:3]


def block_grid(ev: Evidence, window: int = 64) -> Result:
    dy, dx, strength = grid_offset(ev)
    tv, th = ev.cache["grid"][3:]
    res = Result()
    res.views["Phase energy"] = plot([("rows", tv, (80, 200, 255)), ("columns", th, (120, 255, 120))],
                                     "8x8 grid energy by phase (t-statistic); phase 0 = aligned with file", bars=False,
                                     xlabels=("phase 0", "phase 7"))
    res.table = [("Dominant grid origin", f"dx={dx}, dy={dy}"), ("Significance", f"t = {strength:.1f}")]

    (main, foreign, outliers), candidates = foreign_grid(ev, window)
    res.views["Expected grid strength"] = heatmap(main, (ev.height, ev.width), 0, np.percentile(main, 99) + 1e-6)
    res.views["Foreign grid"] = heatmap(foreign, (ev.height, ev.width), 0, max(np.percentile(foreign, 99.5), 0.3))
    if candidates:
        res.table.append(("Foreign grid origin", ", ".join(f"dx={x} dy={y}" for y, x in candidates)))

    significant = strength > 6
    if not significant:
        res.findings.append(Finding(INFO, "No JPEG block grid detected",
            "No measurable 8x8 blocking: never JPEG-compressed, compressed at very high quality, or resized afterwards "
            "(resizing destroys the grid; so does AI upscaling)."))
        res.score = 0.1
        return res
    if (dy, dx) != (0, 0):
        # one direction only: regular scene structure with an 8 px period (blinds, bricks) fakes that; a crop that
        # moves the grid in both directions at once is far less likely to be imitated by content
        sev = WARN if ev.is_jpeg and dy and dx else NOTE
        res.findings.append(Finding(sev, f"Block grid is shifted (dx={dx}, dy={dy})",
            "The dominant 8x8 grid does not start at the image origin: the picture was cropped after an earlier JPEG "
            "compression" + ("" if ev.is_jpeg else " and then stored in a lossless format") + "."))
    elif not ev.is_jpeg:
        res.findings.append(Finding(NOTE, "JPEG history in a non-JPEG file",
            "An aligned 8x8 block grid is present: this image was JPEG-compressed before it was saved in its current format."))
    if min(np.delete(tv, dy).max(), np.delete(th, dx).max()) > 8:
        res.findings.append(Finding(WARN, "Second block grid at a different alignment",
            "Traces of two differently aligned JPEG grids: cropped between two compressions, or content pasted from "
            "another JPEG. Check the Foreign grid view for a localised region."))
    frac = coherent_fraction(outliers, 2) if candidates else 0.0
    res.score = float(np.clip(0.5 + frac * 8, 0, 1)) if frac > 0.004 else (0.45 if dy and dx and ev.is_jpeg else 0.1)
    if frac > 0.004:
        res.findings.append(Finding(WARN, "Region with a misaligned JPEG grid",
            f"{frac:.1%} of the image carries a block grid that does not line up with the rest: typical of content "
            "pasted from another JPEG."))
    elif len(res.findings) == 0:
        res.findings.append(Finding(INFO, "Single, consistently aligned block grid"))
    return res


def foreign_grid(ev: Evidence, window: int = 64):
    """Windows whose strongest 8x8 grid is not the image's dominant grid.
    Returns (main strength map, foreign strength map, outlier mask) and up to two
    candidate (dy, dx) origins of the foreign grid."""
    key = ("foreign_grid", int(window))
    with ev.lock:
        if key not in ev.cache:
            ev.cache[key] = _foreign_grid(ev, int(window))
    return ev.cache[key]


def _phases(ev: Evidence, w: int):
    """(rows, columns) grid energy per phase and window, each minus its median over the 8 phases; and per window the
    strongest phase in each direction."""
    key = ("phase_energy", w)
    with ev.lock:
        if key not in ev.cache:
            lv = _local_phase_energy(_grid_signal(ev.luma, 0), 0, w)
            lh = _local_phase_energy(_grid_signal(ev.luma, 1), 1, w)
            hh, ww = min(lv.shape[1], lh.shape[1]), min(lv.shape[2], lh.shape[2])
            lv, lh = lv[:, :hh, :ww] - np.median(lv[:, :hh, :ww], 0), lh[:, :hh, :ww] - np.median(lh[:, :hh, :ww], 0)
            ev.cache[key] = (lv, lh)
    return ev.cache[key]


def _votes(ev: Evidence, w: int):
    """Windows that show a grid in BOTH directions at an alignment differing from the dominant one in both, grouped
    by alignment: [((dy, dx), count, mask), ...], most frequent first. Scene structure (horizons, facades) votes at
    random and in one direction; the windows of a pasted JPEG agree on one alignment."""
    dy, dx, _ = grid_offset(ev)
    lv, lh = _phases(ev, w)
    pv, ph = lv.argmax(0), lh.argmax(0)
    clear = (lv.max(0) > 0.15) & (lh.max(0) > 0.15) & (pv != dy) & (ph != dx)
    pairs = np.unique(np.stack([pv[clear], ph[clear]], 1), axis=0)
    out = []
    for py, px in pairs:                           # rank by the largest group of NEIGHBOURING windows that agree
        mask = clear & (pv == py) & (ph == px)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)))
        biggest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
        out.append(((int(py), int(px)), int((mask & (labels == biggest)).sum()), labels == biggest))
    return sorted(out, key=lambda v: -v[1])


def _foreign_grid(ev: Evidence, window: int):
    dy, dx, _ = grid_offset(ev)
    w = min(window, ev.height - 8, ev.width - 8) // 8 * 8
    if w < 16:                                     # image smaller than any usable window
        empty = np.zeros((1, 1), np.float32)
        return (empty, empty, empty > 0), []
    lv, lh = _phases(ev, w)
    main = lv[dy] + lh[dx]
    # excess of the best other phase over the dominant phase, per direction; the other
    # direction must show a grid too (at any phase), which rejects plain image texture
    sv = np.clip(np.delete(lv, dy, 0).max(0) - np.maximum(lv[dy], 0), 0, None)
    sh = np.clip(np.delete(lh, dx, 0).max(0) - np.maximum(lh[dx], 0), 0, None)
    foreign = np.maximum(sv * (lh.max(0) > 0.3), sh * (lv.max(0) > 0.3))
    pooled = cv2.blur(foreign, (3, 3))            # a pasted region spans several windows; texture hits are isolated
    spread = 1.4826 * np.median(np.abs(pooled - np.median(pooled)))
    outliers = pooled > max(np.median(pooled) + 6 * spread, 0.2)
    candidates = []
    if coherent_fraction(outliers, 2) > 0.004:
        solid = outliers & (foreign > 0.2)
        pairs = np.stack([lv.argmax(0)[solid], lh.argmax(0)[solid]], 1)
        found, counts = np.unique(pairs, axis=0, return_counts=True)
        # pasted JPEG content has ONE grid origin; texture hits scatter over many phases
        top = int(counts.argmax()) if len(counts) else 0
        if len(counts) and counts[top] >= max(6, 0.5 * counts.sum()) and tuple(found[top]) != (dy, dx):
            candidates = [(int(found[top][0]), int(found[top][1]))]
    if not candidates:
        # weaker grids (insert re-saved at high quality, busy content): several neighbouring windows that agree on
        # one foreign alignment, clearly more often than on any other
        votes = _votes(ev, w)
        # measured on 40 untouched and 40 spliced photographs: untouched ones reach 6 agreeing windows, inserts 8-19
        # a grid leaves a weaker trace half a block away too: an alignment 4 px from the winner is not a rival
        (ty, tx) = votes[0][0] if votes else (0, 0)
        rival = max((n for (py, px), n, _ in votes[1:] if (py - ty) % 4 or (px - tx) % 4), default=0)
        if votes and votes[0][1] >= 8 and votes[0][1] >= 2 * rival:
            (py, px), _, outliers = votes[0]
            candidates = [(py, px)]
    if candidates:                                 # show the grid energy AT the foreign alignment: that is the insert
        py, px = candidates[0]
        foreign = np.clip(lv[py], 0, None) + np.clip(lh[px], 0, None)
    return (main, foreign, outliers), candidates


def grid_hints(ev: Evidence, window: int = 64, limit: int = 6) -> list[tuple[int, int]]:
    """Grid alignments (dy, dx) worth testing for a JPEG ghost. Much weaker evidence than a foreign-grid finding -
    textured content produces such votes too - so they only select what the ghost test examines. A ghost needs the
    exact alignment, so both members of a half-block alias pair are offered."""
    w = min(int(window), ev.height - 8, ev.width - 8) // 8 * 8
    if w < 16:
        return []
    dy, dx, _ = grid_offset(ev)
    out = []
    for (py, px), count, _ in _votes(ev, w):
        if count >= 2:
            out += [(py, px), ((py + 4) % 8, px), (py, (px + 4) % 8), ((py + 4) % 8, (px + 4) % 8)][:2 if count < 6 else 4]
    return [pair for pair in dict.fromkeys(out) if pair != (dy, dx)][:limit]


def _local_phase_energy(g: np.ndarray, axis: int, w: int) -> np.ndarray:
    """(8, H/w, W/w) mean grid signal per phase inside w x w windows."""
    out = []
    for p in range(8):
        start = (p - 2) % 8
        lines = g[start::8] if axis == 0 else g[:, start::8]
        step = (w // 8, w) if axis == 0 else (w, w // 8)
        h, ww = lines.shape[0] // step[0] * step[0], lines.shape[1] // step[1] * step[1]
        out.append(lines[:h, :ww].reshape(h // step[0], step[0], ww // step[1], step[1]).mean(axis=(1, 3)))
    hh, ww = min(o.shape[0] for o in out), min(o.shape[1] for o in out)
    return np.stack([o[:hh, :ww] for o in out])


# ------------------------------------------------------------------ DCT histograms / double quantization

AC_POSITIONS = [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2)]


def block_dct(luma: np.ndarray, dy: int = 0, dx: int = 0) -> np.ndarray:
    """(N, 8, 8) DCT of the 8x8 blocks on the grid starting at (dy, dx)."""
    a = luma[dy:, dx:] - 128.0
    h, w = a.shape[0] // 8 * 8, a.shape[1] // 8 * 8
    blocks = a[:h, :w].reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    return dctn(blocks, axes=(1, 2), norm="ortho")


def _comb_step(coefs: np.ndarray) -> tuple[int, float]:
    """Largest quantization step whose comb explains the coefficient values."""
    best, best_m = 1, 0.0
    for q in range(2, 48):
        c = coefs[np.abs(coefs) > 1.5 * q]
        if c.size < 300:
            break
        m = float(np.abs(np.exp(2j * np.pi * c / q).mean()))
        if m > 0.35 and m >= 0.8 * best_m:
            best, best_m = q, max(m, best_m)
    return best, best_m


def _periodicity(hist: np.ndarray) -> float:
    """Mean |log| ratio between each bin and the average of its neighbours: near 0
    for a smooth single-compression histogram, large for double quantization."""
    h = hist.astype(np.float64)
    mid, nb = h[1:-1], 0.5 * (h[:-2] + h[2:])
    ok = (mid + nb) > 40
    if ok.sum() < 5:
        return 0.0
    return float(np.mean(np.abs(np.log((mid[ok] + 1) / (nb[ok] + 1)))))


def _dq_period(hist: np.ndarray) -> int:
    """Period (in quantization steps) of the comb that double quantization leaves in a coefficient histogram
    (index = k + K), 0 if there is none."""
    K = len(hist) // 2
    logh = np.log(hist + 1.0)
    detr = logh - np.convolve(np.pad(logh, 8, mode="edge"), np.ones(17) / 17, mode="valid")
    detr[K - 1:K + 2] = 0                          # the peak at zero belongs to every histogram
    best, best_gain = 0, 0.0
    for period in range(2, 13):
        on = detr[K % period::period]              # bins at multiples of the period, counted from k = 0
        gain = float(on.mean() - (detr.sum() - on.sum()) / max(detr.size - on.size, 1))
        if gain > 0.5 and gain > 1.15 * best_gain:   # prefer the shortest period that explains the comb
            best, best_gain = period, gain
    return best


def dq_map(coefs: np.ndarray, qtable: np.ndarray, shape: tuple[int, int], count: int = 14, K: int = 40):
    """Lin et al. 2009: per-block probability of NOT sharing the image's double-quantization comb.
    In a double-compressed image the histogram of each DCT frequency is periodic. A coefficient that sits in a
    valley of that comb is unlikely in double-compressed content and ordinary in content compressed only once
    (inserted after the first compression). Returns (probability map over blocks, frequencies used)."""
    log_odds, used = np.zeros(len(coefs)), 0
    for flat in ZIGZAG[1:count + 1]:
        u, v = divmod(int(flat), 8)
        k = np.rint(coefs[:, u, v] / qtable[u, v]).astype(np.int32)
        inside = np.abs(k) <= K
        hist = np.bincount(k[inside] + K, minlength=2 * K + 1).astype(np.float64)
        period = _dq_period(hist)
        if not period:
            continue
        window = np.convolve(np.pad(hist, period, mode="constant"), np.ones(period), mode="same")[period:-period]
        p_double = np.clip(hist / np.maximum(window, 1), 1e-3, 1)      # share of its period that this bin holds
        term = np.log((1 / period) / p_double)                          # single compression: uniform over the period
        log_odds[inside] += term[k[inside] + K]
        used += 1
    if not used:
        return None, 0
    prob = 1 / (1 + np.exp(-np.clip(log_odds, -30, 30)))
    return prob.reshape(shape).astype(np.float32), used


def dct_analysis(ev: Evidence) -> Result:
    res = Result()
    dy, dx, grid_t = grid_offset(ev)
    if ev.is_jpeg:
        dy = dx = 0
    coefs = block_dct(ev.luma, dy, dx)
    qtable = luma_qtable(ev)
    panels, scores, steps = [], [], []
    for (u, v) in AC_POSITIONS:
        c = coefs[:, u, v]
        if qtable is not None:
            q = int(qtable[u, v])
        else:
            q, _ = _comb_step(c)
        steps.append(q)
        k = np.rint(c / q).astype(np.int32)
        k = np.abs(k[np.abs(k) <= 40])
        hist = np.bincount(k, minlength=41)[2:]          # skip the 0/1 peak
        s = _periodicity(hist) if q >= 2 else 0.0
        scores.append(s)
        panels.append(plot([(f"AC({u},{v}) step {q}  periodicity {s:.2f}", hist, (90, 190, 255))],
                           size=(450, 220), bars=True, xlabels=("|k|=2", "40")))
    rows = [np.hstack(panels[i:i + 2]) for i in range(0, 6, 2)]
    res.views["Coefficient histograms"] = np.vstack(rows)
    dq = float(np.median(scores))
    res.table = [("Grid origin used", f"dx={dx}, dy={dy}"), ("Quantization steps (first AC terms)", str(steps)),
                 ("Double-quantization periodicity", f"{dq:.2f}  (single compression is typically < 0.35)")]

    if qtable is None:
        if grid_t > 6 and max(steps) >= 2:
            est = [min(range(1, 101), key=lambda q: abs(int(ijg_table(STD_LUMA, q)[u, v]) - s))
                   for (u, v), s in zip(AC_POSITIONS, steps)]
            res.findings.append(Finding(NOTE, "Quantized DCT coefficients in a lossless file",
                f"The coefficients sit on a comb (steps {steps}): the image was JPEG-compressed before being saved "
                f"in its current format."))
            res.table.append(("Rough earlier quality", f"~{int(np.median(est))}"))
            res.score = 0.3
        else:
            res.findings.append(Finding(INFO, "No JPEG quantization traces"))
            res.score = 0.05
        return res
    if min(steps) < 3:
        res.findings.append(Finding(INFO, "Very high JPEG quality",
            "Quantization steps below 3 make the double-compression test unreliable."))
    res.score = float(np.clip((dq - 0.35) / 0.5, 0, 1))
    if dq > 0.35:
        _localise_dq(ev, res, coefs, qtable)
    if dq > 0.6:
        res.findings.append(Finding(WARN, "Double JPEG compression",
            "The DCT histograms show the periodic gaps/peaks of double quantization: the image was decoded and "
            "re-saved as JPEG at a different quality. Expected for anything edited or re-shared; it shows the file "
            "is not a camera original."))
    elif dq > 0.35:
        res.findings.append(Finding(NOTE, "Possible double JPEG compression", "Weak periodic structure in the DCT histograms."))
    else:
        res.findings.append(Finding(INFO, "Histograms consistent with a single JPEG compression"))
    return res


def _localise_dq(ev: Evidence, res: Result, coefs: np.ndarray, qtable: np.ndarray) -> None:
    shape = (ev.height // 8, ev.width // 8)
    prob, used = dq_map(coefs, qtable, shape)
    if prob is None:
        return
    smooth = cv2.blur(prob, (5, 5))                # single blocks are noise; an insert covers many
    res.views["Double-quantization map"] = heatmap(smooth, (ev.height, ev.width), 0, 1, cv2.COLORMAP_TURBO)
    frac = coherent_fraction(smooth > 0.7, 5)
    res.table += [("Frequencies with a double-quantization comb", str(used)), ("Area lacking the comb", f"{frac:.1%}")]
    res.metrics["dq_single_area"] = round(frac, 4)
    if 0.005 < frac < 0.5:
        res.findings.append(Finding(WARN, "Region without the double-compression traces of the rest",
            f"{frac:.1%} of the image forms a connected region whose DCT coefficients do not follow the double-"
            "quantization comb found elsewhere: that content entered the picture after the first JPEG compression "
            "(splice, inpainting, retouching). Flat and saturated areas carry no information either way."))
        res.score = max(res.score, float(np.clip(0.55 + frac * 3, 0, 0.95)))
