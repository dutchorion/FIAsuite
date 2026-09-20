"""Copy-move (clone) detection.

keypoints  SIFT self-matching + RANSAC affine clustering (Amerini et al. 2011):
           finds clones that were scaled or rotated, needs texture.
blocks     quantized block features, lexicographic sort and shift-vector voting
           (Fridrich et al. 2003, as used by Forensically): finds flat clones
           such as sky or skin, only for pure translation.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import ALERT, INFO, WARN, Evidence, Finding, Result

SAME_NOISE = 0.3          # correlation of the fine detail of two areas above which one is a copy of the other
_COLORS = [(80, 80, 255), (80, 255, 120), (255, 170, 60), (255, 90, 255), (60, 230, 255), (200, 255, 80)]


def _fine(gray: np.ndarray) -> np.ndarray:
    return gray - cv2.GaussianBlur(gray, (0, 0), 1.5)


def same_noise(fine: np.ndarray, a: np.ndarray, b: np.ndarray, size: int, limit: int = 400) -> float:
    """Correlation between the fine detail (noise, micro-texture) of size x size patches at the points a and
    their counterparts b, given as (y, x) rows. A copied area carries the SAME noise as its source, also after
    JPEG compression; things that merely look alike (windows of a facade, waves, an even sky, a repeating
    pattern) have independent noise and correlate near zero. On 100 untouched camera photographs this check
    removed the large majority of false clone reports."""
    a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
    h, w = fine.shape
    inside = ((a >= 0) & (b >= 0) & (a + size <= (h, w)) & (b + size <= (h, w))).all(axis=1)
    a, b = a[inside], b[inside]
    if len(a) == 0:
        return 0.0
    pick = np.linspace(0, len(a) - 1, min(limit, len(a))).astype(int)
    oy, ox = np.mgrid[0:size, 0:size]
    pa = fine[a[pick, 0, None, None] + oy, a[pick, 1, None, None] + ox].ravel()
    pb = fine[b[pick, 0, None, None] + oy, b[pick, 1, None, None] + ox].ravel()
    return float(np.corrcoef(pa, pb)[0, 1]) if pa.std() > 1e-6 and pb.std() > 1e-6 else 0.0


def same_noise_in_flat_parts(luma: np.ndarray, points: np.ndarray, shift: np.ndarray) -> float:
    """Noise correlation between the area spanned by `points` (x, y) and the same area moved by `shift`, measured
    only where the picture is smooth. Keypoints sit on corners and edges, and repeated architecture has correlated
    EDGES; its smooth surfaces, however, carry independent sensor noise, while a copy carries the identical noise."""
    h, w = luma.shape
    hull = cv2.convexHull(points.astype(np.int32))
    x, y, bw, bh = cv2.boundingRect(hull)
    sx, sy = int(shift[0]), int(shift[1])
    x0, y0 = max(x, 0, -sx), max(y, 0, -sy)
    x1, y1 = min(x + bw, w, w - sx), min(y + bh, h, h - sy)
    if x1 - x0 < 16 or y1 - y0 < 16:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, hull, 1)
    a, b = luma[y0:y1, x0:x1], luma[y0 + sy:y1 + sy, x0 + sx:x1 + sx]
    smooth = cv2.GaussianBlur(a, (0, 0), 2)
    busy = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1))
    keep = (mask[y0:y1, x0:x1] > 0) & (busy <= np.median(busy[mask[y0:y1, x0:x1] > 0]))
    if keep.sum() < 200:
        return 0.0
    ra, rb = (a - cv2.GaussianBlur(a, (0, 0), 1))[keep], (b - cv2.GaussianBlur(b, (0, 0), 1))[keep]
    return float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 1e-6 and rb.std() > 1e-6 else 0.0


def _matches(des_a, pts_a, des_b, pts_b, ratio: float, same: bool):
    """Lowe-ratio matches a -> b as two point arrays. same: b is a itself, so the best hit (the point itself) is skipped."""
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    src, dst = [], []
    for m in matcher.knnMatch(des_a, des_b, k=3 if same else 2):
        m = m[1:] if same else m
        if len(m) == 2 and m[0].distance < ratio * m[1].distance:
            src.append(pts_a[m[0].queryIdx])
            dst.append(pts_b[m[0].trainIdx])
    return np.array(src, np.float32).reshape(-1, 2), np.array(dst, np.float32).reshape(-1, 2)


def _groups(src, dst, draw, min_matches: int, limit: int, flip=lambda p: p) -> list[tuple[np.ndarray, np.ndarray]]:
    """Clones as (points, their copies): matches that one similarity transform explains (greedy RANSAC).
    Self-matching sees every clone from both ends (a -> b and b -> a, the inverse transform): both directions are
    merged into one group and each pair is counted once. dst is where the transform is fitted, draw where the
    match lies in the image (they differ for the mirrored search, whose `flip` maps between the two frames)."""
    idx, groups = np.arange(len(src)), []
    while len(idx) >= min_matches and len(groups) < limit:
        m, inliers = cv2.estimateAffinePartial2D(src[idx], dst[idx], method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if m is None or inliers.sum() < min_matches:
            break
        forward = inliers.ravel().astype(bool)
        back = flip(dst[idx]) @ m[:, :2].T + m[:, 2] - flip(src[idx])
        reverse = ~forward & (np.hypot(*back.T) < 4.0)
        f, r = idx[forward], idx[reverse]
        pairs = np.unique(np.rint(np.hstack([np.vstack([src[f], draw[r]]), np.vstack([draw[f], src[r]])])), axis=0)
        groups.append((pairs[:, :2], pairs[:, 2:]))
        idx = idx[~(forward | reverse)]
    return groups


def keypoints(ev: Evidence, min_distance: int = 40, ratio: float = 0.55, min_matches: int = 12,
              mirrored: bool = False) -> Result:
    gray = cv2.cvtColor(ev.bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 2400 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02)
    kps, des = sift.detectAndCompute(gray, None)
    res = Result()
    canvas = (ev.bgr * 0.45).astype(np.uint8)
    if des is None or len(kps) < 10:
        res.views["Clone matches"] = canvas
        res.findings.append(Finding(INFO, "Too few keypoints", "Use the block-based detector for smooth images."))
        return res
    pts = np.array([k.pt for k in kps], np.float32) / scale
    src, dst = _matches(des, pts, des, pts, ratio, same=True)
    far = np.hypot(*(src - dst).T) > min_distance
    src, dst = src[far], dst[far]
    clusters = _groups(src, dst, dst, min_matches, len(_COLORS))

    flipped_groups = 0
    if mirrored:
        # SIFT is not mirror-invariant: match against the flipped image, fit the transform in the flipped frame
        # (where a mirrored copy is an ordinary similarity) and draw in the original frame
        kps2, des2 = sift.detectAndCompute(cv2.flip(gray, 1), None)
        if des2 is not None and len(kps2) >= 10:
            pts2 = np.array([k.pt for k in kps2], np.float32) / scale
            msrc, mflip = _matches(des, pts, des2, pts2, ratio, same=False)
            flip = lambda p: p * np.float32((-1, 1)) + np.float32(((gray.shape[1] - 1) / scale, 0))
            mdst = flip(mflip)
            far = np.hypot(*(msrc - mdst).T) > min_distance
            found = _groups(msrc[far], mflip[far], mdst[far], min_matches, len(_COLORS) - len(clusters), flip)
            clusters += found
            flipped_groups = len(found)

    # a group related by a plain shift can be checked: a copy shares its noise with the source, a look-alike does not
    alike = 0
    for a, b in list(clusters[:len(clusters) - flipped_groups]):
        m, _ = cv2.estimateAffinePartial2D(a.astype(np.float32), b.astype(np.float32))
        if m is None or abs(np.hypot(m[0, 0], m[1, 0]) - 1) > 0.02 or abs(np.degrees(np.arctan2(m[1, 0], m[0, 0]))) > 1:
            continue                               # scaled or rotated copy: interpolation changed the noise
        if same_noise_in_flat_parts(ev.luma, a, np.rint(np.median(b - a, axis=0))) < SAME_NOISE:
            clusters = [c for c in clusters if c[0] is not a]
            alike += 1

    for color, (a, b) in zip(_COLORS, clusters):
        for p, q in zip(a.astype(int), b.astype(int)):
            cv2.line(canvas, tuple(p), tuple(q), color, 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p), 4, color, -1)
            cv2.circle(canvas, tuple(q), 4, color, -1)
        for group in (a, b):
            cv2.polylines(canvas, [cv2.convexHull(group.astype(np.int32))], True, color, 2, cv2.LINE_AA)
    res.views["Clone matches"] = canvas
    best = max((len(a) for a, _ in clusters), default=0)
    res.table = [("Keypoints", str(len(kps))), ("Matched groups", str(len(clusters))), ("Largest group", f"{best} pairs")]
    if mirrored:
        res.table.append(("Mirrored groups", str(flipped_groups)))
    if alike:
        res.table.append(("Look-alike groups discarded", f"{alike}  (same shape, different noise: repeating content)"))
    if best >= min_matches:
        res.score = float(np.clip(0.55 + best / 60, 0, 1))
        res.findings.append(Finding(ALERT if best >= 20 else WARN, "Duplicated region detected",
            f"{len(clusters)} group(s) of matching keypoints related by one geometric transform; the largest has {best} "
            "pairs. Rule out genuinely repeating content (windows, tiles, printed patterns, text) before concluding."
            + (" Mirrored matches also arise from truly symmetric objects (faces, facades, vehicles seen head-on)."
               if flipped_groups else "")))
    else:
        res.score = 0.05
        res.findings.append(Finding(INFO, "No keypoint-based clones found"))
    return res


def blocks(ev: Evidence, block: int = 16, similarity: float = 1.5, min_detail: float = 2.0,
           min_cluster: int = 300, max_size: int = 1600) -> Result:
    scale = min(1.0, max_size / max(ev.height, ev.width))
    gray = ev.luma if scale == 1 else cv2.resize(ev.luma, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    b, cell = int(block), int(block) // 4
    # 16 features per block: means of its 4x4 grid of cells, read from one box-filtered image
    cells = cv2.blur(gray, (cell, cell), anchor=(0, 0), borderType=cv2.BORDER_REFLECT)
    h, w = gray.shape[0] - b, gray.shape[1] - b
    res = Result()
    if h < b or w < b:
        res.findings.append(Finding(INFO, "Image too small"))
        return res
    step = 1
    ys, xs = np.mgrid[0:h:step, 0:w:step]
    ys, xs = ys.ravel(), xs.ravel()
    feats = np.stack([cells[ys + i * cell, xs + j * cell] for i in range(4) for j in range(4)], axis=1)
    # detail = curvature of the 4x4 cell grid: zero for flat areas AND for plain gradients
    grid = feats.reshape(-1, 4, 4)
    detail = (np.abs(grid[:, :, :-2] - 2 * grid[:, :, 1:-1] + grid[:, :, 2:]).mean(axis=(1, 2))
              + np.abs(grid[:, :-2] - 2 * grid[:, 1:-1] + grid[:, 2:]).mean(axis=(1, 2)))
    ok = detail > min_detail
    ys, xs, feats = ys[ok], xs[ok], feats[ok]
    # sort on coarse quadrant means so that noise rarely separates true duplicates,
    # then verify candidates with the full 16-value descriptor
    quads = feats.reshape(-1, 2, 2, 2, 2).mean(axis=(2, 4)).reshape(-1, 4)
    order = np.lexsort(np.floor(quads / 8).astype(np.int32).T[::-1])
    ys, xs, feats = ys[order], xs[order], feats[order]

    pairs = []
    for k in range(1, 9):
        dist = np.abs(feats[k:] - feats[:-k]).mean(axis=1)
        sy, sx = ys[k:] - ys[:-k], xs[k:] - xs[:-k]
        near = (dist < similarity) & (np.hypot(sy, sx) > 2 * b)
        idx = np.nonzero(near)[0]
        pairs.append(np.stack([ys[idx], xs[idx], ys[idx + k], xs[idx + k]], 1))
    pairs = np.concatenate(pairs)
    canvas = (ev.bgr * 0.45).astype(np.uint8)
    found, alike, fine = 0, 0, _fine(gray)
    if len(pairs):
        shift = pairs[:, 2:] - pairs[:, :2]
        flip = (shift[:, 0] < 0) | ((shift[:, 0] == 0) & (shift[:, 1] < 0))
        shift[flip] *= -1
        shifts, inverse, counts = np.unique(shift, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.ravel()
        for ci, si in enumerate(np.argsort(counts)[::-1][:len(_COLORS)]):
            if counts[si] < min_cluster:
                break
            members = pairs[inverse == si]
            # a real clone is ONE compact area: matches on smooth folds are thin strips or scattered specks
            mask = np.zeros(gray.shape, np.uint8)
            mask[members[:, 0], members[:, 1]] = 1
            _, labels = cv2.connectedComponents(cv2.dilate(mask, np.ones((2 * b, 2 * b), np.uint8)))
            sizes = np.bincount(labels[members[:, 0], members[:, 1]])
            blob = members[labels[members[:, 0], members[:, 1]] == sizes.argmax()]
            spread = np.linalg.eigvalsh(np.cov(blob[:, :2].T.astype(np.float64)))
            if len(blob) < max(100, 0.4 * len(members)) or spread[0] < (0.75 * b) ** 2:   # narrower than a block
                continue
            if same_noise(fine, blob[:, :2], blob[:, 2:], b) < SAME_NOISE:
                alike += 1                         # similar blocks with independent noise: sky, water, a pattern
                continue
            found += 1
            for y0, x0, y1, x1 in (members / scale).astype(int):
                s = int(b / scale)
                cv2.rectangle(canvas, (x0, y0), (x0 + s, y0 + s), _COLORS[ci], -1)
                cv2.rectangle(canvas, (x1, y1), (x1 + s, y1 + s), _COLORS[ci], -1)
    res.views["Cloned blocks"] = cv2.addWeighted(canvas, 0.7, ev.bgr, 0.3, 0)
    res.table = [("Blocks compared", f"{len(feats):,}"), ("Shift clusters", str(found)),
                 ("Look-alike clusters discarded", f"{alike}  (similar blocks, different noise)")]
    if found:
        res.score = 0.75
        res.findings.append(Finding(WARN, "Identical blocks at a constant offset",
            f"{found} offset(s) link many near-identical blocks: a region was copied and moved without scaling. "
            "Uniform gradients and repeating patterns can produce the same picture; raise Minimal detail to test."))
    else:
        res.score = 0.05
        res.findings.append(Finding(INFO, "No block-based clones found"))
    return res
