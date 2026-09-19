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

_COLORS = [(80, 80, 255), (80, 255, 120), (255, 170, 60), (255, 90, 255), (60, 230, 255), (200, 255, 80)]


def keypoints(ev: Evidence, min_distance: int = 40, ratio: float = 0.55, min_matches: int = 8) -> Result:
    gray = cv2.cvtColor(ev.bgr, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 2400 / max(gray.shape))
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kps, des = cv2.SIFT_create(nfeatures=12000, contrastThreshold=0.02).detectAndCompute(gray, None)
    res = Result()
    canvas = (ev.bgr * 0.45).astype(np.uint8)
    if des is None or len(kps) < 10:
        res.views["Clone matches"] = canvas
        res.findings.append(Finding(INFO, "Too few keypoints", "Use the block-based detector for smooth images."))
        return res
    pts = np.array([k.pt for k in kps], np.float32) / scale
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    src, dst = [], []
    for m in matcher.knnMatch(des, des, k=3):
        if len(m) < 3:
            continue
        _, first, second = m                      # m[0] is the keypoint itself
        if first.distance < ratio * second.distance:
            a, b = pts[first.queryIdx], pts[first.trainIdx]
            if np.hypot(*(a - b)) > min_distance:
                src.append(a)
                dst.append(b)
    src, dst = np.array(src, np.float32), np.array(dst, np.float32)

    clusters = []
    while len(src) >= min_matches and len(clusters) < len(_COLORS):
        _, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if inliers is None or inliers.sum() < min_matches:
            break
        keep = inliers.ravel().astype(bool)
        clusters.append((src[keep], dst[keep]))
        src, dst = src[~keep], dst[~keep]

    for color, (a, b) in zip(_COLORS, clusters):
        for p, q in zip(a.astype(int), b.astype(int)):
            cv2.line(canvas, tuple(p), tuple(q), color, 1, cv2.LINE_AA)
            cv2.circle(canvas, tuple(p), 4, color, -1)
            cv2.circle(canvas, tuple(q), 4, color, -1)
        for group in (a, b):
            cv2.polylines(canvas, [cv2.convexHull(group.astype(np.int32))], True, color, 2, cv2.LINE_AA)
    res.views["Clone matches"] = canvas
    # symmetric matching reports each pair twice
    best = max((len(a) for a, _ in clusters), default=0) // 2
    res.table = [("Keypoints", str(len(kps))), ("Matched groups", str(len(clusters))), ("Largest group", f"{best} pairs")]
    if best >= min_matches:
        res.score = float(np.clip(0.55 + best / 60, 0, 1))
        res.findings.append(Finding(ALERT if best >= 20 else WARN, "Duplicated region detected",
            f"{len(clusters)} group(s) of matching keypoints related by one geometric transform; the largest has {best} "
            "pairs. Rule out genuinely repeating content (windows, tiles, printed patterns, text) before concluding."))
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
    found = 0
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
            found += 1
            for y0, x0, y1, x1 in (members / scale).astype(int):
                s = int(b / scale)
                cv2.rectangle(canvas, (x0, y0), (x0 + s, y0 + s), _COLORS[ci], -1)
                cv2.rectangle(canvas, (x1, y1), (x1 + s, y1 + s), _COLORS[ci], -1)
    res.views["Cloned blocks"] = cv2.addWeighted(canvas, 0.7, ev.bgr, 0.3, 0)
    res.table = [("Blocks compared", f"{len(feats):,}"), ("Shift clusters", str(found))]
    if found:
        res.score = 0.75
        res.findings.append(Finding(WARN, "Identical blocks at a constant offset",
            f"{found} offset(s) link many near-identical blocks: a region was copied and moved without scaling. "
            "Uniform gradients and repeating patterns can produce the same picture; raise Minimal detail to test."))
    else:
        res.score = 0.05
        res.findings.append(Finding(INFO, "No block-based clones found"))
    return res
