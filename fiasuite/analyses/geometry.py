"""Examiner-drawn geometry checks (Farid; Kee, O'Brien & Farid 2013; O'Brien & Farid 2012).

Physical constraints survive recompression, resizing and screenshots, and image generators do not enforce them:

  shadows       a line from a point on a cast shadow through the point on the object that casts it passes through
                the light source; all such lines in one picture meet in ONE point (the image of the light).
  perspective   lines that are parallel in the scene (building edges, road markings, tiles) meet in ONE vanishing
                point; the vanishing points of all horizontal directions lie on one straight horizon.
  reflections   lines joining a point and its mirror image in a flat mirror or still water are parallel in the
                scene, so they too meet in ONE vanishing point.

The examiner draws the lines; this module intersects them and measures how far each line misses the common point
of the others. It can only show inconsistency, never prove authenticity, and it is as accurate as the annotation.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..core import INFO, NOTE, WARN, Evidence, Finding, Result

SETS = {"A": (70, 70, 255), "B": (90, 220, 110), "C": (255, 170, 60), "D": (60, 230, 255)}   # BGR
TOLERANCE_DEG = 3.0                                # a careful annotation is good to a degree or two


def intersection(lines: np.ndarray) -> np.ndarray | None:
    """Least-squares common point of lines given as rows (x1, y1, x2, y2); None when they are parallel."""
    p, d = lines[:, :2], lines[:, 2:] - lines[:, :2]
    n = np.stack([-d[:, 1], d[:, 0]], axis=1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    a = n.T @ n
    if np.linalg.cond(a) > 1e7:
        return None
    return np.linalg.solve(a, n.T @ np.einsum("ij,ij->i", n, p))


def miss_angle(line: np.ndarray, point: np.ndarray | None, others: np.ndarray) -> float:
    """Degrees by which a line misses `point` (seen from the line's midpoint); for parallel bundles (point None)
    the angle to the mean direction of the other lines."""
    d = line[2:] - line[:2]
    if point is None:
        od = others[:, 2:] - others[:, :2]
        od = od * np.sign(od @ od[0])[:, None]
        target = od.sum(axis=0)
    else:
        target = point - (line[:2] + line[2:]) / 2
    cos = abs(float(d @ target)) / max(float(np.linalg.norm(d) * np.linalg.norm(target)), 1e-9)
    return float(np.degrees(np.arccos(min(cos, 1.0))))


def measure(lines: list[tuple[float, float, float, float]]) -> dict:
    """Common point of a set and how far each line misses it (degrees).
    Three lines: each is compared with the intersection of the other two. Four or more: the common point is taken
    from the most self-consistent subset that leaves one line out, so a single stray line cannot drag the point
    towards itself and hide among the others."""
    arr = np.asarray(lines, np.float64).reshape(-1, 4)
    out = {"point": None, "miss": [], "parallel": False}
    if len(arr) < 2:
        return out
    point = intersection(arr)
    if len(arr) == 3:
        out["miss"] = [miss_angle(arr[i], intersection(np.delete(arr, i, 0)), np.delete(arr, i, 0)) for i in range(3)]
    elif len(arr) > 3:
        subsets = [np.delete(arr, i, axis=0) for i in range(len(arr))]
        spread = [max(miss_angle(line, intersection(rest), rest) for line in rest) for rest in subsets]
        rest = subsets[int(np.argmin(spread))]
        point = intersection(rest)
        out["miss"] = [miss_angle(line, point, rest) for line in arr]
    out["point"], out["parallel"] = (None if point is None else (float(point[0]), float(point[1]))), point is None
    return out


def render(bgr: np.ndarray, sets: dict[str, list], limit: float = 1.5) -> np.ndarray:
    """The image on a canvas large enough to show common points up to `limit` image sizes away."""
    h, w = bgr.shape[:2]
    points = [m["point"] for m in (measure(v) for v in sets.values()) if m["point"] is not None]
    xs = [0, w] + [min(max(p[0], -limit * w), (1 + limit) * w) for p in points]
    ys = [0, h] + [min(max(p[1], -limit * h), (1 + limit) * h) for p in points]
    x0, y0, x1, y1 = int(min(xs)) - 20, int(min(ys)) - 20, int(max(xs)) + 20, int(max(ys)) + 20
    scale = min(1.0, 4000 / max(x1 - x0, y1 - y0))
    canvas = np.full((int((y1 - y0) * scale), int((x1 - x0) * scale), 3), 28, np.uint8)
    small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    oy, ox = int(-y0 * scale), int(-x0 * scale)
    canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    to = lambda x, y: (int((x - x0) * scale), int((y - y0) * scale))
    thick = max(1, int(round(max(canvas.shape[:2]) / 900)))
    for name, lines in sets.items():
        color = SETS[name]
        for x1_, y1_, x2_, y2_ in lines:
            d = np.array([x2_ - x1_, y2_ - y1_], np.float64)
            d = d / max(np.linalg.norm(d), 1e-9) * 10 * max(w, h)
            cv2.line(canvas, to(x1_ - d[0], y1_ - d[1]), to(x1_ + d[0], y1_ + d[1]), color, thick, cv2.LINE_AA)
            cv2.circle(canvas, to(x1_, y1_), 3 * thick, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, to(x2_, y2_), 3 * thick, color, -1, cv2.LINE_AA)
        m = measure(lines)
        if m["point"] is not None and x0 <= m["point"][0] <= x1 and y0 <= m["point"][1] <= y1:
            cv2.drawMarker(canvas, to(*m["point"]), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 14 * thick, 2 * thick, cv2.LINE_AA)
    return canvas


def analyse(ev: Evidence, sets: dict[str, list] | None = None) -> Result:
    res = Result()
    sets = {k: v for k, v in (sets or {}).items() if v}
    if not sets:
        res.views["Image"] = ev.bgr
        res.findings.append(Finding(INFO, "Draw lines on the image",
            "Drag with the left mouse button to draw a line; it is extended across the picture. Put lines that must "
            "meet in one point into the same set: shadow-to-object lines (they meet at the light source), edges that "
            "are parallel in the scene (one vanishing point per direction), or object-to-reflection lines. Use a "
            "second set for a second bundle. Hold Ctrl to pan, wheel to zoom."))
        return res
    res.views["Lines and common points"] = render(ev.bgr, sets)
    horizon = []
    for name, lines in sets.items():
        m = measure(lines)
        label = f"Set {name} ({len(lines)} line{'s' if len(lines) != 1 else ''})"
        if len(lines) < 2:
            res.table.append((label, "one more line needed for an intersection"))
            continue
        where = "parallel in the image: common point at infinity" if m["parallel"] else \
            f"common point at x = {m['point'][0]:.0f}, y = {m['point'][1]:.0f}" + \
            ("" if 0 <= m["point"][0] < ev.width and 0 <= m["point"][1] < ev.height else "  (outside the frame)")
        res.table.append((label, where))
        if m["point"] is not None:
            horizon.append(m["point"])
        for i, angle in enumerate(m["miss"]):
            res.table.append((f"  line {i + 1}", f"misses the common point of the others by {angle:.1f} degrees"))
        if m["miss"]:
            worst = int(np.argmax(m["miss"]))
            if m["miss"][worst] > TOLERANCE_DEG:
                # with three lines any one of them may be the stray one; from four on it can be named
                title = f"Set {name}: line {worst + 1} does not pass through the common point" if len(lines) > 3                     else f"Set {name}: the three lines do not meet in one point"
                res.findings.append(Finding(WARN if m["miss"][worst] > 2 * TOLERANCE_DEG else NOTE, title,
                    f"It misses the point defined by the other lines by {m['miss'][worst]:.1f} degrees. If the line "
                    "is drawn accurately, that object or edge obeys a different light source or perspective than "
                    "the rest of the picture. Check the annotation first: blurred shadow tips and short lines are "
                    "imprecise."))
            else:
                res.findings.append(Finding(INFO, f"Set {name}: all lines agree within {m['miss'][worst]:.1f} degrees",
                                            "Consistent with a single light source / vanishing point. Consistency does "
                                            "not prove authenticity."))
        else:
            res.findings.append(Finding(INFO, f"Set {name}: two lines always meet", "Add a third line to test consistency."))
    if len(horizon) >= 3:                           # vanishing points of horizontal directions lie on one line
        pts = np.asarray(horizon)
        centred = pts - pts.mean(axis=0)
        _, _, vt = np.linalg.svd(centred)
        off = np.abs(centred @ vt[1])
        res.table.append(("Common points on one line (horizon test)",
                          f"largest distance from the fitted line: {off.max():.0f} px over a span of "
                          f"{np.ptp(centred @ vt[0]):.0f} px"))
    return res
