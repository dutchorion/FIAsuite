"""Local neural detectors, run with ONNX Runtime on the CPU (DirectML GPU is opt-in, see use_gpu).

Models are described in models/models.json, so detectors can be added or
replaced without touching the code:

  {"id": "commfor", "name": "...", "file": "commfor_384.onnx", "kind": "classifier",
   "size": 384, "mean": [...], "std": [...], "policy": "crop" | "squash",
   "output": "sigmoid" | "softmax", "fake_index": 1, "license": "...", "source": "..."}

policy "five_crop"  native-resolution crops at the centre and the four corners, logits averaged (B-Free);
                the map is those four corners. Few, large crops: the network is slow.
policy "crop"   the network sees native-resolution crops (pixel-level traces kept);
                the image score is the mean over a grid of crops.
policy "squash" the network sees the whole image resized to its input size.
Both policies also produce a per-crop map that localises generated regions.
policy "whole"  the whole image resized to the input size, one pass and no map: for networks trained that way
                that are too slow to run once per region (OmniAID).
"""
from __future__ import annotations

import json
import threading

import cv2
import numpy as np

from ..core import ALERT, INFO, NOTE, WARN, Evidence, Finding, Result, app_dir, heatmap

_sessions: dict[str, object] = {}
_lock = threading.Lock()
# Neural models run strictly one at a time. Several at once starve the machine, and on an integrated GPU
# that also drives the display they trip the Windows GPU watchdog (driver reset, frozen desktop).
_inference = threading.Lock()


def use_gpu() -> bool:
    """GPU (DirectML) inference is opt-in: config.json {"use_gpu": true}, set under Settings."""
    try:
        return bool(json.loads((app_dir() / "config.json").read_text(encoding="utf-8")).get("use_gpu", False))
    except (OSError, json.JSONDecodeError):
        return False


def _run(sess, feed: dict):
    with _inference:
        return sess.run(None, feed)


def models_dir():
    return app_dir() / "models"


def load_registry() -> list[dict]:
    path = models_dir() / "models.json"
    if not path.exists():
        return []
    entries = json.loads(path.read_text(encoding="utf-8"))
    return [e for e in entries if (models_dir() / e["file"]).exists()]


gpu_problems: dict[str, str] = {}              # model id -> why it was taken off the GPU in this session


def _cpu_session(entry: dict):
    import os

    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)    # leave half the cores to the desktop
    basic = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if entry.get("cpu_optimization") == "basic":
        opts.graph_optimization_level = basic
    path = str(models_dir() / entry["file"])
    try:
        return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    except Exception:                              # ORT's extended fusions fail on some fp16 graphs on the CPU
        opts.graph_optimization_level = basic
        return ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])


def gpu_mismatch(gpu_sess, cpu_sess, entry: dict) -> str | None:
    """One test inference on both devices. A GPU driver in a bad state has returned plausible-looking but wrong
    maps, many times slower than normal, without raising any error: a forensic tool must not trust it blindly."""
    import time
    meta = cpu_sess.get_inputs()[0]
    rng = np.random.default_rng(0)
    size = int(entry["size"])
    x = rng.random((1, 3, size, size), dtype=np.float32)
    x = x.astype(np.float16) if "float16" in meta.type else x
    t = time.perf_counter()
    expected = _run(cpu_sess, {meta.name: x})
    t_cpu = time.perf_counter() - t
    t = time.perf_counter()
    got = _run(gpu_sess, {meta.name: x})
    t_gpu = time.perf_counter() - t
    for e, g in zip(expected, got):
        e, g = np.asarray(e, np.float64), np.asarray(g, np.float64)
        if not np.isfinite(g).all():
            return "the GPU returned non-finite values"
        worst = float(np.abs(e - g).max() / max(np.abs(e).max(), 1e-6))
        if worst > 0.02:
            return f"GPU and CPU results differ by {worst:.0%}"
    if t_gpu > 5 * t_cpu + 5:
        return f"the GPU needed {t_gpu:.0f} s for what the CPU does in {t_cpu:.0f} s"
    return None


def _session(entry: dict):
    gpu = use_gpu()
    key = f"{entry['id']}|{'gpu' if gpu else 'cpu'}"
    with _lock:
        if key not in _sessions:
            sess = _cpu_session(entry)
            if gpu:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                try:
                    candidate = ort.InferenceSession(str(models_dir() / entry["file"]), opts,
                                                     providers=["DmlExecutionProvider", "CPUExecutionProvider"])
                    problem = gpu_mismatch(candidate, sess, entry)
                except Exception as exc:
                    candidate, problem = None, f"the GPU could not run the model ({str(exc)[:120]})"
                if problem is None:
                    sess = candidate
                    gpu_problems.pop(entry["id"], None)
                else:
                    gpu_problems[entry["id"]] = problem
            _sessions[key] = sess
        return _sessions[key]


def _gpu_note(res: Result, entry: dict) -> None:
    if entry["id"] in gpu_problems:
        res.findings.append(Finding(NOTE, f"{entry['name']} ran on the CPU instead of the GPU",
            f"Self-check failed: {gpu_problems[entry['id']]}. This points to a graphics-driver problem; restarting "
            "the PC has cured it before. The result shown here was computed on the CPU and is not affected."))


def _prepare(bgr: np.ndarray, entry: dict) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    rgb = (rgb - np.array(entry["mean"], np.float32)) / np.array(entry["std"], np.float32)
    return rgb.transpose(2, 0, 1)[None]


def _probability(sess, batch: np.ndarray, entry: dict) -> float:
    out = np.asarray(_run(sess, {sess.get_inputs()[0].name: batch})[0], np.float64).reshape(-1)
    if entry["output"] == "sigmoid":
        return float(1 / (1 + np.exp(-out[0])))
    e = np.exp(out - out.max())
    return float(e[entry.get("fake_index", 1)] / e.sum())


def _tile_origins(length: int, size: int, count: int) -> list[int]:
    if length <= size:
        return [0]
    return [int(v) for v in np.linspace(0, length - size, max(count, 2))]


def classify(ev: Evidence, entry: dict, max_tiles: int = 24) -> tuple[float, np.ndarray, float]:
    """(image probability, per-tile probability grid, max tile probability)."""
    sess, size = _session(entry), int(entry["size"])
    policy = entry.get("policy", "crop")
    if policy == "whole":                          # PIL's filtered bilinear resize, as torchvision's Resize
        from PIL import Image
        rgb = Image.fromarray(np.ascontiguousarray(ev.bgr[:, :, ::-1])).resize((size, size), Image.BILINEAR)
        score = _probability(sess, _prepare(np.asarray(rgb)[:, :, ::-1], entry), entry)
        return score, np.array([[score]], np.float32), score
    img = ev.bgr
    if min(ev.height, ev.width) < size:            # never upscale more than needed, keep aspect
        s = size / min(ev.height, ev.width)
        img = cv2.resize(img, (max(size, round(ev.width * s)), max(size, round(ev.height * s))), interpolation=cv2.INTER_CUBIC)
    h, w = img.shape[:2]
    if policy == "five_crop":
        ys, xs = [0, h - size], [0, w - size]
        corners = np.array([[_probability(sess, _prepare(img[y:y + size, x:x + size], entry), entry) for x in xs] for y in ys],
                           np.float32)
        centre = _probability(sess, _prepare(img[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size],
                                             entry), entry)
        p = np.clip(np.append(corners.ravel(), centre), 1e-6, 1 - 1e-6)
        return float(1 / (1 + np.exp(-np.log(p / (1 - p)).mean()))), corners, float(p.max())
    # networks trained on resized whole images get larger windows, scaled down to their input size
    window = size if policy == "crop" else max(size, min(h, w) // 2)
    aspect = w / h
    ny = max(1, min(int(round(np.sqrt(max_tiles / aspect))), -(-2 * h // window)))
    nx = max(1, min(int(round(max_tiles / ny)), -(-2 * w // window)))
    ys, xs = _tile_origins(h, window, ny), _tile_origins(w, window, nx)
    grid = np.zeros((len(ys), len(xs)), np.float32)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            tile = img[y:y + window, x:x + window]
            if window != size:
                tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_AREA)
            grid[i, j] = _probability(sess, _prepare(tile, entry), entry)
    if policy == "squash":
        whole = cv2.resize(ev.bgr, (size, size), interpolation=cv2.INTER_AREA)
        score = _probability(sess, _prepare(whole, entry), entry)
    elif policy == "resize_crop":                  # shorter side to `resize`, then centre crop (torchvision eval transform)
        s = entry["resize"] / min(ev.height, ev.width)
        whole = cv2.resize(ev.bgr, (max(size, round(ev.width * s)), max(size, round(ev.height * s))),
                           interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
        y, x = (whole.shape[0] - size) // 2, (whole.shape[1] - size) // 2
        score = _probability(sess, _prepare(whole[y:y + size, x:x + size], entry), entry)
    else:
        logits = np.log(np.clip(grid, 1e-6, 1 - 1e-6) / np.clip(1 - grid, 1e-6, 1))
        score = float(1 / (1 + np.exp(-logits.mean())))
    return score, grid, float(grid.max())


def detect(ev: Evidence, max_tiles: int = 24) -> Result:
    res = Result()
    registry = [e for e in load_registry() if e.get("kind", "classifier") == "classifier"]
    if not registry:
        res.findings.append(Finding(NOTE, "No local AI detectors installed",
            f"Place ONNX models and models.json in {models_dir()} (see README)."))
        return res
    scores, weights = [], []
    for entry in registry:
        try:
            score, grid, peak = classify(ev, entry, int(max_tiles))
        except Exception as exc:
            res.findings.append(Finding(NOTE, f"{entry['name']} failed", str(exc)[:300]))
            continue
        _gpu_note(res, entry)
        scores.append(score)
        weights.append(float(entry.get("weight", 1.0)))
        res.metrics[entry["id"]] = score
        regions = f"highest region {peak:.1%}, {grid.size} regions" if grid.size > 1 else "whole image only"
        res.table.append((entry["name"], f"{score:.1%} generated   ({regions})"
                          + ("" if weights[-1] else "   - shown for information, not counted in the ensemble")))
        if grid.size > 1:                          # a one-cell map is a flat colour, not a localisation
            res.views[entry["name"]] = heatmap(grid, (ev.height, ev.width), 0, 1, cv2.COLORMAP_TURBO)
        if peak > 0.9 and score < 0.5 and grid.size >= 6:
            res.findings.append(Finding(WARN, f"{entry['name']}: a region looks generated",
                f"The image as a whole scores {score:.0%}, but one region scores {peak:.0%}: possible local AI edit "
                "(inpainting, generative fill, face swap). See the map."))
    counted = [i for i, w in enumerate(weights) if w > 0]       # weight 0: a detector that is reported but not trusted
    if not counted:
        return res
    scores, weights = [scores[i] for i in counted], [weights[i] for i in counted]
    # averaged as log-odds: on 245 images from 16 current generators and 180 photographs this found 68% of the
    # generated ones with no photograph above 50%; averaging the probabilities themselves found 42%
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    res.score = float(1 / (1 + np.exp(-np.average(np.log(p / (1 - p)), weights=weights))))
    res.table.append(("Ensemble (mean of log-odds)", f"{res.score:.1%}"))
    agree = all(s > 0.5 for s in scores) or all(s <= 0.5 for s in scores)
    if res.score > 0.85 and agree:
        res.findings.append(Finding(ALERT, "Detectors agree: AI-generated", f"Ensemble probability {res.score:.0%}."))
    elif res.score > 0.5:
        res.findings.append(Finding(WARN, "Detectors lean towards AI-generated",
            f"Ensemble probability {res.score:.0%}" + ("" if agree else "; the detectors disagree") + "."))
    else:
        res.findings.append(Finding(INFO, "Detectors do not flag the image",
            f"Ensemble probability {res.score:.0%}."))
    res.findings.append(Finding(INFO, "How far to trust this",
        "Open detectors reach roughly 70-80% accuracy on independent benchmarks and miss most output of the newest "
        "commercial generators (Flux, Midjourney v7, Firefly, Imagen) - a low score does NOT establish authenticity. "
        "Heavy JPEG compression, resizing and screenshots reduce accuracy further. Treat the result as one piece of "
        "evidence next to metadata, C2PA and the signal analyses."))
    return res


# ------------------------------------------------------------------ forgery localisation (TruFor)

def localize(ev: Evidence, max_side: int = 1600) -> Result:
    res = Result()
    entry = next((e for e in load_registry() if e.get("kind") == "localizer"), None)
    if entry is None:
        res.findings.append(Finding(NOTE, "Forgery-localisation model not installed",
            f"Place the TruFor ONNX export and its models.json entry in {models_dir()}."))
        return res
    sess = _session(entry)
    _gpu_note(res, entry)
    img = ev.bgr
    scale = min(1.0, max_side / max(ev.height, ev.width))
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    size = entry.get("size")                       # fixed-size export: run on overlapping windows
    h, w = img.shape[:2]
    if size:
        pad = cv2.copyMakeBorder(img, 0, max(0, size - h), 0, max(0, size - w), cv2.BORDER_REFLECT)
        ph, pw = pad.shape[:2]
        tamper, conf, weight = (np.zeros((ph, pw), np.float32) for _ in range(3))
        scores = []
        for y in _tile_origins(ph, size, -(-ph // (size * 3 // 4))):
            for x in _tile_origins(pw, size, -(-pw // (size * 3 // 4))):
                t, c, s = _run_trufor(sess, pad[y:y + size, x:x + size])
                tamper[y:y + size, x:x + size] += t
                conf[y:y + size, x:x + size] += c
                weight[y:y + size, x:x + size] += 1
                scores.append(s)
        tamper, conf, score = (tamper / weight)[:h, :w], (conf / weight)[:h, :w], float(max(scores))
    else:
        tamper, conf, score = _run_trufor(sess, img)
    res.views["Tampering probability"] = heatmap(tamper, (ev.height, ev.width), 0, 1, cv2.COLORMAP_TURBO)
    res.views["Reliability of the map"] = heatmap(conf, (ev.height, ev.width), 0, 1, cv2.COLORMAP_VIRIDIS)
    area = float(((tamper > 0.5) & (conf > 0.5)).mean())
    res.score = score
    res.table = [("Integrity score", f"{score:.2f}  (0 = pristine, 1 = manipulated)"),
                 ("Confidently flagged area", f"{area:.1%}"),
                 ("Processed at", f"{w} x {h}" + ("" if scale == 1 else "  (downscaled: weakens noise traces)"))]
    if score > 0.7:
        res.findings.append(Finding(ALERT if area > 0.005 else WARN, "TruFor: image is likely manipulated",
            f"Integrity score {score:.2f}; {area:.1%} of the image is flagged where the network is confident. "
            "TruFor compares the camera-noise fingerprint (Noiseprint++) across the image and detects splices, "
            "copy-move, inpainting and local AI edits."))
    elif score > 0.4:
        res.findings.append(Finding(NOTE, "TruFor: inconclusive", f"Integrity score {score:.2f}."))
    else:
        res.findings.append(Finding(INFO, "TruFor: no manipulation found", f"Integrity score {score:.2f}."))
    res.findings.append(Finding(INFO, "Reading the maps",
        "Trust red areas of the tampering map only where the reliability map is bright. Fully synthetic images often "
        "score low here, because they are uniformly synthetic: use the AI-generation detectors for that question."))
    return res


def _run_trufor(sess, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32).transpose(2, 0, 1)[None] / 256.0
    out = dict(zip([o.name for o in sess.get_outputs()], _run(sess, {sess.get_inputs()[0].name: rgb})))
    logits = out["pred"][0]
    e = np.exp(logits - logits.max(axis=0, keepdims=True))
    tamper = e[1] / e.sum(axis=0)
    conf = 1 / (1 + np.exp(-out["conf"][0, 0]))
    score = float(1 / (1 + np.exp(-np.ravel(out["det"])[0])))
    return tamper.astype(np.float32), conf.astype(np.float32), score
