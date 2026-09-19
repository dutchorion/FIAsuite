"""File-level provenance: ExifTool metadata with consistency rules, embedded
thumbnail comparison, C2PA Content Credentials, hashes and string extraction."""
from __future__ import annotations

import io
import json
import re
import subprocess
from datetime import datetime

import cv2
import numpy as np

from ..core import ALERT, INFO, NOTE, WARN, Evidence, Finding, Result, app_dir

AI_GENERATORS = re.compile(
    r"\b(midjourney|dall[\-·. ]?e|stable[\- ]?diffusion|sdxl|comfyui|automatic1111|invokeai|novelai|"
    r"adobe firefly|firefly|made with google ai|google ai|imagen ?\d|gemini|openai|chatgpt|gpt-image|"
    r"gpt-4o|sora|grok|black forest labs|flux\.?1|flux[\- ](dev|pro|schnell)|ideogram|leonardo\.?ai|"
    r"playground ai|runway|bing image creator|image creator|copilot designer|dreamstudio|nightcafe|"
    r"craiyon|fooocus|recraft|krea|seedream|qwen-image|hunyuan|ai[\- ]generated|generative fill|"
    r"trainedalgorithmicmedia|trained algorithmic media)\b", re.I)

EDITORS = re.compile(
    r"\b(photoshop|gimp|affinity|pixelmator|snapseed|picsart|facetune|canva|paint\.net|luminar|"
    r"acdsee|photoscape|polarr|remini|faceapp|meitu|topaz|pixlr|fotor|inpaint|paintshop|corel|"
    r"photodirector|lensa|airbrush|beautyplus|retouch)\b", re.I)

RAW_DEVELOPERS = re.compile(r"\b(lightroom|camera raw|capture one|darktable|rawtherapee|dxo|on1|"
                            r"digital photo professional|nx studio|imaging edge)\b", re.I)

MAKERNOTE_BRANDS = ("canon", "nikon", "sony", "fujifilm", "olympus", "om digital", "panasonic",
                    "pentax", "ricoh", "apple", "samsung", "leica", "sigma", "dji", "gopro")

AI_DIMS = {512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472,
           1536, 1792, 2048}

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def exiftool_path():
    return app_dir() / "vendor" / "exiftool" / "exiftool.exe"


def _exiftool(args: list[str], binary: bool = False):
    """Run the bundled ExifTool. Arguments go through a UTF-8 argfile on stdin so
    that non-ASCII file names survive on Windows."""
    argfile = "\n".join(["-charset", "filename=utf8", *args]).encode("utf-8")
    proc = subprocess.run([str(exiftool_path()), "-@", "-"], input=argfile, capture_output=True,
                          creationflags=_NO_WINDOW, timeout=120)
    return proc.stdout if binary else proc.stdout.decode("utf-8", "replace")


def read_metadata(ev: Evidence) -> dict:
    if "meta" not in ev.cache and not exiftool_path().exists():
        ev.cache["meta"] = {}                      # fresh clone without vendor/exiftool: no metadata, no crash
    if "meta" not in ev.cache:
        out = _exiftool(["-j", "-G1", "-a", "-s", "-api", "RequestAll=2", "-api", "LargeFileSupport=1",
                         "-All", "-JPEGDigest", "-JPEGQualityEstimate", str(ev.path)])
        try:
            ev.cache["meta"] = json.loads(out)[0]
        except (json.JSONDecodeError, IndexError):
            ev.cache["meta"] = {}
    return ev.cache["meta"]


def tag(meta: dict, name: str, group: str | None = None):
    """First value whose tag name (and optionally group prefix) matches."""
    name = name.lower()
    for key, value in meta.items():
        grp, _, tname = key.rpartition(":")
        if tname.lower() == name and (group is None or grp.lower().startswith(group.lower())):
            return value
    return None


def _parse_date(value) -> datetime | None:
    m = re.match(r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})", str(value or ""))
    try:
        return datetime(*map(int, m.groups())) if m else None
    except ValueError:
        return None


def run_metadata(ev: Evidence) -> Result:
    res = Result()
    if not exiftool_path().exists():
        res.findings.append(Finding(WARN, "ExifTool not found", f"Expected at {exiftool_path()}"))
        return res
    meta = read_metadata(ev)
    res.table = [(k, _short(v)) for k, v in meta.items() if k != "SourceFile"]
    f = res.findings

    camera_groups = ("IFD0", "ExifIFD", "MakerNotes", "GPS")
    has_exif = any(k.split(":")[0] in camera_groups or k.startswith("MakerNotes") for k in meta)
    make, model = tag(meta, "Make"), tag(meta, "Model")

    # --- declared or leaked AI generation -------------------------------------------------
    dst = str(tag(meta, "DigitalSourceType") or "")
    if re.search(r"compositeWithTrainedAlgorithmicMedia|composite with trained", dst, re.I):
        f.append(Finding(ALERT, "Metadata declares AI-edited content",
                         f"IPTC DigitalSourceType = {dst} (parts of the image were generated, e.g. generative fill)."))
    elif re.search(r"trainedAlgorithmicMedia|trained algorithmic", dst, re.I):
        f.append(Finding(ALERT, "Metadata declares the image as AI-generated", f"IPTC DigitalSourceType = {dst}"))
    for name in ("Parameters", "Prompt", "Workflow", "Dream", "sd-metadata", "invokeai_metadata"):
        value = tag(meta, name)
        if value and re.search(r"steps|sampler|cfg|seed|class_type|KSampler|prompt", str(value), re.I):
            f.append(Finding(ALERT, f"Generator parameters embedded ({name})",
                             "Text chunk written by Stable Diffusion front-ends (A1111, ComfyUI, InvokeAI): "
                             + _short(value, 400)))
    ai_hits = []
    for key, value in meta.items():
        if key.split(":")[0] in ("System", "File", "ExifTool", "ICC_Profile", "ICC-header"):
            continue
        m = AI_GENERATORS.search(str(value))
        if m:
            ai_hits.append(f"{key} = {_short(value, 120)}")
    if ai_hits and not any(x.severity == ALERT for x in f):
        f.append(Finding(ALERT if len(ai_hits) > 1 else WARN, "AI generator named in metadata", "\n".join(ai_hits[:8])))

    # --- software trail ------------------------------------------------------------------
    software = [str(v) for k, v in meta.items()
                if k.rpartition(":")[2] in ("Software", "CreatorTool", "HistorySoftwareAgent", "ProcessingSoftware",
                                            "Producer", "ClaimGenerator", "Claim_generator")]
    soft = "; ".join(dict.fromkeys(_flatten(software)))
    if EDITORS.search(soft):
        f.append(Finding(WARN, "Image editor in software trail",
                         f"{soft}\nThe file was saved by editing software. This shows opportunity, not proof, of manipulation."))
    elif RAW_DEVELOPERS.search(soft):
        f.append(Finding(NOTE, "RAW developer in software trail",
                         f"{soft}\nNormal photographic workflow, but local adjustments and healing are possible."))
    elif soft:
        f.append(Finding(INFO, "Software", soft))

    history = tag(meta, "HistoryAction")
    if history:
        actions = ", ".join(_flatten([history]))
        f.append(Finding(NOTE, "XMP edit history present", f"Actions: {actions[:300]}"))
    ancestors = tag(meta, "DocumentAncestors")
    if ancestors:
        n = len(ancestors) if isinstance(ancestors, list) else 1
        f.append(Finding(WARN, f"{n} document ancestor(s) recorded",
                         "Photoshop lists here the source documents whose content was pasted into this file: "
                         "a direct trace of compositing."))

    # --- dates ---------------------------------------------------------------------------
    original = _parse_date(tag(meta, "DateTimeOriginal"))
    modified = _parse_date(tag(meta, "ModifyDate", "IFD0")) or _parse_date(tag(meta, "ModifyDate", "XMP"))
    if original and modified and abs((modified - original).total_seconds()) > 60:
        f.append(Finding(NOTE, "Modified after capture",
                         f"DateTimeOriginal {original}  vs  ModifyDate {modified}"))
    digitized = _parse_date(tag(meta, "CreateDate", "ExifIFD"))
    if original and digitized and abs((digitized - original).total_seconds()) > 60:
        f.append(Finding(NOTE, "Capture timestamps disagree", f"DateTimeOriginal {original}  vs  CreateDate {digitized}"))

    # --- camera consistency --------------------------------------------------------------
    if not has_exif:
        f.append(Finding(NOTE, "No camera metadata",
                         "EXIF is absent: stripped by a platform (social media and messengers remove it), "
                         "removed deliberately, or it never existed (screenshot, render, AI generator)."))
        if ev.width in AI_DIMS and ev.height in AI_DIMS:
            f.append(Finding(NOTE, f"Generator-typical dimensions ({ev.width} x {ev.height})",
                             "Both sides are multiples of 64 commonly produced by diffusion models. Weak hint only."))
    elif make:
        has_makernotes = any(k.startswith("MakerNotes") or k.split(":")[0].lower() == str(make).split()[0].lower()
                             for k in meta)
        if any(b in str(make).lower() for b in MAKERNOTE_BRANDS) and not has_makernotes:
            f.append(Finding(NOTE, "MakerNotes missing",
                             f"{make} {model or ''} normally writes MakerNotes; they were dropped by software that re-saved the file."))
    ew, eh = tag(meta, "ExifImageWidth"), tag(meta, "ExifImageHeight")
    if isinstance(ew, int) and isinstance(eh, int) and {ew, eh} != {ev.width, ev.height}:
        f.append(Finding(NOTE, "Pixel dimensions differ from EXIF",
                         f"EXIF says {ew} x {eh}, file is {ev.width} x {ev.height}: resized or cropped after capture."))

    digest = tag(meta, "JPEGDigest")
    if digest:
        f.append(Finding(INFO, "Quantization-table signature (JPEGDigest)", str(digest)))
    lat, lon = tag(meta, "GPSLatitude", "Composite"), tag(meta, "GPSLongitude", "Composite")
    if lat and lon:
        f.append(Finding(INFO, "GPS position embedded", f"{lat}, {lon}"))
    if any(k.startswith(("JUMBF", "C2PA")) for k in meta):
        f.append(Finding(INFO, "Content Credentials (C2PA) present", "See the C2PA tool for signature validation."))
    for key in ("ExifTool:Warning", "ExifTool:Error"):
        if key in meta:
            f.append(Finding(NOTE, "ExifTool structural warning", str(meta[key])))

    res.score = res.severity_score()
    return res


def _flatten(values) -> list[str]:
    out = []
    for v in values:
        out += _flatten(v) if isinstance(v, list) else [str(v)]
    return out


def _short(value, limit: int = 600) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[:limit] + " ..."


# ------------------------------------------------------------------ thumbnail

def run_thumbnail(ev: Evidence) -> Result:
    """Compare embedded preview images with the main image. Editors often leave
    the camera's original thumbnail untouched after the pixels were changed."""
    res = Result()
    if not exiftool_path().exists():
        return res
    best = None
    for name in ("ThumbnailImage", "PreviewImage", "JpgFromRaw", "OtherImage"):
        raw = _exiftool(["-b", f"-{name}", str(ev.path)], binary=True)
        if len(raw) > 100:
            thumb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if thumb is not None and (best is None or thumb.shape[1] > best[1].shape[1]):
                best = (name, thumb)
    if best is None:
        res.findings.append(Finding(INFO, "No embedded thumbnail or preview"))
        return res
    name, thumb = best
    th, tw = thumb.shape[:2]
    main = _fit_like_thumbnail(ev.bgr, tw, th)
    diff = cv2.absdiff(cv2.GaussianBlur(thumb, (0, 0), 1.2), cv2.GaussianBlur(main, (0, 0), 1.2))
    dmap = diff.max(axis=2).astype(np.float32)
    mean_diff = float(dmap.mean())
    changed = float((dmap > 40).mean())
    res.views["Thumbnail"] = thumb
    res.views["Main image (scaled)"] = main
    res.views["Difference"] = cv2.applyColorMap(np.clip(dmap * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    res.table = [("Source tag", name), ("Thumbnail size", f"{tw} x {th}"),
                 ("Mean difference", f"{mean_diff:.1f}"), ("Strongly differing area", f"{changed:.1%}")]
    aspect_gap = abs(tw / th - ev.width / ev.height) / (ev.width / ev.height)
    if aspect_gap > 0.03 and not _letterboxed(thumb):
        res.findings.append(Finding(WARN, "Thumbnail aspect ratio differs from the image",
                                    "The main image was cropped after the thumbnail was written."))
    if changed > 0.02 or mean_diff > 12:
        res.findings.append(Finding(ALERT if changed > 0.05 else WARN, "Thumbnail shows different content",
                                    "The embedded preview does not match the main image: the pixels were changed "
                                    "after capture and the old preview was left in place. Inspect the Difference view."))
        res.score = min(1.0, 0.6 + changed * 4)
    else:
        res.findings.append(Finding(INFO, "Thumbnail matches the main image"))
        res.score = 0.05
    return res


def _letterboxed(thumb: np.ndarray) -> bool:
    g = cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY)
    return bool(g[:3].mean() < 8 and g[-3:].mean() < 8) or bool(g[:, :3].mean() < 8 and g[:, -3:].mean() < 8)


def _fit_like_thumbnail(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Scale the main image into the thumbnail frame (cameras letterbox 3:2 into 4:3)."""
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), np.uint8)
    y, x = (th - nh) // 2, (tw - nw) // 2
    canvas[y:y + nh, x:x + nw] = small
    return canvas


# ------------------------------------------------------------------ C2PA

def run_c2pa(ev: Evidence) -> Result:
    res = Result()
    try:
        import c2pa
    except Exception as exc:  # native library missing
        res.findings.append(Finding(NOTE, "C2PA library unavailable", str(exc)))
        return res
    mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
            "tif": "image/tiff", "tiff": "image/tiff", "heic": "image/heic", "heif": "image/heif",
            "avif": "image/avif"}.get(ev.path.suffix.lower().lstrip("."), "image/jpeg")
    try:
        with c2pa.Reader(mime, io.BytesIO(ev.data)) as reader:
            store = json.loads(reader.json())
    except Exception as exc:
        if "ManifestNotFound" in str(exc) or "no JUMBF" in str(exc) or "not found" in str(exc).lower():
            res.findings.append(Finding(INFO, "No Content Credentials",
                "No C2PA manifest is embedded. Most images have none; absence proves nothing."))
            res.score = 0.05
        else:
            res.findings.append(Finding(NOTE, "C2PA data could not be read", str(exc)[:400]))
        return res

    active = store.get("manifests", {}).get(store.get("active_manifest", ""), {})
    sig = active.get("signature_info", {}) or {}
    gen = active.get("claim_generator") or ", ".join(
        f"{g.get('name', '')} {g.get('version', '')}".strip() for g in active.get("claim_generator_info", []) or [])
    res.table = [("Claim generator", gen), ("Signed by", str(sig.get("issuer", ""))),
                 ("Signing time", str(sig.get("time", ""))), ("Title", str(active.get("title", ""))),
                 ("Manifests in store", str(len(store.get("manifests", {})))),
                 ("Validation state", str(store.get("validation_state", "")))]
    res.findings.append(Finding(INFO, "Content Credentials present", f"Generator: {gen}\nSigner: {sig.get('issuer', '?')}"))

    for manifest in store.get("manifests", {}).values():
        for assertion in manifest.get("assertions", []):
            if not str(assertion.get("label", "")).startswith("c2pa.actions"):
                continue
            for action in (assertion.get("data", {}) or {}).get("actions", []):
                name = action.get("action", "")
                agent = action.get("softwareAgent", "")
                agent = agent.get("name", "") if isinstance(agent, dict) else agent
                dst = str(action.get("digitalSourceType", ""))
                res.table.append(("Action", f"{name}  {agent}  {dst.rsplit('/', 1)[-1]}".strip()))
                if "compositeWithTrainedAlgorithmicMedia" in dst:
                    res.findings.append(Finding(ALERT, "Signed record of generative-AI editing", f"{name} by {agent}"))
                elif "trainedAlgorithmicMedia" in dst:
                    res.findings.append(Finding(ALERT, "Signed declaration: created by generative AI", f"{name} by {agent}"))
                elif name not in ("c2pa.created", "c2pa.opened", "c2pa.published", "c2pa.placed") and name:
                    res.findings.append(Finding(NOTE, f"Recorded edit: {name}", str(agent)))
    problems = [s for s in store.get("validation_status", []) or []
                if not str(s.get("code", "")).startswith(("claimSignature.validated", "signingCredential.trusted"))]
    hard = [s for s in problems if "mismatch" in str(s.get("code", "")) or "invalid" in str(s.get("code", "")).lower()]
    if hard:
        res.findings.append(Finding(ALERT, "Content Credentials fail validation",
            "The image no longer matches its signed manifest (changed after signing):\n"
            + "\n".join(f"{s.get('code')}: {s.get('explanation', '')}" for s in hard[:5])))
    elif problems:
        res.findings.append(Finding(NOTE, "Validation remarks",
            "\n".join(f"{s.get('code')}: {s.get('explanation', '')}" for s in problems[:5])))
    res.table.append(("Full manifest", _short(store, 6000)))
    res.score = res.severity_score()
    return res


# ------------------------------------------------------------------ hashes / strings

def run_hashes(ev: Evidence) -> Result:
    res = Result()
    res.table = [("File", str(ev.path)), ("Size", f"{len(ev.data):,} bytes"),
                 ("Pixels", f"{ev.width} x {ev.height}"),
                 *[(k.upper(), v) for k, v in ev.hashes.items()],
                 ("Perceptual hash (pHash)", f"{ev.phash:016x}")]
    return res


INTERESTING = re.compile(r"photoshop|adobe|gimp|lavc|ffmpeg|libjpeg|exif|http|xmp|created|software|"
                         r"prompt|seed|sampler|c2pa|generated|copyright|\.(?:psd|png|jpe?g|tif)", re.I)


def run_strings(ev: Evidence, min_length: int = 6) -> Result:
    res = Result()
    pattern = re.compile(rb"[\x20-\x7e]{%d,}" % int(min_length))
    seen, rows = set(), []
    for m in pattern.finditer(ev.data):
        text = m.group().decode("ascii")
        if text in seen or not INTERESTING.search(text):
            continue
        seen.add(text)
        rows.append((f"@{m.start():,}", text[:400]))
        if len(rows) >= 400:
            break
    res.table = rows or [("", "No notable strings found")]
    return res
