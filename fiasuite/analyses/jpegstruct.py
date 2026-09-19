"""JPEG container analysis: marker map, quantization tables, quality estimate,
chroma subsampling, trailing data. Pure Python, works on the raw file bytes."""
from __future__ import annotations

import struct

import numpy as np

from ..core import INFO, NOTE, WARN, Evidence, Finding, Result

ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52,
    45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99]).reshape(8, 8)

STD_CHROMA = np.array([
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99]).reshape(8, 8)

MARKER_NAMES = {0xD8: "SOI", 0xD9: "EOI", 0xDA: "SOS", 0xDB: "DQT", 0xC4: "DHT", 0xDD: "DRI",
                0xFE: "COM", 0xC0: "SOF0 (baseline)", 0xC1: "SOF1 (extended)",
                0xC2: "SOF2 (progressive)", 0xC3: "SOF3 (lossless)", 0xC9: "SOF9 (arithmetic)",
                0xCA: "SOF10 (progressive, arithmetic)"}


def ijg_table(base: np.ndarray, quality: int) -> np.ndarray:
    scale = 5000 // quality if quality < 50 else 200 - 2 * quality
    return np.clip((base * scale + 50) // 100, 1, 255)


def estimate_quality(table: np.ndarray, base: np.ndarray) -> tuple[int, bool]:
    """(quality, exact): closest IJG quality and whether the table is an exact IJG table."""
    errs = [np.abs(ijg_table(base, q) - table).sum() for q in range(1, 101)]
    q = int(np.argmin(errs)) + 1
    return q, errs[q - 1] == 0


def _app_label(marker: int, payload: bytes) -> str:
    n = marker - 0xE0
    ident = payload[:40]
    known = [(b"JFIF\0", "JFIF"), (b"JFXX\0", "JFIF extension (thumbnail)"), (b"Exif\0", "Exif"),
             (b"http://ns.adobe.com/xap/1.0/", "XMP"), (b"http://ns.adobe.com/xmp/extension/", "Extended XMP"),
             (b"ICC_PROFILE", "ICC profile"), (b"Photoshop 3.0", "Photoshop IRB / IPTC"),
             (b"Adobe", "Adobe (colour transform)"), (b"MPF\0", "Multi-Picture Format"),
             (b"JP", "JUMBF (C2PA / Content Credentials)"), (b"Ducky", "Adobe 'Save for Web' (Ducky)"),
             (b"FPXR", "FlashPix"), (b"HDR_RI", "HDR gain map"), (b"urn:iso", "ISO gain map")]
    for prefix, label in known:
        if ident.startswith(prefix):
            return f"APP{n} {label}"
    text = bytes(c for c in ident[:16] if 32 <= c < 127).decode("ascii", "replace")
    return f"APP{n} {text}".strip()


def parse(data: bytes) -> dict:
    """Walk the marker structure of the primary JPEG image."""
    out = {"segments": [], "qtables": {}, "sof": None, "comments": [], "apps": [],
           "eoi": None, "restart_interval": 0, "scans": 0}
    pos, n = 2, len(data)
    out["segments"].append((0, "SOI", 2))
    while pos + 2 <= n:
        if data[pos] != 0xFF:
            pos += 1
            continue
        marker = data[pos + 1]
        if marker in (0xFF, 0x00) or 0xD0 <= marker <= 0xD7:
            pos += 1 if marker == 0xFF else 2
            continue
        if marker == 0xD9:
            out["segments"].append((pos, "EOI", 2))
            out["eoi"] = pos + 2
            break
        if pos + 4 > n:
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        payload = data[pos + 4:pos + 2 + length]
        name = MARKER_NAMES.get(marker, f"0x{marker:02X}")
        if 0xE0 <= marker <= 0xEF:
            name = _app_label(marker, payload)
            out["apps"].append(name)
        elif marker == 0xDB:
            _read_dqt(payload, out["qtables"])
        elif marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC9, 0xCA):
            prec, h, w, nc = struct.unpack(">BHHB", payload[:6])
            comps = [(payload[6 + 3 * i], payload[7 + 3 * i] >> 4, payload[7 + 3 * i] & 15, payload[8 + 3 * i])
                     for i in range(nc)]
            out["sof"] = {"type": name, "precision": prec, "height": h, "width": w, "components": comps,
                          "progressive": marker in (0xC2, 0xCA)}
        elif marker == 0xFE:
            out["comments"].append(payload.decode("latin-1", "replace"))
        elif marker == 0xDD:
            out["restart_interval"] = struct.unpack(">H", payload[:2])[0]
        out["segments"].append((pos, name, length + 2))
        pos += 2 + length
        if marker == 0xDA:
            out["scans"] += 1
            pos = _skip_entropy(data, pos)
    return out


def _read_dqt(payload: bytes, tables: dict) -> None:
    i = 0
    while i < len(payload):
        pq, tq = payload[i] >> 4, payload[i] & 15
        size = 128 if pq else 64
        raw = payload[i + 1:i + 1 + size]
        if len(raw) < size:
            break
        vals = np.frombuffer(raw, ">u2" if pq else np.uint8).astype(np.int32)
        table = np.zeros(64, np.int32)
        table[ZIGZAG] = vals
        tables[tq] = table.reshape(8, 8)
        i += 1 + size


def _skip_entropy(data: bytes, pos: int) -> int:
    n = len(data)
    while True:
        pos = data.find(b"\xff", pos)
        if pos < 0 or pos + 1 >= n:
            return n
        nxt = data[pos + 1]
        if nxt == 0x00 or 0xD0 <= nxt <= 0xD7 or nxt == 0xFF:
            pos += 1 if nxt == 0xFF else 2
            continue
        return pos


def subsampling(sof: dict) -> str:
    comps = sof["components"]
    if len(comps) == 1:
        return "grayscale"
    if len(comps) < 3:
        return "unknown"
    (_, h0, v0, _), (_, h1, v1, _) = comps[0], comps[1]
    return {(1, 1): "4:4:4 (none)", (2, 1): "4:2:2", (2, 2): "4:2:0", (1, 2): "4:4:0",
            (4, 1): "4:1:1"}.get((h0 // max(h1, 1), v0 // max(v1, 1)), f"{h0}x{v0}/{h1}x{v1}")


def luma_qtable(ev: Evidence) -> np.ndarray | None:
    """Luminance quantization table of a JPEG, or None."""
    if not ev.is_jpeg:
        return None
    info = ev.cache.setdefault("jpeg", parse(ev.data))
    sof = info["sof"]
    tq = sof["components"][0][3] if sof else 0
    return info["qtables"].get(tq)


def run(ev: Evidence) -> Result:
    res = Result()
    if not ev.is_jpeg:
        res.findings.append(Finding(INFO, "Not a JPEG file", "Container analysis applies to JPEG only."))
        return res
    info = ev.cache.setdefault("jpeg", parse(ev.data))
    sof = info["sof"]
    if sof:
        res.table += [("Frame type", sof["type"]), ("Dimensions", f"{sof['width']} x {sof['height']}"),
                      ("Precision", f"{sof['precision']} bit"), ("Chroma subsampling", subsampling(sof)),
                      ("Scans", str(info["scans"]))]
        if sof["progressive"]:
            res.findings.append(Finding(NOTE, "Progressive JPEG",
                "Cameras write baseline JPEG. Progressive encoding means the file was re-encoded by "
                "software (web platforms, editors, optimisers)."))
    for tq, table in sorted(info["qtables"].items()):
        q, exact = estimate_quality(table, STD_LUMA if tq == 0 else STD_CHROMA)
        kind = "standard IJG/libjpeg table" if exact else "custom table (camera firmware, Adobe, or tuned encoder)"
        res.table.append((f"Quantization table {tq}", f"quality ~{q}, {kind}"))
        res.table.append((f"  values {tq}", " ".join(str(v) for v in table.flatten())))
    apps = info["apps"]
    res.table.append(("APP segments", ", ".join(apps) or "none"))
    for c in info["comments"]:
        res.table.append(("Comment", c[:300]))
    res.table.append(("Marker map", "  ".join(f"{name}@{off}" for off, name, _ in info["segments"][:60])))

    if any("Ducky" in a for a in apps):
        res.findings.append(Finding(WARN, "Adobe 'Save for Web' marker",
            "The APP12 Ducky segment is written by Photoshop's Save for Web / Export dialog."))
    elif any("Photoshop" in a or a.endswith("Adobe (colour transform)") for a in apps):
        res.findings.append(Finding(NOTE, "Adobe-style segments present",
            "APP13/APP14 segments are written by Adobe software and by some other tools that embed IPTC. "
            "Indicates the file passed through software after capture; not proof of content changes."))
    if any("JUMBF" in a for a in apps):
        res.findings.append(Finding(INFO, "Content Credentials container found", "See the C2PA tool."))
    eoi = info["eoi"]
    if eoi is None:
        res.findings.append(Finding(WARN, "No end-of-image marker", "File is truncated or malformed."))
    elif eoi < len(ev.data):
        extra = len(ev.data) - eoi
        tail = ev.data[eoi:eoi + 4]
        if any("Multi-Picture" in a or "gain map" in a.lower() for a in apps) or tail[:2] == b"\xff\xd8":
            res.findings.append(Finding(INFO, f"{extra:,} bytes after the primary image",
                "Additional embedded image(s): typical for phone photos (HDR gain map, depth map, preview)."))
        else:
            res.findings.append(Finding(WARN, f"{extra:,} bytes of trailing data after end-of-image",
                "Unexpected data appended to the file (hidden payload, steganography container, or a "
                "sloppy editing tool). Inspect with the Strings tool."))
    if len(info["qtables"]) >= 2 and sof and len(sof["components"]) >= 3:
        if np.array_equal(info["qtables"].get(0), info["qtables"].get(1)):
            res.findings.append(Finding(NOTE, "Identical luminance and chrominance tables",
                "Unusual for cameras; seen with some software encoders."))
    res.score = res.severity_score()
    return res
