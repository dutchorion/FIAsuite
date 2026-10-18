"""Decoder for Adobe TrustMark, the open invisible watermark (MIT licence) that the C2PA lists as a soft binding:
an identifier in the pixels that survives metadata stripping, by which the file's Content Credentials can be
looked up again ('durable Content Credentials').

The decoder networks are Adobe's own ONNX exports; variants are not interchangeable, so both published ones run
(Q, the default, and P, the low-visibility one, which always reads the centre square). A decoder returns 100
values; their signs are the bits. The last two bits name the error-correction schema, the rest is payload plus
BCH check bits, verified with Adobe's BCH code (_trustmark_bch.py).

Adobe's reference decoder accepts any bit pattern the error correction can repair. On 202 unmarked images that
produced 14 false 'watermark found' results, all in the weak BCH_3 / BCH_4 schemas. The decoder's signal strength
(mean absolute output) separated them cleanly: at most 2.1 on unmarked images, at least 5.2 on marked ones, JPEG
quality 85 included. A mark therefore counts only with valid error correction AND a strength of MIN_STRENGTH.
"""
from __future__ import annotations

import re
import threading

import numpy as np
from PIL import Image

from ..core import INFO, NOTE, WARN, Evidence, Finding, Result
from . import ai
from ._trustmark_bch import BCH

BCH_POLYNOMIAL = 137
SCHEMAS = {0: ("BCH_SUPER", 40, 8), 1: ("BCH_5", 61, 5), 2: ("BCH_4", 68, 4), 3: ("BCH_3", 75, 3)}  # name, payload bits, repairable
MIN_STRENGTH = 3.5

# C2PA soft-binding algorithm list (github.com/c2pa-org/softbinding-algorithm-list), snapshot of 18 October 2026
REGISTRY = {
    1: "com.digimarc.validate.1", 2: "org.atsc.a336", 3: "io.iscc.v0", 4: "com.adobe.trustmark.Q",
    5: "com.adobe.trustmark.C", 6: "com.adobe.icn.dense", 7: "ai.steg.api", 8: "ai.trufo.pawprint.watermark",
    9: "ai.trufo.pawprint.fingerprint", 10: "app.overlai.watermark.1", 11: "tv.kinetiq.watercast.48.1",
    12: "castLabs.watermark.1", 13: "com.adobe.trustmark.P", 14: "com.imatag.lamark.v1",
    15: "com.nagra.nexguard.watermark.1", 16: "com.mentaport.watermark.1", 17: "com.digicaps.fingerprint.1",
    18: "com.sonicorigin.watermark.1", 19: "com.microsoft.invismark.1", 20: "com.microsoft.wavmark.1",
    21: "ai.contentlens.image.basewmk", 22: "ai.contentlens.image.ensemblewmk", 23: "ai.contentlens.image.customwmk",
    24: "ai.contentlens.audio.mono", 25: "ai.contentlens.audio.stereo", 26: "es.lumatrace",
    27: "com.markany.watermark.1", 28: "com.verimatrix.watermark.1", 29: "com.writerslogic.zwc-watermark.1",
    30: "com.aiwatermark.videoseal.1", 31: "com.aiwatermark.pixelseal.1", 32: "com.aiwatermark.audioseal.1",
    33: "eu.sasha.softbinding.v1", 34: "ai.verda.watermark.image.1", 35: "ai.verda.watermark.video.1",
    36: "ai.verda.watermark.audio.1", 37: "com.evixar.eaw.1", 38: "com.evixar.efp.type3.1",
    39: "com.cognitive-proof.vsrmark.1", 40: "me.deepmark.audio.vigil.128",
    41: "com.writerslogic.text-fingerprint.1", 42: "com.writerslogic.zwc-watermark.2",
    43: "com.writerslogic.text-structure.1", 44: "com.writerslogic.text-minhash.1",
    45: "io.blockfact.audio.watermark.32", 46: "com.doverunner.watermark.1",
    47: "com.writerslogic.text-watermark-semantic.1", 48: "com.joinmonolith.sha256",
    49: "com.museblossom.contentsdefence.1", 50: "me.reconize.videoseal.1", 51: "io.diker.sigi.v1",
    52: "com.adobe.hiermark.A", 53: "com.adobe.flowmark.A", 54: "be.immostory.videoseal.1",
}

_bch: dict[int, BCH] = {}
_bch_lock = threading.Lock()                     # BCH keeps its working buffers on the object


def _bytes(bits: str) -> bytearray:
    bits += "0" * (-len(bits) % 8)
    return bytearray(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def check_bits(bits) -> tuple[str, bool, int]:
    """(payload bits, error correction succeeded, schema). The same split and padding as Adobe's DataLayer."""
    packet = "".join("1" if b else "0" for b in bits)
    version = int(packet[-2:], 2)
    _, length, t = SCHEMAS[version]
    data, ecc = _bytes(packet[:length]), _bytes(packet[length:96])
    with _bch_lock:
        if t not in _bch:
            _bch[t] = BCH(t, BCH_POLYNOMIAL)
        ok = len(ecc) == _bch[t].get_ecc_bytes() and _bch[t].decode(data, ecc) != -1
    return "".join(format(x, "08b") for x in data)[:length], ok, version


def decoders() -> list[dict]:
    return [e for e in ai.load_registry() if e.get("kind") == "trustmark"]


def read(ev: Evidence, entry: dict) -> tuple[str, bool, int, float]:
    """(payload, error correction succeeded, schema, signal strength) for one decoder variant."""
    rgb = Image.fromarray(np.ascontiguousarray(ev.bgr[:, :, ::-1]))
    w, h = rgb.size
    if entry.get("square") or max(w, h) / min(w, h) > 2:      # Adobe's rule: long images are read in the centre square
        s = min(w, h)
        rgb = rgb.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    size = int(entry["size"])
    # PIL's bilinear resampling filters when it shrinks, like the reference implementation; values in [-1, 1]
    x = np.asarray(rgb.resize((size, size), Image.BILINEAR), np.float32).transpose(2, 0, 1)[None] / 127.5 - 1
    sess = ai._session(entry)
    out = np.asarray(ai._run(sess, {sess.get_inputs()[0].name: x})[0], np.float64).reshape(-1)
    payload, ok, version = check_bits(out > 0)
    return payload, ok, version, float(np.abs(out).mean())


def _as_text(payload: str) -> str | None:
    """Adobe's text mode packs 7-bit ASCII. Binary identifiers often happen to give printable characters too, so a
    reading is shown only when it looks like words: letters, digits and simple punctuation, at least three long."""
    codes = [int(payload[i:i + 7], 2) for i in range(0, len(payload) // 7 * 7, 7)]
    while codes and codes[-1] == 0:
        codes.pop()
    text = "".join(map(chr, codes)).strip()
    return text if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _.,:/@#-]{2,}", text) else None


def detect(ev: Evidence) -> Result:
    res = Result()
    entries = decoders()
    if not entries:
        res.findings.append(Finding(NOTE, "TrustMark decoders not installed",
                                    "Install them with Setup ... in the toolbar (two files of 47 MB)."))
        return res
    hits = []
    for entry in entries:
        payload, ok, version, strength = read(ev, entry)
        found = ok and strength >= MIN_STRENGTH
        res.metrics[entry["id"]] = strength
        note = "found" if found else ("not found (the bits pass the error correction, but the signal is too weak: a "
                                      "chance pattern)" if ok else "not found")
        res.table.append((f"Variant {entry['variant']}", f"{note}   (signal strength {strength:.1f}; marks read "
                                                         f"above {MIN_STRENGTH:g})"))
        if found:
            hits.append((entry, payload, version, strength))
    if not hits:
        res.score = 0.05
        res.findings.append(Finding(INFO, "No TrustMark watermark",
            "Means nothing by itself: few services embed TrustMark, and heavy cropping, strong resizing or "
            "re-generation can remove it."))
        return res
    res.score = 0.6
    for entry, payload, version, strength in hits:
        schema, length, _ = SCHEMAS[version]
        value = int(payload, 2)
        text = _as_text(payload)
        res.table += [(f"Variant {entry['variant']}: schema", f"{schema} ({length}-bit payload)"),
                      (f"Variant {entry['variant']}: payload", f"{value:0{-(-length // 4)}x} (hex)"
                       + (f"   as text: '{text}'" if text else ""))]
        detail = (f"Adobe TrustMark variant {entry['variant']} decoded with valid error correction ({schema}) and "
                  f"signal strength {strength:.1f}. The payload is an identifier: whoever embedded it can look up the "
                  "file's Content Credentials by it, even after the metadata was stripped. Compare with the Content "
                  "Credentials (C2PA) tool: a watermark without a manifest means the metadata was removed after "
                  "watermarking.")
        if version == 0 and value in REGISTRY:
            detail += (f"\n\nRead as a number the payload is {value}, entry '{REGISTRY[value]}' of the C2PA "
                       "soft-binding list. TrustMark's signpost mode uses this schema to point to a second watermark "
                       "of another vendor; its exact payload layout is not documented, so treat the name as a lead.")
        res.findings.append(Finding(WARN, "Content Credentials watermark (TrustMark) found", detail))
    return res
