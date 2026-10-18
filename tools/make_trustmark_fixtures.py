"""Test images for the TrustMark decoder, made with Adobe's reference encoder (run in .venv-convert, which needs
`pip install trustmark`; on Windows set PYTHONUTF8=1 for that install).

    .venv-convert/Scripts/python tools/make_trustmark_fixtures.py

The cover is a CC0 guide picture (fiasuite/guide_examples/SOURCES.md). Writes tests/data/trustmark/.
"""
from pathlib import Path

from PIL import Image
from trustmark import TrustMark

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "tests" / "data" / "trustmark"
OUT.mkdir(parents=True, exist_ok=True)
cover = Image.open(ROOT / "fiasuite" / "guide_examples" / "ela_original.jpg").convert("RGB")
cover = cover.resize((480, cover.height * 480 // cover.width))
E = TrustMark.Encoding
for name, variant, encoding, mode, payload in [("q_text", "Q", E.BCH_5, "text", "FIASuite"),
                                               ("p_text", "P", E.BCH_5, "text", "Adobe P"),
                                               ("q_signpost", "Q", E.BCH_SUPER, "binary", "0" * 39 + "1")]:
    tm = TrustMark(verbose=False, model_type=variant, encoding_type=encoding, loadRemover=False)
    # P is the low-visibility variant: even Adobe's decoder misses it in a JPEG this small, so it stays lossless
    tm.encode(cover, payload, MODE=mode).save(OUT / (f"{name}.png" if variant == "P" else f"{name}.jpg"), quality=90)
cover.save(OUT / "clean.jpg", quality=90)
print("written to", OUT)
