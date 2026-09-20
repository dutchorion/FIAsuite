"""Builds the example pictures shown in the "How to read this" windows (fiasuite/guide_examples/).

Sources are public-domain / CC0 photographs from Wikimedia Commons (see guide_examples/SOURCES.md); the forgeries
are made here, so the ground truth of every example is known. Run from the project root:

    .venv\\Scripts\\python.exe tools\\make_guide_examples.py [folder with the source photos]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from fiasuite.analyses import ai, clone, compression, geometry, interpolation, metadata, noise, spectral, visual  # noqa: E402
from fiasuite.core import Evidence  # noqa: E402

ai.use_gpu = lambda: False
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "bench" / "pd_photos"
OUT = ROOT / "fiasuite" / "guide_examples"
WORK = ROOT / "build" / "guide_work"
OUT.mkdir(exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)


def half(path: Path) -> np.ndarray:
    """Half-size copy: averaging 2x2 pixels removes the source file's own JPEG history, so every compression step
    in an example is one that was applied here."""
    img = cv2.imread(str(path))
    return cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def jpeg(img: np.ndarray, q: int) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def save(name: str, img: np.ndarray, q: int | None) -> Evidence:
    path = WORK / name
    cv2.imwrite(str(path), img, [] if q is None else [cv2.IMWRITE_JPEG_QUALITY, q])
    return Evidence(path)


def panel(name: str, parts: list[tuple[str, np.ndarray]], height: int = 330, box: tuple | None = None) -> None:
    """Side-by-side picture with captions; `box` = (x, y, w, h) of the forged area, outlined on the first part."""
    tiles = []
    for i, (label, img) in enumerate(parts):
        img = img.copy()
        if box and i == 0:
            x, y, w, h = box
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), max(2, img.shape[1] // 300))
        scale = height / img.shape[0]
        tile = cv2.resize(img, (max(1, int(img.shape[1] * scale)), height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST)
        strip = np.full((26, tile.shape[1], 3), 32, np.uint8)
        cv2.putText(strip, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        tiles.append(np.vstack([strip, tile]))
        tiles.append(np.full((height + 26, 6, 3), 18, np.uint8))
    cv2.imwrite(str(OUT / f"{name}.jpg"), np.hstack(tiles[:-1]), [cv2.IMWRITE_JPEG_QUALITY, 84])
    print("wrote", name)


# ------------------------------------------------------------------ material
land = half(SRC / "landscape.jpg")[300:300 + 1056, 600:600 + 1600]          # 1600 x 1056, sky + lake + forest
fruit = half(SRC / "stall.jpg")[620:620 + 512, 0:1600]                       # fruit only, no people
X, Y, W, H = 1003, 403, 320, 240                                             # forged area, off the 8x8 grid on purpose
BOX = (X, Y, W, H)
patch = fruit[100:100 + H, 500:500 + W]

original = save("original.jpg", land, 92)
spliced_px = land.copy()
spliced_px[Y:Y + H, X:X + W] = jpeg(fruit, 60)[100:100 + H, 500:500 + W]     # donor came from a low-quality JPEG
splice = save("splice.jpg", spliced_px, 92)

# ------------------------------------------------------------------ ELA
panel("ela_original", [("camera-like original, saved once", original.bgr),
                       ("ELA: similar surfaces respond alike", compression.ela(original).views["Error level"])])
panel("ela_splice", [("pasted object (outlined)", splice.bgr), ("ELA: the insert responds differently", compression.ela(splice).views["Error level"])], box=BOX)
worn = spliced_px
for _ in range(4):
    worn = jpeg(worn, 75)
worn = save("splice_resaved.jpg", worn, 75)
panel("ela_resaved", [("same forgery after 5 re-saves at Q75", worn.bgr), ("ELA: nothing left to see", compression.ela(worn, quality=75).views["Error level"])], box=BOX)

# ------------------------------------------------------------------ JPEG ghosts, grid
g = compression.ghosts(splice)
panel("ghosts_splice", [("insert from a Q60 JPEG, saved at Q92", splice.bgr), ("Ghost strength", g.views["Ghost strength"])], box=BOX)
big_px = land.copy()                                                           # the grid test needs an insert of several 64 px windows
big_px[371:371 + 320, 931:931 + 480] = jpeg(fruit, 50)[100:420, 500:980]
big = save("splice_large.jpg", big_px, 92)
grid = compression.block_grid(big)
print("grid example:", [f.title for f in grid.findings])
panel("grid_foreign", [("larger insert, pasted off the 8x8 grid", big.bgr), ("Foreign grid", grid.views["Foreign grid"])], box=(931, 371, 480, 320))

# ------------------------------------------------------------------ double quantization
first = jpeg(land, 70)
dq_px = first.copy()
dq_px[Y:Y + H, X:X + W] = patch                                               # never-compressed content in a re-saved JPEG
dq = save("dq_insert.jpg", dq_px, 90)
res = compression.dct_analysis(dq)
panel("dct_map", [("Q70 photo, object added, saved at Q90", dq.bgr), ("Double-quantization map", res.views["Double-quantization map"])], box=BOX)
single = compression.dct_analysis(original).views["Coefficient histograms"][:220, :450]
double = res.views["Coefficient histograms"][:220, :450]
panel("dct_histograms", [("saved once: smooth decay", single), ("saved twice: comb of gaps and peaks", double)], height=220)

# ------------------------------------------------------------------ noise
soft = land.copy()
soft[Y:Y + H, X:X + W] = cv2.GaussianBlur(land, (0, 0), 2.2)[Y:Y + H, X:X + W]      # airbrushed / inpainted area
smooth = save("smoothed.png", soft, None)
panel("noise_smoothed", [("area retouched with a soft brush", smooth.bgr), ("Noise residual: the grain is gone there",
                                                                           noise.residual(smooth, amplitude=40).views["Noise residual"])], box=BOX)
nm = noise.noise_map(smooth)
panel("noisemap_smoothed", [("same retouched picture", smooth.bgr), ("Deviation from trend", nm.views.get("Deviation from trend", smooth.bgr))], box=BOX)

# ------------------------------------------------------------------ resampling
m = cv2.getRotationMatrix2D((W / 2, H / 2), 8, 1.35)
scaled = cv2.warpAffine(fruit[60:60 + H, 900:900 + W], m, (W, H), flags=cv2.INTER_LINEAR)
res_px = land.copy()
res_px[Y:Y + H, X:X + W] = scaled
resampled = save("resampled.png", res_px, None)
panel("resampling_insert", [("insert enlarged 1.35x and turned 8 degrees", resampled.bgr),
                            ("Peak strength", interpolation.resampling(resampled).views["Peak strength"])], box=BOX)

# ------------------------------------------------------------------ copy-move
kp_px = land.copy()
kp_px[640:640 + 300, 1100:1100 + 360] = land[660:660 + 300, 250:250 + 360]         # forest copied onto forest
kp = save("clone_forest.jpg", kp_px, 90)
panel("clone_keypoints", [("a stretch of shoreline duplicated", kp.bgr), ("Clone matches", clone.keypoints(kp).views["Clone matches"])],
      box=(1100, 640, 360, 300))
water_px = land.copy()
water_px[470:470 + 200, 380:380 + 420] = land[440:440 + 200, 820:820 + 420]       # lake surface copied onto lake surface
water = save("clone_water.jpg", water_px, 90)
panel("clone_blocks", [("a patch of water surface duplicated", water.bgr), ("Cloned blocks", clone.blocks(water).views["Cloned blocks"])],
      box=(380, 470, 420, 200))

# ------------------------------------------------------------------ visual tools on the splice
panel("pca_splice", [("pasted object", splice.bgr), ("Principal component 3", visual.pca(splice, component=3).views["Principal component 3"])], box=BOX)
panel("gradient_splice", [("pasted object", splice.bgr), ("Luminance gradient", visual.luminance_gradient(splice).views["Luminance gradient"])], box=BOX)
lut = np.clip((np.arange(256) - 25) * 1.3, 0, 255).astype(np.uint8)
levels = save("levels.png", lut[land], None)
panel("histogram_levels", [("untouched tones: smooth histogram", visual.histogram(save("plain.png", land, None)).views["Histogram"]),
                           ("after a Levels adjustment: comb", visual.histogram(levels).views["Histogram"])], height=260)
rgba = cv2.cvtColor(land, cv2.COLOR_BGR2BGRA)
rgba[Y:Y + H, X:X + W, 3] = 0
hidden = save("erased.png", rgba, None)
shown = land.copy()
shown[Y:Y + H, X:X + W] = 255                                                  # what a viewer displays: a white hole
panel("hidden_pixels", [("as displayed: area 'erased'", shown), ("Hidden pixels only: the content is still in the file",
                                                                 visual.hidden_pixels(hidden).views["Hidden pixels only"])])

# ------------------------------------------------------------------ embedded thumbnail left behind by an editor
if metadata.exiftool_path().exists():
    camera = cv2.imread(str(SRC / "landscape.jpg"))
    edited = camera.copy()
    edited[1900:2500, 2300:3300] = cv2.resize(cv2.imread(str(SRC / "stall.jpg"))[1300:1900, 0:1000], (1000, 600))
    target = WORK / "thumb_edited.jpg"
    cv2.imwrite(str(target), edited, [cv2.IMWRITE_JPEG_QUALITY, 92])
    metadata._exiftool(["-overwrite_original", "-tagsfromfile", str(SRC / "landscape.jpg"), "-all:all", "-ThumbnailImage", str(target)], binary=True)
    t = metadata.run_thumbnail(Evidence(target))
    if "Difference" in t.views:
        views = list(t.views.items())
        panel("thumbnail_edited", [("main image (edited)", Evidence(target).bgr)] + [(k, v) for k, v in views[:2]], height=260)

# ------------------------------------------------------------------ spectrum and AI detectors
photo = Evidence(SRC / "landscape.jpg")
generated = Evidence(SRC / "ai_sdxl.png")
yy, xx = np.indices(land.shape[:2])
lattice = 1.5 * np.cos(np.pi * yy / 2) * np.cos(np.pi * xx / 4)                # transposed-convolution checkerboard, simulated
artifact = save("lattice.png", np.clip(land + lattice[..., None], 0, 255).astype(np.uint8), None)
flat = "Flattened spectrum (peaks marked)"
panel("spectrum_compare", [("camera photo", spectral.spectrum(save("plain_land.png", land, None)).views[flat]),
                           ("simulated up-sampling artifact", spectral.spectrum(artifact).views[flat]),
                           ("real SDXL image: no peaks", spectral.spectrum(generated).views[flat])], height=320)
if ai.load_registry():
    for key, ev, label in (("ai_generated", generated, "generated with SDXL"), ("ai_photo", original, "camera photograph")):
        r = ai.detect(ev)
        first_map = next(iter(r.views.values()))
        panel(key, [(label, ev.bgr), (f"detector map, ensemble {r.score:.0%}", cv2.addWeighted(ev.bgr, 0.35, first_map, 0.65, 0))])
    r = ai.localize(splice)
    if "Tampering probability" in r.views:
        panel("trufor_splice", [("pasted object", splice.bgr), ("TruFor tampering probability", r.views["Tampering probability"]),
                                ("reliability of the map", r.views["Reliability of the map"])], height=260, box=BOX)

# ------------------------------------------------------------------ geometry: parallel edges of one facade
facade = cv2.imread(str(SRC / "facade.jpg"))
lines_file = Path(__file__).with_name("guide_facade_lines.json")
if lines_file.exists():
    sets = {"A": [tuple(line) for line in json.loads(lines_file.read_text())], "B": [], "C": [], "D": []}
    panel("geometry_facade", [("edges that are parallel on the building", geometry.render(facade, sets))], height=520)
print("done ->", OUT)
