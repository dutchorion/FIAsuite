"""Tools added in the audit round (PRNU, geometry, watermark, histogram, hidden pixels, double-quantization map,
mirrored clones) and regressions for the defects found in the code review."""
import re

import cv2
import numpy as np
import pytest

from fiasuite import launder, verdict
from fiasuite.analyses import clone, compare, compression, geometry, metadata, online, prnu, visual, watermark
from fiasuite.core import ALERT, Evidence, Finding, Result
from tests import synth


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return tmp_path_factory.mktemp("new_tools")


@pytest.fixture(scope="module")
def original():
    return synth.camera(synth.scene(seed=0))


# ------------------------------------------------------------------ new tools

def test_double_quantization_map_finds_content_added_between_two_compressions(folder, original):
    donor = synth.camera(synth.scene(seed=5), seed=6)
    base = synth.jpeg_roundtrip(original, 70)
    forged = base.copy()
    forged[248:504, 400:656] = donor[100:356, 200:456]
    clean = compression.dct_analysis(Evidence(synth.save(folder / "dq_clean.jpg", base, 90)))
    hit = compression.dct_analysis(Evidence(synth.save(folder / "dq_forged.jpg", forged, 90)))
    assert clean.metrics["dq_single_area"] < 0.005
    assert 0.04 < hit.metrics["dq_single_area"] < 0.15          # the insert covers 8% of the picture
    assert any("Region without" in f.title for f in hit.findings)


def test_mirrored_clone_needs_the_mirrored_search(folder, original):
    forged = original.copy()
    forged[250:506, 403:659] = original[60:316, 40:296][:, ::-1]
    ev = Evidence(synth.save(folder / "mirror.jpg", forged, 92))
    assert clone.keypoints(ev).score < 0.3
    assert clone.keypoints(ev, mirrored=True).score > 0.5
    assert clone.keypoints(Evidence(synth.save(folder / "plain.jpg", original, 92)), mirrored=True).score < 0.3


def test_a_clone_is_one_group_and_every_pair_counts_once(folder, original):
    forged = original.copy()
    forged[250:506, 403:659] = original[60:316, 40:296]
    table = dict(clone.keypoints(Evidence(synth.save(folder / "clone.jpg", forged, 92))).table)
    assert table["Matched groups"] == "1"
    assert int(table["Largest group"].split()[0]) >= 40


def test_prnu_identifies_the_sensor_and_maps_foreign_content(folder, monkeypatch):
    h, w = 480, 640
    sensor = lambda seed: np.random.default_rng(seed).normal(0, 0.015, (h, w)).astype(np.float32)

    def shoot(scene_seed, k, noise_seed):
        img = synth.scene(h, w, seed=scene_seed) * (1 + k[..., None])
        return np.clip(img + np.random.default_rng(noise_seed).normal(0, 1.5, img.shape), 0, 255).astype(np.uint8)

    ka, kb = sensor(100), sensor(200)
    refs = [str(synth.save(folder / f"ref{i}.png", shoot(10 + i, ka, 500 + i), None)) for i in range(16)]
    monkeypatch.setattr(prnu, "store", lambda: folder)
    prnu.save_fingerprint("camA", prnu.build_fingerprint(refs), "test")
    assert "camA" in prnu.stored()
    same, other = shoot(77, ka, 900), shoot(77, kb, 901)
    forged = same.copy()
    forged[150:330, 250:470] = shoot(78, kb, 902)[150:330, 250:470]
    pce = lambda img, name: prnu.identify(Evidence(synth.save(folder / name, img, None)), "camA")
    assert pce(same, "same.png").metrics["pce"] > 1000
    assert pce(other, "other.png").metrics["pce"] < prnu.PCE_MATCH
    crop = pce(same[64:400, 96:560], "crop.png")
    assert crop.metrics["pce"] > 1000 and "x=96, y=64" in dict(crop.table)["Position in the sensor frame"]
    hit = pce(forged, "forged.png")
    assert hit.metrics["prnu_missing_area"] > 0.03 and hit.score > 0.5
    assert pce(same, "same.png").metrics.get("prnu_missing_area", 0) < 0.005


def test_geometry_names_the_line_that_misses_the_common_point():
    point = np.array([500.0, -300.0])

    def through(x, y, off=0.0):
        d = (point - (x, y)) / np.linalg.norm(point - (x, y))
        a = np.array([x, y]) + np.array([-d[1], d[0]]) * off
        return (*a, *(a + d * 120))

    good = [through(100, 600), through(400, 650), through(800, 500)]
    m = geometry.measure(good)
    assert np.allclose(m["point"], point, atol=1e-6) and max(m["miss"]) < 0.01
    m = geometry.measure(good + [through(600, 600, off=60)])
    assert int(np.argmax(m["miss"])) == 3 and m["miss"][3] > geometry.TOLERANCE_DEG and max(m["miss"][:3]) < 0.01
    assert geometry.measure([(0, 0, 100, 0), (0, 50, 100, 50)])["parallel"]


def test_stable_diffusion_watermark_round_trip(folder, original):
    for message in watermark.MESSAGES.values():
        marked = Evidence(synth.save(folder / "wm.png", synth.embed_watermark(original, message), None))
        assert watermark.detect(marked).score > 0.9
    assert watermark.detect(Evidence(synth.save(folder / "nowm.png", original, None))).score < 0.1


def test_histogram_comb_after_a_levels_adjustment(folder, original):
    lut = np.clip((np.arange(256) - 30) * 1.35, 0, 255).astype(np.uint8)
    assert visual.histogram(Evidence(synth.save(folder / "levels.png", lut[original], None))).worst() == "note"
    assert visual.histogram(Evidence(synth.save(folder / "untouched.png", original, None))).worst() == "info"


def test_hidden_pixels_under_transparency(folder, original):
    rgba = cv2.cvtColor(original, cv2.COLOR_BGR2BGRA)
    rgba[100:300, 100:400, 3] = 0
    cv2.imwrite(str(folder / "alpha.png"), rgba)
    assert visual.hidden_pixels(Evidence(folder / "alpha.png")).worst() == "note"
    assert "No transparency" in visual.hidden_pixels(Evidence(synth.save(folder / "opaque.png", original, None))).findings[0].title


# ------------------------------------------------------------------ regressions from the code review

@pytest.mark.parametrize("caption", ["Models walk the runway at Paris Fashion Week", "Gemini constellation over the desert",
                                     "A firefly at dusk", "OpenAI CEO speaks at conference", "Sora Aoi attends premiere"])
def test_ordinary_caption_words_are_not_ai_declarations(caption):
    assert not metadata.AI_GENERATORS.search(caption)
    assert metadata.AI_GENERATORS_IN_SOFTWARE_TAGS.search("Made with Gemini")       # still caught in a software tag
    assert metadata.AI_GENERATORS.search("created with Midjourney v6")


def test_compare_sees_small_edits_and_edits_on_black(folder, original):
    base = original.copy()
    base[500:760, 0:400] = 0                                     # a pure black area
    edited = base.copy()
    edited[560:660, 100:300] = 200                               # object pasted onto the black
    edited[100:124, 700:724] = 255 - edited[100:124, 700:724]    # 24 px edit elsewhere
    ref, ev = Evidence(synth.save(folder / "cmp_ref.png", base, None)), Evidence(synth.save(folder / "cmp_new.png", edited, None))
    res = compare.compare(ev, ref)
    assert any(f.severity in ("warn", "alert") for f in res.findings), [f.title for f in res.findings]


def test_ai_declaration_counts_as_an_analysed_axis():
    res = Result(score=0.95, findings=[Finding(ALERT, "Metadata declares the image as AI-generated")])
    assert verdict.summarize({"metadata": res})["ai"].level != "Not analysed"


def test_laundering_steps_are_range_checked():
    assert launder.parse_chain("jpeg85 + resize0.5") == [("jpeg", 85.0), ("resize", 0.5)]
    for bad in ("jpeg0", "jpeg500", "resize50", "resize0"):
        with pytest.raises(ValueError):
            launder.parse_chain(bad)


def test_config_is_replaced_atomically_and_tolerates_a_read_only_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(online, "config_path", lambda: tmp_path / "config.json")
    assert online.save_config({"a": 1}) and online.load_config() == {"a": 1}
    (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")                # valid JSON, wrong shape
    assert online.load_config() == {}
    monkeypatch.setattr(online, "config_path", lambda: tmp_path / "missing" / "config.json")
    assert online.save_config({"a": 1}) is False


def test_pdf_break_opportunities_do_not_split_html_entities():
    from fiasuite import report
    text = 'x' * 27 + '"&<' + 'y' * 40
    assert not re.search(r"&[a-z]*​[a-z]*;", report.soften(text))
    assert "&quot;" in report.soften(text) and "​" in report.soften(text)


def test_tiny_images_fail_softly(folder):
    tiny = Evidence(synth.save(folder / "tiny.png", np.full((40, 30, 3), 128, np.uint8), None))
    for fn in (compression.block_grid, compression.ghosts):
        assert fn(tiny).findings
