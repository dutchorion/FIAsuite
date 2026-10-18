"""TrustMark: the ported error correction behaves like Adobe's, marks made by Adobe's reference encoder are found
with their payload, and unmarked images are not reported (the reference decoder alone reported 14 of 202)."""
from pathlib import Path

import numpy as np
import pytest

from fiasuite.analyses import ai, trustmark
from fiasuite.core import Evidence
from tests import synth

DATA = Path(__file__).parent / "data" / "trustmark"
INSTALLED = {e["variant"] for e in trustmark.decoders()}
needs_decoders = pytest.mark.skipif(INSTALLED != {"Q", "P"}, reason="TrustMark decoders not installed")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(ai, "use_gpu", lambda: False)


def packet(payload: str, version: int) -> list[int]:
    """100 bits as Adobe's encoder lays them out: payload, BCH check bits, four version bits."""
    _, length, t = trustmark.SCHEMAS[version]
    bch = trustmark.BCH(t, trustmark.BCH_POLYNOMIAL)
    ecc = bch.encode(trustmark._bytes(payload))
    ecc_bits = "".join(format(x, "08b") for x in ecc)[:96 - length]
    return [int(b) for b in payload + ecc_bits + format(version, "04b")][:100]


@pytest.mark.parametrize("version", [0, 1, 2, 3])
def test_error_correction_repairs_up_to_its_limit(version):
    _, length, t = trustmark.SCHEMAS[version]
    payload = "".join(np.random.default_rng(version).choice(["0", "1"], length))
    bits = packet(payload, version)
    assert trustmark.check_bits(bits) == (payload, True, version)
    for i in range(t):                                # t flipped payload bits are repaired
        bits[3 + 7 * i] ^= 1
    assert trustmark.check_bits(bits) == (payload, True, version)


def test_text_reading_only_for_words():
    as_bits = lambda text: "".join(format(ord(c), "07b") for c in text).ljust(61, "0")
    assert trustmark._as_text(as_bits("FIASuite")) == "FIASuite"
    assert trustmark._as_text("1011" * 15 + "1") is None               # binary identifier, printable by accident


@needs_decoders
@pytest.mark.parametrize("name, variant, text", [("q_text.jpg", "Q", "FIASuite"), ("p_text.png", "P", "Adobe P")])
def test_marks_from_the_reference_encoder_are_found(name, variant, text):
    res = trustmark.detect(Evidence(str(DATA / name)))
    assert res.findings[0].title == "Content Credentials watermark (TrustMark) found"
    table = dict(res.table)
    assert f"as text: '{text}'" in table[f"Variant {variant}: payload"]
    assert res.metrics[f"trustmark_{variant.lower()}"] >= trustmark.MIN_STRENGTH


@needs_decoders
def test_signpost_names_the_registry_entry():
    res = trustmark.detect(Evidence(str(DATA / "q_signpost.jpg")))
    assert "BCH_SUPER" in dict(res.table)["Variant Q: schema"]
    assert "com.digimarc.validate.1" in res.findings[0].detail and "lead" in res.findings[0].detail


@needs_decoders
def test_unmarked_images_are_not_reported(tmp_path):
    images = [DATA / "clean.jpg"] + sorted((Path(trustmark.__file__).parents[1] / "guide_examples").glob("*.jpg"))
    for seed in range(8):
        images.append(synth.save(tmp_path / f"s{seed}.jpg", synth.camera(synth.scene(384, 512, seed=seed), seed=seed), 90))
    for path in images:
        res = trustmark.detect(Evidence(str(path)))
        assert res.findings[0].title == "No TrustMark watermark", path.name
