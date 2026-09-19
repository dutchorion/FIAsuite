"""Neural detectors through the ONNX path. Skipped when the (unversioned) models are not installed."""
from pathlib import Path

import pytest

from fiasuite.analyses import ai
from fiasuite.core import Evidence
from tests import synth

REGISTRY = {e["id"]: e for e in ai.load_registry()}
SAMPLES = sorted((ai.models_dir() / "_src" / "Community-Forensics" / "test_images").glob("*.png"))


def test_inference_defaults_to_cpu():
    """GPU (DirectML) inference froze a PC with integrated graphics: it must stay opt-in."""
    assert ai.use_gpu() is False
    for entry in REGISTRY.values():
        assert ai._session(entry).get_providers() == ["CPUExecutionProvider"]


@pytest.mark.skipif("trufor" not in REGISTRY, reason="TruFor model not installed")
def test_trufor_localises_the_splice(tmp_path):
    ev = Evidence(synth.build(Path(tmp_path))["splice"])
    sess = ai._session(REGISTRY["trufor"])
    import cv2
    pad = cv2.copyMakeBorder(ev.bgr, 0, 1024 - ev.height, 0, 0, cv2.BORDER_REFLECT)
    tamper, conf, score = ai._run_trufor(sess, pad)
    ys, xs = synth.REGION
    assert score > 0.7
    assert tamper[ys, xs].mean() > 0.9 and conf[ys, xs].mean() > 0.7
    assert tamper[600:, :].mean() < 0.3               # untouched strip below the forgery


@pytest.mark.skipif("commfor" not in REGISTRY or len(SAMPLES) < 5, reason="detector or sample images not installed")
def test_community_forensics_flags_generated_samples():
    scores = [ai.classify(Evidence(p), REGISTRY["commfor"], max_tiles=1)[0] for p in SAMPLES[:5]]
    assert sum(s > 0.5 for s in scores) >= 4
