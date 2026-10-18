"""Neural detectors through the ONNX path. Skipped when the (unversioned) models are not installed."""
from pathlib import Path

import pytest

from fiasuite.analyses import ai
from fiasuite.core import Evidence
from tests import synth

REGISTRY = {e["id"]: e for e in ai.load_registry()}
SAMPLES = sorted((ai.models_dir() / "_src" / "Community-Forensics" / "test_images").glob("*.png"))


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """The tests must not depend on the settings of whoever runs them (config.json may enable the GPU)."""
    monkeypatch.setattr(ai, "use_gpu", lambda: False)


def test_gpu_is_opt_in(tmp_path, monkeypatch):
    """GPU (DirectML) inference froze a PC with integrated graphics: without a saved setting it must stay off."""
    monkeypatch.undo()                                        # look at the real function
    monkeypatch.setattr(ai, "app_dir", lambda: tmp_path)      # a fresh installation: no config.json
    assert ai.use_gpu() is False
    (tmp_path / "config.json").write_text('{"use_gpu": true}', encoding="utf-8")
    assert ai.use_gpu() is True


def test_cpu_sessions_when_gpu_is_off():
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


# ------------------------------------------------------------------ GPU self-check (stand-in sessions, no GPU needed)

class _FakeSession:
    class _Input:
        name, type = "x", "tensor(float)"

    def __init__(self, transform):
        self.transform = transform

    def get_inputs(self):
        return [self._Input()]

    def run(self, _names, feed):
        return [self.transform(feed["x"].mean(axis=(2, 3)))]


ENTRY = {"id": "fake", "name": "Fake", "size": 8}


def test_gpu_selfcheck_accepts_matching_results():
    cpu, gpu = _FakeSession(lambda v: v), _FakeSession(lambda v: v * 1.0001)
    assert ai.gpu_mismatch(gpu, cpu, ENTRY) is None


def test_gpu_selfcheck_rejects_wrong_results():
    import numpy as np
    cpu = _FakeSession(lambda v: v)
    assert "differ" in ai.gpu_mismatch(_FakeSession(lambda v: v * 0.2), cpu, ENTRY)
    assert "non-finite" in ai.gpu_mismatch(_FakeSession(lambda v: v * np.nan), cpu, ENTRY)


def test_gpu_selfcheck_rejects_a_crawling_gpu(monkeypatch):
    import time
    ticks = iter([0.0, 1.0, 1.0, 30.0])                      # cpu: 1 s, gpu: 29 s
    monkeypatch.setattr(time, "perf_counter", lambda: next(ticks))
    cpu, gpu = _FakeSession(lambda v: v), _FakeSession(lambda v: v)
    assert "needed 29 s" in ai.gpu_mismatch(gpu, cpu, ENTRY)


@pytest.mark.skipif("omniaid" not in REGISTRY, reason="OmniAID not installed")
def test_whole_image_detector_is_reported_but_not_counted(tmp_path):
    ev = Evidence(synth.save(tmp_path / "photo.jpg", synth.camera(synth.scene(300, 400)), 90))
    score, grid, peak = ai.classify(ev, REGISTRY["omniaid"])
    assert grid.shape == (1, 1) and score == peak                   # one pass over the whole picture, no map
    res = ai.detect(ev, max_tiles=1)
    row = dict(res.table)[REGISTRY["omniaid"]["name"]]
    assert "whole image only" in row and "not counted" in row
    assert REGISTRY["omniaid"]["name"] not in res.views
