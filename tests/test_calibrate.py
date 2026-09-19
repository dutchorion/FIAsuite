"""Laundering chains and the calibration statistics."""
import numpy as np
import pytest

from fiasuite import calibrate, launder
from fiasuite.core import Evidence
from tests import synth


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return synth.build(tmp_path_factory.mktemp("cal"))


def test_parse_chain():
    assert launder.parse_chain("none") == []
    assert launder.parse_chain("resize0.5+jpeg75") == [("resize", 0.5), ("jpeg", 75.0)]
    with pytest.raises(ValueError):
        launder.parse_chain("blur3")


def test_apply(files):
    ev = Evidence(files["original_png"])
    out = launder.apply(ev, "resize0.5+jpeg75")
    assert out.is_jpeg and (out.width, out.height) == (ev.width // 2, ev.height // 2)
    assert launder.apply(ev, "resize0.5").is_png
    assert launder.apply(ev, "none") is ev


def test_roc_summary():
    real, fake = np.arange(100.0), np.arange(100.0) + 80
    s = calibrate.roc_summary(real, fake)
    assert 0.9 < s["auc"] < 1 and s["thr@0.01"] == 99 and s["tpr@0.01"] == pytest.approx(0.8)
    assert calibrate.roc_summary(real, real)["auc"] == pytest.approx(0.5)


def test_laundering_destroys_the_planted_trace(files):
    """The point of the harness: a detector can be perfect on pristine files and useless after a platform round-trip."""
    chains = ["none", "resize0.5+jpeg75"]
    fake = calibrate.measure(str(files["periodic"]), "spectrum", chains)
    real = calibrate.measure(str(files["original_png"]), "spectrum", chains)
    assert fake["none"]["n_peaks_other"] >= 2 and real["none"]["n_peaks_other"] == 0
    assert fake["resize0.5+jpeg75"]["n_peaks_other"] == 0
    rows = calibrate.summarize([real], [fake], chains)
    best = {(r["laundering"], r["metric"]): r["auc"] for r in rows}
    assert best[("none", "n_peaks_other")] == 1.0 and best[("resize0.5+jpeg75", "n_peaks_other")] == 0.5
