"""Ground-truth checks: every detector must separate its forgery from the untouched original."""
from pathlib import Path

import numpy as np
import pytest

from fiasuite.analyses import clone, compression, interpolation, jpegstruct, noise, spectral
from fiasuite.core import Evidence
from tests import synth


@pytest.fixture(scope="session")
def files(tmp_path_factory):
    return synth.build(Path(tmp_path_factory.mktemp("evidence")))


@pytest.fixture(scope="session")
def ev(files):
    cache = {}
    return lambda name: cache.setdefault(name, Evidence(files[name]))


def test_jpeg_structure(ev):
    res = jpegstruct.run(ev("original"))
    rows = dict(res.table)
    assert "quality ~92, standard" in rows["Quantization table 0"]
    assert rows["Chroma subsampling"].startswith("4:2:0")


def test_ghosts(ev):
    assert compression.ghosts(ev("splice")).score > 0.5
    assert compression.ghosts(ev("original")).score < 0.3


def test_double_quantization(ev):
    assert compression.dct_analysis(ev("double")).score > 0.5
    assert compression.dct_analysis(ev("original")).score < 0.2


def test_grid(ev):
    assert compression.grid_offset(ev("original"))[:2] == (0, 0)
    dy, dx, t = compression.grid_offset(ev("cropped"))
    assert (dy, dx) == (5, 3) and t > 6           # 8 - crop offset
    assert compression.grid_offset(ev("original_png"))[2] < 6
    assert compression.block_grid(ev("splice")).score > compression.block_grid(ev("original")).score


def test_jpeg_history_in_png(ev):
    res = compression.dct_analysis(ev("cropped"))
    assert any("lossless" in f.title for f in res.findings)


def test_noise(ev):
    assert noise.noise_map(ev("smoothed")).score > 0.3
    assert noise.noise_map(ev("original_png")).score < 0.15


def test_resampling(ev):
    local = interpolation.resampling(ev("resampled"))
    assert local.score > 0.5 and "Locally" in local.findings[0].title
    assert "Whole image" in interpolation.resampling(ev("resized")).findings[0].title
    assert interpolation.resampling(ev("original_png")).score < 0.2


def test_cfa(ev):
    clean = interpolation.cfa(ev("original_png"))
    assert any("traces present" in f.title for f in clean.findings) and clean.score < 0.2
    assert interpolation.cfa(ev("resampled")).score > 0.3      # textured insert that never saw this sensor
    assert any("No demosaicing" in f.title for f in interpolation.cfa(ev("resized")).findings)


def test_clone(ev):
    assert clone.keypoints(ev("clone")).score > 0.5
    assert clone.keypoints(ev("original")).score < 0.2
    assert clone.blocks(ev("clone")).score > 0.5
    assert clone.blocks(ev("original")).score < 0.2


def test_spectral(ev):
    assert spectral.spectrum(ev("periodic")).score > 0.5
    assert spectral.spectrum(ev("original_png")).score < 0.3


def test_views_are_images(ev):
    for fn in (compression.ela, compression.ghosts, compression.block_grid, compression.dct_analysis,
               noise.residual, noise.noise_map, interpolation.resampling, interpolation.cfa, spectral.spectrum):
        for view in fn(ev("splice")).views.values():
            assert view.dtype == np.uint8 and view.ndim == 3 and view.shape[2] == 3
