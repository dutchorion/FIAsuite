"""PDF report layout: every image fits the column and no image block crosses a page boundary."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from fiasuite import report
from fiasuite.core import Evidence, Finding, Result
from fiasuite.verdict import summarize
from tests import synth


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    ev = Evidence(synth.save(tmp_path_factory.mktemp("rep") / "e.jpg", synth.camera(synth.scene(240, 320))))
    rng = np.random.default_rng(0)
    shapes = [(300, 2400), (2600, 500), (900, 1200), (1500, 1500), (120, 160), (700, 3000), (2000, 640)] * 2
    results = {}
    for i, (h, w) in enumerate(shapes):                       # very wide, very tall and tiny views
        view = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        results[f"stress{i}"] = Result(views={"map": view, "second": view[: h // 2]}, score=0.1,
                                       findings=[Finding("info", "stress " * 30, "detail " * 60)],
                                       table=[("key", "0123456789abcdef" * 12)] * 4)
    return ev, results, summarize(results), {}


def _image_blocks(doc, height):
    layout, block = doc.documentLayout(), doc.begin()
    while block.isValid():
        it = block.begin()
        while not it.atEnd():
            fmt = it.fragment().charFormat()
            if fmt.isImageFormat():
                yield fmt.toImageFormat(), layout.blockBoundingRect(block)
            it += 1
        block = block.next()


def test_pdf_images_fit_and_stay_whole(qapp, case):
    doc, width, height = report.layout_pdf(lambda w, h: report.build_html(*case, page=(w, h)))
    blocks = list(_image_blocks(doc, height))
    assert len(blocks) == 29                                  # evidence preview + 14 tools x 2 views
    for image, rect in blocks:
        assert image.width() <= width and image.height() <= height * 0.72 + 1
        assert int(rect.top() // height) == int((rect.bottom() - 1) // height), "image block crosses a page boundary"


def test_aspect_ratio_is_kept(case):
    tag = report._img_tag(np.zeros((300, 2400, 3), np.uint8), 680, 700, 2.0)
    assert 'width="680" height="85"' in tag


def test_pdf_is_written(qapp, case, tmp_path):
    target = tmp_path / "r.pdf"
    pages = report.write_pdf(str(target), *case)
    assert pages >= 10 and target.read_bytes()[:5] == b"%PDF-"


def test_html_variant_keeps_hashes_copyable(case):
    page = report.build_html(*case)
    assert "&#8203;" not in page and case[0].hashes["sha256"] in page
