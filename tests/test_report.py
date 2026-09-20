"""PDF report: sections start on new pages, continuation pages are titled, images fit the column and stay
whole, and - checked on the painted pages, not only on the layout - nothing disappears."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QImage, QPainter
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
                                       table=[("key", "0123456789abcdef" * 12)] * (60 if i == 3 else 4))
    return ev, results, summarize(results), {}


def _layout(case, include=None):
    ev, results, verdicts, params = case

    def make_html(page, breaks, placeholder):
        return report.build_html(ev, results, verdicts, params, "", "", include, page, breaks, placeholder)
    return report.layout_pdf(make_html)


def _ink(doc, width, height, page) -> float:
    img = QImage(int(width), int(height), QImage.Format_RGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    p.translate(0, -page * height)
    doc.drawContents(p, QRectF(0, page * height, width, height))
    p.end()
    a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4]
    return float((a.reshape(img.height(), img.width(), 4)[:, :, :3].min(axis=2) < 128).mean())


def test_images_fit_and_stay_whole(qapp, case):
    doc, width, height = _layout(case)
    images = [(b, r) for kind, _, b, r in report._blocks(doc) if kind == "image"]
    assert len(images) == 29                                  # evidence preview + 14 analyses x 2 views
    for _, rect in images:
        assert rect.width() <= width + 1 and rect.height() < height
        assert int((rect.top() + 0.5) // height) == int((rect.bottom() - 1) // height), "image block crosses a page boundary"


def test_every_analysis_starts_a_page_and_nothing_vanishes(qapp, case):
    doc, width, height = _layout(case)
    sections = [(b.text(), r) for kind, _, b, r in report._blocks(doc) if kind == "heading" and b.text().startswith("stress")]
    assert len(sections) == 14
    for title, rect in sections:
        assert (rect.top() + 0.5) % height < 40, f"{title} does not start at the top of a page"   # allows a paragraph margin
    assert doc.pageCount() >= 15
    # the regression this guards: editing a laid-out document made Qt stop painting paragraphs and images
    assert all(_ink(doc, width, height, page) > 0.0005 for page in range(doc.pageCount()))   # no blank page
    last = sections[-1][1]
    assert _ink(doc, width, height, int((last.top() + 0.5) // height)) > 0.01     # the last section's text is painted


def test_continuation_pages_are_titled(qapp, case):
    doc, _, height = _layout(case)
    continued = report.continued_titles(doc, height)
    starts = {int((r.top() + 0.5) // height) for kind, _, b, r in report._blocks(doc) if kind == "heading" and b.text().startswith("stress")}
    assert continued and not starts & set(continued)          # pages that open a section carry no running title
    assert "stress3" in continued.values()                    # the analysis with the 60-row table runs over


def test_selection_limits_the_report(qapp, case):
    doc, _, _ = _layout(case, include={"stress1", "stress5"})
    titles = [b.text() for kind, _, b, _ in report._blocks(doc) if kind == "heading"]
    assert [t for t in titles if t.startswith("stress")] == ["stress1", "stress5"]
    assert set(report.build_json(*case, include={"stress1"})["tools"]) == {"stress1"}


def test_aspect_ratio_is_kept():
    tag = report._img_tag(np.zeros((300, 2400, 3), np.uint8), 680, 700, 2.0)
    assert 'width="680" height="85"' in tag


def test_pdf_is_written(qapp, case, tmp_path):
    target = tmp_path / "r.pdf"
    pages = report.write_pdf(str(target), *case)
    assert pages >= 15 and target.read_bytes()[:5] == b"%PDF-"


def test_html_variant_keeps_hashes_copyable(case):
    page = report.build_html(*case)
    assert "&#8203;" not in page and "page-break" not in page and case[0].hashes["sha256"] in page
