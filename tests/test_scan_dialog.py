"""Scan dialog: nothing runs when an image is opened, the dialog runs exactly what was ticked, and group check
boxes tick their whole group. Sightengine: every score in the reply is shown, and the reply is kept per file
hash so reopening the file shows it again without a new upload."""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

from fiasuite.analyses import online
from fiasuite.core import Evidence
from fiasuite.ui import dialogs
from fiasuite.ui.main import ALL_TOOLS, MainWindow, result_color
from tests import synth

REPLY = {"status": "success", "type": {"ai_generated": 0.97,
                                       "ai_generators": {"diffusion": 0.9, "gan": 0.06, "other": 0.04}}}


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def app_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(online, "app_dir", lambda: tmp_path)          # config and stored replies live here
    return tmp_path


@pytest.fixture()
def image(tmp_path):
    return synth.save(tmp_path / "photo.jpg", synth.camera(synth.scene(256, 320)), 90)


@pytest.fixture()
def window(qapp, app_dir):
    w = MainWindow()
    yield w
    w.pool.clear()
    w.pool.waitForDone()
    w.close()


def choose(monkeypatch, ids, upload=False):
    """Stand-in for the user: tick these tools in the dialog, and answer the upload question."""
    monkeypatch.setattr(dialogs.ScanDialog, "exec", lambda self: QDialog.Accepted)
    monkeypatch.setattr(dialogs.ScanDialog, "selected", lambda self: set(ids))
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes if upload else QMessageBox.No)


def test_opening_an_image_runs_nothing(window, image):
    window.open_file(str(image))
    assert not window.pending and not window.results
    assert window.btn_scan.text() == "Scan ..."


def test_scan_runs_only_the_ticked_tools(window, image, monkeypatch):
    window.open_file(str(image))
    choose(monkeypatch, {"hashes", "histogram"})
    window.scan()
    assert window.pending == {"hashes", "histogram"}
    assert window.btn_scan.text() == "Re-run scan ..."


def test_declined_upload_runs_the_rest(window, image, monkeypatch):
    window.open_file(str(image))
    choose(monkeypatch, {"hashes", "online:Sightengine"}, upload=False)
    window.scan()
    assert window.pending == {"hashes"}


def test_dialog_groups_select_all_and_none(qapp):
    dialog = dialogs.ScanDialog(ALL_TOOLS, {"metadata"})
    assert dialog.selected() == {"metadata"}
    provenance = next(g for g in dialog.groups if g.text(0) == "File & provenance")
    assert provenance.checkState(0) == Qt.PartiallyChecked
    provenance.setCheckState(0, Qt.Checked)                           # the group box ticks the whole group
    assert dialog.selected() == {t.id for t in ALL_TOOLS if t.group == "File & provenance"}
    dialog.set_all(Qt.Unchecked)
    assert dialog.selected() == set() and not dialog.run.isEnabled()
    dialog.set_all(Qt.Checked)
    assert dialog.selected() == {t.id for t in ALL_TOOLS} and dialog.run.isEnabled()


def test_every_sightengine_score_is_shown():
    res = online.sightengine_result(REPLY)
    assert res.score == pytest.approx(0.97)
    table = dict(res.table)
    assert table["  diffusion"] == "90.0%" and table["  gan"] == "6.0%" and table["  other"] == "4.0%"
    assert "diffusion 90%" in res.findings[0].detail


def test_sightengine_reply_is_kept_and_recalled(app_dir, image, monkeypatch):
    online.save_config({"sightengine": {"api_user": "u", "api_secret": "s"}})

    class Reply:
        def json(self):
            return REPLY
    monkeypatch.setattr(online.requests, "post", lambda *a, **k: Reply())
    ev = Evidence(str(image))
    online.sightengine(ev)
    monkeypatch.setattr(online.requests, "post", lambda *a, **k: pytest.fail("recall must not upload"))
    recalled = online.recall_sightengine(Evidence(str(image)))
    assert recalled.score == pytest.approx(0.97)
    assert any(f.title.startswith("Stored Sightengine result") for f in recalled.findings)


def test_reopened_image_shows_the_stored_result(window, image):
    online.remember(Evidence(str(image)), "Sightengine", REPLY)
    window.open_file(str(image))
    assert window.results["online:Sightengine"].score == pytest.approx(0.97)
    assert not window.pending


def finish(qapp, window, seconds=60):
    end = time.monotonic() + seconds
    while window.pending and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not window.pending, "analyses did not finish"


def test_scan_shows_progress_and_outlines_finished_tests(qapp, window, image, monkeypatch):
    window.open_file(str(image))
    assert window.progress.isHidden()                                  # nothing has run on this image
    choose(monkeypatch, {"hashes", "histogram"})
    window.scan()
    dialog = window.progress_dialog
    assert dialog.isVisible() and list(dialog.widgets) == ["hashes", "histogram"]
    assert window.progress.isVisibleTo(window) and window.progress.isTextVisible()
    assert window.progress.text() == "in progress"
    finish(qapp, window)
    for tool_id in ("hashes", "histogram"):
        frame, _, bar = dialog.widgets[tool_id]
        assert result_color(window.results[tool_id]) in frame.styleSheet()
        assert bar.value() == bar.maximum() == 1
    assert dialog.summary.text() == "2 of 2 analyses finished"
    assert not window.progress.isTextVisible()                        # the text only shows while something runs
    assert window.progress.value() == window.progress.maximum() == 2


def test_progress_bar_click_reopens_the_overview(qapp, window, image, monkeypatch):
    window.open_file(str(image))
    choose(monkeypatch, {"hashes"})
    window.scan()
    finish(qapp, window)
    window.progress_dialog.close()
    QTest.mouseClick(window.progress, Qt.LeftButton)
    assert window.progress_dialog.isVisible()
    window.select_tool("histogram")                                   # a single run joins the scan's overview
    assert list(window.progress_dialog.widgets) == ["hashes", "histogram"]
    finish(qapp, window)
