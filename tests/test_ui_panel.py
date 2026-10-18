"""Analysis panel: the table search hides the rows that do not match, and a click on a parameter
slider jumps to that point instead of paging towards it."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QPushButton, QSlider, QSpinBox,
                               QTableWidgetItem, QToolButton)

from fiasuite.tools import BY_ID, defaults
from fiasuite.ui.main import ClickSlider, MainWindow
from fiasuite.ui.viewer import MAX_ZOOM, MIN_ZOOM, ZOOM_STEP, ImageView


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def window(qapp):
    w = MainWindow()
    w.table.setRowCount(2)
    for row, (prop, value) in enumerate([("GPSLatitude", "52.1 N"), ("Make", "Canon")]):
        w.table.setItem(row, 0, QTableWidgetItem(prop))
        w.table.setItem(row, 1, QTableWidgetItem(value))
    yield w
    w.close()


def hidden(window):
    return [window.table.isRowHidden(i) for i in range(window.table.rowCount())]


def test_search_matches_property_and_value(window):
    window.table_search.setText("gpslat")                 # property, any case
    assert hidden(window) == [False, True]
    window.table_search.setText("canon")                  # value, any case
    assert hidden(window) == [True, False]
    window.table_search.setText("Nikon")
    assert hidden(window) == [True, True]


def test_empty_search_shows_every_row(window):
    window.table_search.setText("canon")
    window.table_search.clear()
    assert hidden(window) == [False, False]


def test_every_slider_jumps_to_the_click(window):
    sliders = window.findChildren(QSlider)
    assert sliders and all(isinstance(s, ClickSlider) for s in sliders)


def click(slider, x):
    point = slider.rect().center()
    point.setX(x)
    event = QMouseEvent(QMouseEvent.Type.MouseButtonPress, point, slider.mapToGlobal(point),
                        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(slider, event)


@pytest.mark.parametrize("slider_type, jumps", [(ClickSlider, True), (QSlider, False)])
def test_click_on_the_groove_jumps_to_that_point(qapp, slider_type, jumps):
    slider = slider_type(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(0)
    slider.resize(200, 20)
    click(slider, slider.rect().center().x())
    assert (abs(slider.value() - 50) <= 2) is jumps      # the plain slider only pages a step towards the click
    slider.deleteLater()


def test_click_on_the_handle_leaves_the_value_alone(qapp):
    slider = ClickSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)
    slider.resize(200, 20)
    click(slider, slider.rect().center().x())
    assert slider.value() == 50


def option_buttons(window):
    resets = [b for b in window.form_host.findChildren(QToolButton) if b.toolTip().startswith("Reset to")]
    reset_all = next(b for b in window.form_host.findChildren(QPushButton) if b.text() == "Reset all")
    return resets, reset_all


def test_reset_buttons_are_live_only_for_changed_options(window):
    window.select_tool("ela")
    resets, reset_all = option_buttons(window)
    assert len(resets) == len(BY_ID["ela"].params)
    assert not any(b.isEnabled() for b in resets) and not reset_all.isEnabled()
    window.form_host.findChildren(QCheckBox)[0].setChecked(False)        # "Automatic error scale" defaults to on
    assert [b.isEnabled() for b in resets] == [p.key == "auto_scale" for p in BY_ID["ela"].params]
    assert reset_all.isEnabled()
    next(b for b in resets if b.isEnabled()).click()
    assert window.params["ela"]["auto_scale"] is True
    assert not any(b.isEnabled() for b in resets) and not reset_all.isEnabled()


def test_reset_all_restores_every_kind_of_option(window):
    window.select_tool("clone_kp")                    # int sliders, a float box and a check box
    window.form_host.findChildren(QSpinBox)[0].setValue(200)
    window.form_host.findChildren(QDoubleSpinBox)[0].setValue(0.75)
    window.form_host.findChildren(QCheckBox)[0].setChecked(True)
    assert window.params["clone_kp"] != defaults(BY_ID["clone_kp"])
    option_buttons(window)[1].click()
    assert window.params["clone_kp"] == defaults(BY_ID["clone_kp"])
    window.select_tool("ela")                         # a combo box
    window.form_host.findChildren(QComboBox)[0].setCurrentIndex(1)
    option_buttons(window)[1].click()
    assert window.params["ela"] == defaults(BY_ID["ela"])


@pytest.fixture()
def view(qapp):
    v = ImageView()
    v.resize(800, 600)
    v.set_images(np.zeros((300, 400, 3), np.uint8))
    v.actual_size()
    return v


def test_one_zoom_step_is_one_wheel_notch(view):
    view.zoom_step(1)
    assert view.zoom_factor() == pytest.approx(ZOOM_STEP)
    view.zoom_step(-1)
    assert view.zoom_factor() == pytest.approx(1)


def test_zoom_stays_between_5_and_1000_percent(view):
    for _ in range(40):
        view.zoom_step(1)
    assert view.zoom_factor() == pytest.approx(MAX_ZOOM)
    for _ in range(80):
        view.zoom_step(-1)
    assert view.zoom_factor() == pytest.approx(MIN_ZOOM)


def test_fitted_tiny_image_zooms_out_by_one_step_not_to_the_limit(qapp):
    v = ImageView()
    v.resize(800, 800)
    v.set_images(np.zeros((8, 8, 3), np.uint8))                     # fitted far above 1000 %
    fitted = v.zoom_factor()
    v.zoom_step(1)
    assert v.zoom_factor() == pytest.approx(fitted)                  # no further in
    v.zoom_step(-1)
    assert v.zoom_factor() == pytest.approx(fitted / ZOOM_STEP)      # one normal step out


def test_zoom_buttons_step_like_the_wheel(window):
    window.stack.setCurrentIndex(1)
    window.viewer.set_images(np.zeros((300, 400, 3), np.uint8))
    window.viewer.actual_size()
    buttons = {b.text(): b for b in window.findChildren(QPushButton)}
    buttons["+"].click()
    assert window.viewer.zoom_factor() == pytest.approx(ZOOM_STEP)
    buttons["−"].click()
    buttons["−"].click()
    assert window.viewer.zoom_factor() == pytest.approx(1 / ZOOM_STEP)


def test_window_starts_at_most_of_the_screen(window):
    screen = QApplication.primaryScreen().availableGeometry()
    assert window.width() == int(screen.width() * 0.9) and window.height() == int(screen.height() * 0.9)


def test_watermark_check_only_opens_the_synthid_page(window, monkeypatch):
    from fiasuite.ui import main as main_module
    opened = []
    monkeypatch.setattr(main_module.webbrowser, "open", opened.append)
    monkeypatch.setattr(main_module.QMessageBox, "information", lambda *a: pytest.fail("no pop-up expected"))
    QApplication.clipboard().setText("left alone")
    window.btn_marks.setEnabled(True)
    window.btn_marks.click()
    assert opened == ["https://synthid.com/"]
    assert QApplication.clipboard().text() == "left alone"
