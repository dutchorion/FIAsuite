"""Analysis panel: the table search hides the rows that do not match, and a click on a parameter
slider jumps to that point instead of paging towards it."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QSlider, QTableWidgetItem

from fiasuite.ui.main import ClickSlider, MainWindow


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
