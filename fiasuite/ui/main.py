"""Main window: tool tree, viewer, parameter/finding panel and the dashboard."""
from __future__ import annotations

import html
import sys
import traceback
import webbrowser
from pathlib import Path

import cv2
from PySide6.QtCore import Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QDoubleSpinBox,
                               QFileDialog, QFormLayout,
                               QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMenu, QMessageBox,
                               QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider, QSpinBox, QSplitter,
                               QStackedWidget, QStyle, QStyleOptionSlider, QTableWidget, QTableWidgetItem, QTextBrowser,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .. import APP_NAME, __version__, components
from ..analyses import geometry, online
from ..core import IMAGE_EXTS, SEVERITY_ORDER, Evidence, Result, app_dir
from ..tools import TOOLS, Tool, defaults
from ..verdict import headline, summarize
from . import theme
from .viewer import ImageView, to_pixmap
from .worker import Job, Signals

REVERSE_SEARCH = {"Google Lens": "https://lens.google.com/", "TinEye": "https://tineye.com/",
                  "Bing Visual Search": "https://www.bing.com/visualsearch", "Yandex Images": "https://yandex.com/images/"}

# sites without a usable API: the image goes on the clipboard and the page opens; the user pastes it there
PROVENANCE_SITES = {
    "Google Gemini (checks for Google's SynthID watermark)": (
        "https://gemini.google.com/app",
        "paste the image with Ctrl+V and ask:  Was this image created or edited with Google AI?\n\n"
        "Gemini looks for Google's invisible SynthID watermark. It recognises content from Google's own tools only."),
    "Google SynthID Detector (portal, access by waitlist)": (
        "https://deepmind.google/models/synthid/",
        "open the SynthID Detector from this page (journalists and researchers can join the waitlist) and paste or "
        "upload the image there."),
    "OpenAI verify page (no API key needed)": (
        "https://openai.com/research/verify/",
        "paste or upload the image. The page checks for OpenAI's Content Credentials and SynthID watermark, the same "
        "check as the 'OpenAI provenance check' tool."),
}

ONLINE_TOOLS = [Tool(f"online:{name}", f"{name} (online)", "Online services", fn,
                     f"{about} Nothing is sent until you press the button; the API key is set under Settings.",
                     axis="ai", weight=1.2, manual=True)
                for name, (fn, about) in online.PROVIDERS.items()]
ALL_TOOLS = TOOLS + ONLINE_TOOLS


def dot(color: str) -> QIcon:
    pix = QPixmap(14, 14)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.NoPen)
    p.drawEllipse(2, 2, 10, 10)
    p.end()
    return QIcon(pix)


def result_color(res: Result | None) -> str:
    if res is None:
        return "#3a3d44"
    worst = res.worst()
    if res.score is not None and res.score >= 0.55 and SEVERITY_ORDER[worst] < 2:
        worst = "warn"
    return {"info": "#46a758", "note": "#f5d90a", "warn": "#f76b15", "alert": "#e5484d"}[worst]


class ClickSlider(QSlider):
    """Horizontal slider that jumps to the clicked point instead of paging towards it."""

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)
            style, point = self.style(), event.position().toPoint()
            handle = style.subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderHandle, self)
            if not handle.contains(point):
                groove = style.subControlRect(QStyle.CC_Slider, opt, QStyle.SC_SliderGroove, self)
                span = groove.width() - handle.width()
                offset = point.x() - groove.x() - handle.width() // 2
                self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), offset, span,
                                                             opt.upsideDown))
        super().mousePressEvent(event)      # the handle now sits under the cursor: dragging continues from there


class ClickProgressBar(QProgressBar):
    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()


class ProgressDialog(QDialog):
    """Progress of every analysis since the last scan started. A finished analysis is outlined in the colour of
    its result, a failed one in grey."""

    def __init__(self, main: "MainWindow"):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Scan progress")
        self.setModal(True)
        self.resize(440, 600)
        lay = QVBoxLayout(self)
        self.summary = QLabel("")
        self.summary.setObjectName("h2")
        self.overall = QProgressBar()
        self.overall.setTextVisible(False)
        body = QWidget()
        self.rows = QVBoxLayout(body)
        self.rows.setContentsMargins(0, 0, 4, 0)
        self.rows.setSpacing(6)
        self.rows.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(body)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(self.summary)
        lay.addWidget(self.overall)
        lay.addWidget(scroll, 1)
        lay.addWidget(buttons)
        self.widgets: dict[str, tuple[QFrame, QLabel, QProgressBar]] = {}
        self.refresh()

    def _row(self, tool_id: str):
        frame = QFrame()
        frame.setObjectName("run")
        row = QVBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        title = QLabel(next(t.title for t in ALL_TOOLS if t.id == tool_id))
        status = QLabel("")
        status.setObjectName("muted")
        bar = QProgressBar()
        bar.setTextVisible(False)
        row.addWidget(title)
        row.addWidget(status)
        row.addWidget(bar)
        self.rows.insertWidget(self.rows.count() - 1, frame)        # above the stretch
        self.widgets[tool_id] = frame, status, bar

    def refresh(self):
        log = self.main.run_log
        if list(self.widgets) != list(log):                          # a new round of analyses: rebuild the list
            for frame, _, _ in self.widgets.values():
                frame.deleteLater()
            self.widgets.clear()
            for tool_id in log:
                self._row(tool_id)
        for tool_id, state in log.items():
            frame, status, bar = self.widgets[tool_id]
            if state == "done":
                res = self.main.results.get(tool_id)
                top = max(res.findings, key=lambda f: SEVERITY_ORDER[f.severity], default=None) if res else None
                score = "" if res is None or res.score is None else f"  ·  {res.score:.2f}"
                status.setText((top.title if top else "done") + score)
                border = result_color(res)
            elif state == "failed":
                error = self.main.errors.get(tool_id, "").strip().splitlines()
                status.setText("failed: " + error[-1][:90] if error else "failed")
                border = "#8b8d98"
            else:
                status.setText(state)
                border = theme.BORDER
            bar.setRange(0, 0 if state == "running" else 1)          # 0..0: busy animation while it runs
            bar.setValue(1 if state in ("done", "failed") else 0)
            frame.setStyleSheet(f"QFrame#run {{ background: {theme.PANEL}; border: 2px solid {border}; "
                                "border-radius: 8px; }")
        done = sum(state in ("done", "failed") for state in log.values())
        self.overall.setRange(0, max(len(log), 1))
        self.overall.setValue(done)
        self.summary.setText(f"{done} of {len(log)} analyses finished" if log else "No analyses have run yet.")


class Card(QFrame):
    clicked = Signal(str)
    SIDE = 250
    THUMB = (228, 146)                                          # box the preview is fitted into

    def __init__(self, tool: Tool):
        super().__init__()
        self.setObjectName("card")
        self.tool_id = tool.id
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(self.SIDE, self.SIDE)                  # square
        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        self.thumb = QLabel()
        self.thumb.setFixedSize(*self.THUMB)
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setTextFormat(Qt.PlainText)                  # finding titles can quote text from the file
        self.thumb.setStyleSheet(f"background:{theme.BG}; border-radius:6px; color:{theme.MUTED};")
        self.title = QLabel(tool.title)
        self.title.setObjectName("h2")
        self.status = QLabel("waiting")
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        self.status.setAlignment(Qt.AlignTop)
        lay.addWidget(self.thumb)
        lay.addWidget(self.title)
        lay.addWidget(self.status, 1)

    def set_idle(self):
        self.thumb.setText("not run")
        self.status.setText("not run - choose it under Scan")

    def set_running(self):
        self.thumb.setText("analysing ...")
        self.status.setText("running")

    def set_result(self, res: Result | None, error: str = ""):
        if res is None:
            self.thumb.setText("failed")
            self.status.setText(error.strip().splitlines()[-1][:90] if error else "failed")
            return
        if res.views:
            view = next(iter(res.views.values()))
            # fit inside the box with the original aspect ratio (letterboxed), at the screen's pixel density
            dpr = self.devicePixelRatioF()
            h, w = view.shape[:2]
            scale = min(self.THUMB[0] * dpr / w, self.THUMB[1] * dpr / h)
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            pix = to_pixmap(cv2.resize(view, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR))
            pix.setDevicePixelRatio(dpr)
            self.thumb.setPixmap(pix)
        else:                                      # text-only tool: show its findings instead of a picture
            lines = [f.title for f in sorted(res.findings, key=lambda f: -SEVERITY_ORDER[f.severity])[:4]]
            self.thumb.setWordWrap(True)
            self.thumb.setText("\n".join(lines) or "no findings")
        top = max(res.findings, key=lambda f: SEVERITY_ORDER[f.severity], default=None)
        color = result_color(res)
        score = "" if res.score is None else f"  ·  {res.score:.2f}"
        self.status.setText(f'<span style="color:{color}">&#9679;</span> {html.escape(top.title if top else "done")}{score}')

    def mousePressEvent(self, event):
        self.clicked.emit(self.tool_id)


class Dashboard(QScrollArea):
    tool_requested = Signal(str)

    def __init__(self):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        self.setWidget(body)
        lay = QVBoxLayout(body)
        lay.setContentsMargins(24, 20, 24, 24)
        lay.setSpacing(14)
        self.title = QLabel("Drop an image here or press Open")
        self.title.setObjectName("h1")
        self.subtitle = QLabel(f"{APP_NAME} {__version__} - forensic image analysis. All processing is local unless you "
                               "use an online service.")
        self.subtitle.setObjectName("muted")
        self.subtitle.setWordWrap(True)
        self.headline = QLabel("")
        self.headline.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.subtitle)
        lay.addWidget(self.headline)
        row = QHBoxLayout()
        self.axis_cards: dict[str, tuple[QLabel, QLabel]] = {}
        for axis, label in (("ai", "AI generation"), ("edit", "Local manipulation"), ("provenance", "File history")):
            frame = QFrame()
            frame.setObjectName("card")
            v = QVBoxLayout(frame)
            name = QLabel(label)
            name.setObjectName("h2")
            level = QLabel("-")
            level.setStyleSheet("font-size:15pt; font-weight:600;")
            reasons = QLabel("")
            reasons.setWordWrap(True)
            reasons.setObjectName("muted")
            reasons.setAlignment(Qt.AlignTop)
            reasons.setMinimumHeight(96)
            v.addWidget(name)
            v.addWidget(level)
            v.addWidget(reasons, 1)
            row.addWidget(frame, 1)
            self.axis_cards[axis] = (level, reasons)
        lay.addLayout(row)
        self.grid = QGridLayout()
        self.grid.setSpacing(12)
        self.cards: dict[str, Card] = {}
        for tool in [t for t in TOOLS if t.auto]:
            card = Card(tool)
            card.clicked.connect(self.tool_requested)
            self.cards[tool.id] = card
        lay.addLayout(self.grid)
        lay.addStretch(1)
        self._columns = 0

    def resizeEvent(self, event):
        super().resizeEvent(event)
        columns = max(1, (self.viewport().width() - 48) // 262)
        if columns != self._columns:
            self._columns = columns
            for i, card in enumerate(self.cards.values()):
                self.grid.addWidget(card, i // columns, i % columns, Qt.AlignLeft | Qt.AlignTop)

    def set_evidence(self, ev: Evidence):
        self.title.setText(ev.path.name)
        self.subtitle.setText(f"{ev.width} x {ev.height} px  ·  {len(ev.data):,} bytes  ·  SHA-256 {ev.hashes['sha256']}")
        self.headline.setText("No analyses have run yet. Press Scan to choose them.")
        for level, reasons in self.axis_cards.values():
            level.setText("-")
            level.setStyleSheet("font-size:15pt; font-weight:600;")
            reasons.setText("")
        for card in self.cards.values():
            card.thumb.clear()
            card.set_idle()

    def set_verdicts(self, verdicts, pending: int):
        self.headline.setText(headline(verdicts) if pending == 0 else f"Analysing ... {pending} tools still running.")
        for axis, (level, reasons) in self.axis_cards.items():
            v = verdicts[axis]
            level.setText(v.level)
            level.setStyleSheet(f"font-size:15pt; font-weight:600; color:{v.color};")
            lines = [f'<span style="color:{theme.SEVERITY_COLORS[s]}">&#9679;</span> {html.escape(title)} '
                     f'<span style="color:#6b6e76">({html.escape(tool)})</span>' for s, tool, title in v.reasons[:5]]
            reasons.setText("<br>".join(lines) or "No findings.")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1560, 960)
        self.setAcceptDrops(True)
        self.ev: Evidence | None = None
        self.generation = 0
        self.results: dict[str, Result] = {}
        self.errors: dict[str, str] = {}
        self.params: dict[str, dict] = {t.id: defaults(t) for t in ALL_TOOLS}
        self.pending: set[str] = set()
        self.current: Tool | None = None
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(3)
        self.signals = Signals()
        self.signals.done.connect(self._on_done)
        self.signals.started.connect(self._on_started)
        self.run_log: dict[str, str] = {}                        # tool id -> waiting | running | done | failed
        self.progress_dialog: ProgressDialog | None = None
        self.result_params: dict[str, dict] = {}                 # options each stored result was computed with
        self.scan_selection = {t.id for t in TOOLS if t.auto}    # last choice in the scan dialog
        self.lines: dict[str, list] = {name: [] for name in geometry.SETS}   # examiner-drawn lines per set
        self._dirty: Tool | None = None                          # tool whose options changed, awaiting the debounce
        self.debounce = QTimer(self, singleShot=True, interval=350)
        self.debounce.timeout.connect(self._run_dirty)

        self._build_toolbar()
        self._build_tree()
        self._build_center()
        self._build_details()
        self.progress = ClickProgressBar()
        self.progress.setObjectName("status")
        self.progress.setFixedWidth(180)
        self.progress.setFormat("in progress")
        self.progress.setCursor(Qt.PointingHandCursor)
        self.progress.setToolTip("Show the progress of every analysis")
        self.progress.clicked.connect(self.show_progress)
        self.progress.hide()
        self.pixel_label = QLabel("")
        self.statusBar().addPermanentWidget(self.pixel_label)
        self.statusBar().addPermanentWidget(self.progress)
        self._set_enabled(False)
        widths = online.load_config().get("sidebars", [290, 440])      # remembered from the last session
        QTimer.singleShot(0, lambda: self.resizeDocks([self.tree_dock, self.details_dock], list(widths), Qt.Horizontal))

    # ------------------------------------------------------------------ construction
    def _build_toolbar(self):
        bar = self.addToolBar("Main")
        bar.setMovable(False)
        bar.setContextMenuPolicy(Qt.PreventContextMenu)

        def button(text, slot=None, primary=False, menu=None, tip=""):
            b = QToolButton()
            b.setText(text)
            b.setToolTip(tip)
            if primary:
                b.setObjectName("primary")
            if menu:
                b.setMenu(menu)
                b.setPopupMode(QToolButton.InstantPopup)
            if slot:
                b.clicked.connect(slot)
            bar.addWidget(b)
            return b

        button("Open image", self.open_dialog, primary=True, tip="Ctrl+O, or drop a file on the window")
        self.btn_scan = button("Scan ...", self.scan, primary=True, tip="Choose the analyses to run on this image")
        bar.addSeparator()
        self.btn_compare = button("Compare ...", self.open_compare, tip="Compare with a second version of the picture")
        button("Batch triage ...", self.open_batch, tip="Screen a whole folder")
        button("Camera fingerprint ...", self.open_fingerprints,
               tip="Build a sensor-noise (PRNU) fingerprint from reference photos of one camera")
        button("Calibrate ...", self.open_calibration,
               tip="Measure how well a detector separates your own authentic and manipulated images")
        self.btn_report = button("Export report ...", self.export_report)
        bar.addSeparator()
        menu = QMenu(self)
        for name, url in REVERSE_SEARCH.items():
            menu.addAction(name, lambda url=url, name=name: self.reverse_search(name, url))
        self.btn_reverse = button("Reverse image search", menu=menu,
                                  tip="Copies the image to the clipboard and opens the search page: press Ctrl+V there")
        marks = QMenu(self)
        for name, (url, how) in PROVENANCE_SITES.items():
            marks.addAction(name, lambda url=url, name=name, how=how: self.open_site_with_image(name, url, how))
        self.btn_marks = button("Watermark check", menu=marks,
                                tip="Google SynthID and OpenAI provenance checks that only exist as web pages")
        button("Settings ...", self.open_settings)
        button("Setup ...", self.open_setup, tip="Install ExifTool and the neural models")
        self.addAction(QAction(self, shortcut=QKeySequence.Open, triggered=self.open_dialog))

    def _build_tree(self):
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.items: dict[str, QTreeWidgetItem] = {}
        self.home = QTreeWidgetItem(self.tree, ["Dashboard"])
        groups: dict[str, QTreeWidgetItem] = {}
        for tool in ALL_TOOLS:
            if tool.group not in groups:
                groups[tool.group] = QTreeWidgetItem(self.tree, [tool.group])
                groups[tool.group].setFlags(Qt.ItemIsEnabled)
                groups[tool.group].setForeground(0, QColor(theme.MUTED))
            item = QTreeWidgetItem(groups[tool.group], [tool.title])
            item.setData(0, Qt.UserRole, tool.id)
            item.setIcon(0, dot("#3a3d44"))
            item.setToolTip(0, tool.help)
            self.items[tool.id] = item
        self.tree.expandAll()
        self.tree.currentItemChanged.connect(self._on_tree)
        dock = QDockWidget("Tools")
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(self.tree)
        dock.setMinimumWidth(160)                  # drag the separator to resize
        self.tree_dock = dock
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

    def _build_center(self):
        self.stack = QStackedWidget()
        self.dashboard = Dashboard()
        self.dashboard.tool_requested.connect(self.select_tool)
        self.stack.addWidget(self.dashboard)

        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 6, 10, 6)
        self.view_combo = QComboBox()
        self.view_combo.setMinimumWidth(240)
        self.view_combo.currentIndexChanged.connect(lambda: self._show_view(keep=True))
        self.opacity = ClickSlider(Qt.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(100)
        self.opacity.setFixedWidth(160)
        self.opacity.valueChanged.connect(lambda v: self.viewer.set_opacity(v / 100))
        self.hold = QPushButton("Hold: original")
        self.hold.setToolTip("Hold to flip back to the unprocessed image")
        self.hold.pressed.connect(lambda: self.viewer.set_opacity(0))
        self.hold.released.connect(lambda: self.viewer.set_opacity(self.opacity.value() / 100))
        self.side = QCheckBox("Side by side")
        self.side.toggled.connect(self._toggle_side)
        fit = QPushButton("Fit")
        fit.clicked.connect(lambda: self.viewer.fit())
        full = QPushButton("1:1")
        full.clicked.connect(lambda: self.viewer.actual_size())
        self.overlay_label = QLabel("Overlay")
        for w in (QLabel("View"), self.view_combo, self.overlay_label, self.opacity, self.hold, self.side):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(fit)
        bar.addWidget(full)
        bar_host = QWidget()
        bar_host.setLayout(bar)
        bar_host.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)   # lets the sidebars be dragged wide
        lay.addWidget(bar_host)
        self.viewer = ImageView()
        self.reference = ImageView()
        self.viewer.link(self.reference)
        self.reference.hide()
        self.viewer.pixel_hovered.connect(self._on_pixel)
        self.viewer.line_drawn.connect(self._on_line)
        split = self.view_split = QSplitter(Qt.Horizontal)
        split.addWidget(self.viewer)
        split.addWidget(self.reference)
        lay.addWidget(split, 1)
        self.stack.addWidget(page)
        self.setCentralWidget(self.stack)

    def _build_details(self):
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 10, 12, 10)
        self.tool_title = QLabel("")
        self.tool_title.setObjectName("h2")
        self.tool_help = QLabel("")
        self.tool_help.setWordWrap(True)
        self.tool_help.setObjectName("muted")
        self.form_host = QWidget()
        self.form = QFormLayout(self.form_host)
        self.form.setContentsMargins(0, 6, 0, 6)
        self.geometry_bar = QWidget()                # controls of the line-drawing tool
        row = QHBoxLayout(self.geometry_bar)
        row.setContentsMargins(0, 0, 0, 0)
        self.line_set = QComboBox()
        self.line_set.addItems([f"Set {name}" for name in geometry.SETS])
        row.addWidget(QLabel("Draw into"))
        row.addWidget(self.line_set, 1)
        for text, slot in (("Undo line", lambda: self._edit_lines("undo")), ("Clear set", lambda: self._edit_lines("set")),
                           ("Clear all", lambda: self._edit_lines("all"))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        self.geometry_bar.hide()
        self.run_button = QPushButton("Upload and analyse")
        self.run_button.clicked.connect(self._run_manual)
        self.run_button.hide()
        self.findings = QTextBrowser()
        self.findings.setOpenExternalLinks(True)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Property", "Value"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setWordWrap(True)
        self.table_search = QLineEdit()
        self.table_search.setPlaceholderText("Search properties and values")
        self.table_search.setClearButtonEnabled(True)
        self.table_search.textChanged.connect(self._filter_table)
        table_host = QWidget()
        table_lay = QVBoxLayout(table_host)
        table_lay.setContentsMargins(0, 0, 0, 0)
        table_lay.setSpacing(6)
        table_lay.addWidget(self.table_search)
        table_lay.addWidget(self.table, 1)
        self.guide_button = QPushButton("How to read this ...")
        self.guide_button.setToolTip("What the output shows, how to interpret it, what the options do, and the "
                                     "sources of the technique")
        self.guide_button.clicked.connect(self.open_guide)
        self.guide_button.hide()
        top = QHBoxLayout()
        top.addWidget(self.tool_title, 1)
        top.addWidget(self.guide_button)
        lay.addLayout(top)
        lay.addWidget(self.tool_help)
        lay.addWidget(self.form_host)
        lay.addWidget(self.geometry_bar)
        lay.addWidget(self.run_button)
        split = QSplitter(Qt.Vertical)
        split.addWidget(self.findings)
        split.addWidget(table_host)
        split.setSizes([320, 380])
        lay.addWidget(split, 1)
        dock = QDockWidget("Analysis")
        dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        dock.setWidget(panel)
        dock.setMinimumWidth(240)
        self.details_dock = dock
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _set_enabled(self, on: bool):
        for w in (self.btn_scan, self.btn_compare, self.btn_report, self.btn_reverse, self.btn_marks):
            w.setEnabled(on)

    def closeEvent(self, event):
        self.pool.clear()                            # queued analyses must not keep the process alive
        self.generation += 1
        cfg = online.load_config()
        cfg["sidebars"] = [self.tree_dock.width(), self.details_dock.width()]
        online.save_config(cfg)
        super().closeEvent(event)

    # ------------------------------------------------------------------ loading
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls()]
        if paths:
            self.open_file(paths[0])

    def open_dialog(self):
        exts = " ".join(f"*{e}" for e in sorted(IMAGE_EXTS))
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", f"Images ({exts});;All files (*)")
        if path:
            self.open_file(path)

    def open_file(self, path: str):
        try:
            ev = Evidence(path)
        except Exception as exc:
            QMessageBox.warning(self, APP_NAME, f"Cannot open {path}\n\n{exc}")
            return
        self.ev = ev
        ev.luma                                      # shared by most tools: compute once, not once per thread
        self.pool.clear()                            # analyses still queued for the previous image
        self.generation += 1
        self.results.clear()
        self.result_params.clear()
        self.lines = {name: [] for name in geometry.SETS}
        self.errors.clear()
        self.pending.clear()
        self.run_log.clear()
        self.setWindowTitle(f"{ev.path.name} - {APP_NAME}")
        self._set_enabled(True)
        for item in self.items.values():
            item.setIcon(0, dot("#3a3d44"))
        self.dashboard.set_evidence(ev)
        self.btn_scan.setText("Scan ...")
        self.tree.setCurrentItem(self.home)
        self.stack.setCurrentIndex(0)
        stored = online.recall_sightengine(ev)
        if stored is not None:
            self._on_done(self._job_key("online:Sightengine"), stored, "", self.generation)

    def scan(self):
        """Ask which analyses to run, then run them."""
        if self.ev is None:
            return
        from .dialogs import ScanDialog
        dialog = ScanDialog([t for t in ALL_TOOLS if t.id != "geometry"], self.scan_selection, self)
        if dialog.exec() != ScanDialog.Accepted:
            return
        self.scan_selection = dialog.selected()
        chosen = [t for t in ALL_TOOLS if t.id in self.scan_selection]
        uploads = [t.title for t in chosen if t.manual]
        confirmed = bool(uploads) and QMessageBox.question(
            self, APP_NAME, f"This uploads\n\n{self.ev.path.name}\n\nto {', '.join(uploads)}: third-party "
            "services. Continue?\n\nNo runs the other analyses without uploading.") == QMessageBox.Yes
        self.pool.clear()                            # drop queued duplicates; whatever was pending is queued again
        again = [t for t in ALL_TOOLS if t.id in self.scan_selection or (t.id in self.pending and not t.manual)]
        self.pending.clear()
        self.run_log.clear()                         # the progress overview starts again with every scan
        for tool in again:
            self.run_tool(tool, confirmed=confirmed)
        self.btn_scan.setText("Re-run scan ...")
        if self.pending:
            self.show_progress()

    def show_progress(self):
        if self.progress_dialog is None:
            self.progress_dialog = ProgressDialog(self)
        self.progress_dialog.refresh()
        self.progress_dialog.show()
        self.progress_dialog.raise_()

    # ------------------------------------------------------------------ running tools
    def _run_dirty(self):
        tool, self._dirty = self._dirty, None
        if tool is not None:
            self.run_tool(tool)

    def run_tool(self, tool: Tool, confirmed: bool = False):
        """confirmed: the user agreed to the upload; tools that send the image away never start without it."""
        if self.ev is None or (tool.manual and not confirmed):
            return
        self.run_log[tool.id] = "waiting"
        self.pending.add(tool.id)
        self.items[tool.id].setIcon(0, dot(theme.ACCENT))
        if tool.id in self.dashboard.cards:
            self.dashboard.cards[tool.id].set_running()
        key = self._job_key(tool.id)
        self.pool.start(Job(key, tool.fn, (self.ev,), dict(self.params[tool.id]), self.generation, self.signals))
        self._update_progress()

    def _job_key(self, tool_id: str) -> str:
        return f"{tool_id}|{sorted(self.params[tool_id].items())}"

    def _on_started(self, key: str, generation: int):
        tool_id = key.partition("|")[0]
        if generation == self.generation and key == self._job_key(tool_id) and tool_id in self.pending:
            self.run_log[tool_id] = "running"
            self._update_progress()

    def _on_done(self, key: str, res, error: str, generation: int):
        tool_id, _, used = key.partition("|")
        if generation != self.generation:
            return
        if used != str(sorted(self.params[tool_id].items())):
            return                                   # parameters changed while it ran; a newer job is on its way
        self.pending.discard(tool_id)
        if tool_id in self.run_log:                  # a recalled online result is not a run
            self.run_log[tool_id] = "failed" if res is None else "done"
        if res is None:
            self.errors[tool_id] = error
            self.results.pop(tool_id, None)
            self.items[tool_id].setIcon(0, dot("#8b8d98"))
        else:
            self.results[tool_id] = res
            self.result_params[tool_id] = dict(self.params[tool_id])
            self.items[tool_id].setIcon(0, dot(result_color(res)))
        if tool_id in self.dashboard.cards:
            self.dashboard.cards[tool_id].set_result(res, error)
        self.dashboard.set_verdicts(summarize(self.results), len(self.pending))
        if self.current and self.current.id == tool_id:
            self._show_result()
        self._update_progress()

    def _update_progress(self):
        n = len(self.pending)
        done = sum(state in ("done", "failed") for state in self.run_log.values())
        self.progress.setVisible(bool(self.run_log))  # stays after the round finishes, to reopen the overview
        self.progress.setRange(0, max(len(self.run_log), 1))
        self.progress.setValue(done)
        self.progress.setTextVisible(n > 0)          # "in progress" over the bar while anything runs
        if self.progress_dialog is not None and self.progress_dialog.isVisible():
            self.progress_dialog.refresh()
        self.statusBar().showMessage(f"Running {n} analysis job(s) ..." if n else "Ready", 0 if n else 4000)

    # ------------------------------------------------------------------ tool selection
    def select_tool(self, tool_id: str):
        self.tree.setCurrentItem(self.items[tool_id])

    def _on_tree(self, item: QTreeWidgetItem | None, _prev=None):
        if item is None:
            return
        tool_id = item.data(0, Qt.UserRole)
        self.table_search.clear()
        drawing = tool_id == "geometry" and self.ev is not None
        self.geometry_bar.setVisible(tool_id == "geometry")
        self.viewer.set_draw_mode(drawing)
        if not drawing:
            self.viewer.set_lines([], [])
        if tool_id is None:
            self.current = None
            self.stack.setCurrentIndex(0)
            self.tool_title.setText("Dashboard")
            self.guide_button.hide()
            self.tool_help.setText("Overview of every automatic analysis. Click a card or pick a tool on the left.")
            self._build_form(None)
            self.findings.clear()
            self.table.setRowCount(0)
            return
        self.current = next(t for t in ALL_TOOLS if t.id == tool_id)
        self.stack.setCurrentIndex(1)
        self.tool_title.setText(self.current.title)
        self.guide_button.show()
        self.tool_help.setText(self.current.help)
        self._build_form(self.current)
        self.run_button.setVisible(self.current.manual)
        if self.ev is None:
            return
        if drawing:
            self._update_geometry()
            return
        if tool_id in self.results or tool_id in self.errors:
            self._show_result()
        elif tool_id not in self.pending and not self.current.manual:
            self.viewer.set_images(self.ev.bgr)
            self.findings.setHtml("<i>Analysing ...</i>")
            self.table.setRowCount(0)
            self.run_tool(self.current)
        else:
            self.viewer.set_images(self.ev.bgr)
            self.findings.setHtml("<i>Analysing ...</i>" if tool_id in self.pending else "")
            self.table.setRowCount(0)

    # ------------------------------------------------------------------ geometry tool (lines drawn in the viewer)
    def _on_line(self, x1: float, y1: float, x2: float, y2: float):
        self.lines[self.line_set.currentText()[-1]].append((x1, y1, x2, y2))
        self._update_geometry()

    def _edit_lines(self, what: str):
        current = self.lines[self.line_set.currentText()[-1]]
        if what == "undo" and current:
            current.pop()
        elif what == "set":
            current.clear()
        elif what == "all":
            self.lines = {name: [] for name in geometry.SETS}
        self._update_geometry()

    def _update_geometry(self):
        res = geometry.analyse(self.ev, self.lines)
        self.results["geometry"], self.result_params["geometry"] = res, {}
        self.items["geometry"].setIcon(0, dot(result_color(res)))
        self._show_result()

    def _run_manual(self):
        if self.ev is None or self.current is None:
            return
        name = self.current.title
        ok = QMessageBox.question(self, APP_NAME, f"This uploads\n\n{self.ev.path.name}\n\nto {name}, a third-party service. "
                                  "Continue?") == QMessageBox.Yes
        if ok:
            self.findings.setHtml("<i>Uploading ...</i>")
            self.run_tool(self.current, confirmed=True)

    def _build_form(self, tool: Tool | None):
        while self.form.rowCount():
            self.form.removeRow(0)
        self.run_button.hide()
        if tool is None:
            return
        values = self.params[tool.id]

        def changed(key, value):
            values[key] = value
            if self._dirty not in (None, tool):      # options of another tool are still waiting: run that one now
                self._run_dirty()
            self._dirty = tool
            self.debounce.start()

        for p in tool.params:
            if p.choices:
                w = QComboBox()
                w.addItems(p.choices() if callable(p.choices) else p.choices)
                w.setCurrentText(str(values[p.key]))
                w.currentTextChanged.connect(lambda v, k=p.key: changed(k, v))
            elif isinstance(p.default, bool):
                w = QCheckBox()
                w.setChecked(bool(values[p.key]))
                w.toggled.connect(lambda v, k=p.key: changed(k, v))
            elif isinstance(p.default, float):
                w = QDoubleSpinBox()
                w.setRange(p.lo, p.hi)
                w.setSingleStep(p.step)
                w.setValue(float(values[p.key]))
                w.valueChanged.connect(lambda v, k=p.key: changed(k, v))
            else:
                w = self._int_slider(p, values, changed)
            self.form.addRow(p.label, w)

    def _int_slider(self, p, values, changed) -> QWidget:
        host = QWidget()
        row = QHBoxLayout(host)
        row.setContentsMargins(0, 0, 0, 0)
        slider = ClickSlider(Qt.Horizontal)
        step = max(int(p.step), 1)
        slider.setRange(int(p.lo) // step, int(p.hi) // step)
        slider.setValue(int(values[p.key]) // step)
        box = QSpinBox()
        box.setRange(int(p.lo), int(p.hi))
        box.setSingleStep(step)
        box.setValue(int(values[p.key]))
        slider.valueChanged.connect(lambda v: box.setValue(v * step))
        box.valueChanged.connect(lambda v: (slider.blockSignals(True), slider.setValue(v // step),
                                            slider.blockSignals(False),
                                            changed(p.key, max(int(p.lo), v // step * step))))
        row.addWidget(slider, 1)
        row.addWidget(box)
        return host

    # ------------------------------------------------------------------ showing results
    def _show_result(self):
        tool = self.current
        res = self.results.get(tool.id)
        if res is None:
            self.findings.setHtml(f"<b>Analysis failed</b><pre>{html.escape(self.errors.get(tool.id, ''))}</pre>")
            self.table.setRowCount(0)
            return
        previous = self.view_combo.currentText()
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        self.view_combo.addItems(list(res.views))
        if previous in res.views:
            self.view_combo.setCurrentText(previous)
        self.view_combo.blockSignals(False)
        self._show_view(keep=True)

        parts = []
        if res.score is not None:
            parts.append(f'<p style="color:{theme.MUTED}">Screening score <b style="color:{result_color(res)}">{res.score:.2f}</b> '
                         "(0 = nothing found, 1 = strong indication)</p>")
        for f in sorted(res.findings, key=lambda f: -SEVERITY_ORDER[f.severity]):
            color = theme.SEVERITY_COLORS[f.severity]
            detail = html.escape(f.detail).replace("\n", "<br>")
            parts.append(f'<p><span style="color:{color}">&#9679; <b>{html.escape(f.title)}</b></span><br>'
                         f'<span style="color:#c4c6cc">{detail}</span></p>')
        self.findings.setHtml("".join(parts))
        self.table.setRowCount(len(res.table))
        for i, (k, v) in enumerate(res.table):
            self.table.setItem(i, 0, QTableWidgetItem(str(k)))
            self.table.setItem(i, 1, QTableWidgetItem(str(v)))
        self.table.resizeColumnToContents(0)
        self.table.setColumnWidth(0, min(self.table.columnWidth(0), 210))
        self._filter_table()

    def _filter_table(self):
        """Hide the rows that do not contain the search text, in either column."""
        needle = self.table_search.text().strip().lower()
        for i in range(self.table.rowCount()):
            row = " ".join(self.table.item(i, c).text() for c in (0, 1) if self.table.item(i, c))
            self.table.setRowHidden(i, needle not in row.lower())

    def _show_view(self, keep: bool = False):
        if self.ev is None or self.current is None:
            return
        if self.current.id == "geometry":            # the lines are drawn live on the image itself
            self.viewer.set_images(self.ev.bgr, keep_view=keep)
            vectors = [(*line, geometry.SETS[name]) for name, lines in self.lines.items() for line in lines]
            points = [m["point"] for m in map(geometry.measure, self.lines.values()) if m["point"] is not None]
            self.viewer.set_lines(vectors, points)
            for w in (self.opacity, self.hold, self.overlay_label, self.side):
                w.setEnabled(False)
            self.reference.hide()
            return
        res = self.results.get(self.current.id)
        view = res.views.get(self.view_combo.currentText()) if res else None
        if view is None:
            self.viewer.set_images(self.ev.bgr, keep_view=keep)
            aligned = False
        else:
            aligned = view.shape[:2] == self.ev.bgr.shape[:2]
            if aligned:
                self.viewer.set_images(self.ev.bgr, view, self.opacity.value() / 100, keep_view=keep)
            else:
                self.viewer.set_images(view, keep_view=False)
        for w in (self.opacity, self.hold, self.overlay_label, self.side):
            w.setEnabled(aligned)
        self.reference.setVisible(aligned and self.side.isChecked())
        if self.reference.isVisible():
            self.reference.set_images(self.ev.bgr, keep_view=True)

    def _toggle_side(self, on: bool):
        self._show_view(keep=True)
        if on:
            self.view_split.setSizes([10_000, 10_000])   # equal halves
            QTimer.singleShot(0, self.viewer.fit)

    def _on_pixel(self, x: int, y: int):
        if not self.viewer.shows(self.ev.bgr if self.ev is not None else None):
            self.pixel_label.setText("")             # a plot or spectrum is shown: its pixels are not the evidence's
            return
        if self.ev is not None and 0 <= x < self.ev.width and 0 <= y < self.ev.height:
            b, g, r = self.ev.bgr[y, x]
            self.pixel_label.setText(f"x {x}  y {y}   RGB {r} {g} {b}   zoom {self.viewer.zoom_factor():.0%}   ")

    # ------------------------------------------------------------------ toolbar actions
    def reverse_search(self, name: str, url: str):
        self.open_site_with_image(name, url, "click the image-search box and press Ctrl+V.")

    def open_site_with_image(self, name: str, url: str, how: str):
        """Clipboard hand-over for services without an API. The pixels are copied, not the file: metadata does not
        travel, a pixel watermark does."""
        if self.ev is None:
            return
        QApplication.clipboard().setPixmap(to_pixmap(self.ev.bgr))
        webbrowser.open(url)
        QMessageBox.information(self, APP_NAME, f"The image is on the clipboard and {name.split(' (')[0]} is opening.\n\n"
                                f"There, {how}\n\nThe picture reaches that service only when you paste it.")

    def export_report(self):
        from .dialogs import ReportDialog
        ReportDialog(self).exec()

    def open_compare(self):
        from .dialogs import CompareWindow
        path, _ = QFileDialog.getOpenFileName(self, "Reference image to compare with", str(self.ev.path.parent))
        if path:
            self._compare = CompareWindow(self.ev, path, self)
            self._compare.setAttribute(Qt.WA_DeleteOnClose)
            self._compare.show()

    def open_batch(self):
        from .dialogs import BatchWindow
        self._batch = BatchWindow(self)
        self._batch.setAttribute(Qt.WA_DeleteOnClose)
        self._batch.show()

    def open_fingerprints(self):
        from .dialogs import FingerprintWindow
        self._fingerprints = FingerprintWindow(self)
        self._fingerprints.setAttribute(Qt.WA_DeleteOnClose)
        self._fingerprints.show()

    def open_calibration(self):
        from .dialogs import CalibrationWindow
        self._calibration = CalibrationWindow(self)
        self._calibration.setAttribute(Qt.WA_DeleteOnClose)
        self._calibration.show()

    def open_setup(self, first_run: bool = False):
        from .setup_wizard import SetupWizard
        before = len(components.missing())
        SetupWizard(self, first_run=bool(first_run)).exec()
        if self.ev is not None and len(components.missing()) < before:
            self.scan()                              # newly installed tools apply to the open image

    def first_run_check(self):
        """Offer the setup wizard at start-up while components are missing, unless the user opted out."""
        if components.missing() and not online.load_config().get("setup_skip", False):
            self.open_setup(first_run=True)

    def open_guide(self):
        if self.current is not None:
            from .dialogs import GuideDialog
            GuideDialog(self.current, self).exec()

    def open_settings(self):
        from .dialogs import SettingsDialog
        SettingsDialog(self).exec()


def selftest(image: str, out: str) -> int:
    """Headless check of a build: run every automatic tool once and write the scores as JSON."""
    import json
    ev, report = Evidence(image), {"_components_missing": [c.id for c in components.missing()]}
    for tool in TOOLS:
        if tool.auto:
            try:
                res = tool.fn(ev, **defaults(tool))
                report[tool.id] = {"score": res.score, "findings": [f.title for f in res.findings]}
            except Exception as exc:
                report[tool.id] = {"error": repr(exc)}
    Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


def headless_install(ids: str, out: str) -> int:
    """Build verification: install components (comma-separated ids) without the GUI, report as JSON."""
    import json
    report = {}
    for cid in ids.split(","):
        try:
            components.install(components.BY_ID[cid], lambda *a: None, lambda: False)
            report[cid] = "installed"
        except Exception as exc:
            report[cid] = f"failed: {exc}"
    Path(out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


def _report_crash(kind, value, trace):
    """pythonw and the windowed exe have no console: write the traceback next to the app and show it."""
    text = "".join(traceback.format_exception(kind, value, trace))
    try:
        (app_dir() / "fiasuite-error.log").write_text(text, encoding="utf-8")
    except OSError:
        pass
    QMessageBox.critical(None, APP_NAME, f"Unexpected error (also written to fiasuite-error.log):\n\n{text[-1500:]}")


def run(argv: list[str]) -> int:
    if len(argv) == 4 and argv[1] == "--selftest":
        return selftest(argv[2], argv[3])
    if len(argv) == 4 and argv[1] == "--install":
        return headless_install(argv[2], argv[3])
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    sys.excepthook = _report_crash
    theme.apply(app)
    win = MainWindow()
    win.show()
    QTimer.singleShot(300, win.first_run_check)
    files = [a for a in argv[1:] if Path(a).is_file()]
    if files:
        QTimer.singleShot(0, lambda: win.open_file(files[0]))
    return app.exec()
