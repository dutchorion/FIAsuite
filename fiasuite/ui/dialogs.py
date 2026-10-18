"""Secondary windows: report export, two-image comparison, batch triage, settings."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from PySide6.QtCore import Qt, QThreadPool
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QTreeWidget, QTreeWidgetItem,
                               QVBoxLayout, QWidget)

from .. import APP_NAME, calibrate, guide, launder
from ..analyses import compare, online, prnu
from ..core import IMAGE_EXTS, SEVERITY_ORDER, Evidence
from ..report import build_html, build_json, ordered, title_of, write_pdf
from ..tools import BY_ID, TOOLS, defaults
from ..verdict import summarize
from . import theme
from .viewer import ImageView
from .worker import Job, Signals

FORBIDDEN = '\\/:*?"<>|'                            # characters Windows refuses in file names
SLOW_TOOLS = {"ghosts", "trufor", "clone_blocks", "resampling", "clone_kp"}


# ---------------------------------------------------------------------- scan selection

class ScanDialog(QDialog):
    """Choose the analyses to run, grouped as in the tool tree. A group's check box ticks its whole group."""

    def __init__(self, tools, selected: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scan")
        self.resize(460, 640)
        lay = QVBoxLayout(self)
        note = QLabel("Choose the analyses to run on this image. Online services upload the image to a third "
                      "party; you are asked to confirm before anything is sent.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.items: dict[str, QTreeWidgetItem] = {}
        groups: dict[str, QTreeWidgetItem] = {}
        for tool in tools:
            if tool.group not in groups:
                groups[tool.group] = QTreeWidgetItem(self.tree, [tool.group])
                groups[tool.group].setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsAutoTristate)
            item = QTreeWidgetItem(groups[tool.group], [tool.title])
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if tool.id in selected else Qt.Unchecked)
            item.setToolTip(0, tool.help)
            self.items[tool.id] = item
        self.groups = list(groups.values())
        self.tree.expandAll()
        self.tree.itemChanged.connect(self._update_run)
        row = QHBoxLayout()
        for text, state in (("Select all", Qt.Checked), ("Select none", Qt.Unchecked)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, state=state: self.set_all(state))
            row.addWidget(b)
        row.addStretch(1)
        buttons = QDialogButtonBox()
        self.run = buttons.addButton("Run scan", QDialogButtonBox.AcceptRole)
        self.run.setObjectName("primary")
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        row.addWidget(buttons)
        lay.addWidget(note)
        lay.addWidget(self.tree, 1)
        lay.addLayout(row)
        self._update_run()

    def set_all(self, state):
        for group in self.groups:
            group.setCheckState(0, state)

    def selected(self) -> set[str]:
        return {tool_id for tool_id, item in self.items.items() if item.checkState(0) == Qt.Checked}

    def _update_run(self, *_):
        self.run.setEnabled(bool(self.selected()))


# ---------------------------------------------------------------------- report

class ReportDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Export report")
        self.resize(560, 680)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.examiner = QLineEdit(online.load_config().get("examiner", ""))
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Case number, source of the file, questions asked, observations ...")
        self.notes.setMaximumHeight(110)
        form.addRow("Examiner", self.examiner)
        form.addRow("Notes", self.notes)
        lay.addLayout(form)

        head = QHBoxLayout()
        label = QLabel("Analyses to include")
        label.setObjectName("h2")
        check_all, check_none = QPushButton("Check all"), QPushButton("Check none")
        check_all.clicked.connect(lambda: self._set_all(Qt.Checked))
        check_none.clicked.connect(lambda: self._set_all(Qt.Unchecked))
        head.addWidget(label)
        head.addStretch(1)
        head.addWidget(check_all)
        head.addWidget(check_none)
        lay.addLayout(head)
        self.tests = QListWidget()
        for tool_id in ordered(main.results):                  # every analysis that has been run, in report order
            res = main.results[tool_id]
            worst = res.worst()
            note = "" if worst == "info" else f"   - {max(res.findings, key=lambda f: SEVERITY_ORDER[f.severity]).title}"
            item = QListWidgetItem(title_of(tool_id) + note)
            item.setData(Qt.UserRole, tool_id)
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setForeground(QColor(theme.SEVERITY_COLORS[worst] if worst != "info" else theme.TEXT))
            self.tests.addItem(item)
        lay.addWidget(self.tests, 1)
        info = QLabel("Only analyses that have been run are listed; open a tool first to add it. In the PDF every "
                      "analysis starts on a new page." + (f"  {len(main.pending)} still running." if main.pending else ""))
        info.setWordWrap(True)
        info.setObjectName("muted")
        lay.addWidget(info)
        buttons = QDialogButtonBox()
        buttons.addButton("Save HTML ...", QDialogButtonBox.AcceptRole).clicked.connect(lambda: self.save("html"))
        buttons.addButton("Save PDF ...", QDialogButtonBox.AcceptRole).clicked.connect(lambda: self.save("pdf"))
        buttons.addButton("Save JSON ...", QDialogButtonBox.AcceptRole).clicked.connect(lambda: self.save("json"))
        buttons.addButton(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        lay.addWidget(buttons)

    def _set_all(self, state):
        for i in range(self.tests.count()):
            self.tests.item(i).setCheckState(state)

    def selected(self) -> set[str]:
        return {self.tests.item(i).data(Qt.UserRole) for i in range(self.tests.count())
                if self.tests.item(i).checkState() == Qt.Checked}

    def save(self, kind: str):
        ev = self.main.ev
        if not self.selected():
            QMessageBox.information(self, APP_NAME, "Tick at least one analysis to include.")
            return
        target, _ = QFileDialog.getSaveFileName(self, "Save report", str(ev.path.with_suffix(f".report.{kind}")),
                                                f"{kind.upper()} (*.{kind})")
        if not target:
            return
        # the summary verdict always reflects everything that was run; the selection limits the detail sections
        args = (ev, self.main.results, summarize(self.main.results), self.main.result_params,
                self.examiner.text(), self.notes.toPlainText(), self.selected())
        self.setCursor(Qt.WaitCursor)
        try:
            if kind == "json":
                Path(target).write_text(json.dumps(build_json(*args), indent=1, ensure_ascii=False), encoding="utf-8")
            elif kind == "html":
                Path(target).write_text(build_html(*args), encoding="utf-8")
            else:
                write_pdf(target, *args)
        except OSError as exc:
            QMessageBox.warning(self, APP_NAME, f"The report could not be written to\n{target}\n\n{exc}\n\n"
                                "Is the file open in another program, or the folder read-only?")
            return
        finally:
            self.unsetCursor()
        cfg = online.load_config()
        cfg["examiner"] = self.examiner.text()
        online.save_config(cfg)
        self.accept()
        QMessageBox.information(self.main, APP_NAME, f"Report saved to\n{target}")


# ---------------------------------------------------------------------- compare

class CompareWindow(QMainWindow):
    def __init__(self, ev: Evidence, reference_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Compare - {ev.path.name}  vs  {Path(reference_path).name}")
        self.resize(1400, 860)
        self.ev = ev
        self.result = None
        body = QWidget()
        lay = QVBoxLayout(body)
        bar = QHBoxLayout()
        self.combo = QComboBox()
        self.combo.setMinimumWidth(260)
        self.combo.currentTextChanged.connect(self._show)
        bar.addWidget(QLabel("Right-hand view"))
        bar.addWidget(self.combo)
        bar.addStretch(1)
        lay.addLayout(bar)
        self.left, self.right = ImageView(), ImageView()
        self.left.link(self.right)
        views = QSplitter(Qt.Horizontal)
        views.addWidget(self.left)
        views.addWidget(self.right)
        views.setSizes([10_000, 10_000])             # equal halves
        self.text = QTextBrowser()
        self.text.setHtml("<i>Aligning and comparing ...</i>")
        split = QSplitter(Qt.Vertical)
        split.addWidget(views)
        split.addWidget(self.text)
        split.setSizes([620, 240])
        lay.addWidget(split, 1)
        self.setCentralWidget(body)
        self.left.set_images(ev.bgr)
        self.signals = Signals()
        self.signals.done.connect(self._done)
        QThreadPool.globalInstance().start(Job("compare", self._work, (reference_path,), {}, 0, self.signals))

    def _work(self, reference_path: str):
        return compare.compare(self.ev, Evidence(reference_path))

    def _done(self, _key, res, error, _gen):
        if res is None:
            self.text.setHtml(f"<b>Comparison failed</b><pre>{html.escape(error)}</pre>")
            return
        self.result = res
        self.combo.addItems(list(res.views))
        rows = "".join(f"<tr><td style='color:{theme.MUTED}; padding-right:14px'>{html.escape(str(k))}</td>"
                       f"<td>{html.escape(str(v))}</td></tr>" for k, v in res.table)
        finds = "".join(f"<p><b style='color:{theme.SEVERITY_COLORS[f.severity]}'>&#9679; {html.escape(f.title)}</b><br>"
                        f"{html.escape(f.detail)}</p>" for f in res.findings)
        self.text.setHtml(f"{finds}<table>{rows}</table>")

    def _show(self, name: str):
        if self.result and name in self.result.views:
            self.right.set_images(self.result.views[name], keep_view=True)


# ---------------------------------------------------------------------- batch

def triage(path: str, thorough: bool) -> dict:
    ev = Evidence(path)
    results, failed = {}, []
    for tool in TOOLS:
        if tool.auto and (thorough or tool.id not in SLOW_TOOLS):
            try:
                results[tool.id] = tool.fn(ev, **defaults(tool))
            except Exception:
                failed.append(tool.title)
    verdicts = summarize(results)
    flagged = [(f.severity, f.title) for r in results.values() for f in r.findings]
    top = max(flagged, key=lambda f: SEVERITY_ORDER[f[0]], default=("info", ""))
    return {"path": path, "verdicts": verdicts, "top": top, "failed": failed}


class NumberItem(QTableWidgetItem):
    def __lt__(self, other):
        return float(self.data(Qt.UserRole) or 0) < float(other.data(Qt.UserRole) or 0)


class BatchWindow(QMainWindow):
    AXES = ("ai", "edit", "provenance")

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Batch triage")
        self.resize(1100, 700)
        body = QWidget()
        lay = QVBoxLayout(body)
        bar = QHBoxLayout()
        choose = QPushButton("Choose folder ...")
        choose.setDefault(True)
        choose.clicked.connect(self.choose)
        self.thorough = QCheckBox("Thorough (adds ghosts, clones, resampling, TruFor - much slower)")
        self.recursive = QCheckBox("Include sub-folders")
        self.export = QPushButton("Export CSV ...")
        self.export.clicked.connect(self.export_csv)
        self.export.setEnabled(False)
        for w in (choose, self.thorough, self.recursive):
            bar.addWidget(w)
        bar.addStretch(1)
        bar.addWidget(self.export)
        lay.addLayout(bar)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["File", "AI generation", "Manipulation", "File history", "Most serious finding"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)
        self.table.cellDoubleClicked.connect(self._open_row)
        lay.addWidget(self.table, 1)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        hint = QLabel("Double-click a row to open the image in the main window. Click a column header to sort.")
        hint.setObjectName("muted")
        lay.addWidget(hint)
        self.setCentralWidget(body)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.signals = Signals()
        self.signals.done.connect(self._done)
        self.generation = 0

    def choose(self):
        folder = QFileDialog.getExistingDirectory(self, "Folder with images")
        if not folder:
            return
        pattern = "**/*" if self.recursive.isChecked() else "*"
        files = sorted(str(p) for p in Path(folder).glob(pattern) if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        self.pool.clear()                            # a previous folder that is still queued
        self.generation += 1
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        self.progress.setRange(0, max(len(files), 1))
        self.progress.setValue(0)
        self.export.setEnabled(False)
        for path in files:
            self.pool.start(Job(path, triage, (path, self.thorough.isChecked()), {}, self.generation, self.signals))
        if not files:
            QMessageBox.information(self, APP_NAME, "No images found in that folder.")

    def _done(self, path, info, _error, generation):
        if generation != self.generation:
            return
        self.progress.setValue(self.progress.value() + 1)
        self.table.setSortingEnabled(False)
        row = self.table.rowCount()
        self.table.insertRow(row)
        name = QTableWidgetItem(Path(path).name)
        name.setData(Qt.UserRole, path)
        name.setToolTip(path)
        self.table.setItem(row, 0, name)
        if info is None:
            self.table.setItem(row, 4, QTableWidgetItem("could not be analysed"))
        else:
            for col, axis in enumerate(self.AXES, start=1):
                v = info["verdicts"][axis]
                item = NumberItem(f"{v.score:.2f}  {v.level}")
                item.setData(Qt.UserRole, v.score)
                item.setForeground(QColor(v.color))
                self.table.setItem(row, col, item)
            note = f"   [failed: {', '.join(info['failed'])}]" if info["failed"] else ""
            self.table.setItem(row, 4, QTableWidgetItem(info["top"][1] + note))
        if self.progress.value() >= self.progress.maximum():
            self.table.setSortingEnabled(True)
            self.table.sortItems(1, Qt.DescendingOrder)
            self.table.resizeColumnsToContents()
            self.export.setEnabled(True)

    def closeEvent(self, event):
        self.pool.clear()
        self.generation += 1
        super().closeEvent(event)

    def _open_row(self, row, _col):
        self.main.open_file(self.table.item(row, 0).data(Qt.UserRole))
        self.main.raise_()
        self.main.activateWindow()

    def export_csv(self):
        target, _ = QFileDialog.getSaveFileName(self, "Export CSV", "triage.csv", "CSV (*.csv)")
        if not target:
            return
        with open(target, "w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["path", "ai", "manipulation", "file_history", "top_finding"])
            for r in range(self.table.rowCount()):
                cells = [self.table.item(r, c) for c in range(5)]
                writer.writerow([cells[0].data(Qt.UserRole)] + [c.text() if c else "" for c in cells[1:]])


# ---------------------------------------------------------------------- calibration

class CalibrationWindow(QMainWindow):
    """AUC and detection rate at fixed false-alarm rates on the user's own corpus, per laundering chain."""
    COLUMNS = ["Laundering", "Metric", "Authentic", "Manipulated", "AUC", "Detected at 1% false alarms", "Threshold",
               "Detected at 5% false alarms", "Threshold"]

    def __init__(self, main):
        super().__init__(main)
        self.setWindowTitle("Calibrate a detector on your own images")
        self.resize(1150, 680)
        body = QWidget()
        lay = QVBoxLayout(body)
        intro = QLabel("Scores are screening aids until they have been measured. Point this tool at a folder of images you "
                       "know to be authentic and a folder you know to be manipulated or AI-generated. Each image is also "
                       "laundered (resized / recompressed the way a platform would) to show how much of the detector "
                       "survives. AUC 0.5 = useless, 1.0 = perfect. Use 50+ images per folder for meaningful numbers.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        lay.addWidget(intro)
        form = QFormLayout()
        self.real, self.fake = QLineEdit(), QLineEdit()
        for edit, label in ((self.real, "Authentic images"), (self.fake, "Manipulated / synthetic images")):
            row = QHBoxLayout()
            browse = QPushButton("Browse ...")
            browse.clicked.connect(lambda _, e=edit: e.setText(QFileDialog.getExistingDirectory(self, "Folder") or e.text()))
            row.addWidget(edit, 1)
            row.addWidget(browse)
            form.addRow(label, row)
        self.tool = QComboBox()
        for tool_id in calibrate.PIXEL_TOOLS:
            self.tool.addItem(BY_ID[tool_id].title, tool_id)
        self.chains = QLineEdit("none, jpeg85, resize0.5+jpeg75")
        self.chains.setToolTip("Comma-separated chains of jpegNN and resizeF steps joined with +, e.g. resize0.5+jpeg75")
        form.addRow("Detector", self.tool)
        form.addRow("Laundering chains", self.chains)
        lay.addLayout(form)
        bar = QHBoxLayout()
        self.start = QPushButton("Run calibration")
        self.start.setDefault(True)
        self.start.clicked.connect(self.run)
        self.export = QPushButton("Export CSV ...")
        self.export.setEnabled(False)
        self.export.clicked.connect(self.export_csv)
        bar.addWidget(self.start)
        bar.addStretch(1)
        bar.addWidget(self.export)
        lay.addLayout(bar)
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)
        self.progress = QProgressBar()
        lay.addWidget(self.progress)
        self.setCentralWidget(body)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(1)               # one image at a time: calibration runs for a long while
        self.signals = Signals()
        self.signals.done.connect(self._done)
        self.generation = 0
        self.rows: list[dict] = []

    def run(self):
        try:
            self.chain_list = [c.strip() or "none" for c in self.chains.text().split(",")]
            for chain in self.chain_list:
                launder.parse_chain(chain)
        except ValueError as exc:
            QMessageBox.warning(self, APP_NAME, str(exc))
            return
        sets = {}
        for kind, edit in (("real", self.real), ("fake", self.fake)):
            folder = Path(edit.text().strip())
            usable = bool(edit.text().strip()) and folder.is_absolute() and folder.is_dir()   # '' would mean the current folder
            sets[kind] = sorted(str(p) for p in folder.rglob("*")
                                if p.suffix.lower() in IMAGE_EXTS and p.is_file()) if usable else []
        if not sets["real"] or not sets["fake"]:
            QMessageBox.warning(self, APP_NAME, "Both folders must contain images.")
            return
        self.generation += 1
        self.records = {"real": [], "fake": []}
        self.failures: list[str] = []
        self.pool.clear()
        self.progress.setRange(0, len(sets["real"]) + len(sets["fake"]))
        self.progress.setValue(0)
        self.start.setEnabled(False)
        tool_id = self.tool.currentData()
        for kind, paths in sets.items():
            for path in paths:
                self.pool.start(Job(kind, calibrate.measure, (path, tool_id, self.chain_list), {}, self.generation, self.signals))

    def closeEvent(self, event):
        self.pool.clear()
        self.generation += 1
        super().closeEvent(event)

    def _done(self, kind, record, error, generation):
        if generation != self.generation:
            return
        if record is not None:
            self.records[kind].append(record)
        else:
            self.failures.append(error.strip().splitlines()[-1] if error.strip() else "unknown error")
        self.progress.setValue(self.progress.value() + 1)
        if self.progress.value() < self.progress.maximum():
            return
        self.start.setEnabled(True)
        self.rows = calibrate.summarize(self.records["real"], self.records["fake"], self.chain_list)
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            def fmt(key, pattern):
                return "-" if row.get(key) is None else pattern.format(row[key])
            cells = [row["laundering"], row["metric"], str(row["n_real"]), str(row["n_fake"]), fmt("auc", "{:.3f}"),
                     fmt("tpr@0.01", "{:.1%}"), fmt("thr@0.01", "{:.4g}"), fmt("tpr@0.05", "{:.1%}"), fmt("thr@0.05", "{:.4g}")]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col == 4 and row.get("auc") is not None:
                    item.setForeground(QColor("#46a758" if row["auc"] >= 0.9 else "#f5d90a" if row["auc"] >= 0.7 else "#e5484d"))
                self.table.setItem(i, col, item)
        self.table.resizeColumnsToContents()
        self.export.setEnabled(bool(self.rows))
        if self.failures:
            QMessageBox.warning(self, APP_NAME, f"{len(self.failures)} image(s) could not be measured and are not in "
                                f"the table. First error:\n\n{self.failures[0]}")

    def export_csv(self):
        target, _ = QFileDialog.getSaveFileName(self, "Export CSV", "calibration.csv", "CSV (*.csv)")
        if target:
            with open(target, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=sorted({k for r in self.rows for k in r}))
                writer.writeheader()
                writer.writerows(self.rows)


# ---------------------------------------------------------------------- camera fingerprints (PRNU)

class FingerprintWindow(QMainWindow):
    """Build and manage sensor-noise fingerprints; they are used by the 'Camera fingerprint (PRNU)' tool."""

    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Camera fingerprints (PRNU)")
        self.resize(760, 560)
        body = QWidget()
        lay = QVBoxLayout(body)
        intro = QLabel("Every image sensor leaves its own fixed noise pattern in each photo. Collect 20 or more photos "
                       "taken with ONE camera - ideally bright, evenly lit, low-detail subjects such as sky or a wall, "
                       "straight from the camera at full resolution and never edited or resized - and build a "
                       "fingerprint from them. The 'Camera fingerprint (PRNU)' tool then tests whether an image came "
                       "from that camera and which regions lack its pattern.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        lay.addWidget(intro)
        form = QFormLayout()
        self.folder, self.name = QLineEdit(), QLineEdit()
        self.name.setPlaceholderText("e.g. Canon 60D - serial 1234")
        row = QHBoxLayout()
        browse = QPushButton("Browse ...")
        browse.clicked.connect(lambda: self.folder.setText(QFileDialog.getExistingDirectory(self, "Folder") or self.folder.text()))
        row.addWidget(self.folder, 1)
        row.addWidget(browse)
        form.addRow("Reference photos", row)
        form.addRow("Fingerprint name", self.name)
        lay.addLayout(form)
        self.build_button = QPushButton("Build fingerprint")
        self.build_button.clicked.connect(self.build)
        self.progress = QProgressBar()
        self.progress.hide()
        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.message.setTextFormat(Qt.PlainText)
        lay.addWidget(self.build_button)
        lay.addWidget(self.progress)
        lay.addWidget(self.message)
        label = QLabel("Stored fingerprints")
        label.setObjectName("h2")
        lay.addWidget(label)
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        remove = QPushButton("Delete selected fingerprint")
        remove.clicked.connect(self.remove)
        lay.addWidget(remove)
        self.setCentralWidget(body)
        self.signals = Signals()
        self.signals.done.connect(self._done)
        self.signals.progress.connect(lambda done, total: (self.progress.setRange(0, total), self.progress.setValue(done)))
        self.cancelled = False
        self.refresh()

    def refresh(self):
        self.list.clear()
        for name in prnu.stored()[1:]:
            k, info = prnu.load_fingerprint(name)
            self.list.addItem(f"{name}   -   {info['count']} photos, {k.shape[1]} x {k.shape[0]} px")
            self.list.item(self.list.count() - 1).setData(Qt.UserRole, name)

    def build(self):
        folder, name = Path(self.folder.text().strip()), self.name.text().strip()
        if not self.folder.text().strip() or not folder.is_dir():
            QMessageBox.warning(self, APP_NAME, "Choose the folder with the reference photos.")
            return
        if not name or any(c in name for c in FORBIDDEN):
            QMessageBox.warning(self, APP_NAME, f"Give the fingerprint a name without any of  {FORBIDDEN}")
            return
        files = sorted(str(p) for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file())
        if len(files) < 5:
            QMessageBox.warning(self, APP_NAME, "At least 5 photos are needed; 20 or more give a reliable fingerprint.")
            return
        self.build_button.setEnabled(False)
        self.progress.show()
        self.message.setText(f"Reading {len(files)} photos ...")
        self.cancelled = False

        def work():
            fp = prnu.build_fingerprint(files, self.signals.progress.emit, lambda: self.cancelled)
            prnu.save_fingerprint(name, fp, str(folder))
            return fp

        QThreadPool.globalInstance().start(Job(name, work, (), {}, 0, self.signals))

    def _done(self, name, fp, error, _generation):
        self.build_button.setEnabled(True)
        self.progress.hide()
        if fp is None:
            self.message.setText("Failed: " + (error.strip().splitlines()[-1] if error.strip() else "unknown error"))
            return
        skipped = f" {len(fp['skipped'])} skipped (other dimensions or unreadable)." if fp["skipped"] else ""
        few = "  Fewer than 20 photos: expect weaker matches." if fp["count"] < 20 else ""
        self.message.setText(f"'{name}' built from {fp['count']} photos.{skipped}{few}")
        self.refresh()

    def remove(self):
        item = self.list.currentItem()
        if item is None:
            return
        name = item.data(Qt.UserRole)
        if QMessageBox.question(self, APP_NAME, f"Delete the fingerprint '{name}'?") == QMessageBox.Yes:
            (prnu.store() / f"{name}.npz").unlink(missing_ok=True)
            self.refresh()

    def closeEvent(self, event):
        self.cancelled = True
        super().closeEvent(event)


# ---------------------------------------------------------------------- guide

class GuideDialog(QDialog):
    """How to read a tool's output and use its options, with links to the origin of the technique."""

    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"How to read: {tool.title}")
        self.resize(800, 820)
        lay = QVBoxLayout(self)
        text = QTextBrowser()
        text.setOpenExternalLinks(True)                        # sources open in the web browser
        text.document().setDefaultStyleSheet(
            f"h2 {{ font-size:15pt; }} h3 {{ font-size:11pt; color:{theme.MUTED}; margin-top:14px; }} "
            f"li {{ margin-bottom:6px; }} a {{ color:#7aa2ff; }}")
        text.setSearchPaths([str(guide.EXAMPLES)])             # the example pictures are referenced by file name
        text.setHtml(guide.render(tool.id, tool.title, [p.label for p in tool.params]))
        lay.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)


# ---------------------------------------------------------------------- settings

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(540, 600)
        self.cfg = online.load_config()
        se = self.cfg.get("sightengine", {})
        lay = QVBoxLayout(self)
        title = QLabel("Sightengine (optional online AI detector)")
        title.setObjectName("h2")
        note = QLabel("Create a free account at sightengine.com and copy the API user and secret from its dashboard. "
                      "Keys are stored in plain text in config.json next to the program. Images are uploaded only "
                      "when you press 'Upload and analyse' in the Sightengine tool.")
        note.setWordWrap(True)
        note.setObjectName("muted")
        form = QFormLayout()
        self.user = QLineEdit(se.get("api_user", ""))
        self.secret = QLineEdit(se.get("api_secret", ""))
        self.secret.setEchoMode(QLineEdit.Password)
        form.addRow("API user", self.user)
        form.addRow("API secret", self.secret)
        lay.addWidget(title)
        lay.addWidget(note)
        lay.addLayout(form)
        oa_title = QLabel("OpenAI provenance check (optional)")
        oa_title.setObjectName("h2")
        oa_note = QLabel("Checks for OpenAI's Content Credentials and SynthID watermark. Needs an API key from "
                         "platform.openai.com; it is stored in plain text in config.json. The service is strictly "
                         "rate-limited and uploads are not covered by OpenAI's zero-data-retention option.")
        oa_note.setWordWrap(True)
        oa_note.setObjectName("muted")
        self.openai_key = QLineEdit(self.cfg.get("openai", {}).get("api_key", ""))
        self.openai_key.setEchoMode(QLineEdit.Password)
        oa_form = QFormLayout()
        oa_form.addRow("API key", self.openai_key)
        lay.addWidget(oa_title)
        lay.addWidget(oa_note)
        lay.addLayout(oa_form)
        gpu_title = QLabel("Neural models")
        gpu_title.setObjectName("h2")
        self.gpu = QCheckBox("Run neural models on the GPU (DirectML)")
        self.gpu.setChecked(bool(self.cfg.get("use_gpu", False)))
        gpu_note = QLabel("Off by default: the models then run on the CPU (about half as fast, always stable). When "
                          "switched on, every model is first run once on both GPU and CPU; if the results differ or the "
                          "GPU crawls, that model stays on the CPU and the tool says so. Background: a graphics driver "
                          "in a bad state once froze the desktop and returned wrong maps without any error message. "
                          "If that happens, switch this off and restart the PC.")
        gpu_note.setWordWrap(True)
        gpu_note.setObjectName("muted")
        lay.addWidget(gpu_title)
        lay.addWidget(self.gpu)
        lay.addWidget(gpu_note)
        lay.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.save)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def save(self):
        self.cfg = {**online.load_config(), **self.cfg}       # keep what other windows stored meanwhile
        self.cfg["sightengine"] = {"api_user": self.user.text().strip(), "api_secret": self.secret.text().strip()}
        self.cfg["openai"] = {"api_key": self.openai_key.text().strip()}
        self.cfg["use_gpu"] = self.gpu.isChecked()
        if not online.save_config(self.cfg):
            QMessageBox.warning(self, APP_NAME, f"Settings could not be saved: {online.config_path()} is not writable.")
        self.accept()
