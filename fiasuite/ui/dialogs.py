"""Secondary windows: report export, two-image comparison, batch triage, settings."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from PySide6.QtCore import QMarginsF, Qt, QThreadPool
from PySide6.QtGui import QColor, QPageLayout, QPageSize, QTextDocument
from PySide6.QtPrintSupport import QPrinter
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QHeaderView,
                               QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QTableWidget, QTableWidgetItem, QTextBrowser, QVBoxLayout, QWidget)

from .. import APP_NAME, calibrate, launder
from ..analyses import compare, online
from ..core import IMAGE_EXTS, SEVERITY_ORDER, Evidence
from ..report import build_html, build_json
from ..tools import BY_ID, TOOLS, defaults
from ..verdict import summarize
from . import theme
from .viewer import ImageView
from .worker import Job, Signals

SLOW_TOOLS = {"ghosts", "trufor", "clone_blocks", "resampling", "clone_kp"}


# ---------------------------------------------------------------------- report

class ReportDialog(QDialog):
    def __init__(self, main):
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Export report")
        self.resize(520, 360)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.examiner = QLineEdit(online.load_config().get("examiner", ""))
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Case number, source of the file, questions asked, observations ...")
        form.addRow("Examiner", self.examiner)
        form.addRow("Notes", self.notes)
        lay.addLayout(form)
        done = len(main.results)
        info = QLabel(f"The report contains the {done} analyses run so far, with their parameters, findings and images."
                      + (f"  {len(main.pending)} still running." if main.pending else ""))
        info.setWordWrap(True)
        info.setObjectName("muted")
        lay.addWidget(info)
        buttons = QDialogButtonBox()
        buttons.addButton("Save HTML ...", QDialogButtonBox.AcceptRole).clicked.connect(lambda: self.save("html"))
        buttons.addButton("Save PDF ...", QDialogButtonBox.AcceptRole).clicked.connect(lambda: self.save("pdf"))
        buttons.addButton("Save JSON ...", QDialogButtonBox.AcceptRole).clicked.connect(lambda: self.save("json"))
        buttons.addButton(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        lay.addWidget(buttons)

    def save(self, kind: str):
        ev = self.main.ev
        target, _ = QFileDialog.getSaveFileName(self, "Save report", str(ev.path.with_suffix(f".report.{kind}")),
                                                f"{kind.upper()} (*.{kind})")
        if not target:
            return
        cfg = online.load_config()
        cfg["examiner"] = self.examiner.text()
        online.save_config(cfg)
        args = (ev, self.main.results, summarize(self.main.results), self.main.params,
                self.examiner.text(), self.notes.toPlainText())
        doc = build_html(*args) if kind != "json" else ""
        if kind == "json":
            Path(target).write_text(json.dumps(build_json(*args), indent=1, ensure_ascii=False), encoding="utf-8")
        elif kind == "html":
            Path(target).write_text(doc, encoding="utf-8")
        else:
            printer = QPrinter(QPrinter.HighResolution)
            printer.setOutputFormat(QPrinter.PdfFormat)
            printer.setOutputFileName(target)
            printer.setPageLayout(QPageLayout(QPageSize(QPageSize.A4), QPageLayout.Portrait, QMarginsF(14, 14, 14, 14),
                                              QPageLayout.Millimeter))
            text = QTextDocument()
            text.setHtml(doc)
            text.print_(printer)
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
    results = {}
    for tool in TOOLS:
        if tool.auto and (thorough or tool.id not in SLOW_TOOLS):
            try:
                results[tool.id] = tool.fn(ev, **defaults(tool))
            except Exception:
                pass
    verdicts = summarize(results)
    flagged = [(f.severity, f.title) for r in results.values() for f in r.findings]
    top = max(flagged, key=lambda f: SEVERITY_ORDER[f[0]], default=("info", ""))
    return {"path": path, "verdicts": verdicts, "top": top}


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
            self.table.setItem(row, 4, QTableWidgetItem(info["top"][1]))
        if self.progress.value() >= self.progress.maximum():
            self.table.setSortingEnabled(True)
            self.table.sortItems(1, Qt.DescendingOrder)
            self.table.resizeColumnsToContents()
            self.export.setEnabled(True)

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
            folder = Path(edit.text())
            sets[kind] = sorted(str(p) for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS) if folder.is_dir() else []
        if not sets["real"] or not sets["fake"]:
            QMessageBox.warning(self, APP_NAME, "Both folders must contain images.")
            return
        self.generation += 1
        self.records = {"real": [], "fake": []}
        self.progress.setRange(0, len(sets["real"]) + len(sets["fake"]))
        self.progress.setValue(0)
        self.start.setEnabled(False)
        tool_id = self.tool.currentData()
        for kind, paths in sets.items():
            for path in paths:
                self.pool.start(Job(kind, calibrate.measure, (path, tool_id, self.chain_list), {}, self.generation, self.signals))

    def _done(self, kind, record, _error, generation):
        if generation != self.generation:
            return
        if record is not None:
            self.records[kind].append(record)
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

    def export_csv(self):
        target, _ = QFileDialog.getSaveFileName(self, "Export CSV", "calibration.csv", "CSV (*.csv)")
        if target:
            with open(target, "w", newline="", encoding="utf-8-sig") as fh:
                writer = csv.DictWriter(fh, fieldnames=sorted({k for r in self.rows for k in r}))
                writer.writeheader()
                writer.writerows(self.rows)


# ---------------------------------------------------------------------- settings

class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 420)
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
        self.cfg["sightengine"] = {"api_user": self.user.text().strip(), "api_secret": self.secret.text().strip()}
        self.cfg["use_gpu"] = self.gpu.isChecked()
        online.save_config(self.cfg)
        self.accept()
