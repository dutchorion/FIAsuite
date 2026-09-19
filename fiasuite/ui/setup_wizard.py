"""First-run setup: shows which optional components are missing and installs the ticked ones."""
from __future__ import annotations

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
                               QPushButton, QVBoxLayout)

from .. import APP_NAME, components
from ..analyses import online
from . import theme


class _Signals(QObject):
    progress = Signal(str, int, int)              # component id, bytes done, bytes total
    finished = Signal(str, str)                   # component id, error text ("" = installed, "cancelled")


class _Install(QRunnable):
    def __init__(self, todo: list[components.Component], token: str | None, signals: _Signals):
        super().__init__()
        self.todo, self.token, self.signals = todo, token, signals
        self.cancel = False

    def run(self):
        for c in self.todo:
            if self.cancel:
                self.signals.finished.emit(c.id, "cancelled")
                continue
            try:
                components.install(c, lambda done, total, c=c: self.signals.progress.emit(c.id, done, total),
                                   lambda: self.cancel, self.token)
                self.signals.finished.emit(c.id, "")
            except components.Cancelled:
                self.signals.finished.emit(c.id, "cancelled")
            except components.SetupError as exc:
                self.signals.finished.emit(c.id, str(exc))
            except Exception as exc:              # never take the application down with a failed download
                self.signals.finished.emit(c.id, f"Unexpected error: {exc}")


class SetupWizard(QDialog):
    def __init__(self, parent=None, first_run: bool = False):
        super().__init__(parent)
        self.setWindowTitle(f"Set up {APP_NAME}")
        self.resize(760, 640)
        self.job: _Install | None = None
        self.remaining = 0
        self.signals = _Signals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)

        lay = QVBoxLayout(self)
        lay.setSpacing(10)
        title = QLabel("Some components are not installed yet" if components.missing() else "All components are installed")
        title.setObjectName("h1")
        intro = QLabel(f"{APP_NAME} works without them, but the tools listed below stay switched off. Tick what you "
                       "want and press Install: the files are downloaded, checked against their published checksums "
                       "and placed next to the program. Nothing else on this PC is changed.")
        intro.setWordWrap(True)
        intro.setObjectName("muted")
        lay.addWidget(title)
        lay.addWidget(intro)

        self.rows: dict[str, tuple[QCheckBox, QLabel]] = {}
        for c in components.COMPONENTS:
            frame = QFrame()
            frame.setObjectName("card")
            grid = QGridLayout(frame)
            grid.setColumnStretch(1, 1)
            box = QCheckBox()
            name = QLabel(f"{c.title}   <span style='color:{theme.MUTED}'>{c.size_mb} MB</span>")
            name.setObjectName("h2")
            status = QLabel()
            detail = QLabel(f"{c.enables}<br><span style='color:{theme.MUTED}'>Source: {c.source} &nbsp;·&nbsp; "
                            f"Licence: {c.licence}</span>")
            detail.setWordWrap(True)
            grid.addWidget(box, 0, 0)
            grid.addWidget(name, 0, 1)
            grid.addWidget(status, 0, 2, Qt.AlignRight)
            grid.addWidget(detail, 1, 1, 1, 2)
            lay.addWidget(frame)
            self.rows[c.id] = (box, status)

        self.github = QFrame()
        self.github.setObjectName("card")
        g = QVBoxLayout(self.github)
        self.github_status = QLabel()
        self.github_status.setWordWrap(True)
        self.token = QLineEdit()
        self.token.setEchoMode(QLineEdit.Password)
        self.token.setPlaceholderText("GitHub access token (used for this download only, never saved)")
        self.token.setToolTip(components.TOKEN_HELP)
        g.addWidget(self.github_status)
        g.addWidget(self.token)
        lay.addWidget(self.github)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        self.bar = QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.hide()
        lay.addWidget(self.message)
        lay.addWidget(self.bar)
        lay.addStretch(1)

        buttons = QHBoxLayout()
        self.dont_ask = QCheckBox("Do not show this at start-up again")
        self.dont_ask.setChecked(bool(online.load_config().get("setup_skip", False)))
        self.dont_ask.setVisible(first_run or self.dont_ask.isChecked())
        self.install_button = QPushButton("Install selected")
        self.install_button.setDefault(True)
        self.install_button.clicked.connect(self.start)
        self.close_button = QPushButton("Not now" if first_run else "Close")
        self.close_button.clicked.connect(self.reject)
        buttons.addWidget(self.dont_ask)
        buttons.addStretch(1)
        buttons.addWidget(self.close_button)
        buttons.addWidget(self.install_button)
        lay.addLayout(buttons)
        self.refresh()

    # ------------------------------------------------------------------ state
    def refresh(self):
        any_missing = False
        for c in components.COMPONENTS:
            box, status = self.rows[c.id]
            done = components.installed(c)
            any_missing |= not done
            box.setEnabled(not done)
            box.setChecked(not done and c.default)
            status.setText("<span style='color:#46a758'>&#10003; installed</span>" if done
                           else f"<span style='color:{theme.SEVERITY_COLORS['note']}'>not installed</span>")
        needs_github = any(c.kind == "github" and not components.installed(c) for c in components.COMPONENTS)
        self.github.setVisible(needs_github)
        if needs_github:
            signed_in = components.github_cli_token() is not None
            self.token.setVisible(not signed_in)
            self.github_status.setText(
                "<b>GitHub access:</b> signed in through GitHub CLI - the two models from the private repository can "
                "be downloaded." if signed_in else
                "<b>GitHub access needed</b> for the two models kept in the private repository "
                f"{components.RELEASE_REPO}. Paste an access token below, or install GitHub CLI and run "
                f"'gh auth login'.<br><span style='color:{theme.MUTED}'>{components.TOKEN_HELP}</span>")
        self.install_button.setEnabled(any_missing)

    def start(self):
        todo = [c for c in components.COMPONENTS if self.rows[c.id][0].isChecked() and not components.installed(c)]
        if not todo:
            self.message.setText("Nothing is ticked.")
            return
        self.errors: list[str] = []
        self.remaining = len(todo)
        self.install_button.setEnabled(False)
        self.close_button.setText("Cancel")
        self.bar.show()
        self.job = _Install(todo, self.token.text().strip() or None, self.signals)
        QThreadPool.globalInstance().start(self.job)

    def _on_progress(self, cid: str, done: int, total: int):
        c = components.BY_ID[cid]
        self.bar.setRange(0, max(total, 0))
        self.bar.setValue(done)
        size = f"{done / 2**20:.0f} of {total / 2**20:.0f} MB" if total else f"{done / 2**20:.0f} MB"
        self.message.setText(f"Downloading {c.title} ... {size}")
        self.rows[cid][1].setText(f"<span style='color:{theme.ACCENT}'>downloading</span>")

    def _on_finished(self, cid: str, error: str):
        self.remaining -= 1
        if error and error != "cancelled":
            self.errors.append(f"<b>{components.BY_ID[cid].title}:</b> {error}")
        if self.remaining:
            return
        self.job = None
        self.bar.hide()
        self.close_button.setText("Close")
        self.refresh()
        if self.errors:
            self.message.setText("<span style='color:#e5484d'>" + "<br>".join(self.errors) + "</span>")
        else:
            left = components.missing()
            self.message.setText("Done. " + ("Everything is installed." if not left else
                                             "Installed. You can add the remaining components later via Setup."))

    # ------------------------------------------------------------------ closing
    def reject(self):
        if self.job is not None:                  # a download is running: cancel it, keep the window open
            self.job.cancel = True
            self.message.setText("Cancelling ...")
            return
        cfg = online.load_config()
        if self.dont_ask.isVisible() and cfg.get("setup_skip", False) != self.dont_ask.isChecked():
            cfg["setup_skip"] = self.dont_ask.isChecked()
            online.save_config(cfg)
        super().reject()
