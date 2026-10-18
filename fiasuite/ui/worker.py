"""Run analyses off the GUI thread."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QObject, QRunnable, Signal


class Signals(QObject):
    done = Signal(str, object, str, int)          # key, result | None, error text, generation
    started = Signal(str, int)                    # key, generation: the job left the queue
    progress = Signal(int, int)                   # steps done, steps in total (long jobs that report it)


class Job(QRunnable):
    def __init__(self, key: str, fn, args: tuple, kwargs: dict, generation: int, signals: Signals):
        super().__init__()
        self.key, self.fn, self.args, self.kwargs = key, fn, args, kwargs
        self.generation, self.signals = generation, signals

    def run(self):
        self.signals.started.emit(self.key, self.generation)
        try:
            self.signals.done.emit(self.key, self.fn(*self.args, **self.kwargs), "", self.generation)
        except Exception:
            self.signals.done.emit(self.key, None, traceback.format_exc(limit=4), self.generation)
