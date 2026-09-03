"""Run lifecycle regressions on CI without any model weights or inference."""
from threading import Event

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication
from shiboken6 import isValid

import ui.auto_label_dialog as dialog_module


class _HeldWorker(QThread):
    progress = Signal(int, int)
    image_done = Signal(str, list)
    error = Signal(str)
    finished_all = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.entered = Event()
        self.release = Event()
        self.aborted = False

    def run(self):
        self.entered.set()
        # A worker's own completion signal is earlier than native thread exit.
        self.finished_all.emit()
        self.release.wait(5)

    def abort(self):
        self.aborted = True


class _Model:
    is_loaded = True

    def get_model_type(self):
        return "YOLO"

    def get_model_path(self):
        return "fixture.pt"


def test_cancel_and_close_retain_worker_until_native_thread_finish(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(dialog_module, "AutoLabelWorker", _HeldWorker)
    dialog = dialog_module.AutoLabelDialog(_Model(), ["fixture.png"])
    dialog.show()
    dialog._on_start()
    worker = dialog._worker
    try:
        assert worker.entered.wait(2), "CI fixture worker did not start"
        app.processEvents()
        assert dialog._worker is worker, "finished_all must not release a running QThread"
        assert worker.isRunning()
        dialog.reject()
        assert worker.aborted
        assert dialog._worker is worker
        assert not dialog._start_btn.isEnabled()
        close_event = QCloseEvent()
        dialog.closeEvent(close_event)
        assert not close_event.isAccepted()
        assert dialog._worker is worker

        worker.release.set()
        assert worker.wait(2000), "CI fixture worker did not stop"
        app.processEvents()
        assert dialog._worker is None
        assert dialog._start_btn.isEnabled()
        dialog.reject()
        assert not dialog.isVisible()
    finally:
        if isValid(worker):
            worker.release.set()
            worker.wait(2000)
        app.processEvents()
        dialog.deleteLater()
        app.processEvents()
