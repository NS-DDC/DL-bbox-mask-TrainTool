"""Regression coverage for stale thumbnail delivery and shutdown ownership."""
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication
import pytest

from ui.file_list_widget import FileListWidget


@pytest.fixture
def panel(monkeypatch):
    app = QApplication.instance() or QApplication([])
    widget = FileListWidget()
    monkeypatch.setattr(widget, "_start_thumbnail_loader", lambda paths: None)
    yield widget
    widget.deleteLater()
    app.processEvents()


def test_replaced_folder_cannot_receive_old_thumbnail(panel):
    panel.set_image_list(["old-folder/image.png"])
    old_generation = panel._thumbnail_generation
    panel.set_image_list(["new-folder/image.png"])
    image = QImage(4, 4, QImage.Format.Format_RGB32)
    image.fill(0xff00ff)
    panel._on_thumbnail_ready(old_generation, 0, "old-folder/image.png", image)
    assert panel._list_widget.item(0).icon().isNull()
    panel._on_thumbnail_ready(panel._thumbnail_generation, 0, "wrong-image.png", image)
    assert panel._list_widget.item(0).icon().isNull()
    panel._on_thumbnail_ready(panel._thumbnail_generation, 0, "new-folder/image.png", image)
    assert not panel._list_widget.item(0).icon().isNull()


def test_shutdown_retains_a_decoder_that_has_not_finished(panel):
    class Decoder:
        cancelled = False

        def cancel(self):
            self.cancelled = True

    class Thread:
        running = True

        def isRunning(self):
            return self.running

    worker, thread = Decoder(), Thread()
    panel._thumbnail_jobs[1] = (thread, worker)
    assert panel.shutdown() is False
    assert worker.cancelled is True
    assert panel._thumbnail_jobs[1] == (thread, worker)
    thread.running = False
    assert panel.shutdown() is True
    panel._thumbnail_jobs.clear()
