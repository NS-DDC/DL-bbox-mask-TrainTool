"""File list panel showing image files with thumbnails and label status."""

import os
import logging
from threading import Event

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QAbstractItemView,
)
from PySide6.QtGui import QPixmap, QIcon, QColor, QImage, QImageReader
from PySide6.QtCore import Signal, Slot, Qt, QSize, QThread, QObject

from i18n import tr


class _ThumbnailWorker(QObject):
    """Loads thumbnails in a background thread one at a time.

    QPixmap is not thread-safe; only QImage work is done here.
    The main-thread slot converts QImage → QPixmap → QIcon.
    """
    thumbnail_ready = Signal(int, int, str, QImage)  # generation, index, source, image
    finished = Signal()

    def __init__(self, image_paths: list[str], generation: int, thumb_size: int = 64):
        super().__init__()
        self._image_paths = list(image_paths)
        self._generation = generation
        self._thumb_size = thumb_size
        self._cancelled = Event()

    def cancel(self):
        self._cancelled.set()

    @Slot()
    def run(self):
        try:
            for i, path in enumerate(self._image_paths):
                if self._cancelled.is_set():
                    break
                try:
                    reader = QImageReader(path)
                    original = reader.size()
                    if original.isValid():
                        reader.setScaledSize(original.scaled(
                            QSize(self._thumb_size, self._thumb_size),
                            Qt.AspectRatioMode.KeepAspectRatio,
                        ))
                    image = reader.read()
                    if not image.isNull() and not self._cancelled.is_set():
                        scaled = image.scaled(
                            self._thumb_size, self._thumb_size,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.FastTransformation,
                        )
                        self.thumbnail_ready.emit(self._generation, i, path, scaled)
                except Exception:
                    logging.getLogger(__name__).exception("Thumbnail read failed: %s", path)
        finally:
            self.finished.emit()


class FileListWidget(QWidget):
    image_selected = Signal(int)  # index

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._image_paths: list[str] = []
        self._thumb_thread: QThread | None = None
        self._thumb_worker: _ThumbnailWorker | None = None
        self._thumbnail_generation = 0
        self._thumbnail_jobs: dict[int, tuple[QThread, _ThumbnailWorker]] = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._title_label = QLabel(tr("file_panel_title"))
        self._title_label.setStyleSheet("font-weight: bold; padding: 4px;")
        layout.addWidget(self._title_label)

        self._count_label = QLabel(tr("file_no_folder"))
        self._count_label.setStyleSheet("padding: 2px 4px; color: gray;")
        layout.addWidget(self._count_label)

        self._list_widget = QListWidget()
        self._list_widget.setIconSize(QSize(64, 64))
        self._list_widget.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list_widget)

    def set_image_list(self, image_paths: list[str]):
        # Stop any running thumbnail loader
        self._stop_thumbnail_loader()
        self._thumbnail_generation += 1

        self._image_paths = list(image_paths)
        self._list_widget.clear()

        # Add items immediately with just filenames (no thumbnails yet)
        for path in image_paths:
            filename = os.path.basename(path)
            item = QListWidgetItem(filename)
            item.setToolTip(path)
            self._list_widget.addItem(item)

        count = len(image_paths)
        self._count_label.setText(
            tr("file_count").format(count=count) if count > 0 else tr("file_no_folder")
        )

        # Start background thumbnail loading
        if image_paths:
            self._start_thumbnail_loader(image_paths)

    def _start_thumbnail_loader(self, image_paths: list[str]):
        self._thumb_thread = QThread(self)
        self._thumb_worker = _ThumbnailWorker(image_paths, self._thumbnail_generation)
        self._thumbnail_jobs[self._thumbnail_generation] = (self._thumb_thread, self._thumb_worker)
        self._thumb_worker.moveToThread(self._thumb_thread)

        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumb_worker.finished.connect(self._thumb_thread.quit, Qt.ConnectionType.DirectConnection)
        self._thumb_worker.finished.connect(self._thumb_worker.deleteLater)
        self._thumb_thread.finished.connect(self._on_thumb_thread_done)
        self._thumb_thread.finished.connect(self._thumb_thread.deleteLater)

        self._thumb_thread.start()

    def _stop_thumbnail_loader(self):
        # Decoding can outlive a timeout. Keep every thread until native finished.
        for _, worker in self._thumbnail_jobs.values():
            worker.cancel()
        self._thumb_worker = None
        self._thumb_thread = None

    def shutdown(self) -> bool:
        """Request cancellation; the owner should defer closing while False."""
        self._stop_thumbnail_loader()
        self._thumbnail_generation += 1
        return not any(thread.isRunning() for thread, _ in self._thumbnail_jobs.values())

    @Slot(int, int, str, QImage)
    def _on_thumbnail_ready(self, generation: int, index: int, source: str, image: QImage):
        """Convert QImage to QIcon on the main thread (QPixmap is not thread-safe)."""
        if (generation == self._thumbnail_generation
                and 0 <= index < self._list_widget.count()
                and index < len(self._image_paths)
                and source == self._image_paths[index]):
            pixmap = QPixmap.fromImage(image)
            self._list_widget.item(index).setIcon(QIcon(pixmap))

    @Slot()
    def _on_thumb_thread_done(self):
        finished_thread = self.sender()
        for generation, (thread, _) in list(self._thumbnail_jobs.items()):
            if thread is finished_thread:
                del self._thumbnail_jobs[generation]
                break
        if self._thumb_thread is finished_thread:
            self._thumb_worker = None
            self._thumb_thread = None

    def update_label_status(self, index: int, has_labels: bool):
        if 0 <= index < self._list_widget.count():
            item = self._list_widget.item(index)
            if has_labels:
                item.setForeground(QColor("#2ecc71"))
            else:
                item.setForeground(QColor("#cccccc"))

    def select_image(self, index: int):
        if 0 <= index < self._list_widget.count():
            self._list_widget.setCurrentRow(index)

    def current_index(self) -> int:
        return self._list_widget.currentRow()

    def _on_row_changed(self, row: int):
        if row >= 0:
            self.image_selected.emit(row)

    def retranslate(self):
        self._title_label.setText(tr("file_panel_title"))
        count = len(self._image_paths)
        self._count_label.setText(
            tr("file_count").format(count=count) if count > 0 else tr("file_no_folder")
        )
