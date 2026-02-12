"""File list panel showing image files with thumbnails and label status."""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QLabel, QAbstractItemView,
)
from PySide6.QtGui import QPixmap, QIcon, QColor, QImage
from PySide6.QtCore import Signal, Qt, QSize, QThread, QObject

from i18n import tr


class _ThumbnailWorker(QObject):
    """Loads thumbnails in a background thread one at a time."""
    thumbnail_ready = Signal(int, QIcon)  # index, icon
    finished = Signal()

    def __init__(self, image_paths: list[str], thumb_size: int = 64):
        super().__init__()
        self._image_paths = image_paths
        self._thumb_size = thumb_size
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        for i, path in enumerate(self._image_paths):
            if self._cancelled:
                break
            try:
                reader = QImage(path)
                if not reader.isNull():
                    scaled = reader.scaled(
                        self._thumb_size, self._thumb_size,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.FastTransformation,
                    )
                    pixmap = QPixmap.fromImage(scaled)
                    self.thumbnail_ready.emit(i, QIcon(pixmap))
            except Exception:
                pass
        self.finished.emit()


class FileListWidget(QWidget):
    image_selected = Signal(int)  # index

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._image_paths: list[str] = []
        self._thumb_thread: QThread | None = None
        self._thumb_worker: _ThumbnailWorker | None = None
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

        self._image_paths = image_paths
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
        self._thumb_thread = QThread()
        self._thumb_worker = _ThumbnailWorker(image_paths)
        self._thumb_worker.moveToThread(self._thumb_thread)

        self._thumb_thread.started.connect(self._thumb_worker.run)
        self._thumb_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self._thumb_worker.finished.connect(self._thumb_thread.quit)
        self._thumb_worker.finished.connect(self._thumb_worker.deleteLater)
        self._thumb_thread.finished.connect(self._thumb_thread.deleteLater)
        self._thumb_thread.finished.connect(self._on_thumb_thread_done)

        self._thumb_thread.start()

    def _stop_thumbnail_loader(self):
        if self._thumb_worker:
            self._thumb_worker.cancel()
        if self._thumb_thread and self._thumb_thread.isRunning():
            self._thumb_thread.quit()
            self._thumb_thread.wait(2000)
        self._thumb_worker = None
        self._thumb_thread = None

    def _on_thumbnail_ready(self, index: int, icon: QIcon):
        if 0 <= index < self._list_widget.count():
            self._list_widget.item(index).setIcon(icon)

    def _on_thumb_thread_done(self):
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
