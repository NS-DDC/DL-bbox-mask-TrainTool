"""Cancelable inference with explicit existing-label policy and error reporting."""

import os
import time
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QDoubleSpinBox, QSpinBox, QPushButton, QProgressBar,
    QMessageBox, QComboBox, QPlainTextEdit,
)
from i18n import tr
from core.auto_labeler import AutoLabelWorker
from core.model_manager import DEFAULT_INFER_SIZE


def _fmt_seconds(secs: float) -> str:
    minutes, seconds = divmod(max(0, int(secs)), 60)
    return f"{minutes}m {seconds:02d}s" if minutes else f"{seconds}s"


class AutoLabelDialog(QDialog):
    labels_generated = Signal(str, list)

    def __init__(self, model_manager, image_paths, current_index=0, parent=None,
                 *, existing_paths=None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._image_paths = list(image_paths)
        self._current_index = current_index
        self._existing_paths = set(existing_paths or [])
        self._worker = None
        self._cancelled = False
        self._success = self._empty = self._errors = self._skipped = 0
        self._start_time = 0.0
        self._active_policy = "skip"
        self._setup_ui()

    @property
    def apply_policy(self):
        return self._active_policy

    def _setup_ui(self):
        self.setWindowTitle(tr("auto_label_title"))
        self.setMinimumWidth(580)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        model = QLabel(f"{self._model_manager.get_model_type()} / "
                       f"{os.path.basename(self._model_manager.get_model_path() or '')}")
        model.setWordWrap(True)
        form.addRow(tr("auto_label_model"), model)
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.01, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setValue(0.25)
        form.addRow(tr("auto_label_confidence"), self._confidence_spin)
        self._infer_size_spin = QSpinBox()
        self._infer_size_spin.setRange(32, 4096)
        self._infer_size_spin.setSingleStep(32)
        self._infer_size_spin.setValue(DEFAULT_INFER_SIZE)
        form.addRow(tr("auto_label_infer_size"), self._infer_size_spin)
        self._device_combo = QComboBox()
        self._device_combo.setEditable(True)
        self._device_combo.addItems(["auto", "cpu", "0"])
        self._device_combo.setToolTip(tr("improved_device_hint"))
        form.addRow(tr("auto_label_device"), self._device_combo)
        self._output_combo = QComboBox()
        for title, value in [("improved_output_auto", "auto"),
                             ("improved_output_bbox", "bbox"),
                             ("improved_output_polygon", "polygon"),
                             ("improved_output_mask", "mask")]:
            self._output_combo.addItem(tr(title), value)
        form.addRow(tr("improved_output"), self._output_combo)
        self._scope_combo = QComboBox()
        self._scope_combo.addItems([tr("auto_label_current"), tr("auto_label_all")])
        form.addRow(tr("auto_label_scope"), self._scope_combo)
        self._policy_combo = QComboBox()
        for title, value in [("improved_policy_skip", "skip"),
                             ("improved_policy_append", "append"),
                             ("improved_policy_replace", "replace")]:
            self._policy_combo.addItem(tr(title), value)
        form.addRow(tr("improved_existing"), self._policy_combo)
        layout.addLayout(form)
        hint = QLabel(tr("improved_auto_hint"))
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._progress_bar = QProgressBar()
        self._progress_bar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_bar.setValue(0)
        layout.addWidget(self._progress_bar)
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        self._error_log = QPlainTextEdit()
        self._error_log.setReadOnly(True)
        self._error_log.setMaximumBlockCount(200)
        self._error_log.setMaximumHeight(150)
        self._error_log.hide()
        layout.addWidget(self._error_log)
        buttons = QHBoxLayout()
        self._start_btn = QPushButton(tr("auto_label_start"))
        self._start_btn.clicked.connect(self._on_start)
        buttons.addWidget(self._start_btn)
        self._cancel_btn = QPushButton(tr("auto_label_cancel"))
        self._cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(self._cancel_btn)
        layout.addLayout(buttons)

    def _set_running(self, running):
        self._start_btn.setEnabled(not running)
        for widget in (self._confidence_spin, self._infer_size_spin,
                       self._device_combo, self._output_combo,
                       self._scope_combo, self._policy_combo):
            widget.setEnabled(not running)

    def _on_start(self):
        if self._worker is not None or not self._model_manager.is_loaded:
            return
        if self._scope_combo.currentIndex() == 0:
            if not 0 <= self._current_index < len(self._image_paths):
                return
            paths = [self._image_paths[self._current_index]]
        else:
            paths = self._image_paths[:]
        self._active_policy = self._policy_combo.currentData()
        self._skipped = 0
        if self._active_policy == "skip":
            filtered = [p for p in paths if p not in self._existing_paths]
            self._skipped = len(paths) - len(filtered)
            paths = filtered
        if not paths:
            self._status_label.setText(tr("improved_no_unlabeled"))
            return
        if self._active_policy == "replace" and any(p in self._existing_paths for p in paths):
            reply = QMessageBox.question(self, tr("warning"), tr("improved_replace_confirm"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        self._success = self._empty = self._errors = 0
        self._cancelled = False
        self._error_log.clear()
        self._error_log.hide()
        self._start_time = time.monotonic()
        self._progress_bar.setRange(0, len(paths))
        self._progress_bar.setValue(0)
        self._set_running(True)
        self._worker = AutoLabelWorker(
            self._model_manager, paths,
            confidence=self._confidence_spin.value(),
            score_threshold=self._confidence_spin.value(),
            infer_size=self._infer_size_spin.value(),
            device=self._device_combo.currentText().strip(),
            output_type=self._output_combo.currentData(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.error.connect(self._on_error)
        # QThread.finished occurs AFTER run() exits; finished_all does not.
        self._worker.finished.connect(self._on_finished)
        self._worker.start()

    @Slot(int, int)
    def _on_progress(self, current, total):
        self._progress_bar.setValue(current)
        self._status_label.setText(
            f"{current} / {total} · {_fmt_seconds(time.monotonic() - self._start_time)}")

    @Slot(str, list)
    def _on_image_done(self, image_path, labels):
        self._success += 1
        if not labels:
            self._empty += 1
        self.labels_generated.emit(image_path, labels)
        self._existing_paths.add(image_path)

    @Slot(str)
    def _on_error(self, message):
        self._errors += 1
        self._error_log.show()
        self._error_log.appendPlainText(message)

    @Slot()
    def _on_finished(self):
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._set_running(False)
        self._status_label.setText(tr("improved_auto_summary").format(
            success=self._success, empty=self._empty, failed=self._errors,
            skipped=self._skipped,
            state=tr("improved_cancelled" if self._cancelled else "improved_finished")))

    def reject(self):
        if self._worker is not None:
            self._cancelled = True
            self._worker.abort()
            self._status_label.setText(tr("improved_cancelling"))
            return
        super().reject()

    def closeEvent(self, event):
        if self._worker is not None:
            self.reject()
            event.ignore()
            return
        super().closeEvent(event)
