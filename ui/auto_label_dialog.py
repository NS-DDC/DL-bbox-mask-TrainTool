"""Auto labeling dialog for batch inference."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QDoubleSpinBox, QRadioButton,
    QPushButton, QProgressBar, QButtonGroup, QGroupBox,
    QMessageBox, QWidget,
)
from PySide6.QtCore import Signal, Slot

from i18n import tr
from core.auto_labeler import AutoLabelWorker


class AutoLabelDialog(QDialog):
    labels_generated = Signal(str, list)  # image_path, list[LabelItem]

    def __init__(self, model_manager, image_paths: list[str],
                 current_index: int = 0, parent: QWidget = None):
        super().__init__(parent)
        self._model_manager = model_manager
        self._image_paths = image_paths
        self._current_index = current_index
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(tr("auto_label_title"))
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)

        # Model info
        form = QFormLayout()
        self._model_label = QLabel(
            self._model_manager.get_model_type() if self._model_manager.is_loaded else tr("auto_label_no_model")
        )
        form.addRow(tr("auto_label_model"), self._model_label)

        # Confidence threshold
        self._confidence_spin = QDoubleSpinBox()
        self._confidence_spin.setRange(0.01, 1.0)
        self._confidence_spin.setSingleStep(0.05)
        self._confidence_spin.setValue(0.5)
        self._confidence_spin.setDecimals(2)
        form.addRow(tr("auto_label_confidence"), self._confidence_spin)

        layout.addLayout(form)

        # Scope selection
        scope_group = QGroupBox(tr("auto_label_scope"))
        scope_layout = QVBoxLayout(scope_group)
        self._scope_btn_group = QButtonGroup(self)

        self._current_radio = QRadioButton(tr("auto_label_current"))
        self._current_radio.setChecked(True)
        self._scope_btn_group.addButton(self._current_radio, 0)
        scope_layout.addWidget(self._current_radio)

        self._all_radio = QRadioButton(tr("auto_label_all"))
        self._scope_btn_group.addButton(self._all_radio, 1)
        scope_layout.addWidget(self._all_radio)

        layout.addWidget(scope_group)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._status_label = QLabel("")
        layout.addWidget(self._status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton(tr("auto_label_start"))
        self._start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self._start_btn)

        self._cancel_btn = QPushButton(tr("auto_label_cancel"))
        self._cancel_btn.clicked.connect(self._on_cancel)
        btn_layout.addWidget(self._cancel_btn)

        layout.addLayout(btn_layout)

    def _on_start(self):
        if not self._model_manager.is_loaded:
            QMessageBox.warning(self, tr("warning"), tr("auto_label_no_model"))
            return

        # Determine which images to process
        if self._scope_btn_group.checkedId() == 0:
            paths = [self._image_paths[self._current_index]]
        else:
            paths = self._image_paths

        confidence = self._confidence_spin.value()

        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(len(paths))
        self._progress_bar.setValue(0)
        self._start_btn.setEnabled(False)

        self._worker = AutoLabelWorker(self._model_manager, paths, confidence)
        self._worker.progress.connect(self._on_progress)
        self._worker.image_done.connect(self._on_image_done)
        self._worker.finished_all.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    @Slot(int, int)
    def _on_progress(self, current: int, total: int):
        self._progress_bar.setValue(current)
        self._status_label.setText(
            tr("auto_label_progress").format(current=current, total=total)
        )

    @Slot(str, list)
    def _on_image_done(self, image_path: str, labels: list):
        self.labels_generated.emit(image_path, labels)

    @Slot()
    def _on_finished(self):
        self._start_btn.setEnabled(True)
        self._status_label.setText(tr("auto_label_complete").format(count=""))
        self._worker = None

    @Slot(str)
    def _on_error(self, msg: str):
        self._start_btn.setEnabled(True)
        QMessageBox.critical(self, tr("error"), msg)
        self._worker = None

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(3000)
            self._worker = None
        self.reject()

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(3000)
        super().closeEvent(event)
