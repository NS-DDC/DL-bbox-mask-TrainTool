"""Training configuration and execution dialog."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QProgressBar, QTextEdit, QLineEdit,
    QFileDialog, QGroupBox, QWidget, QMessageBox, QCheckBox,
)
from PySide6.QtCore import Signal, Slot

from i18n import tr
from core.trainer import TrainWorker, generate_data_yaml


class TrainingDialog(QDialog):
    training_complete = Signal(str)  # best model path

    def __init__(self, class_names: list[str], project_dir: str = "",
                 parent: QWidget = None):
        super().__init__(parent)
        self._class_names = class_names
        self._project_dir = project_dir
        self._worker = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle(tr("training_title"))
        self.setMinimumSize(550, 600)
        layout = QVBoxLayout(self)

        # --- Model settings ---
        model_group = QGroupBox(tr("training_title"))
        model_form = QFormLayout(model_group)

        self._model_type_combo = QComboBox()
        self._model_type_combo.addItems(["RT-DETR", "YOLOv8", "YOLOv11"])
        model_form.addRow(tr("training_model_type"), self._model_type_combo)

        self._base_model_edit = QLineEdit("rtdetr-l.pt")
        base_model_layout = QHBoxLayout()
        base_model_layout.addWidget(self._base_model_edit)
        self._browse_model_btn = QPushButton(tr("training_browse"))
        self._browse_model_btn.clicked.connect(self._browse_base_model)
        base_model_layout.addWidget(self._browse_model_btn)
        model_form.addRow(tr("training_base_model"), base_model_layout)

        layout.addWidget(model_group)

        # --- Dataset settings ---
        data_group = QGroupBox(tr("training_dataset"))
        data_form = QFormLayout(data_group)

        self._dataset_edit = QLineEdit(self._project_dir)
        dataset_layout = QHBoxLayout()
        dataset_layout.addWidget(self._dataset_edit)
        self._browse_dataset_btn = QPushButton(tr("training_browse"))
        self._browse_dataset_btn.clicked.connect(self._browse_dataset)
        dataset_layout.addWidget(self._browse_dataset_btn)
        data_form.addRow(tr("training_dataset"), dataset_layout)

        self._generate_yaml_check = QCheckBox(tr("training_generate_yaml"))
        self._generate_yaml_check.setChecked(True)
        data_form.addRow(self._generate_yaml_check)

        self._classes_label = QLabel(", ".join(self._class_names) if self._class_names else "-")
        self._classes_label.setWordWrap(True)
        data_form.addRow(tr("training_classes"), self._classes_label)

        layout.addWidget(data_group)

        # --- Hyperparameters ---
        hyper_group = QGroupBox("Hyperparameters")
        hyper_form = QFormLayout(hyper_group)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 1000)
        self._epochs_spin.setValue(50)
        hyper_form.addRow(tr("training_epochs"), self._epochs_spin)

        self._batch_spin = QSpinBox()
        self._batch_spin.setRange(1, 128)
        self._batch_spin.setValue(16)
        hyper_form.addRow(tr("training_batch_size"), self._batch_spin)

        self._imgsz_combo = QComboBox()
        self._imgsz_combo.addItems(["320", "416", "512", "640", "800", "1024"])
        self._imgsz_combo.setCurrentText("640")
        hyper_form.addRow(tr("training_img_size"), self._imgsz_combo)

        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setRange(0.00001, 0.1)
        self._lr_spin.setSingleStep(0.0001)
        self._lr_spin.setValue(0.001)
        self._lr_spin.setDecimals(5)
        hyper_form.addRow(tr("training_lr"), self._lr_spin)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "cpu", "0", "0,1"])
        hyper_form.addRow(tr("training_device"), self._device_combo)

        layout.addWidget(hyper_group)

        # --- Log output ---
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setMaximumHeight(200)
        layout.addWidget(QLabel(tr("training_log")))
        layout.addWidget(self._log_text)

        # --- Progress bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        # --- Buttons ---
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton(tr("training_start"))
        self._start_btn.clicked.connect(self._on_start)
        btn_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton(tr("training_stop"))
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._on_stop)
        btn_layout.addWidget(self._stop_btn)

        self._close_btn = QPushButton(tr("training_close"))
        self._close_btn.clicked.connect(self.close)
        btn_layout.addWidget(self._close_btn)

        layout.addLayout(btn_layout)

    def _browse_base_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("select_model"), "", tr("model_files")
        )
        if path:
            self._base_model_edit.setText(path)

    def _browse_dataset(self):
        path = QFileDialog.getExistingDirectory(self, tr("select_folder"))
        if path:
            self._dataset_edit.setText(path)

    def _on_start(self):
        dataset_path = self._dataset_edit.text().strip()
        if not dataset_path:
            QMessageBox.warning(self, tr("warning"), "Dataset path is required.")
            return

        model_type_text = self._model_type_combo.currentText()
        if model_type_text == "RT-DETR":
            model_type = "RT-DETR"
        else:
            model_type = "YOLO"

        base_model = self._base_model_edit.text().strip()
        epochs = self._epochs_spin.value()
        batch_size = self._batch_spin.value()
        imgsz = int(self._imgsz_combo.currentText())
        lr = self._lr_spin.value()
        device = self._device_combo.currentText()
        if device == "auto":
            device = ""

        # Generate data.yaml if requested
        import os
        data_yaml_path = os.path.join(dataset_path, "data.yaml")
        if self._generate_yaml_check.isChecked():
            train_path = os.path.join(dataset_path, "images", "train")
            val_path = os.path.join(dataset_path, "images", "val")
            if not os.path.isdir(train_path):
                train_path = os.path.join(dataset_path, "images")
            if not os.path.isdir(val_path):
                val_path = train_path
            generate_data_yaml(train_path, val_path, self._class_names, data_yaml_path)
            self._log_text.append(f"Generated data.yaml at: {data_yaml_path}")

        self._log_text.clear()
        self._progress_bar.setVisible(True)
        self._progress_bar.setMaximum(epochs)
        self._progress_bar.setValue(0)
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = TrainWorker(
            model_type=model_type,
            base_model_path=base_model,
            data_yaml_path=data_yaml_path,
            epochs=epochs,
            batch_size=batch_size,
            imgsz=imgsz,
            lr=lr,
            device=device,
        )
        self._worker.log_message.connect(self._on_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_training.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._worker.wait(3000)
            self._worker = None
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log_text.append("Training stopped by user.")

    @Slot(str)
    def _on_log(self, msg: str):
        self._log_text.append(msg)

    @Slot(int, int, float)
    def _on_progress(self, epoch: int, total: int, loss: float):
        self._progress_bar.setValue(epoch)
        self._log_text.append(
            tr("training_progress").format(epoch=epoch, total=total, loss=loss)
        )

    @Slot(str)
    def _on_finished(self, best_path: str):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log_text.append(tr("training_complete").format(path=best_path))
        self.training_complete.emit(best_path)

    @Slot(str)
    def _on_error(self, msg: str):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log_text.append(f"ERROR: {msg}")
        QMessageBox.critical(self, tr("error"), msg)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            reply = QMessageBox.question(
                self, tr("warning"),
                "Training is running. Stop and close?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                event.ignore()
                return
            self._worker.terminate()
            self._worker.wait(3000)
        super().closeEvent(event)
