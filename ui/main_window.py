"""Main window for VisionAce application."""

import os

from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QFileDialog, QMessageBox,
    QStatusBar, QMenuBar, QWidget, QApplication, QDockWidget,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt, Slot

from i18n import tr, set_language, get_language
from config import get_config
from core.project_manager import ProjectManager
from core.label_manager import LabelManager, LabelItem
from core.model_manager import ModelManager
from core.export_manager import ExportManager
from ui.canvas_widget import CanvasWidget
from ui.file_list_widget import FileListWidget
from ui.label_list_widget import LabelListWidget
from ui.toolbar_widget import ToolbarWidget, ToolMode
from ui.auto_label_dialog import AutoLabelDialog
from ui.training_dialog import TrainingDialog
from ui.help_dialog import HelpDialog
from ui.help_panel import HelpPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._config = get_config()
        self._project = ProjectManager()
        self._labels = LabelManager()
        self._model = ModelManager()
        self._current_image_path = ""

        self._setup_ui()
        self._setup_menu()
        self._setup_connections()
        self._update_status_bar()

        # Restore window size
        self.resize(self._config.window_width, self._config.window_height)
        self.setWindowTitle(tr("app_title"))

        # Initialize toolbar extension from config
        self._toolbar.set_save_extension(self._config.default_save_extension)

    def _setup_ui(self):
        # Toolbar
        self._toolbar = ToolbarWidget(self)
        self.addToolBar(self._toolbar)

        # Central layout: 3-panel splitter
        self._splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self._file_list = FileListWidget(self)
        self._canvas = CanvasWidget(self)
        self._label_list = LabelListWidget(self)

        self._splitter.addWidget(self._file_list)
        self._splitter.addWidget(self._canvas)
        self._splitter.addWidget(self._label_list)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 4)
        self._splitter.setStretchFactor(2, 1)
        self._splitter.setSizes([220, 800, 220])

        self.setCentralWidget(self._splitter)

        # Status bar
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Help dock widget (collapsible panel)
        self._help_dock = QDockWidget("도움말 (F1 또는 ? 버튼)", self)
        self._help_panel = HelpPanel(self)
        self._help_dock.setWidget(self._help_panel)
        self._help_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self._help_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._help_dock)
        # Show help on first launch
        self._help_dock.show()  # Visible by default for new users

    def _setup_menu(self):
        menubar = self.menuBar()

        # --- File menu ---
        file_menu = menubar.addMenu(tr("menu_file"))

        self._action_open = QAction(tr("action_open_folder"), self)
        self._action_open.setShortcut(QKeySequence("Ctrl+O"))
        self._action_open.triggered.connect(self._on_open_folder)
        file_menu.addAction(self._action_open)

        # Recent directories submenu
        self._recent_dirs_menu = file_menu.addMenu("최근 폴더")
        self._update_recent_directories_menu()

        file_menu.addSeparator()

        self._action_load_model = QAction(tr("action_load_model"), self)
        self._action_load_model.triggered.connect(self._on_load_model)
        file_menu.addAction(self._action_load_model)

        # Recent models submenu
        self._recent_models_menu = file_menu.addMenu("최근 모델")
        self._update_recent_models_menu()

        file_menu.addSeparator()

        self._action_save = QAction(tr("action_save_labels"), self)
        self._action_save.setShortcut(QKeySequence("Ctrl+S"))
        self._action_save.triggered.connect(self._on_save_labels)
        file_menu.addAction(self._action_save)

        self._action_export_mask = QAction(tr("action_export_masks"), self)
        self._action_export_mask.triggered.connect(self._on_export_masks)
        file_menu.addAction(self._action_export_mask)

        file_menu.addSeparator()

        self._action_exit = QAction(tr("action_exit"), self)
        self._action_exit.setShortcut(QKeySequence("Ctrl+Q"))
        self._action_exit.triggered.connect(self.close)
        file_menu.addAction(self._action_exit)

        # --- Edit menu ---
        edit_menu = menubar.addMenu(tr("menu_edit"))

        self._action_undo = QAction(tr("action_undo"), self)
        self._action_undo.setShortcut(QKeySequence("Ctrl+Z"))
        self._action_undo.triggered.connect(self._on_undo)
        edit_menu.addAction(self._action_undo)

        self._action_redo = QAction(tr("action_redo"), self)
        self._action_redo.setShortcut(QKeySequence("Ctrl+Y"))
        self._action_redo.triggered.connect(self._on_redo)
        edit_menu.addAction(self._action_redo)

        edit_menu.addSeparator()

        self._action_delete = QAction(tr("action_delete_label"), self)
        self._action_delete.setShortcut(QKeySequence("Delete"))
        self._action_delete.triggered.connect(self._on_delete_selected)
        edit_menu.addAction(self._action_delete)

        edit_menu.addSeparator()

        # Navigation shortcuts
        self._action_prev_image = QAction("이전 이미지 (A)", self)
        self._action_prev_image.setShortcut(QKeySequence("A"))
        self._action_prev_image.triggered.connect(self._on_prev_image)
        edit_menu.addAction(self._action_prev_image)

        self._action_next_with_save = QAction("저장 후 다음 이미지 (S)", self)
        self._action_next_with_save.setShortcut(QKeySequence("S"))
        self._action_next_with_save.triggered.connect(self._on_next_with_save)
        edit_menu.addAction(self._action_next_with_save)

        self._action_next_no_save = QAction("저장 안하고 다음 이미지 (D)", self)
        self._action_next_no_save.setShortcut(QKeySequence("D"))
        self._action_next_no_save.triggered.connect(self._on_next_without_save)
        edit_menu.addAction(self._action_next_no_save)

        self._action_exclude_from_training = QAction("학습에서 제외 (F)", self)
        self._action_exclude_from_training.setShortcut(QKeySequence("F"))
        self._action_exclude_from_training.triggered.connect(self._on_exclude_from_training)
        edit_menu.addAction(self._action_exclude_from_training)

        # --- Tools menu ---
        tools_menu = menubar.addMenu(tr("menu_tools"))

        self._action_auto_label = QAction(tr("action_auto_label"), self)
        self._action_auto_label.triggered.connect(self._on_auto_label)
        tools_menu.addAction(self._action_auto_label)

        self._action_train = QAction(tr("action_training"), self)
        self._action_train.triggered.connect(self._on_training)
        tools_menu.addAction(self._action_train)

        # --- Settings menu ---
        settings_menu = menubar.addMenu(tr("menu_settings"))

        self._action_set_label_dir = QAction(tr("action_set_label_dir") if tr("action_set_label_dir") != "action_set_label_dir" else "Set Label Folder...", self)
        self._action_set_label_dir.triggered.connect(self._on_set_label_dir)
        settings_menu.addAction(self._action_set_label_dir)

        self._action_set_save_extension = QAction("저장 이미지 확장자 설정...", self)
        self._action_set_save_extension.triggered.connect(self._on_set_save_extension)
        settings_menu.addAction(self._action_set_save_extension)

        settings_menu.addSeparator()

        self._action_lang_ko = QAction(tr("action_lang_ko"), self)
        self._action_lang_ko.triggered.connect(lambda: self._switch_language("ko"))
        settings_menu.addAction(self._action_lang_ko)

        self._action_lang_en = QAction(tr("action_lang_en"), self)
        self._action_lang_en.triggered.connect(lambda: self._switch_language("en"))
        settings_menu.addAction(self._action_lang_en)

        # --- Help menu ---
        help_menu = menubar.addMenu(tr("menu_help") if tr("menu_help") != "menu_help" else "&Help")

        self._action_toggle_help_panel = QAction("도움말 패널 표시", self)
        self._action_toggle_help_panel.setCheckable(True)
        self._action_toggle_help_panel.toggled.connect(self._on_toggle_help_panel)
        help_menu.addAction(self._action_toggle_help_panel)

        self._action_help_dialog = QAction(tr("action_help_dialog") if tr("action_help_dialog") != "action_help_dialog" else "Help Dialog", self)
        self._action_help_dialog.triggered.connect(self._on_help)
        help_menu.addAction(self._action_help_dialog)

    def _setup_connections(self):
        # Toolbar mode change and brush size
        self._toolbar.mode_changed.connect(self._on_mode_changed)
        self._toolbar.brush_size_changed.connect(self._on_brush_size_changed)
        self._toolbar.brush_shape_changed.connect(self._on_brush_shape_changed)
        self._toolbar.bbox_mode_changed.connect(self._on_bbox_mode_changed)
        self._toolbar.finish_polygon_requested.connect(self._on_finish_polygon)

        # Navigation buttons
        self._toolbar.prev_image_requested.connect(self._on_prev_image)
        self._toolbar.next_without_save_requested.connect(self._on_next_without_save)
        self._toolbar.next_with_save_requested.connect(self._on_next_with_save)

        # Save extension change
        self._toolbar.save_extension_changed.connect(self._on_save_extension_changed)

        # Help - toggle help panel instead of dialog
        self._toolbar.help_requested.connect(self._on_toolbar_help)

        # Help dock visibility tracking
        self._help_dock.visibilityChanged.connect(self._on_help_dock_visibility_changed)

        # File list selection
        self._file_list.image_selected.connect(self._on_image_selected)

        # Canvas signals
        self._canvas.label_created.connect(self._on_label_created)
        self._canvas.label_selected.connect(self._on_canvas_label_selected)
        self._canvas.label_updated.connect(self._on_label_updated)
        self._canvas.cursor_moved.connect(self._on_cursor_moved)
        self._canvas.zoom_changed.connect(self._on_zoom_changed)
        self._canvas.skip_image_requested.connect(self._on_skip_image)
        self._canvas.label_delete_requested.connect(self._on_delete_instance)

        # Label list signals
        self._label_list.class_selected.connect(self._on_class_selected)
        self._label_list.instance_selected.connect(self._on_instance_selected)
        self._label_list.delete_instance_requested.connect(self._on_delete_instance)

        # Label manager changes
        self._labels.labels_changed.connect(self._on_labels_changed)

        # Model loaded
        self._model.model_loaded.connect(self._on_model_loaded)

        # Project changes
        self._project.folder_changed.connect(self._on_folder_loaded)

    # --- Menu action handlers ---

    def _on_open_folder(self):
        path = QFileDialog.getExistingDirectory(self, tr("select_folder"),
                                                 self._config.recent_image_dir)
        if path:
            self._config.recent_image_dir = path
            self._config.add_recent_directory(path)
            self._project.open_folder(path)
            self._update_recent_directories_menu()

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("select_model"), self._config.recent_model_path,
            "Model Files (*.pt *.h5);;PyTorch Models (*.pt);;Keras Models (*.h5);;All Files (*.*)"
        )
        if path:
            self._config.recent_model_path = os.path.dirname(path)
            self._config.add_recent_model(path)
            self._update_recent_models_menu()
            try:
                # Detect model type from file extension and filename
                basename = os.path.basename(path).lower()
                if path.endswith('.h5'):
                    model_type = "KERAS"
                elif "rtdetr" in basename or "rt-detr" in basename:
                    model_type = "RT-DETR"
                else:
                    model_type = "YOLO"
                self._model.load_model(path, model_type)
            except Exception as e:
                QMessageBox.critical(self, tr("error"), str(e))

    def _on_save_labels(self):
        if not self._project.image_dir:
            return

        import cv2
        from pathlib import Path
        import shutil

        # Get extension from toolbar
        extension = self._toolbar.get_save_extension()

        # Create folders
        gt_image_dir = Path(self._project.image_dir) / "gt_image"
        images_dir = Path(self._project.image_dir) / "images"
        gt_image_dir.mkdir(parents=True, exist_ok=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        label_count = 0
        gt_count = 0
        image_count = 0

        # Process each image
        for img_path in self._project.image_list:
            labels = self._labels.get_labels(img_path)
            if not labels:
                continue

            # Separate bbox/polygon and mask labels
            bbox_polygon_labels = [l for l in labels if l.label_type in ("bbox", "polygon")]
            mask_labels = [l for l in labels if l.label_type == "mask"]

            # Save YOLO txt labels for bbox/polygon
            if bbox_polygon_labels:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    label_path = self._project.get_label_path(img_path)
                    if label_path:
                        ExportManager.save_yolo_txt(bbox_polygon_labels, w, h, label_path)
                        label_count += 1

            # Save GT images for segmentation masks (organized by class)
            if mask_labels:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    img_file = Path(img_path)

                    # Group masks by class name
                    for label in mask_labels:
                        class_dir = gt_image_dir / label.class_name
                        class_dir.mkdir(parents=True, exist_ok=True)
                        # Use selected extension or original
                        if extension:
                            gt_filename = img_file.stem + extension
                        else:
                            gt_filename = img_file.name
                        gt_image_path = class_dir / gt_filename
                        # Save individual mask for this class
                        ExportManager.save_semantic_mask([label], w, h, str(gt_image_path), multi_label=False)
                    gt_count += 1

            # Copy original image to images folder
            img_file = Path(img_path)
            # Use selected extension or original
            if extension:
                dest_filename = img_file.stem + extension
                dest_path = images_dir / dest_filename
                # Read and save with new extension
                img = cv2.imread(img_path)
                if img is not None and not dest_path.exists():
                    cv2.imwrite(str(dest_path), img)
                    image_count += 1
            else:
                dest_path = images_dir / img_file.name
                if not dest_path.exists():
                    shutil.copy2(img_path, dest_path)
                    image_count += 1

        # Show detailed status message
        status_parts = []
        if label_count > 0:
            status_parts.append(f"{label_count} label files")
        if gt_count > 0:
            status_parts.append(f"{gt_count} GT images (by class)")
        if image_count > 0:
            status_parts.append(f"{image_count} images copied")

        if status_parts:
            self._status_bar.showMessage(f"✓ Saved: {', '.join(status_parts)}", 5000)
        else:
            self._status_bar.showMessage("No labels to save", 3000)

        # Update file list label status
        for i, img_path in enumerate(self._project.image_list):
            has = len(self._labels.get_labels(img_path)) > 0
            self._file_list.update_label_status(i, has)

    def _on_export_masks(self):
        if not self._project.image_dir:
            return

        # Ask user for mask format
        reply = QMessageBox.question(
            self,
            tr("export_mask_format_title") if hasattr(self, "tr") else "Mask Format Selection",
            tr("export_mask_format_message") if hasattr(self, "tr") else
            "Export as multi-label semantic mask?\n"
            "Yes: Pixel value = Class ID + 1 (semantic segmentation)\n"
            "No: Binary mask (foreground=255, background=0)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        multi_label = (reply == QMessageBox.StandardButton.Yes)

        import cv2
        from pathlib import Path

        # Create gt_image folder
        gt_image_dir = Path(self._project.image_dir) / "gt_image"
        gt_image_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for img_path in self._project.image_list:
            labels = self._labels.get_labels(img_path)
            if labels:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    img_file = Path(img_path)
                    # Save mask in gt_image folder with same name and extension
                    mask_path = gt_image_dir / img_file.name
                    ExportManager.save_semantic_mask(labels, w, h, str(mask_path), multi_label)
                    count += 1

        mask_type = "semantic" if multi_label else "binary"
        self._status_bar.showMessage(
            f"Exported {count} {mask_type} mask files to gt_image/", 3000
        )

    def _on_undo(self):
        self._labels.undo_stack.undo()

    def _on_redo(self):
        self._labels.undo_stack.redo()

    def _on_delete_selected(self):
        # Delete currently selected label from canvas
        if self._current_image_path:
            selected_idx = self._canvas.get_selected_index()
            if selected_idx >= 0:
                self._labels.remove_label(self._current_image_path, selected_idx)

    def _on_auto_label(self):
        if not self._model.is_loaded:
            QMessageBox.warning(self, tr("warning"), tr("auto_label_no_model"))
            return
        if not self._project.image_list:
            return

        dialog = AutoLabelDialog(
            self._model,
            self._project.image_list,
            self._file_list.current_index(),
            self,
        )
        dialog.labels_generated.connect(self._on_auto_labels_received)
        dialog.exec()

    def _on_training(self):
        classes = self._label_list.get_classes()
        class_names = [c["name"] for c in classes]
        dialog = TrainingDialog(
            class_names,
            self._project.image_dir or "",
            self,
        )
        dialog.training_complete.connect(self._on_training_complete)
        dialog.exec()

    def _on_help(self):
        """Show help dialog (from menu)."""
        dialog = HelpDialog(self)
        dialog.exec()

    def _on_toolbar_help(self):
        """Toggle help panel (from toolbar)."""
        is_visible = self._help_dock.isVisible()
        if is_visible:
            self._help_dock.hide()
        else:
            self._help_dock.show()
            self._help_dock.raise_()
            # Force focus
            self._help_dock.activateWindow()

    def _on_toggle_help_panel(self, visible: bool):
        """Toggle help panel visibility from menu/shortcut."""
        self._help_dock.setVisible(visible)
        if visible:
            self._help_dock.raise_()

    def _on_help_dock_visibility_changed(self, visible: bool):
        """Sync help panel toggle action with dock visibility."""
        self._action_toggle_help_panel.setChecked(visible)

    def _on_set_label_dir(self):
        """Set a custom label directory."""
        if not self._project.image_dir:
            QMessageBox.warning(
                self,
                tr("warning"),
                tr("label_dir_no_project") if tr("label_dir_no_project") != "label_dir_no_project" else "Please open an image folder first."
            )
            return

        # Provide option to use default or custom
        reply = QMessageBox.question(
            self,
            tr("label_dir_title") if tr("label_dir_title") != "label_dir_title" else "Label Folder",
            tr("label_dir_message") if tr("label_dir_message") != "label_dir_message" else
            "Do you want to specify a custom label folder?\n\n"
            "Yes: Choose a custom folder\n"
            "No: Use default labels/ subfolder",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Cancel:
            return
        elif reply == QMessageBox.StandardButton.No:
            # Use default labels/ subdirectory
            self._project.set_custom_label_dir("")
            self._status_bar.showMessage("Using default labels/ subfolder", 3000)
        else:
            # Choose custom directory
            path = QFileDialog.getExistingDirectory(
                self,
                tr("select_label_folder") if tr("select_label_folder") != "select_label_folder" else "Select Label Folder",
                str(self._project.image_dir)
            )
            if path:
                if self._project.set_custom_label_dir(path):
                    self._status_bar.showMessage(f"Label folder set to: {path}", 3000)
                else:
                    QMessageBox.critical(
                        self,
                        tr("error"),
                        tr("label_dir_error") if tr("label_dir_error") != "label_dir_error" else "Failed to set label folder."
                    )

    def _on_set_save_extension(self):
        """Set default save image extension."""
        from PySide6.QtWidgets import QInputDialog

        current_ext = self._config.default_save_extension
        current_display = "원본 확장자" if not current_ext else current_ext.upper()

        extensions = ["원본 확장자 사용", "PNG", "JPG", "BMP", "TIFF"]
        # Find current index
        current_idx = 0
        if current_ext == ".png":
            current_idx = 1
        elif current_ext == ".jpg":
            current_idx = 2
        elif current_ext == ".bmp":
            current_idx = 3
        elif current_ext == ".tiff":
            current_idx = 4

        extension, ok = QInputDialog.getItem(
            self,
            "기본 저장 이미지 확장자 설정",
            f"현재 설정: {current_display}\n\n자동 저장 시 사용할 이미지 형식을 선택하세요:\n(Ctrl+S로 전체 저장 시에는 별도 선택 가능)",
            extensions,
            current_idx,
            False
        )

        if ok:
            if extension == "원본 확장자 사용":
                self._config.default_save_extension = ""
            else:
                self._config.default_save_extension = f".{extension.lower()}"
            self._config.save()
            self._status_bar.showMessage(f"기본 저장 확장자: {extension}", 3000)

    # --- Signal handlers ---

    @Slot(str)
    def _on_folder_loaded(self):
        images = self._project.image_list
        self._file_list.set_image_list(images)

        # Mark label status by checking if label files exist (fast, no image read)
        for i, img_path in enumerate(images):
            has = self._project.has_labels(img_path)
            self._file_list.update_label_status(i, has)

        # Select first image (labels loaded lazily on selection)
        if images:
            self._file_list.select_image(0)

    @Slot(int)
    def _on_image_selected(self, index: int):
        img_path = self._project.get_image_path(index)
        if not img_path:
            return

        # Finalize any unfinished brush/mask work before switching
        if self._current_image_path and self._canvas.has_unfinished_mask():
            self._canvas.finish_current_shape()

        # Auto-save previous image labels
        if self._current_image_path and self._config.auto_save:
            self._save_current_labels()

        self._current_image_path = img_path
        self._canvas.load_image(img_path)

        # Lazy-load labels from disk if not yet loaded
        self._load_labels_from_disk(img_path)

        # Display labels
        labels = self._labels.get_labels(img_path)
        self._canvas.display_labels(labels)
        w, h = self._canvas.get_image_size()
        self._label_list.set_image_size(w, h)
        self._label_list.set_instances(labels)

        # Update status bar
        w, h = self._canvas.get_image_size()
        filename = os.path.basename(img_path)
        self._status_bar.showMessage(
            tr("status_image_info").format(filename=filename, width=w, height=h)
        )

    @Slot()
    def _on_skip_image(self):
        """Skip to next image without saving labels."""
        current_idx = self._file_list.current_index()
        if current_idx >= 0 and current_idx < self._project.image_count - 1:
            # Temporarily disable auto-save
            auto_save_backup = self._config.auto_save
            self._config.auto_save = False

            # Move to next image
            self._file_list.select_image(current_idx + 1)

            # Restore auto-save setting
            self._config.auto_save = auto_save_backup

            self._status_bar.showMessage("Skipped to next image (not saved)", 2000)

    @Slot()
    def _on_next_without_save(self):
        """Move to next image without saving current labels."""
        current_idx = self._file_list.current_index()
        if current_idx >= 0 and current_idx < self._project.image_count - 1:
            # Temporarily disable auto-save
            auto_save_backup = self._config.auto_save
            self._config.auto_save = False

            # Move to next image
            self._file_list.select_image(current_idx + 1)

            # Restore auto-save setting
            self._config.auto_save = auto_save_backup

            self._status_bar.showMessage("다음 이미지 (저장 안함)", 2000)

    @Slot()
    def _on_next_with_save(self):
        """Save current labels and move to next image."""
        current_idx = self._file_list.current_index()
        if current_idx >= 0 and current_idx < self._project.image_count - 1:
            # Save current labels
            if self._current_image_path:
                self._save_current_labels()

            # Move to next image
            self._file_list.select_image(current_idx + 1)

            self._status_bar.showMessage("저장 후 다음 이미지로 이동", 2000)

    @Slot()
    def _on_prev_image(self):
        """Move to previous image with auto-save."""
        current_idx = self._file_list.current_index()
        if current_idx > 0:
            # Auto-save will happen in _on_image_selected if enabled
            self._file_list.select_image(current_idx - 1)
            self._status_bar.showMessage("이전 이미지로 이동", 2000)

    @Slot()
    def _on_exclude_from_training(self):
        """Exclude current image from training by deleting image, labels, and GT images."""
        if not self._current_image_path or not self._project.image_dir:
            return

        # Confirm deletion
        from pathlib import Path
        img_name = Path(self._current_image_path).name
        reply = QMessageBox.question(
            self,
            "학습에서 제외",
            f"'{img_name}'을(를) 학습에서 제외하시겠습니까?\n\n"
            "다음 항목들이 삭제됩니다:\n"
            "- 원본 이미지\n"
            "- 라벨 파일 (.txt)\n"
            "- GT 이미지 파일들\n"
            "- images 폴더의 복사본",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        current_idx = self._file_list.current_index()

        # Delete label file
        label_path = self._project.get_label_path(self._current_image_path)
        if label_path and os.path.exists(label_path):
            os.remove(label_path)

        # Delete GT images
        gt_image_dir = Path(self._project.image_dir) / "gt_image"
        if gt_image_dir.exists():
            img_file = Path(self._current_image_path)
            # Check all class subdirectories
            for class_dir in gt_image_dir.iterdir():
                if class_dir.is_dir():
                    # Check all possible extensions
                    for ext in ['', '.png', '.jpg', '.bmp', '.tiff']:
                        if ext:
                            gt_file = class_dir / (img_file.stem + ext)
                        else:
                            gt_file = class_dir / img_file.name
                        if gt_file.exists():
                            gt_file.unlink()

        # Delete from images folder
        images_dir = Path(self._project.image_dir) / "images"
        if images_dir.exists():
            img_file = Path(self._current_image_path)
            for ext in ['', '.png', '.jpg', '.bmp', '.tiff']:
                if ext:
                    dest_file = images_dir / (img_file.stem + ext)
                else:
                    dest_file = images_dir / img_file.name
                if dest_file.exists():
                    dest_file.unlink()

        # Delete original image file
        if os.path.exists(self._current_image_path):
            os.remove(self._current_image_path)

        # Remove from label manager
        self._labels.clear_labels(self._current_image_path)

        # Reload project to update file list
        self._project.open_folder(self._project.image_dir)

        self._status_bar.showMessage(f"'{img_name}' 학습에서 제외됨 (삭제 완료)", 3000)

    @Slot(str)
    def _on_save_extension_changed(self, extension: str):
        """Update default save extension when toolbar combo changes."""
        self._config.default_save_extension = extension
        self._config.save()
        ext_display = extension if extension else "원본"
        self._status_bar.showMessage(f"저장 확장자: {ext_display}", 2000)

    @Slot(str)
    def _on_mode_changed(self, mode: str):
        self._canvas.set_mode(mode)

    @Slot(int)
    def _on_brush_size_changed(self, size: int):
        """Update canvas brush size."""
        self._canvas.set_brush_size(size)

    @Slot(str)
    def _on_brush_shape_changed(self, shape: str):
        """Update canvas brush shape."""
        self._canvas.set_brush_shape(shape)

    @Slot(str)
    def _on_bbox_mode_changed(self, mode: str):
        """Update canvas bbox mode."""
        self._canvas.set_bbox_mode(mode)

    @Slot()
    def _on_finish_polygon(self):
        """Finish current polygon drawing."""
        self._canvas.finish_current_shape()

    @Slot(object)
    def _on_label_created(self, label: LabelItem):
        if self._current_image_path:
            self._labels.add_label(self._current_image_path, label)

    @Slot(int)
    def _on_canvas_label_selected(self, index: int):
        self._label_list.select_instance(index)

    @Slot(int, object)
    def _on_label_updated(self, index: int, updated_label: LabelItem):
        """Handle label update from canvas editing."""
        if self._current_image_path:
            self._labels.update_label(self._current_image_path, index, updated_label)

    @Slot(int, int)
    def _on_cursor_moved(self, x: int, y: int):
        mode = self._toolbar.current_mode()
        self._status_bar.showMessage(
            f"{tr('status_mode').format(mode=mode)}  |  {tr('status_cursor').format(x=x, y=y)}"
        )

    @Slot(float)
    def _on_zoom_changed(self, zoom: float):
        pass  # Could update status bar zoom indicator

    @Slot(int)
    def _on_class_selected(self, class_id: int):
        classes = self._label_list.get_classes()
        if 0 <= class_id < len(classes):
            cls = classes[class_id]
            self._canvas.set_current_class(class_id, cls["name"], cls["color"])

    @Slot(int)
    def _on_instance_selected(self, index: int):
        self._canvas.highlight_label(index)

    @Slot(int)
    def _on_delete_instance(self, index: int):
        if self._current_image_path:
            self._labels.remove_label(self._current_image_path, index)

    @Slot(str)
    def _on_labels_changed(self, image_path: str):
        if image_path == self._current_image_path:
            labels = self._labels.get_labels(image_path)
            self._canvas.display_labels(labels)
            w, h = self._canvas.get_image_size()
            self._label_list.set_image_size(w, h)
            self._label_list.set_instances(labels)

        # Update file list icon
        idx = self._project.get_image_index(image_path)
        if idx >= 0:
            has = len(self._labels.get_labels(image_path)) > 0
            self._file_list.update_label_status(idx, has)

    @Slot(str)
    def _on_model_loaded(self, path: str):
        name = os.path.basename(path)
        self._status_bar.showMessage(
            tr("status_model_loaded").format(name=name), 5000
        )
        # Import model class names as label classes
        class_names = self._model.get_class_names()
        if class_names and self._label_list.get_class_count() == 0:
            from ui.label_list_widget import DEFAULT_COLORS
            classes = []
            for cid, cname in class_names.items():
                classes.append({
                    "name": cname,
                    "color": DEFAULT_COLORS[int(cid) % len(DEFAULT_COLORS)],
                })
            self._label_list.set_classes(classes)

    @Slot(str, list)
    def _on_auto_labels_received(self, image_path: str, labels: list):
        for label in labels:
            self._labels.add_label(image_path, label)

    @Slot(str)
    def _on_training_complete(self, best_path: str):
        QMessageBox.information(
            self, tr("info"),
            tr("training_complete").format(path=best_path)
        )

    # --- Helpers ---

    def _load_labels_from_disk(self, image_path: str):
        """Load labels from YOLO txt file if exists (lazy, uses canvas size)."""
        if self._labels.get_labels(image_path):
            return  # Already loaded

        label_path = self._project.get_label_path(image_path)
        if not label_path or not os.path.isfile(label_path):
            return

        # Use canvas image size (already loaded) instead of cv2.imread
        w, h = self._canvas.get_image_size()
        if w == 0 or h == 0:
            return

        classes = self._label_list.get_classes()
        class_names = {i: c["name"] for i, c in enumerate(classes)}

        labels = ExportManager.load_yolo_txt(label_path, w, h, class_names)
        if labels:
            for label in labels:
                label.color = self._label_list.get_class_color(label.class_id)
            self._labels.set_labels(image_path, labels)

    def _save_current_labels(self):
        """Save labels for current image to disk."""
        if not self._current_image_path or not self._project.image_dir:
            return

        labels = self._labels.get_labels(self._current_image_path)
        w, h = self._canvas.get_image_size()
        if w == 0 or h == 0:
            return

        from pathlib import Path
        import shutil

        # If no labels, delete existing txt file and GT images
        if not labels:
            label_path = self._project.get_label_path(self._current_image_path)
            if label_path and os.path.exists(label_path):
                os.remove(label_path)

            # Remove GT images for this image
            gt_image_dir = Path(self._project.image_dir) / "gt_image"
            if gt_image_dir.exists():
                img_file = Path(self._current_image_path)
                # Check all class subdirectories
                for class_dir in gt_image_dir.iterdir():
                    if class_dir.is_dir():
                        gt_file = class_dir / img_file.name
                        if gt_file.exists():
                            gt_file.unlink()
            return

        # Separate bbox/polygon labels and mask labels
        bbox_polygon_labels = [l for l in labels if l.label_type in ("bbox", "polygon")]
        mask_labels = [l for l in labels if l.label_type == "mask"]

        saved_items = []

        # Save YOLO txt labels for bbox/polygon
        if bbox_polygon_labels:
            label_path = self._project.get_label_path(self._current_image_path)
            if label_path:
                ExportManager.save_yolo_txt(bbox_polygon_labels, w, h, label_path)
                saved_items.append(f"{len(bbox_polygon_labels)} labels")
        else:
            # No bbox/polygon labels, delete txt file if exists
            label_path = self._project.get_label_path(self._current_image_path)
            if label_path and os.path.exists(label_path):
                os.remove(label_path)

        # Get default extension from config
        extension = self._config.default_save_extension

        # Save GT image for segmentation masks (organized by class)
        if mask_labels:
            gt_image_dir = Path(self._project.image_dir) / "gt_image"
            gt_image_dir.mkdir(parents=True, exist_ok=True)

            img_file = Path(self._current_image_path)

            # Save each mask in its class-specific folder
            for label in mask_labels:
                class_dir = gt_image_dir / label.class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                # Use selected extension or original
                if extension:
                    gt_filename = img_file.stem + extension
                else:
                    gt_filename = img_file.name
                gt_image_path = class_dir / gt_filename
                # Save individual mask for this class
                ExportManager.save_semantic_mask([label], w, h, str(gt_image_path), multi_label=False)
            saved_items.append(f"GT images ({len(mask_labels)} classes)")

        # Copy original image to images folder if any labels exist
        if labels:
            import cv2
            images_dir = Path(self._project.image_dir) / "images"
            images_dir.mkdir(parents=True, exist_ok=True)

            img_file = Path(self._current_image_path)

            # Use selected extension or original
            if extension:
                dest_filename = img_file.stem + extension
                dest_path = images_dir / dest_filename
                # Read and save with new extension
                img = cv2.imread(self._current_image_path)
                if img is not None and not dest_path.exists():
                    cv2.imwrite(str(dest_path), img)
                    saved_items.append("image")
            else:
                dest_path = images_dir / img_file.name
                # Copy only if not already in images folder
                if not dest_path.exists():
                    shutil.copy2(self._current_image_path, dest_path)
                    saved_items.append("image")

        # Show status message
        if saved_items:
            img_name = Path(self._current_image_path).name
            self._status_bar.showMessage(f"Saved {img_name}: {', '.join(saved_items)}", 2000)

    def _switch_language(self, lang: str):
        set_language(lang)
        self._config.language = lang
        self._config.save()
        self._retranslate_ui()

    def _retranslate_ui(self):
        self.setWindowTitle(tr("app_title"))
        self._toolbar.retranslate()
        self._file_list.retranslate()
        self._label_list.retranslate()
        self._canvas.retranslate()
        self._help_panel.retranslate()

        # Re-create menus
        self.menuBar().clear()
        self._setup_menu()

    def _update_recent_directories_menu(self):
        """Update the recent directories menu with current list."""
        self._recent_dirs_menu.clear()

        recent_dirs = self._config.recent_directories
        if not recent_dirs:
            no_recent_action = QAction("(없음)", self)
            no_recent_action.setEnabled(False)
            self._recent_dirs_menu.addAction(no_recent_action)
            return

        for path in recent_dirs:
            if os.path.exists(path):
                # Show only the directory name (last part of path)
                dir_name = os.path.basename(path) or path
                action = QAction(f"{dir_name}  ({path})", self)
                action.triggered.connect(lambda checked=False, p=path: self._on_open_recent_directory(p))
                self._recent_dirs_menu.addAction(action)

        # Add separator and clear option
        self._recent_dirs_menu.addSeparator()
        clear_action = QAction("목록 지우기", self)
        clear_action.triggered.connect(self._on_clear_recent_directories)
        self._recent_dirs_menu.addAction(clear_action)

    def _on_open_recent_directory(self, path: str):
        """Open a recently used directory."""
        if os.path.exists(path):
            self._config.recent_image_dir = path
            self._config.add_recent_directory(path)
            self._project.open_folder(path)
            self._update_recent_directories_menu()
        else:
            QMessageBox.warning(
                self,
                "폴더를 찾을 수 없음",
                f"폴더가 존재하지 않습니다:\n{path}"
            )

    def _on_clear_recent_directories(self):
        """Clear the recent directories list."""
        self._config.recent_directories = []
        self._config.save()
        self._update_recent_directories_menu()

    def _update_recent_models_menu(self):
        """Update the recent models menu with current list."""
        self._recent_models_menu.clear()

        recent_models = self._config.recent_models
        if not recent_models:
            no_recent_action = QAction("(없음)", self)
            no_recent_action.setEnabled(False)
            self._recent_models_menu.addAction(no_recent_action)
            return

        for path in recent_models:
            if os.path.exists(path):
                # Show model filename and type
                model_name = os.path.basename(path)
                action = QAction(f"{model_name}  ({path})", self)
                action.triggered.connect(lambda checked=False, p=path: self._on_open_recent_model(p))
                self._recent_models_menu.addAction(action)

        # Add separator and clear option
        self._recent_models_menu.addSeparator()
        clear_action = QAction("목록 지우기", self)
        clear_action.triggered.connect(self._on_clear_recent_models)
        self._recent_models_menu.addAction(clear_action)

    def _on_open_recent_model(self, path: str):
        """Open a recently used model."""
        if os.path.exists(path):
            self._config.recent_model_path = os.path.dirname(path)
            self._config.add_recent_model(path)
            self._update_recent_models_menu()
            try:
                # Detect model type from file extension and filename
                basename = os.path.basename(path).lower()
                if path.endswith('.h5'):
                    model_type = "KERAS"
                elif "rtdetr" in basename or "rt-detr" in basename:
                    model_type = "RT-DETR"
                else:
                    model_type = "YOLO"
                self._model.load_model(path, model_type)
            except Exception as e:
                QMessageBox.critical(self, tr("error"), str(e))
        else:
            QMessageBox.warning(
                self,
                "모델 파일을 찾을 수 없음",
                f"모델 파일이 존재하지 않습니다:\n{path}"
            )

    def _on_clear_recent_models(self):
        """Clear the recent models list."""
        self._config.recent_models = []
        self._config.save()
        self._update_recent_models_menu()

    def _update_status_bar(self):
        model_status = (
            tr("status_model_loaded").format(name=os.path.basename(self._model.get_model_path()))
            if self._model.is_loaded
            else tr("status_no_model")
        )
        self._status_bar.showMessage(f"{tr('status_ready')}  |  {model_status}")

    def closeEvent(self, event):
        # Save labels on exit
        if self._current_image_path and self._config.auto_save:
            self._save_current_labels()
        self._config.window_width = self.width()
        self._config.window_height = self.height()
        self._config.save()
        super().closeEvent(event)
