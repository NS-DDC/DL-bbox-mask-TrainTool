"""Main window for VisionAce application."""

import os
import shutil
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QSplitter, QFileDialog, QMessageBox,
    QStatusBar, QMenuBar, QWidget, QApplication, QDockWidget, QInputDialog,
)
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtCore import Qt, Slot, QThread

from i18n import tr, set_language, get_language
from config import get_config
from core.project_manager import ProjectManager
from core.label_manager import LabelManager, LabelItem
from core.model_manager import ModelManager
from core.export_manager import ExportManager
from core.save_manager import SaveManager
from core.image_io import read_image
from core.project_metadata import load_classes, save_classes
from ui.canvas_widget import CanvasWidget
from ui.file_list_widget import FileListWidget
from ui.label_list_widget import LabelListWidget
from ui.toolbar_widget import ToolbarWidget, ToolMode
from ui.auto_label_dialog import AutoLabelDialog
from ui.help_dialog import HelpDialog
from ui.help_panel import HelpPanel


class _ModelLoadWorker(QThread):
    def __init__(self, manager, path, model_type, parent=None):
        super().__init__(parent)
        self.manager, self.path, self.model_type = manager, path, model_type
        self.success = False

    def run(self):
        self.success = self.manager.load_model(self.path, self.model_type)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._config = get_config()
        self._project = ProjectManager()
        self._labels = LabelManager()
        self._model = ModelManager()
        self._saver = SaveManager(self._labels, self._project)
        self._current_image_path = ""
        self._model_loader = None
        self._opening_classes = []
        self._restoring_project = False
        self._skip_auto_load_mask = False  # suppress auto-load during explicit mask edit
        self._discard_pending_mask = False  # discard (not finalize) mask on next image switch

        self._setup_ui()
        self._setup_menu()
        self._setup_connections()
        self._update_status_bar()

        # Restore window size
        self.resize(self._config.window_width, self._config.window_height)
        self.setWindowTitle(tr("app_title"))

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
        self._help_dock = QDockWidget(tr("help_dock_title"), self)
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
        self._recent_dirs_menu = file_menu.addMenu(tr("menu_recent_dirs"))
        self._update_recent_directories_menu()

        file_menu.addSeparator()

        self._action_load_model = QAction(tr("action_load_model"), self)
        self._action_load_model.triggered.connect(self._on_load_model)
        file_menu.addAction(self._action_load_model)

        # Recent models submenu
        self._recent_models_menu = file_menu.addMenu(tr("menu_recent_models"))
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

        # Import external labels/GT
        self._action_import_labels = QAction(tr("action_import_labels"), self)
        self._action_import_labels.triggered.connect(self._on_import_external_labels)
        file_menu.addAction(self._action_import_labels)

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
        self._action_prev_image = QAction(tr("action_prev_image"), self)
        self._action_prev_image.setShortcut(QKeySequence("A"))
        self._action_prev_image.triggered.connect(self._on_prev_image)
        edit_menu.addAction(self._action_prev_image)

        self._action_next_with_save = QAction(tr("action_next_save"), self)
        self._action_next_with_save.setShortcut(QKeySequence("S"))
        self._action_next_with_save.triggered.connect(self._on_next_with_save)
        edit_menu.addAction(self._action_next_with_save)

        self._action_next_no_save = QAction(tr("action_next_no_save"), self)
        self._action_next_no_save.setShortcut(QKeySequence("D"))
        self._action_next_no_save.triggered.connect(self._on_next_without_save)
        edit_menu.addAction(self._action_next_no_save)

        self._action_exclude_from_training = QAction(tr("action_exclude_training"), self)
        self._action_exclude_from_training.setShortcut(QKeySequence("F"))
        self._action_exclude_from_training.triggered.connect(self._on_exclude_from_training)
        edit_menu.addAction(self._action_exclude_from_training)

        view_menu = menubar.addMenu(tr("improved_view"))
        actual = QAction(tr("improved_actual_size"), self)
        actual.setShortcut("Ctrl+1")
        actual.triggered.connect(self._canvas.actual_size)
        view_menu.addAction(actual)
        fit = QAction(tr("improved_fit"), self)
        fit.setShortcut("Ctrl+0")
        fit.triggered.connect(self._canvas.fit_to_window)
        view_menu.addAction(fit)

        # --- Tools menu ---
        tools_menu = menubar.addMenu(tr("menu_tools"))

        self._action_auto_label = QAction(tr("action_auto_label"), self)
        self._action_auto_label.triggered.connect(self._on_auto_label)
        tools_menu.addAction(self._action_auto_label)

        # --- Settings menu ---
        settings_menu = menubar.addMenu(tr("menu_settings"))

        self._action_set_label_dir = QAction(tr("action_set_label_dir"), self)
        self._action_set_label_dir.triggered.connect(self._on_set_label_dir)
        settings_menu.addAction(self._action_set_label_dir)

        self._copy_original_action = QAction(tr("improved_copy_original"), self)
        self._copy_original_action.setCheckable(True)
        self._copy_original_action.setChecked(self._config.copy_original_on_save)
        self._copy_original_action.toggled.connect(self._on_copy_original_toggled)
        settings_menu.addAction(self._copy_original_action)
        settings_menu.addSeparator()

        self._action_lang_ko = QAction(tr("action_lang_ko"), self)
        self._action_lang_ko.triggered.connect(lambda: self._switch_language("ko"))
        settings_menu.addAction(self._action_lang_ko)

        self._action_lang_en = QAction(tr("action_lang_en"), self)
        self._action_lang_en.triggered.connect(lambda: self._switch_language("en"))
        settings_menu.addAction(self._action_lang_en)

        # --- Help menu ---
        help_menu = menubar.addMenu(tr("menu_help"))

        self._action_toggle_help_panel = QAction(tr("action_toggle_help_panel"), self)
        self._action_toggle_help_panel.setCheckable(True)
        self._action_toggle_help_panel.toggled.connect(self._on_toggle_help_panel)
        help_menu.addAction(self._action_toggle_help_panel)

        self._action_help_dialog = QAction(tr("action_help_dialog"), self)
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
        self._canvas.brush_size_changed_from_canvas.connect(self._on_canvas_brush_size_changed)
        self._canvas.edit_mask_requested.connect(self._on_edit_mask_requested)
        self._canvas.class_switch_requested.connect(self._on_class_switch_by_number)

        # Label list signals
        self._label_list.class_selected.connect(self._on_class_selected)
        self._label_list.instance_selected.connect(self._on_instance_selected)
        self._label_list.delete_instance_requested.connect(self._on_delete_instance)
        self._label_list.visibility_changed.connect(self._canvas.set_label_visible)
        self._label_list.classes_changed.connect(self._on_classes_changed)
        self._label_list.can_remove_class = self._can_remove_class

        # Label manager changes
        self._labels.labels_changed.connect(self._on_labels_changed)

        # Model loaded
        self._model.model_loaded.connect(self._on_model_loaded)

        # Project changes
        self._project.folder_changed.connect(self._on_folder_loaded)

    # --- Menu action handlers ---

    def _on_open_folder(self):
        path = QFileDialog.getExistingDirectory(self, tr("select_folder"), self._config.recent_image_dir)
        if path:
            self._open_project(path)

    def _open_project(self, path):
        try:
            if self._project.image_dir and not self._save_project():
                return
            self._opening_classes = load_classes(Path(path) / "labels")
            if not self._project.open_folder(path):
                raise ValueError(tr("folder_not_found_msg").format(path=path))
            self._config.recent_image_dir = path
            self._config.add_recent_directory(path)
            self._update_recent_directories_menu()
        except Exception as exc:
            QMessageBox.critical(self, tr("error"), str(exc))

    def _on_load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, tr("select_model"), self._config.recent_model_path,
            "Model Files (*.pt *.pth *.h5 *.keras);;All Files (*.*)")
        if path:
            self._choose_and_load_model(path)

    def _choose_and_load_model(self, path):
        if self._model_loader is not None:
            return
        options = ["AUTO", "YOLO", "RT-DETR", "KERAS", "DINOv3"]
        preferred = self._config.model_types.get(path, "AUTO")
        model_type, ok = QInputDialog.getItem(self, tr("select_model"),
            tr("improved_model_type_hint"), options,
            options.index(preferred) if preferred in options else 0, False)
        if not ok:
            return
        if model_type == "DINOv3":
            QMessageBox.information(self, "DINOv3", tr("improved_dinov3_help"))
            return
        self._model_loader = _ModelLoadWorker(self._model, path, model_type, self)
        self._action_load_model.setEnabled(False)
        self._recent_models_menu.setEnabled(False)
        self._action_auto_label.setEnabled(False)
        self._status_bar.showMessage(tr("improved_model_loading"))
        self._model_loader.finished.connect(self._on_model_load_finished)
        self._model_loader.start()

    def _on_model_load_finished(self):
        worker = self._model_loader
        self._model_loader = None
        self._action_load_model.setEnabled(True)
        self._recent_models_menu.setEnabled(True)
        self._action_auto_label.setEnabled(True)
        if worker.success:
            self._config.model_types[worker.path] = worker.model_type
            self._config.recent_model_path = str(Path(worker.path).parent)
            self._config.add_recent_model(worker.path)
            self._update_recent_models_menu()
        else:
            QMessageBox.critical(self, tr("error"), self._model.last_error)
            self._update_status_bar()
        worker.deleteLater()

    def _on_save_labels(self):
        self._save_project()

    def _save_project(self):
        if not self._project.image_dir:
            return True
        try:
            if self._canvas.has_unfinished_mask():
                self._canvas.finalize_pending_mask()
            classes = self._label_list.get_classes()
            save_classes(self._project.label_dir, classes)
            counts = self._saver.save_all_images(
                {i: c["name"] for i, c in enumerate(classes)},
                copy_original=self._config.copy_original_on_save)
            self._status_bar.showMessage(tr("improved_saved").format(
                labels=counts[0], masks=counts[1], images=counts[2]), 5000)
            return True
        except Exception as exc:
            QMessageBox.critical(self, tr("improved_save_failed"), str(exc))
            return False

    def _on_copy_original_toggled(self, enabled):
        self._config.copy_original_on_save = enabled
        self._config.save()

    def _on_classes_changed(self):
        if self._project.label_dir and not self._restoring_project:
            try:
                save_classes(self._project.label_dir, self._label_list.get_classes())
            except Exception as exc:
                QMessageBox.critical(self, tr("improved_save_failed"), str(exc))

    def _can_remove_class(self, index):
        if any(self._labels.get_labels(p) for p in self._labels.loaded_image_paths):
            return False
        if self._project.label_dir:
            if any(p.name != "classes.txt" for p in self._project.label_dir.glob("*.txt")):
                return False
        return not (self._project.image_dir and (self._project.image_dir / "gt_image").exists())

    def _on_export_masks(self):
        if not self._project.image_dir:
            return

        # Ask user for mask format
        reply = QMessageBox.question(
            self,
            tr("export_mask_format_title"),
            tr("export_mask_format_message"),
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
                img = read_image(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    img_file = Path(img_path)
                    # Save mask as PNG (lossless) to avoid JPEG lossy compression
                    mask_path = gt_image_dir / (img_file.stem + ".png")
                    ExportManager.save_semantic_mask(labels, w, h, str(mask_path), multi_label)
                    count += 1

        mask_type = "semantic" if multi_label else "binary"
        self._status_bar.showMessage(
            f"Exported {count} {mask_type} mask files to gt_image/", 3000
        )

    def _on_import_external_labels(self):
        """Import matching annotations only after mapping/file preflight."""
        if not self._project.image_dir:
            QMessageBox.warning(self, tr("warning"), tr("import_no_project"))
            return
        ext_dir = QFileDialog.getExistingDirectory(
            self, tr("import_select_folder"), self._config.recent_image_dir
        )
        if not ext_dir:
            return
        # Save edits and deletions first. A freshly opened empty image has no
        # work to save; don't create a negative TXT that would block its import.
        has_work = self._canvas.has_unfinished_mask() or any(
            self._labels.get_labels(path) or self._project.has_labels(path)
            for path in self._labels.loaded_image_paths)
        if has_work and not self._save_project():
            return
        try:
            from core.import_manager import import_annotations
            label_count, gt_count, classes = import_annotations(
                ext_dir, self._project, self._label_list.get_classes())
            self._restoring_project = True
            try:
                self._label_list.set_classes(classes)
            finally:
                self._restoring_project = False
            self._canvas.discard_pending_mask()
            self._labels.clear()  # Invalidates undo commands from the old data.
            for index, path in enumerate(self._project.image_list):
                self._file_list.update_label_status(index, self._project.has_labels(path))
            if self._current_image_path:
                self._load_labels_from_disk(self._current_image_path)
                self._canvas.setEnabled(True)
                if self._canvas._mode == ToolMode.SEGMENTATION:
                    self._auto_load_mask_for_segmentation(self._current_image_path)
                labels = self._labels.get_labels(self._current_image_path)
                self._canvas.display_labels(labels)
                self._label_list.set_instances(labels)
            self._status_bar.showMessage(tr("import_complete").format(labels=label_count, gt=gt_count), 5000)
        except Exception as exc:
            # If disk I/O failed after copying some new files, invalidate old
            # empty caches so a later auto-save cannot overwrite that import.
            self._canvas.discard_pending_mask()
            self._labels.clear()
            try:
                self._restoring_project = True
                restored = load_classes(self._project.label_dir)
                if restored:
                    self._label_list.set_classes(restored)
                if self._current_image_path:
                    self._load_labels_from_disk(self._current_image_path)
            except Exception:
                self._canvas.setEnabled(False)
            finally:
                self._restoring_project = False
            QMessageBox.critical(self, tr("error"), str(exc))

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
        if not self._model.is_loaded or self._model_loader is not None:
            QMessageBox.warning(self, tr("warning"), tr("auto_label_no_model"))
            return
        if not self._project.image_list:
            return
        if self._canvas.has_unfinished_mask():
            self._canvas.finalize_pending_mask()
        existing = {p for p in self._project.image_list
                    if self._project.has_labels(p) or self._labels.label_count(p) > 0}
        dialog = AutoLabelDialog(self._model, self._project.image_list,
            self._file_list.current_index(), self, existing_paths=existing)
        def apply(image_path, labels):
            try:
                self._apply_auto_labels(image_path, labels, dialog.apply_policy)
            except Exception as exc:
                dialog._on_error(f"{image_path}: {exc}")
                if dialog._worker is not None:
                    dialog._worker.abort()
        dialog.labels_generated.connect(apply)
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
            QMessageBox.warning(self, tr("warning"), tr("label_dir_no_project"))
            return

        path = QFileDialog.getExistingDirectory(self, tr("improved_label_dir_hint"),
                                                str(self._project.label_dir))
        if not path:
            return
        try:
            classes = load_classes(path)
            if not self._save_project():
                return
            if not self._project.set_custom_label_dir(path):
                raise ValueError(tr("label_dir_error"))
            self._opening_classes = classes
            self._on_folder_loaded()
        except Exception as exc:
            QMessageBox.critical(self, tr("error"), str(exc))

    # --- Signal handlers ---

    @Slot()
    def _on_folder_loaded(self):
        self._current_image_path = ""
        self._labels.clear()
        self._canvas.clear_canvas()
        self._restoring_project = True
        try:
            self._label_list.set_classes(self._opening_classes)
        finally:
            self._restoring_project = False
        images = self._project.image_list
        self._file_list.set_image_list(images)
        for i, image in enumerate(images):
            self._file_list.update_label_status(i, self._project.has_labels(image))
        if images:
            self._file_list.select_image(0)

    @Slot(int)
    def _on_image_selected(self, index: int):
        img_path = self._project.get_image_path(index)
        if not img_path:
            return

        # Handle pending mask work before switching images
        if self._current_image_path and self._canvas.has_unfinished_mask():
            if self._discard_pending_mask:
                self._canvas.discard_pending_mask()
            else:
                self._canvas.finalize_pending_mask()

        # Auto-save previous image labels
        if self._current_image_path and self._config.auto_save:
            if not self._save_current_labels():
                self._restore_file_selection()
                return

        # Force save mask labels for previous image even if auto_save is off
        # (mask labels created by finalize above need to be persisted in label_manager)
        # This is already handled by label_created signal -> add_label

        if not self._canvas.load_image(img_path):
            QMessageBox.critical(self, tr("error"), tr("improved_image_failed").format(path=img_path))
            self._restore_file_selection()
            return
        self._current_image_path = img_path

        # Lazy-load labels from disk if not yet loaded
        try:
            self._load_labels_from_disk(img_path)
        except Exception as exc:
            QMessageBox.critical(self, tr("error"), str(exc))
            self._canvas.setEnabled(False)
            return
        self._canvas.setEnabled(True)

        # If in SEGMENTATION mode, extract mask labels into the brush canvas
        # BEFORE displaying so the user doesn't see a brief flash of mask
        # graphics that immediately disappear.
        if self._canvas._mode == ToolMode.SEGMENTATION:
            self._auto_load_mask_for_segmentation(img_path)

        # Display labels (mask labels already removed if in SEGMENTATION)
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
        self._navigate_without_save(1)

    @Slot()
    def _on_next_without_save(self):
        self._navigate_without_save(1)

    def _restore_file_selection(self):
        self._file_list.blockSignals(True)
        self._file_list.select_image(self._project.get_image_index(self._current_image_path))
        self._file_list.blockSignals(False)

    def _navigate_without_save(self, delta):
        index = self._file_list.current_index() + delta
        if not 0 <= index < self._project.image_count:
            return
        self._canvas.discard_pending_mask()
        self._labels.remove_image(self._current_image_path)
        self._labels.undo_stack.clear()
        self._current_image_path = ""
        self._file_list.select_image(index)

    @Slot()
    def _on_next_with_save(self):
        if not self._save_current_labels():
            return
        index = self._file_list.current_index()
        if 0 <= index < self._project.image_count - 1:
            self._file_list.select_image(index + 1)

    @Slot()
    def _on_prev_image(self):
        self._navigate_without_save(-1)

    @Slot()
    def _on_exclude_from_training(self):
        """Hide an image for this session, preserving source and annotations."""
        if not self._current_image_path or not self._save_current_labels():
            return
        current_idx = self._file_list.current_index()
        hidden = self._current_image_path
        self._current_image_path = ""
        self._project.remove_image(hidden)
        self._canvas.clear_canvas()
        self._file_list.set_image_list(self._project.image_list)
        if self._project.image_count:
            self._file_list.select_image(min(current_idx, self._project.image_count - 1))
        self._status_bar.showMessage(tr("improved_hidden"), 5000)

    @Slot(str)
    def _on_mode_changed(self, mode: str):
        if self._current_image_path and not self._labels.is_image_loaded(self._current_image_path):
            return
        # Finalize pending mask only when LEAVING segmentation.
        # When entering segmentation, _auto_load_mask_for_segmentation
        # handles it and would conflict with an early finalize (wrong class).
        if mode != ToolMode.SEGMENTATION and self._canvas.has_unfinished_mask():
            self._canvas.finalize_pending_mask()
        self._canvas.set_mode(mode)
        # When switching to SEGMENTATION mode, auto-load existing mask for current image
        # (skipped when _on_edit_mask_requested already loaded a specific mask)
        if mode == ToolMode.SEGMENTATION and self._current_image_path:
            if not self._skip_auto_load_mask:
                self._auto_load_mask_for_segmentation(self._current_image_path)

    def _auto_load_mask_for_segmentation(self, img_path: str):
        """If in SEGMENTATION mode and image has mask labels, auto-load them into the brush canvas."""
        if not self._labels.is_image_loaded(img_path):
            return
        # Use get_labels_ref so id()-based removal below works on the
        # actual internal objects (get_labels returns copies).
        labels = self._labels.get_labels_ref(img_path)
        mask_labels = [l for l in labels if l.label_type == "mask"]
        if not mask_labels:
            return
        import numpy as np
        w, h = self._canvas.get_image_size()
        if w == 0 or h == 0:
            return
        selected_class_id = self._label_list.selected_class_id()
        selected_masks = [l for l in mask_labels if l.class_id == selected_class_id]
        if not selected_masks:
            selected_masks = mask_labels[:1]
        combined = np.zeros((h, w), dtype=np.uint8)
        for ml in selected_masks:
            if ml.mask_data is not None:
                combined = np.maximum(combined, ml.mask_data)
        label_to_edit = selected_masks[0]
        self._canvas.load_mask_for_editing(
            combined, label_to_edit.color,
            class_id=label_to_edit.class_id,
            class_name=label_to_edit.class_name,
        )
        # Sync label-list class selection to match the loaded mask
        self._label_list.select_class(label_to_edit.class_id)

        # Remove the merged mask labels from label manager in one batch
        # to avoid N separate labels_changed emissions (one per remove).
        # Use get_labels_ref so we work on the actual internal list objects
        # (get_labels returns copies whose id() would never match).
        selected_set = set(id(m) for m in selected_masks)
        ref_labels = self._labels.get_labels_ref(img_path)
        remaining = [l for l in ref_labels if id(l) not in selected_set]
        self._labels.set_labels(img_path, remaining)

    @Slot(int)
    def _on_brush_size_changed(self, size: int):
        """Update canvas brush size."""
        self._canvas.set_brush_size(size)

    @Slot(int)
    def _on_canvas_brush_size_changed(self, size: int):
        """Update toolbar slider when canvas changes brush size via +/- keys."""
        self._toolbar.set_brush_size(size)

    @Slot(int)
    def _on_edit_mask_requested(self, index: int):
        """Handle request to edit an existing mask label."""
        if not self._current_image_path:
            return
        labels = self._labels.get_labels(self._current_image_path)
        if 0 <= index < len(labels):
            label = labels[index]
            if label.label_type == "mask" and label.mask_data is not None:
                # Load mask into canvas for editing (with class info so
                # finalize_pending_mask uses the correct class).
                self._canvas.load_mask_for_editing(
                    label.mask_data.copy(), label.color,
                    class_id=label.class_id,
                    class_name=label.class_name,
                )
                # Remove the old label (use set_labels to update without losing others)
                self._labels.remove_label(self._current_image_path, index)
                # Switch to segmentation mode.  Suppress auto-load so that
                # _auto_load_mask_for_segmentation does NOT overwrite the mask
                # we just loaded for editing.
                self._skip_auto_load_mask = True
                self._toolbar.set_mode(ToolMode.SEGMENTATION)
                self._skip_auto_load_mask = False
                # Sync class selection to match the mask's class
                classes = self._label_list.get_classes()
                for i, cls in enumerate(classes):
                    if cls["name"] == label.class_name:
                        self._label_list.select_class(i)
                        break
                self._status_bar.showMessage(tr("mask_edit_status"), 3000)

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
        if self._current_image_path and self._labels.is_image_loaded(self._current_image_path):
            if not 0 <= label.class_id < self._label_list.get_class_count():
                QMessageBox.warning(self, tr("warning"), tr("improved_add_class_first"))
                return
            label.class_name = self._label_list.get_class_name(label.class_id)
            label.color = self._label_list.get_class_color(label.class_id)
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
    def _on_class_switch_by_number(self, class_id: int):
        """Handle number key 1-0 → switch to class 0-9."""
        if 0 <= class_id < self._label_list.get_class_count():
            self._label_list.select_class(class_id)
            self._on_class_selected(class_id)

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
            # Remember selection so we can restore after refresh
            prev_selected = self._canvas.get_selected_index()

            labels = self._labels.get_labels(image_path)
            self._canvas.display_labels(labels)
            w, h = self._canvas.get_image_size()
            self._label_list.set_image_size(w, h)
            self._label_list.set_instances(labels)

            # Re-apply preserved visibility state to canvas items
            for i, vis in enumerate(self._label_list.get_visibility()):
                self._canvas.set_label_visible(i, vis)

            # Restore selection when the index is still valid (e.g. after
            # an update_label edit) so edit handles persist.
            if 0 <= prev_selected < len(labels):
                self._canvas.highlight_label(prev_selected)
                self._label_list.select_instance(prev_selected)

        # Update file list icon
        idx = self._project.get_image_index(image_path)
        if idx >= 0:
            has = len(self._labels.get_labels(image_path)) > 0
            self._file_list.update_label_status(idx, has)

    @Slot(str)
    def _on_model_loaded(self, path):
        self._status_bar.showMessage(tr("status_model_loaded").format(name=Path(path).name), 5000)
        # Class IDs belong to the project; model classes are mapped by name when applied.

    @Slot(str, list)
    def _on_auto_labels_received(self, image_path, labels):
        self._apply_auto_labels(image_path, labels, "append")

    def _apply_auto_labels(self, image_path, labels, policy):
        # Default skip never overwrites labels that appeared after the dialog opened.
        if policy == "skip" and (self._labels.label_count(image_path) or self._project.has_labels(image_path)):
            return
        self._load_labels_from_disk(image_path)
        mapped = []
        for label in labels:
            label = label.copy()
            label.class_id = self._label_list.add_class(label.class_name)
            label.class_name = self._label_list.get_class_name(label.class_id)
            label.color = self._label_list.get_class_color(label.class_id)
            mapped.append(label)
        previous = self._labels.get_labels(image_path)
        self._labels.replace_labels(image_path, previous + mapped if policy == "append" else mapped)

    def _load_labels_from_disk(self, image_path):
        if self._labels.is_image_loaded(image_path):
            return
        if image_path == self._current_image_path:
            size = self._canvas.get_image_size()
        else:
            image = read_image(image_path)
            if image is None:
                raise ValueError(tr("improved_image_failed").format(path=image_path))
            size = (image.shape[1], image.shape[0])
        classes = self._label_list.get_classes()
        class_names = {i: c["name"] for i, c in enumerate(classes)}
        labels = self._saver.load_labels_from_disk(image_path, size, class_names,
                                                  self._label_list.add_class)
        for label in labels:
            label.color = self._label_list.get_class_color(label.class_id)
        self._labels.set_labels(image_path, labels)

    def _save_current_labels(self):
        if not self._current_image_path or not self._project.image_dir:
            return True
        if not self._labels.is_image_loaded(self._current_image_path):
            return False
        try:
            if self._canvas.has_unfinished_mask():
                self._canvas.finalize_pending_mask()
            classes = self._label_list.get_classes()
            save_classes(self._project.label_dir, classes)
            self._saver.save_image_labels(self._current_image_path,
                {i: c["name"] for i, c in enumerate(classes)},
                self._canvas.get_image_size(),
                copy_original=self._config.copy_original_on_save)
            return True
        except Exception as exc:
            QMessageBox.critical(self, tr("improved_save_failed"), str(exc))
            return False

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
        self._help_dock.setWindowTitle(tr("help_dock_title"))

        # Re-create menus
        self.menuBar().clear()
        self._setup_menu()

    def _update_recent_directories_menu(self):
        """Update the recent directories menu with current list."""
        self._recent_dirs_menu.clear()

        recent_dirs = self._config.recent_directories
        if not recent_dirs:
            no_recent_action = QAction(tr("menu_recent_none"), self)
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
        clear_action = QAction(tr("menu_recent_clear"), self)
        clear_action.triggered.connect(self._on_clear_recent_directories)
        self._recent_dirs_menu.addAction(clear_action)

    def _on_open_recent_directory(self, path):
        self._open_project(path)

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
            no_recent_action = QAction(tr("menu_recent_none"), self)
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
        clear_action = QAction(tr("menu_recent_clear"), self)
        clear_action.triggered.connect(self._on_clear_recent_models)
        self._recent_models_menu.addAction(clear_action)

    def _on_open_recent_model(self, path):
        self._choose_and_load_model(path)

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
        if self._model_loader is not None:
            self._status_bar.showMessage(tr("improved_model_loading"))
            event.ignore()
            return
        if self._config.auto_save and not self._save_project():
            event.ignore()
            return
        if not self._file_list.shutdown():
            self._status_bar.showMessage(tr("improved_thumbnail_stopping"))
            event.ignore()
            return
        self._config.window_width = self.width()
        self._config.window_height = self.height()
        self._config.save()
        super().closeEvent(event)
