"""Label list panel showing classes and label instances."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
    QPushButton, QLabel, QInputDialog, QColorDialog, QGroupBox,
    QAbstractItemView, QMessageBox, QComboBox,
)
from PySide6.QtGui import QColor, QPixmap, QIcon
from PySide6.QtCore import Signal, Qt

from config import get_config
from i18n import tr
from core.project_metadata import validate_class_name


# Default color palette for classes
DEFAULT_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#800000", "#aaffc3",
    "#808000", "#ffd8b1", "#000075", "#a9a9a9", "#000000",
]


class LabelListWidget(QWidget):
    class_selected = Signal(int)  # class index
    instance_selected = Signal(int)  # label index within current image
    class_added = Signal(str, str)  # name, color
    class_removed = Signal(int)  # class index
    classes_changed = Signal()
    delete_instance_requested = Signal(int)  # label index
    visibility_changed = Signal(int, bool)  # label index, visible

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._classes: list[dict] = []  # [{name, color}, ...]
        self._current_labels: list = []  # store for re-rendering on format change
        self._image_size: tuple[int, int] = (0, 0)
        self._visibility: list[bool] = []  # per-label visibility state
        self._config = get_config()
        self.can_remove_class = lambda index: True
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        # --- Classes section ---
        self._classes_group = QGroupBox(tr("label_classes_title"))
        classes_layout = QVBoxLayout(self._classes_group)

        self._class_list = QListWidget()
        self._class_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._class_list.currentRowChanged.connect(self._on_class_row_changed)
        self._class_list.itemDoubleClicked.connect(self._on_class_double_clicked)
        classes_layout.addWidget(self._class_list)

        btn_layout = QHBoxLayout()
        self._add_btn = QPushButton(tr("label_add_class"))
        self._add_btn.clicked.connect(self._on_add_class)
        btn_layout.addWidget(self._add_btn)

        self._remove_btn = QPushButton(tr("label_remove_class"))
        self._remove_btn.clicked.connect(self._on_remove_class)
        btn_layout.addWidget(self._remove_btn)
        classes_layout.addLayout(btn_layout)

        layout.addWidget(self._classes_group)

        # --- Instances section ---
        self._instances_group = QGroupBox(tr("label_instances_title"))
        inst_layout = QVBoxLayout(self._instances_group)

        # Coordinate format selector
        coord_layout = QHBoxLayout()
        self._coord_label = QLabel(tr("bbox_coord_format"))
        self._coord_label.setStyleSheet("font-size: 11px;")
        coord_layout.addWidget(self._coord_label)
        self._coord_combo = QComboBox()
        self._coord_combo.addItem(tr("bbox_coord_absolute"), "absolute")
        self._coord_combo.addItem(tr("bbox_coord_relative"), "relative")
        # Restore saved preference
        if self._config.bbox_coord_format == "relative":
            self._coord_combo.setCurrentIndex(1)
        self._coord_combo.currentIndexChanged.connect(self._on_coord_format_changed)
        coord_layout.addWidget(self._coord_combo)
        inst_layout.addLayout(coord_layout)

        self._instance_list = QListWidget()
        self._instance_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._instance_list.currentRowChanged.connect(self._on_instance_row_changed)
        self._instance_list.itemChanged.connect(self._on_item_changed)
        inst_layout.addWidget(self._instance_list)

        self._delete_inst_btn = QPushButton(tr("action_delete_label"))
        self._delete_inst_btn.clicked.connect(self._on_delete_instance)
        inst_layout.addWidget(self._delete_inst_btn)

        vis_btn_layout = QHBoxLayout()
        self._show_all_btn = QPushButton(tr("label_show_all"))
        self._show_all_btn.clicked.connect(self._on_show_all)
        vis_btn_layout.addWidget(self._show_all_btn)
        self._hide_all_btn = QPushButton(tr("label_hide_all"))
        self._hide_all_btn.clicked.connect(self._on_hide_all)
        vis_btn_layout.addWidget(self._hide_all_btn)
        inst_layout.addLayout(vis_btn_layout)

        self._no_labels_label = QLabel(tr("label_no_labels"))
        self._no_labels_label.setStyleSheet("color: gray; padding: 4px;")
        self._no_labels_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inst_layout.addWidget(self._no_labels_label)

        layout.addWidget(self._instances_group)

    # --- Class management ---

    def set_classes(self, classes: list[dict]):
        """Set class list. Each dict has 'name' and 'color' keys."""
        self._classes = classes
        self._refresh_class_list()
        if classes:
            self._class_list.setCurrentRow(0)

    def get_classes(self) -> list[dict]:
        return self._classes.copy()

    def get_class_count(self) -> int:
        return len(self._classes)

    def get_class_color(self, class_id: int) -> str:
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id]["color"]
        return DEFAULT_COLORS[class_id % len(DEFAULT_COLORS)]

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self._classes):
            return self._classes[class_id]["name"]
        return f"class_{class_id}"

    def selected_class_id(self) -> int:
        return self._class_list.currentRow()

    def add_class(self, name: str, color: str = None) -> int:
        """Add a class programmatically. Returns the new class index."""
        name = validate_class_name(name)
        # Check if already exists
        for i, cls in enumerate(self._classes):
            if cls["name"].casefold() == name.casefold():
                return i
        if color is None:
            color_idx = len(self._classes) % len(DEFAULT_COLORS)
            color = DEFAULT_COLORS[color_idx]
        self._classes.append({"name": name, "color": color})
        self._refresh_class_list()
        self._class_list.setCurrentRow(len(self._classes) - 1)
        self.classes_changed.emit()
        return len(self._classes) - 1

    def select_class(self, index: int):
        """Select a class by index programmatically (without triggering class_selected signal)."""
        if 0 <= index < self._class_list.count():
            self._class_list.blockSignals(True)
            self._class_list.setCurrentRow(index)
            self._class_list.blockSignals(False)

    def _refresh_class_list(self):
        self._class_list.clear()
        for cls in self._classes:
            item = QListWidgetItem(cls["name"])
            color = QColor(cls["color"])
            pixmap = QPixmap(16, 16)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))
            self._class_list.addItem(item)

    def _on_add_class(self):
        name, ok = QInputDialog.getText(self, tr("label_add_class"), tr("label_class_name"))
        if ok and name.strip():
            try:
                index = self.add_class(name)
                self.class_added.emit(self._classes[index]["name"], self._classes[index]["color"])
            except ValueError as exc:
                QMessageBox.warning(self, tr("warning"), str(exc))

    def _on_remove_class(self):
        row = self._class_list.currentRow()
        if row >= 0:
            if row != len(self._classes) - 1 or not self.can_remove_class(row):
                QMessageBox.warning(self, tr("warning"), tr("improved_class_remove_blocked"))
                return
            reply = QMessageBox.question(
                self, tr("warning"), tr("confirm_delete"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._classes.pop(row)
                self._refresh_class_list()
                self.class_removed.emit(row)
                self.classes_changed.emit()

    def _on_class_row_changed(self, row: int):
        if row >= 0:
            self.class_selected.emit(row)

    def _on_class_double_clicked(self, item: QListWidgetItem):
        row = self._class_list.row(item)
        if 0 <= row < len(self._classes):
            current_color = QColor(self._classes[row]["color"])
            color = QColorDialog.getColor(current_color, self)
            if color.isValid():
                self._classes[row]["color"] = color.name()
                self._refresh_class_list()
                self.classes_changed.emit()

    # --- Instance management ---

    def set_image_size(self, width: int, height: int):
        """Store current image dimensions for coordinate calculations."""
        self._image_size = (width, height)

    def set_instances(self, labels: list):
        """Set label instances for current image. Each item has class_name, label_type.

        Visibility is preserved when the label count is unchanged (e.g. after
        editing a bbox handle) so that the user's show/hide choices survive
        label-update refreshes.
        """
        self._current_labels = labels
        new_count = len(labels)
        if len(self._visibility) != new_count:
            # Count changed (add / delete / new image) → reset to all-visible
            self._visibility = [True] * new_count
        # else: same count → keep existing visibility
        self._refresh_instance_list()

    def get_visibility(self) -> list[bool]:
        """Return the current per-label visibility list."""
        return list(self._visibility)

    def _refresh_instance_list(self):
        """Re-render instance list with current coordinate format."""
        self._instance_list.blockSignals(True)
        self._instance_list.clear()
        labels = self._current_labels
        if not labels:
            self._no_labels_label.show()
            self._instance_list.blockSignals(False)
            return

        self._no_labels_label.hide()
        fmt = self._coord_combo.currentData()
        img_w, img_h = self._image_size

        for i, label in enumerate(labels):
            type_tag = "□" if label.label_type == "bbox" else "◇"
            coord_info = self._format_coords(label, fmt, img_w, img_h)
            text = f"{type_tag} [{label.class_id}] {label.class_name}  {coord_info}"
            item = QListWidgetItem(text)
            color = QColor(label.color)
            pixmap = QPixmap(12, 12)
            pixmap.fill(color)
            item.setIcon(QIcon(pixmap))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            visible = self._visibility[i] if i < len(self._visibility) else True
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            self._instance_list.addItem(item)

        self._instance_list.blockSignals(False)

    def _format_coords(self, label, fmt: str, img_w: int, img_h: int) -> str:
        """Format coordinate info string based on selected format."""
        if label.label_type == "mask":
            if label.mask_data is not None:
                pixel_count = int((label.mask_data > 0).sum())
                return f"mask ({pixel_count}px)"
            return "mask"
        if label.label_type == "bbox" and len(label.points) == 4:
            xs = [p[0] for p in label.points]
            ys = [p[1] for p in label.points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            if fmt == "relative" and img_w > 0 and img_h > 0:
                cx = ((x1 + x2) / 2.0) / img_w
                cy = ((y1 + y2) / 2.0) / img_h
                w = (x2 - x1) / img_w
                h = (y2 - y1) / img_h
                return f"[{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]"
            else:
                return f"[{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]"
        else:
            n = len(label.points)
            if fmt == "relative":
                return f"{n}pts (norm)"
            return f"{n}pts"

    def _on_coord_format_changed(self, index: int):
        fmt = self._coord_combo.currentData()
        self._config.bbox_coord_format = fmt
        self._config.save()
        self._refresh_instance_list()

    def clear_instances(self):
        self._instance_list.clear()
        self._no_labels_label.show()

    def select_instance(self, index: int):
        if 0 <= index < self._instance_list.count():
            self._instance_list.setCurrentRow(index)

    def _on_instance_row_changed(self, row: int):
        if row >= 0:
            self.instance_selected.emit(row)

    def _on_delete_instance(self):
        row = self._instance_list.currentRow()
        if row >= 0:
            self.delete_instance_requested.emit(row)

    def _on_item_changed(self, item: QListWidgetItem):
        index = self._instance_list.row(item)
        if 0 <= index < len(self._visibility):
            visible = item.checkState() == Qt.CheckState.Checked
            self._visibility[index] = visible
            self.visibility_changed.emit(index, visible)

    def _on_show_all(self):
        self._visibility = [True] * len(self._current_labels)
        self._refresh_instance_list()
        for i in range(len(self._current_labels)):
            self.visibility_changed.emit(i, True)

    def _on_hide_all(self):
        self._visibility = [False] * len(self._current_labels)
        self._refresh_instance_list()
        for i in range(len(self._current_labels)):
            self.visibility_changed.emit(i, False)

    def retranslate(self):
        self._classes_group.setTitle(tr("label_classes_title"))
        self._instances_group.setTitle(tr("label_instances_title"))
        self._add_btn.setText(tr("label_add_class"))
        self._remove_btn.setText(tr("label_remove_class"))
        self._delete_inst_btn.setText(tr("action_delete_label"))
        self._show_all_btn.setText(tr("label_show_all"))
        self._hide_all_btn.setText(tr("label_hide_all"))
        self._no_labels_label.setText(tr("label_no_labels"))
        self._coord_label.setText(tr("bbox_coord_format"))
        self._coord_combo.setItemText(0, tr("bbox_coord_absolute"))
        self._coord_combo.setItemText(1, tr("bbox_coord_relative"))
