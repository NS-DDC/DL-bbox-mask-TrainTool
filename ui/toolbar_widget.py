"""Toolbar widget for mode selection and tools."""

from PySide6.QtWidgets import QToolBar, QWidget, QSlider, QLabel, QWidgetAction, QComboBox, QPushButton
from PySide6.QtGui import QActionGroup, QAction, QIcon, QPixmap, QPainter, QColor, QFont
from PySide6.QtCore import Signal, Qt, QRect

from i18n import tr


def _make_help_icon() -> QIcon:
    """Create a simple '?' help icon programmatically."""
    size = 24
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    # Circle
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#4a90d9"))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    # Question mark
    painter.setPen(QColor("#ffffff"))
    font = QFont("Arial", 14, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(QRect(0, -1, size, size), Qt.AlignmentFlag.AlignCenter, "?")
    painter.end()
    return QIcon(pixmap)


class ToolMode:
    SELECT = "select"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"


class ToolbarWidget(QToolBar):
    mode_changed = Signal(str)
    brush_size_changed = Signal(int)
    brush_shape_changed = Signal(str)
    bbox_mode_changed = Signal(str)
    finish_polygon_requested = Signal()
    help_requested = Signal()
    next_without_save_requested = Signal()
    next_with_save_requested = Signal()
    prev_image_requested = Signal()
    save_extension_changed = Signal(str)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setMovable(False)
        self._current_mode = ToolMode.SELECT
        self._brush_size = 20
        self._brush_shape = "circle"
        self._bbox_mode = "rectangle"
        self._setup_actions()

    def _setup_actions(self):
        self._action_group = QActionGroup(self)
        self._action_group.setExclusive(True)

        self._select_action = QAction(tr("tool_select"), self)
        self._select_action.setCheckable(True)
        self._select_action.setChecked(True)
        self._select_action.setShortcut("Q")
        self._select_action.triggered.connect(lambda: self._set_mode(ToolMode.SELECT))
        self._action_group.addAction(self._select_action)
        self.addAction(self._select_action)

        self._detection_action = QAction(tr("tool_detection"), self)
        self._detection_action.setCheckable(True)
        self._detection_action.setShortcut("W")
        self._detection_action.triggered.connect(lambda: self._set_mode(ToolMode.DETECTION))
        self._action_group.addAction(self._detection_action)
        self.addAction(self._detection_action)

        self._seg_action = QAction(tr("tool_segmentation"), self)
        self._seg_action.setCheckable(True)
        self._seg_action.setShortcut("E")
        self._seg_action.triggered.connect(lambda: self._set_mode(ToolMode.SEGMENTATION))
        self._action_group.addAction(self._seg_action)
        self.addAction(self._seg_action)

        # BBox mode selector (rectangle/polygon)
        self.addSeparator()
        self._bbox_label = QLabel("BBox Mode:")
        self._bbox_label.setStyleSheet("padding: 0 6px; color: #ccc;")
        self.addWidget(self._bbox_label)

        self._bbox_mode_combo = QComboBox()
        self._bbox_mode_combo.addItems(["Rectangle", "Polygon"])
        self._bbox_mode_combo.setFixedWidth(100)
        self._bbox_mode_combo.currentTextChanged.connect(self._on_bbox_mode_changed)
        self.addWidget(self._bbox_mode_combo)

        # Segmentation tools
        self.addSeparator()
        self._seg_label = QLabel("Brush:")
        self._seg_label.setStyleSheet("padding: 0 6px; color: #ccc;")
        self.addWidget(self._seg_label)

        # Brush shape selector
        self._brush_shape_combo = QComboBox()
        self._brush_shape_combo.addItems(["Circle", "Square"])
        self._brush_shape_combo.setFixedWidth(80)
        self._brush_shape_combo.currentTextChanged.connect(self._on_brush_shape_changed)
        self.addWidget(self._brush_shape_combo)

        # Brush size slider
        self._brush_size_label = QLabel(f"{self._brush_size}px")
        self._brush_size_label.setStyleSheet("padding: 0 6px; color: #ccc;")
        self.addWidget(self._brush_size_label)

        self._brush_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._brush_size_slider.setMinimum(5)
        self._brush_size_slider.setMaximum(100)
        self._brush_size_slider.setValue(self._brush_size)
        self._brush_size_slider.setFixedWidth(100)
        self._brush_size_slider.valueChanged.connect(self._on_brush_size_changed)
        self.addWidget(self._brush_size_slider)

        # Finish polygon button
        self.addSeparator()
        self._finish_btn = QPushButton("Finish [Enter]")
        self._finish_btn.setFixedHeight(24)
        self._finish_btn.setStyleSheet("padding: 2px 8px;")
        self._finish_btn.clicked.connect(self.finish_polygon_requested.emit)
        self.addWidget(self._finish_btn)

        # Save extension selector
        self.addSeparator()
        self._extension_label = QLabel("저장 확장자:")
        self._extension_label.setStyleSheet("padding: 0 6px; color: #ccc;")
        self.addWidget(self._extension_label)

        self._extension_combo = QComboBox()
        self._extension_combo.addItems(["원본", "PNG", "JPG", "BMP", "TIFF"])
        self._extension_combo.setFixedWidth(80)
        self._extension_combo.setToolTip("이미지 저장 시 사용할 확장자")
        self._extension_combo.currentTextChanged.connect(self._on_extension_changed)
        self.addWidget(self._extension_combo)

        # Navigation buttons
        self.addSeparator()
        self._prev_image_btn = QPushButton("◀ 이전 [A]")
        self._prev_image_btn.setFixedHeight(24)
        self._prev_image_btn.setStyleSheet("padding: 2px 8px; background-color: #3498db; color: white;")
        self._prev_image_btn.setToolTip("이전 이미지로 이동 (단축키: A)")
        self._prev_image_btn.clicked.connect(self.prev_image_requested.emit)
        self.addWidget(self._prev_image_btn)

        self._next_with_save_btn = QPushButton("저장O 다음 [S]")
        self._next_with_save_btn.setFixedHeight(24)
        self._next_with_save_btn.setStyleSheet("padding: 2px 8px; background-color: #27ae60; color: white;")
        self._next_with_save_btn.setToolTip("현재 작업을 저장하고 다음 이미지로 이동 (단축키: S)")
        self._next_with_save_btn.clicked.connect(self.next_with_save_requested.emit)
        self.addWidget(self._next_with_save_btn)

        self._next_without_save_btn = QPushButton("저장X 다음 [D]")
        self._next_without_save_btn.setFixedHeight(24)
        self._next_without_save_btn.setStyleSheet("padding: 2px 8px; background-color: #e74c3c; color: white;")
        self._next_without_save_btn.setToolTip("현재 작업을 저장하지 않고 다음 이미지로 이동 (단축키: D)")
        self._next_without_save_btn.clicked.connect(self.next_without_save_requested.emit)
        self.addWidget(self._next_without_save_btn)

        # Separator + Help button
        self.addSeparator()
        self._help_action = QAction(self)
        self._help_action.setIcon(_make_help_icon())
        self._help_action.setToolTip("도움말 열기/닫기 (F1)")
        self._help_action.triggered.connect(self.help_requested.emit)
        self.addAction(self._help_action)

    def _set_mode(self, mode: str):
        self._current_mode = mode
        self.mode_changed.emit(mode)

    def set_mode(self, mode: str):
        """Programmatically set the current mode and update UI."""
        self._current_mode = mode
        if mode == ToolMode.SELECT:
            self._select_action.setChecked(True)
        elif mode == ToolMode.DETECTION:
            self._detection_action.setChecked(True)
        elif mode == ToolMode.SEGMENTATION:
            self._seg_action.setChecked(True)
        self.mode_changed.emit(mode)

    def _on_brush_size_changed(self, value: int):
        self._brush_size = value
        self._brush_size_label.setText(f"{value}px")
        self.brush_size_changed.emit(value)

    def _on_brush_shape_changed(self, text: str):
        self._brush_shape = text.lower()
        self.brush_shape_changed.emit(self._brush_shape)

    def _on_bbox_mode_changed(self, text: str):
        self._bbox_mode = text.lower()
        self.bbox_mode_changed.emit(self._bbox_mode)

    def _on_extension_changed(self, text: str):
        """Handle save extension change."""
        ext_map = {
            "원본": "",
            "PNG": ".png",
            "JPG": ".jpg",
            "BMP": ".bmp",
            "TIFF": ".tiff"
        }
        extension = ext_map.get(text, "")
        self.save_extension_changed.emit(extension)

    def current_mode(self) -> str:
        return self._current_mode

    def get_brush_size(self) -> int:
        return self._brush_size

    def get_brush_shape(self) -> str:
        return self._brush_shape

    def get_bbox_mode(self) -> str:
        return self._bbox_mode

    def get_save_extension(self) -> str:
        """Get current save extension."""
        text = self._extension_combo.currentText()
        ext_map = {
            "원본": "",
            "PNG": ".png",
            "JPG": ".jpg",
            "BMP": ".bmp",
            "TIFF": ".tiff"
        }
        return ext_map.get(text, "")

    def set_save_extension(self, extension: str):
        """Set save extension programmatically."""
        ext_to_text = {
            "": "원본",
            ".png": "PNG",
            ".jpg": "JPG",
            ".bmp": "BMP",
            ".tiff": "TIFF"
        }
        text = ext_to_text.get(extension, "원본")
        index = self._extension_combo.findText(text)
        if index >= 0:
            self._extension_combo.setCurrentIndex(index)

    def retranslate(self):
        self._select_action.setText(tr("tool_select"))
        self._detection_action.setText(tr("tool_detection"))
        self._seg_action.setText(tr("tool_segmentation"))
        self._bbox_label.setText("BBox Mode:")
        self._seg_label.setText("Brush:")
        self._brush_size_label.setText(f"{self._brush_size}px")
        self._finish_btn.setText("Finish [Enter]")
