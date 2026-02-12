"""Canvas widget for image display and label drawing using QGraphicsView."""

import os
from typing import Optional
import numpy as np
import cv2

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsPolygonItem, QGraphicsEllipseItem,
    QWidget, QVBoxLayout, QLabel,
)
from PySide6.QtGui import (
    QPixmap, QPen, QBrush, QColor, QPainter, QPolygonF, QWheelEvent,
    QMouseEvent, QKeyEvent, QCursor, QImage,
)
from PySide6.QtCore import Signal, Qt, QPointF, QRectF

from core.label_manager import LabelItem
from ui.toolbar_widget import ToolMode
from i18n import tr


class LabelGraphicsItem:
    """Wrapper linking a QGraphicsItem to a LabelItem."""

    def __init__(self, label: LabelItem, graphics_item):
        self.label = label
        self.graphics_item = graphics_item


class CanvasWidget(QWidget):
    label_created = Signal(object)  # LabelItem
    label_selected = Signal(int)  # index
    label_updated = Signal(int, object)  # index, updated LabelItem
    label_delete_requested = Signal(int)  # index - Delete key pressed
    cursor_moved = Signal(int, int)  # x, y in image coords
    zoom_changed = Signal(float)
    skip_image_requested = Signal()  # X key pressed to skip

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self._mode = ToolMode.SELECT
        self._current_class_id = 0
        self._current_class_name = "object"
        self._current_color = "#e6194b"
        self._image_path: Optional[str] = None
        self._image_pixmap: Optional[QPixmap] = None
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._label_items: list[LabelGraphicsItem] = []
        self._selected_index = -1

        # Drawing state
        self._drawing = False
        self._draw_start: Optional[QPointF] = None
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._polygon_points: list[QPointF] = []
        self._polygon_dots: list[QGraphicsEllipseItem] = []
        self._temp_polygon: Optional[QGraphicsPolygonItem] = None

        # Editing state
        self._editing = False
        self._edit_label_index = -1
        self._edit_handle_index = -1  # bbox: 0-3=corners, 4=center; polygon: vertex index
        self._edit_start_pos: Optional[QPointF] = None
        self._handle_items: list[QGraphicsEllipseItem] = []

        # Brush/mask state
        self._brush_size = 20
        self._brush_shape = "circle"
        self._brushing = False
        self._erasing = False
        self._current_mask: Optional[np.ndarray] = None
        self._current_mask_color: Optional[str] = None  # Color when mask was started
        self._mask_pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._brush_cursor = None  # Can be QGraphicsEllipseItem or QGraphicsRectItem
        self._brush_snapshot: Optional[np.ndarray] = None  # For undo

        # BBox mode
        self._bbox_mode = "rectangle"  # "rectangle" or "polygon"

        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scene = QGraphicsScene(self)
        self._view = _GraphicsView(self._scene, self)
        self._view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._view.setDragMode(QGraphicsView.DragMode.NoDrag)

        # Forward signals from custom view
        self._view.mouse_pressed.connect(self._on_mouse_press)
        self._view.mouse_moved.connect(self._on_mouse_move)
        self._view.mouse_released.connect(self._on_mouse_release)
        self._view.mouse_double_clicked.connect(self._on_mouse_double_click)
        self._view.wheel_zoomed.connect(self._on_wheel_zoom)
        self._view.key_pressed.connect(self._on_key_press)

        self._placeholder = QLabel(tr("canvas_no_image"))
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setStyleSheet("color: gray; font-size: 16px;")

        layout.addWidget(self._placeholder)
        layout.addWidget(self._view)
        self._view.hide()

    # --- Public API ---

    def load_image(self, image_path: str):
        if not os.path.isfile(image_path):
            return
        self._image_path = image_path
        self._image_pixmap = QPixmap(image_path)
        if self._image_pixmap.isNull():
            return

        # Save current mode before clearing
        current_mode = self._mode

        self._scene.clear()
        self._label_items.clear()
        self._reset_drawing_state()

        # Reset mask and mask display for new image
        self._current_mask = None
        self._mask_pixmap_item = None
        self._brush_cursor = None

        self._pixmap_item = self._scene.addPixmap(self._image_pixmap)
        self._scene.setSceneRect(QRectF(self._image_pixmap.rect()))

        # Restore brush cursor if in segmentation mode
        if current_mode == ToolMode.SEGMENTATION:
            w, h = self._image_pixmap.width(), self._image_pixmap.height()
            self._current_mask = np.zeros((h, w), dtype=np.uint8)
            self._show_brush_cursor()
            self._view.setCursor(Qt.CursorShape.BlankCursor)

        self._placeholder.hide()
        self._view.show()
        self._view.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def get_image_size(self) -> tuple[int, int]:
        if self._image_pixmap and not self._image_pixmap.isNull():
            return self._image_pixmap.width(), self._image_pixmap.height()
        return 0, 0

    def get_selected_index(self) -> int:
        """Return the index of the currently selected label, or -1 if none."""
        return self._selected_index

    def set_mode(self, mode: str):
        self._mode = mode
        self._reset_drawing_state()
        if mode == ToolMode.SELECT:
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.ArrowCursor)
            self._hide_brush_cursor()
        elif mode == ToolMode.DETECTION:
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.CrossCursor)
            self._hide_brush_cursor()
        elif mode == ToolMode.SEGMENTATION:
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.BlankCursor)
            self._show_brush_cursor()
            # Initialize mask if needed
            if self._current_mask is None and self._image_pixmap:
                w, h = self._image_pixmap.width(), self._image_pixmap.height()
                self._current_mask = np.zeros((h, w), dtype=np.uint8)
                self._current_mask_color = self._current_color  # Save color when mask starts

    def set_current_class(self, class_id: int, class_name: str, color: str):
        # If class is changing and there's an active mask, finalize it first
        if (self._mode == ToolMode.SEGMENTATION and
            self._current_mask is not None and
            self._current_mask.max() > 0 and
            (class_id != self._current_class_id or class_name != self._current_class_name)):
            # Save current mask before changing class
            self._finalize_mask()

        self._current_class_id = class_id
        self._current_class_name = class_name
        self._current_color = color
        # Update brush cursor color if it exists
        if self._brush_cursor:
            pen = QPen(QColor(color), 3, Qt.PenStyle.SolidLine)
            if hasattr(self._brush_cursor, 'setPen'):
                self._brush_cursor.setPen(pen)

    def set_brush_size(self, size: int):
        """Update brush size."""
        self._brush_size = size
        if self._brush_cursor:
            self._update_brush_cursor_size()

    def set_brush_shape(self, shape: str):
        """Update brush shape (circle or square)."""
        self._brush_shape = shape
        if self._brush_cursor:
            # Recreate cursor with new shape
            if hasattr(self._brush_cursor, 'rect'):
                center = self._brush_cursor.rect().center()
            else:
                center = QPointF(0, 0)
            self._hide_brush_cursor()
            self._show_brush_cursor()
            if self._brush_cursor:
                self._update_brush_cursor_pos(center)

    def set_bbox_mode(self, mode: str):
        """Set BBox drawing mode (rectangle or polygon)."""
        self._bbox_mode = mode

    def finish_current_shape(self):
        """Finish drawing current polygon or mask."""
        if self._mode == ToolMode.SEGMENTATION:
            # Check if in polygon mode
            if len(self._polygon_points) >= 3:
                self._finalize_polygon()
            # Check if in mask mode
            elif self._current_mask is not None and self._current_mask.max() > 0:
                self._finalize_mask()

    def has_unfinished_mask(self) -> bool:
        """Check if there's an unfinished mask drawing."""
        return (self._mode == ToolMode.SEGMENTATION and
                self._current_mask is not None and
                self._current_mask.max() > 0)

    def display_labels(self, labels: list[LabelItem]):
        # Remove old label graphics
        for li in self._label_items:
            if li.graphics_item.scene():
                self._scene.removeItem(li.graphics_item)
        self._label_items.clear()

        # Clear selection and handles
        self._selected_index = -1
        self._clear_edit_handles()

        for label in labels:
            gfx = self._create_label_graphics(label)
            self._label_items.append(LabelGraphicsItem(label, gfx))

    def highlight_label(self, index: int):
        # Reset previous highlight
        if 0 <= self._selected_index < len(self._label_items):
            self._set_label_highlight(self._selected_index, False)
            self._clear_edit_handles()
        self._selected_index = index
        if 0 <= index < len(self._label_items):
            self._set_label_highlight(index, True)
            # Show edit handles when in SELECT mode
            if self._mode == ToolMode.SELECT:
                self._show_edit_handles(index)
        else:
            # If index is -1 (deselected), clear handles
            self._clear_edit_handles()

    def _show_edit_handles(self, index: int):
        """Show edit handles for the selected label."""
        self._clear_edit_handles()

        if not (0 <= index < len(self._label_items)):
            return

        label = self._label_items[index].label

        if label.label_type == "bbox":
            # 4 corners + center handle + 4 edge midpoints for easier resizing
            xs = [p[0] for p in label.points]
            ys = [p[1] for p in label.points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # Calculate bbox size for dynamic handle sizing
            bbox_width = abs(x2 - x1)
            bbox_height = abs(y2 - y1)
            bbox_size = min(bbox_width, bbox_height)

            # Dynamic handle size based on bbox size (min 10, max 20)
            base_handle_size = max(10, min(20, bbox_size * 0.05))
            # Center handle is 1.5x larger for easier dragging
            center_handle_size = base_handle_size * 1.5

            handle_positions = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2),  # 4 corners
                (cx, y1), (x2, cy), (cx, y2), (x1, cy),  # 4 edge midpoints
                (cx, cy)  # center
            ]
        else:  # polygon
            handle_positions = label.points
            base_handle_size = 10
            center_handle_size = 10

        # Draw handles
        for i, pos in enumerate(handle_positions):
            # Use larger size for center handle (index 8 for bbox)
            if label.label_type == "bbox" and i == 8:
                handle_size = center_handle_size
                pen_width = 3
                brush_color = QColor("#ff6b6b")  # Red for center handle
            else:
                handle_size = base_handle_size if label.label_type == "bbox" else 10
                pen_width = 2
                brush_color = QColor("#4a90d9")

            handle = self._scene.addEllipse(
                pos[0] - handle_size/2, pos[1] - handle_size/2, handle_size, handle_size,
                QPen(QColor("#ffffff"), pen_width),
                QBrush(brush_color)
            )
            handle.setZValue(100)  # Above other items
            self._handle_items.append(handle)

    def _clear_edit_handles(self):
        """Remove all edit handles from the scene."""
        for handle in self._handle_items:
            if handle.scene():
                self._scene.removeItem(handle)
        self._handle_items.clear()

    def clear_canvas(self):
        self._scene.clear()
        self._label_items.clear()
        self._pixmap_item = None
        self._image_pixmap = None
        self._image_path = None
        self._reset_drawing_state()
        self._view.hide()
        self._placeholder.show()

    # --- Drawing helpers ---

    def _create_label_graphics(self, label: LabelItem):
        color = QColor(label.color)
        pen = QPen(color, 2)
        brush_color = QColor(color)
        brush_color.setAlpha(40)
        brush = QBrush(brush_color)

        if label.label_type == "bbox" and len(label.points) == 4:
            xs = [p[0] for p in label.points]
            ys = [p[1] for p in label.points]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            rect = QRectF(x1, y1, x2 - x1, y2 - y1)
            item = self._scene.addRect(rect, pen, brush)
            return item
        elif label.label_type == "mask" and label.mask_data is not None:
            # Render mask as pixmap overlay
            h, w = label.mask_data.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[:, :, 0] = color.blue()
            overlay[:, :, 1] = color.green()
            overlay[:, :, 2] = color.red()
            overlay[:, :, 3] = (label.mask_data * 0.5).astype(np.uint8)

            qimage = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
            pixmap = QPixmap.fromImage(qimage)
            item = self._scene.addPixmap(pixmap)
            item.setZValue(10)
            return item
        else:
            # Polygon
            polygon = QPolygonF([QPointF(p[0], p[1]) for p in label.points])
            item = self._scene.addPolygon(polygon, pen, brush)
            return item

    def _set_label_highlight(self, index: int, highlighted: bool):
        li = self._label_items[index]
        color = QColor(li.label.color)
        if highlighted:
            pen = QPen(QColor("#ffffff"), 3)
            brush_color = QColor(color)
            brush_color.setAlpha(80)
        else:
            pen = QPen(color, 2)
            brush_color = QColor(color)
            brush_color.setAlpha(40)

        item = li.graphics_item
        if isinstance(item, QGraphicsRectItem):
            item.setPen(pen)
            item.setBrush(QBrush(brush_color))
        elif isinstance(item, QGraphicsPolygonItem):
            item.setPen(pen)
            item.setBrush(QBrush(brush_color))

    def _update_bbox_with_handle(self, label: LabelItem, handle_idx: int, new_pos: QPointF):
        """Update bbox points when dragging a handle."""
        xs = [p[0] for p in label.points]
        ys = [p[1] for p in label.points]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        if handle_idx == 8:  # Center handle - move entire bbox
            delta_x = new_pos.x() - (x1 + x2) / 2
            delta_y = new_pos.y() - (y1 + y2) / 2
            label.points = [
                (x1 + delta_x, y1 + delta_y),
                (x2 + delta_x, y1 + delta_y),
                (x2 + delta_x, y2 + delta_y),
                (x1 + delta_x, y2 + delta_y),
            ]
        elif handle_idx < 4:  # Corner handles - resize
            if handle_idx == 0:  # Top-left
                x1, y1 = new_pos.x(), new_pos.y()
            elif handle_idx == 1:  # Top-right
                x2, y1 = new_pos.x(), new_pos.y()
            elif handle_idx == 2:  # Bottom-right
                x2, y2 = new_pos.x(), new_pos.y()
            elif handle_idx == 3:  # Bottom-left
                x1, y2 = new_pos.x(), new_pos.y()

            label.points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]
        else:  # Edge midpoint handles - resize along one axis
            if handle_idx == 4:  # Top edge
                y1 = new_pos.y()
            elif handle_idx == 5:  # Right edge
                x2 = new_pos.x()
            elif handle_idx == 6:  # Bottom edge
                y2 = new_pos.y()
            elif handle_idx == 7:  # Left edge
                x1 = new_pos.x()

            label.points = [
                (x1, y1),
                (x2, y1),
                (x2, y2),
                (x1, y2),
            ]

    def _update_polygon_vertex(self, label: LabelItem, vertex_idx: int, new_pos: QPointF):
        """Update polygon vertex when dragging a handle."""
        if 0 <= vertex_idx < len(label.points):
            label.points[vertex_idx] = (new_pos.x(), new_pos.y())

    def _refresh_label_graphics(self, index: int):
        """Refresh the graphics item for a label after editing."""
        if not (0 <= index < len(self._label_items)):
            return

        li = self._label_items[index]
        # Remove old graphics item
        if li.graphics_item.scene():
            self._scene.removeItem(li.graphics_item)

        # Create new graphics item
        new_gfx = self._create_label_graphics(li.label)
        li.graphics_item = new_gfx

        # Restore highlight if selected
        if index == self._selected_index:
            self._set_label_highlight(index, True)

    def _show_brush_cursor(self):
        """Show brush cursor (circle or square) with current class color."""
        if not self._brush_cursor:
            # Use current class color for brush cursor
            pen = QPen(QColor(self._current_color), 3, Qt.PenStyle.SolidLine)
            if self._brush_shape == "circle":
                self._brush_cursor = self._scene.addEllipse(
                    0, 0, self._brush_size, self._brush_size, pen
                )
            else:  # square
                self._brush_cursor = self._scene.addRect(
                    0, 0, self._brush_size, self._brush_size, pen
                )
            self._brush_cursor.setZValue(200)

    def _hide_brush_cursor(self):
        """Hide brush cursor."""
        if self._brush_cursor and self._brush_cursor.scene():
            self._scene.removeItem(self._brush_cursor)
            self._brush_cursor = None

    def _update_brush_cursor_size(self):
        """Update brush cursor size."""
        if self._brush_cursor:
            if hasattr(self._brush_cursor, 'rect'):
                rect = self._brush_cursor.rect()
                rect.setWidth(self._brush_size)
                rect.setHeight(self._brush_size)
                self._brush_cursor.setRect(rect)

    def _update_brush_cursor_pos(self, pos):
        """Update brush cursor position."""
        if self._brush_cursor:
            if isinstance(pos, QPointF):
                x, y = pos.x(), pos.y()
            else:
                x, y = pos.x(), pos.y()
            r = self._brush_size / 2
            if hasattr(self._brush_cursor, 'setRect'):
                self._brush_cursor.setRect(x - r, y - r, self._brush_size, self._brush_size)
            else:  # Ellipse has setRect too
                self._brush_cursor.setRect(x - r, y - r, self._brush_size, self._brush_size)

    def _draw_on_mask(self, pos: QPointF, erase: bool = False):
        """Draw or erase on the current mask."""
        if self._current_mask is None:
            return

        x, y = int(pos.x()), int(pos.y())
        h, w = self._current_mask.shape
        if not (0 <= x < w and 0 <= y < h):
            return

        value = 0 if erase else 255
        radius = self._brush_size // 2

        if self._brush_shape == "circle":
            cv2.circle(self._current_mask, (x, y), radius, value, -1)
        else:  # square
            x1 = max(0, x - radius)
            y1 = max(0, y - radius)
            x2 = min(w, x + radius)
            y2 = min(h, y + radius)
            self._current_mask[y1:y2, x1:x2] = value

        # Update mask display
        self._update_mask_display()

    def _update_mask_display(self):
        """Update the mask overlay display."""
        if self._current_mask is None:
            return

        # Create colored overlay using the color when mask was started
        h, w = self._current_mask.shape
        # Use saved mask color, or current color if not set
        mask_color = self._current_mask_color if self._current_mask_color else self._current_color
        color = QColor(mask_color)
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        overlay[:, :, 0] = color.blue()
        overlay[:, :, 1] = color.green()
        overlay[:, :, 2] = color.red()
        overlay[:, :, 3] = (self._current_mask * 0.5).astype(np.uint8)  # 50% opacity

        # Convert to QPixmap
        qimage = QImage(overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        pixmap = QPixmap.fromImage(qimage)

        # Update or create pixmap item
        if self._mask_pixmap_item is None:
            self._mask_pixmap_item = self._scene.addPixmap(pixmap)
            self._mask_pixmap_item.setZValue(50)  # Above labels, below handles
        else:
            self._mask_pixmap_item.setPixmap(pixmap)

    def _finalize_mask(self):
        """Convert current mask to a LabelItem and emit."""
        if self._current_mask is None or self._current_mask.max() == 0:
            return

        # Use the color when mask was started
        mask_color = self._current_mask_color if self._current_mask_color else self._current_color

        label = LabelItem(
            class_id=self._current_class_id,
            class_name=self._current_class_name,
            label_type="mask",
            points=[],
            color=mask_color,
            mask_data=self._current_mask.copy(),
        )
        self.label_created.emit(label)

        # Reset mask and color for next drawing
        self._current_mask = np.zeros_like(self._current_mask)
        self._current_mask_color = None  # Reset color
        if self._mask_pixmap_item and self._mask_pixmap_item.scene():
            self._scene.removeItem(self._mask_pixmap_item)
            self._mask_pixmap_item = None

    def _reset_drawing_state(self):
        self._drawing = False
        self._draw_start = None
        if self._temp_rect and self._temp_rect.scene():
            self._scene.removeItem(self._temp_rect)
        self._temp_rect = None
        if self._temp_polygon and self._temp_polygon.scene():
            self._scene.removeItem(self._temp_polygon)
        self._temp_polygon = None
        for dot in self._polygon_dots:
            if dot.scene():
                self._scene.removeItem(dot)
        self._polygon_dots.clear()
        self._polygon_points.clear()

    def _scene_pos(self, view_pos) -> Optional[QPointF]:
        """Convert view position to scene coordinates, clamped to image bounds."""
        scene_pos = self._view.mapToScene(view_pos)
        if self._image_pixmap:
            w, h = self._image_pixmap.width(), self._image_pixmap.height()
            x = max(0, min(scene_pos.x(), w))
            y = max(0, min(scene_pos.y(), h))
            return QPointF(x, y)
        return scene_pos

    # --- Mouse handlers ---

    def _on_mouse_press(self, pos, button=Qt.MouseButton.LeftButton):
        scene_pos = self._scene_pos(pos)
        if not scene_pos or not self._image_pixmap:
            return

        # Segmentation mode - brush or polygon
        if self._mode == ToolMode.SEGMENTATION:
            # Check if Ctrl is pressed - if so, polygon mode
            from PySide6.QtWidgets import QApplication
            modifiers = QApplication.keyboardModifiers()
            if modifiers & Qt.KeyboardModifier.ControlModifier:
                # Polygon mode
                if button == Qt.MouseButton.LeftButton:
                    self._polygon_points.append(scene_pos)
                    dot = self._scene.addEllipse(
                        scene_pos.x() - 3, scene_pos.y() - 3, 6, 6,
                        QPen(QColor(self._current_color)),
                        QBrush(QColor(self._current_color)),
                    )
                    self._polygon_dots.append(dot)

                    if self._temp_polygon and self._temp_polygon.scene():
                        self._scene.removeItem(self._temp_polygon)
                    if len(self._polygon_points) >= 2:
                        polygon = QPolygonF(self._polygon_points)
                        pen = QPen(QColor(self._current_color), 2, Qt.PenStyle.DashLine)
                        self._temp_polygon = self._scene.addPolygon(polygon, pen)
                elif button == Qt.MouseButton.RightButton:
                    # Right click in polygon mode = finish polygon
                    if len(self._polygon_points) >= 3:
                        self._finalize_polygon()
            else:
                # Brush mode
                if button == Qt.MouseButton.RightButton:
                    # Save snapshot for undo before erasing
                    if self._current_mask is not None:
                        self._brush_snapshot = self._current_mask.copy()
                    self._erasing = True
                    self._draw_on_mask(scene_pos, erase=True)
                elif button == Qt.MouseButton.LeftButton:
                    # Save snapshot for undo before drawing
                    if self._current_mask is not None:
                        self._brush_snapshot = self._current_mask.copy()
                        # Save color when first drawing starts
                        if self._current_mask_color is None or self._current_mask.max() == 0:
                            self._current_mask_color = self._current_color
                    self._brushing = True
                    self._draw_on_mask(scene_pos, erase=False)
            return

        # Check if clicking on a handle (for editing)
        if self._mode == ToolMode.SELECT and self._selected_index >= 0:
            for i, handle in enumerate(self._handle_items):
                # Check if click is within handle bounds
                handle_rect = handle.rect()
                handle_center = handle_rect.center()
                distance = ((scene_pos.x() - handle_center.x())**2 +
                           (scene_pos.y() - handle_center.y())**2)**0.5
                if distance <= handle_rect.width() / 2 + 5:  # Add 5px tolerance
                    self._editing = True
                    self._edit_label_index = self._selected_index
                    self._edit_handle_index = i
                    self._edit_start_pos = scene_pos
                    return

        if self._mode == ToolMode.DETECTION:
            if self._bbox_mode == "rectangle":
                # Rectangle mode
                if button == Qt.MouseButton.LeftButton:
                    self._drawing = True
                    self._draw_start = scene_pos
                    color = QColor(self._current_color)
                    pen = QPen(color, 2, Qt.PenStyle.DashLine)
                    self._temp_rect = self._scene.addRect(
                        QRectF(scene_pos, scene_pos), pen
                    )
            else:
                # Polygon mode for BBox
                if button == Qt.MouseButton.LeftButton:
                    self._polygon_points.append(scene_pos)
                    dot = self._scene.addEllipse(
                        scene_pos.x() - 3, scene_pos.y() - 3, 6, 6,
                        QPen(QColor(self._current_color)),
                        QBrush(QColor(self._current_color)),
                    )
                    self._polygon_dots.append(dot)

                    if self._temp_polygon and self._temp_polygon.scene():
                        self._scene.removeItem(self._temp_polygon)
                    if len(self._polygon_points) >= 2:
                        polygon = QPolygonF(self._polygon_points)
                        pen = QPen(QColor(self._current_color), 2, Qt.PenStyle.DashLine)
                        self._temp_polygon = self._scene.addPolygon(polygon, pen)
                elif button == Qt.MouseButton.RightButton:
                    # Right click = finish polygon bbox
                    if len(self._polygon_points) >= 3:
                        self._finalize_polygon_as_bbox()

        elif self._mode == ToolMode.SELECT:
            # Try to select a label under cursor
            items_at = self._scene.items(scene_pos)
            for item in items_at:
                for i, li in enumerate(self._label_items):
                    if li.graphics_item is item:
                        self.highlight_label(i)
                        self.label_selected.emit(i)
                        return

    def _on_mouse_move(self, pos):
        scene_pos = self._scene_pos(pos)
        if not scene_pos:
            return

        # Emit cursor position in image coords
        self.cursor_moved.emit(int(scene_pos.x()), int(scene_pos.y()))

        # Update brush cursor position in segmentation mode
        if self._mode == ToolMode.SEGMENTATION:
            self._update_brush_cursor_pos(scene_pos)
            # Continue drawing/erasing if mouse is pressed
            if self._brushing:
                self._draw_on_mask(scene_pos, erase=False)
                return
            elif self._erasing:
                self._draw_on_mask(scene_pos, erase=True)
                return

        # Handle editing mode (dragging handles)
        if self._editing and self._edit_label_index >= 0 and self._edit_start_pos:
            label = self._label_items[self._edit_label_index].label

            if label.label_type == "bbox":
                self._update_bbox_with_handle(label, self._edit_handle_index, scene_pos)
            else:  # polygon
                self._update_polygon_vertex(label, self._edit_handle_index, scene_pos)

            self._edit_start_pos = scene_pos
            # Refresh graphics
            self._refresh_label_graphics(self._edit_label_index)
            self._show_edit_handles(self._edit_label_index)
            return

        if self._mode == ToolMode.DETECTION and self._drawing and self._draw_start and self._bbox_mode == "rectangle":
            rect = QRectF(self._draw_start, scene_pos).normalized()
            if self._temp_rect:
                self._temp_rect.setRect(rect)

    def _on_mouse_release(self, pos):
        scene_pos = self._scene_pos(pos)
        if not scene_pos:
            return

        # Finish brushing/erasing - create undo command
        if self._brushing or self._erasing:
            if self._brushing:
                self._brushing = False
            if self._erasing:
                self._erasing = False
            # Create undo command for brush stroke
            if self._brush_snapshot is not None and self._current_mask is not None:
                # Emit brush stroke completed signal (will be handled by main window)
                # For now, just clear snapshot
                self._brush_snapshot = None
            return

        # Finish editing
        if self._editing:
            label = self._label_items[self._edit_label_index].label
            self.label_updated.emit(self._edit_label_index, label)
            self._editing = False
            self._edit_label_index = -1
            self._edit_handle_index = -1
            self._edit_start_pos = None
            return

        if self._mode == ToolMode.DETECTION and self._bbox_mode == "rectangle" and self._drawing and self._draw_start:
            self._drawing = False
            rect = QRectF(self._draw_start, scene_pos).normalized()

            # Remove temp rect
            if self._temp_rect and self._temp_rect.scene():
                self._scene.removeItem(self._temp_rect)
            self._temp_rect = None

            # Minimum size check
            if rect.width() > 5 and rect.height() > 5:
                points = [
                    (rect.x(), rect.y()),
                    (rect.x() + rect.width(), rect.y()),
                    (rect.x() + rect.width(), rect.y() + rect.height()),
                    (rect.x(), rect.y() + rect.height()),
                ]
                label = LabelItem(
                    class_id=self._current_class_id,
                    class_name=self._current_class_name,
                    label_type="bbox",
                    points=points,
                    color=self._current_color,
                )
                self.label_created.emit(label)

            self._draw_start = None

    def _on_mouse_double_click(self, pos):
        """Finalize polygon on double click."""
        # Detection mode polygon
        if self._mode == ToolMode.DETECTION and self._bbox_mode == "polygon" and len(self._polygon_points) >= 3:
            self._finalize_polygon_as_bbox()
            return

        # Segmentation mode polygon (Ctrl+Click mode)
        if self._mode == ToolMode.SEGMENTATION and len(self._polygon_points) >= 3:
            self._finalize_polygon()

    def _finalize_polygon(self):
        """Finalize and emit polygon label."""
        if len(self._polygon_points) < 3:
            return

        points = [(p.x(), p.y()) for p in self._polygon_points]

        # Cleanup temp graphics
        if self._temp_polygon and self._temp_polygon.scene():
            self._scene.removeItem(self._temp_polygon)
        self._temp_polygon = None
        for dot in self._polygon_dots:
            if dot.scene():
                self._scene.removeItem(dot)
        self._polygon_dots.clear()
        self._polygon_points.clear()

        label = LabelItem(
            class_id=self._current_class_id,
            class_name=self._current_class_name,
            label_type="polygon",
            points=points,
            color=self._current_color,
        )
        self.label_created.emit(label)

    def _finalize_polygon_as_bbox(self):
        """Finalize polygon as bbox label (for Detection mode with polygon option)."""
        if len(self._polygon_points) < 3:
            return

        points = [(p.x(), p.y()) for p in self._polygon_points]

        # Cleanup temp graphics
        if self._temp_polygon and self._temp_polygon.scene():
            self._scene.removeItem(self._temp_polygon)
        self._temp_polygon = None
        for dot in self._polygon_dots:
            if dot.scene():
                self._scene.removeItem(dot)
        self._polygon_dots.clear()
        self._polygon_points.clear()

        label = LabelItem(
            class_id=self._current_class_id,
            class_name=self._current_class_name,
            label_type="bbox",  # Saved as bbox type
            points=points,
            color=self._current_color,
        )
        self.label_created.emit(label)

    def _on_wheel_zoom(self, delta: int, modifiers=None):
        """Handle wheel zoom or brush size change with Ctrl."""
        from PySide6.QtWidgets import QApplication
        if modifiers is None:
            modifiers = QApplication.keyboardModifiers()

        # Ctrl + Wheel = Brush size change
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            if self._mode == ToolMode.SEGMENTATION:
                change = 5 if delta > 0 else -5
                new_size = max(5, min(100, self._brush_size + change))
                self.set_brush_size(new_size)
                # Emit signal to update toolbar
                from ui.main_window import MainWindow
                # Find parent main window and update slider
                parent = self.parent()
                while parent and not isinstance(parent, QWidget):
                    parent = parent.parent()
                if parent:
                    # Update slider value through main window
                    pass  # Will be handled by signal connection
            return

        # Normal zoom
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._view.scale(factor, factor)
        # Calculate zoom level
        transform = self._view.transform()
        zoom = transform.m11() * 100
        self.zoom_changed.emit(zoom)

    def _on_key_press(self, event: QKeyEvent):
        """Handle keyboard shortcuts."""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # Finish current shape
            self.finish_current_shape()
        elif event.key() == Qt.Key.Key_Escape:
            # Cancel current drawing
            self._reset_drawing_state()
        elif event.key() == Qt.Key.Key_V:
            # Switch to SELECT mode
            parent = self.parent()
            while parent:
                if hasattr(parent, '_toolbar'):
                    parent._toolbar.set_mode(ToolMode.SELECT)
                    break
                parent = parent.parent() if hasattr(parent, 'parent') else None
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            # Delete selected label (no modifier keys required)
            if self._selected_index >= 0:
                # Emit signal to request deletion
                self.label_delete_requested.emit(self._selected_index)
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key.Key_X:
            # Skip to next image without saving
            self.skip_image_requested.emit()

    def retranslate(self):
        self._placeholder.setText(tr("canvas_no_image"))


class _GraphicsView(QGraphicsView):
    """Custom QGraphicsView with signal forwarding for mouse events."""

    mouse_pressed = Signal(object, object)  # pos, button
    mouse_moved = Signal(object)
    mouse_released = Signal(object)
    mouse_double_clicked = Signal(object)
    wheel_zoomed = Signal(int, object)  # delta, modifiers
    key_pressed = Signal(object)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._panning = False
        self._pan_start = None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        self.mouse_pressed.emit(event.position().toPoint(), event.button())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning and self._pan_start:
            delta = event.position().toPoint() - self._pan_start
            self._pan_start = event.position().toPoint()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        self.mouse_moved.emit(event.position().toPoint())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self._pan_start = None
            self.unsetCursor()
            event.accept()
            return
        self.mouse_released.emit(event.position().toPoint())
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_double_clicked.emit(event.position().toPoint())
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        from PySide6.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        self.wheel_zoomed.emit(event.angleDelta().y(), modifiers)
        event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        self.key_pressed.emit(event)
        super().keyPressEvent(event)
