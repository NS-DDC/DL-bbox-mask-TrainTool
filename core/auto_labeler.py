"""Auto-labeling engine that runs model inference in a background thread."""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.label_manager import LabelItem
from core.model_manager import ModelManager

logger = logging.getLogger(__name__)

# Default color palette for auto-generated labels (cycled if needed).
_DEFAULT_COLORS: list[str] = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def _color_for_class(class_id: int) -> str:
    """Return a deterministic hex color for a class id."""
    return _DEFAULT_COLORS[class_id % len(_DEFAULT_COLORS)]


def _mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
) -> list[tuple[float, float]]:
    """Convert a binary mask to a simplified polygon via OpenCV.

    Args:
        mask: 2-D uint8 array where non-zero pixels are foreground.
        epsilon_ratio: Fraction of the arc length used by ``approxPolyDP``
            for contour simplification.

    Returns:
        List of ``(x, y)`` tuples forming the polygon, or an empty list if
        no contour could be found.
    """
    if mask is None or mask.size == 0:
        return []

    # Ensure binary uint8.
    binary = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Pick the largest contour by area.
    contour = max(contours, key=cv2.contourArea)

    # Simplify.
    epsilon = epsilon_ratio * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)

    # approx has shape (N, 1, 2) – flatten to list of tuples.
    points: list[tuple[float, float]] = [
        (float(pt[0][0]), float(pt[0][1])) for pt in approx
    ]
    return points


def _xyxy_to_four_corners(
    x1: float, y1: float, x2: float, y2: float
) -> list[tuple[float, float]]:
    """Convert (x1, y1, x2, y2) box to four corner points.

    Order: top-left, top-right, bottom-right, bottom-left.
    """
    return [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]


class AutoLabelWorker(QThread):
    """Background worker that runs auto-labeling on a list of images.

    Signals:
        progress(int, int): ``(current_index, total_count)``
        image_done(str, list): ``(image_path, list_of_LabelItem)``
        finished_all(): Emitted when all images have been processed.
        error(str): Emitted when an unrecoverable error occurs.
    """

    progress = Signal(int, int)
    image_done = Signal(str, list)
    finished_all = Signal()
    error = Signal(str)

    def __init__(
        self,
        model_manager: ModelManager,
        image_paths: list[str],
        confidence: float = 0.25,
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._model_manager = model_manager
        self._image_paths = list(image_paths)
        self._confidence = confidence
        self._abort = False

    # -- Control -------------------------------------------------------------

    def abort(self) -> None:
        """Request the worker to stop after the current image."""
        self._abort = True

    # -- Thread entry --------------------------------------------------------

    def run(self) -> None:  # noqa: D401 (imperative mood ok for QThread.run)
        """Process each image through the model and emit results."""
        if not self._model_manager.is_loaded:
            self.error.emit("No model is loaded.")
            return

        total = len(self._image_paths)
        class_names = self._model_manager.get_class_names()

        for idx, image_path in enumerate(self._image_paths):
            if self._abort:
                break

            try:
                labels = self._process_image(image_path, class_names)
                self.image_done.emit(image_path, labels)
            except Exception as exc:
                logger.exception("Auto-label failed for %s", image_path)
                self.error.emit(f"Error processing {image_path}: {exc}")

            self.progress.emit(idx + 1, total)

        self.finished_all.emit()

    # -- Internal helpers ----------------------------------------------------

    def _process_image(
        self,
        image_path: str,
        class_names: dict[int, str],
    ) -> list[LabelItem]:
        """Run inference on a single image and return labels."""
        results = self._model_manager.predict(image_path, self._confidence)
        if results is None:
            return []

        labels: list[LabelItem] = []

        for result in results:
            # --- Bounding boxes -------------------------------------------
            if result.boxes is not None and len(result.boxes):
                boxes = result.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    cls_name = class_names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                    points = _xyxy_to_four_corners(x1, y1, x2, y2)

                    label = LabelItem(
                        class_id=cls_id,
                        class_name=cls_name,
                        label_type="bbox",
                        points=points,
                        color=_color_for_class(cls_id),
                    )
                    labels.append(label)

            # --- Masks (instance segmentation) ----------------------------
            if result.masks is not None and len(result.masks):
                masks = result.masks
                for i in range(len(masks)):
                    cls_id = int(result.boxes.cls[i].item())
                    cls_name = class_names.get(cls_id, str(cls_id))

                    mask_data = masks.data[i].cpu().numpy()
                    polygon_pts = _mask_to_polygon(mask_data)
                    if len(polygon_pts) < 3:
                        continue

                    # Scale polygon points from mask resolution to original
                    # image resolution if needed.
                    orig_shape = result.orig_shape  # (height, width)
                    mask_h, mask_w = mask_data.shape[:2]
                    if (mask_h, mask_w) != tuple(orig_shape):
                        scale_x = orig_shape[1] / mask_w
                        scale_y = orig_shape[0] / mask_h
                        polygon_pts = [
                            (px * scale_x, py * scale_y)
                            for px, py in polygon_pts
                        ]

                    label = LabelItem(
                        class_id=cls_id,
                        class_name=cls_name,
                        label_type="polygon",
                        points=polygon_pts,
                        color=_color_for_class(cls_id),
                    )
                    labels.append(label)

        return labels
