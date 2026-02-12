"""Label export and import utilities for YOLO format and binary masks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.label_manager import LabelItem, LabelManager
from core.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class ExportManager:
    """Handles reading and writing label files in YOLO format as well as
    generating binary segmentation masks."""

    # ------------------------------------------------------------------
    # YOLO TXT – save
    # ------------------------------------------------------------------

    @staticmethod
    def save_yolo_txt(
        labels: list[LabelItem],
        image_width: int,
        image_height: int,
        output_path: str,
    ) -> None:
        """Save labels in YOLO annotation format.

        For **bbox** labels the line format is::

            <class_id> <cx> <cy> <w> <h>

        where all values are normalized to [0, 1].

        For **polygon** and **mask** labels the line format is::

            <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

        where all coordinates are normalized. Masks are converted to polygons.

        Args:
            labels: List of ``LabelItem`` instances.
            image_width: Width of the source image in pixels.
            image_height: Height of the source image in pixels.
            output_path: Destination ``.txt`` file path.
        """
        if image_width <= 0 or image_height <= 0:
            logger.error("Invalid image dimensions: %dx%d", image_width, image_height)
            return

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for label in labels:
            if label.label_type == "bbox":
                line = ExportManager._bbox_to_yolo_line(
                    label, image_width, image_height
                )
            elif label.label_type == "polygon":
                line = ExportManager._polygon_to_yolo_line(
                    label, image_width, image_height
                )
            elif label.label_type == "mask":
                # Convert mask to polygon contours
                line = ExportManager._mask_to_yolo_line(
                    label, image_width, image_height
                )
            else:
                logger.warning("Unknown label type: %s", label.label_type)
                continue
            if line:
                lines.append(line)

        with open(out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
            if lines:
                fh.write("\n")

    @staticmethod
    def _bbox_to_yolo_line(
        label: LabelItem, img_w: int, img_h: int
    ) -> str:
        """Convert a bbox LabelItem to a YOLO detection line."""
        xs = [p[0] for p in label.points]
        ys = [p[1] for p in label.points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = ((x_min + x_max) / 2.0) / img_w
        cy = ((y_min + y_max) / 2.0) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h

        return f"{label.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def _polygon_to_yolo_line(
        label: LabelItem, img_w: int, img_h: int
    ) -> str:
        """Convert a polygon LabelItem to a YOLO segmentation line."""
        coords: list[str] = []
        for x, y in label.points:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        return f"{label.class_id} " + " ".join(coords)

    @staticmethod
    def _mask_to_yolo_line(
        label: LabelItem, img_w: int, img_h: int
    ) -> str:
        """Convert a mask LabelItem to YOLO segmentation line by finding contours."""
        if label.mask_data is None:
            return ""

        # Find contours in the mask
        contours, _ = cv2.findContours(
            label.mask_data, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return ""

        # Use the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Simplify contour to reduce point count
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)

        # Convert to normalized coordinates
        coords: list[str] = []
        for point in approx:
            x, y = point[0]
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")

        if len(coords) < 6:  # At least 3 points
            return ""

        return f"{label.class_id} " + " ".join(coords)

    # ------------------------------------------------------------------
    # YOLO TXT – load
    # ------------------------------------------------------------------

    @staticmethod
    def load_yolo_txt(
        txt_path: str,
        image_width: int,
        image_height: int,
        class_names: dict[int, str],
    ) -> list[LabelItem]:
        """Load labels from a YOLO-format ``.txt`` annotation file.

        Args:
            txt_path: Path to the annotation file.
            image_width: Image width in pixels (for de-normalizing).
            image_height: Image height in pixels (for de-normalizing).
            class_names: Mapping of class id to class name.

        Returns:
            List of ``LabelItem`` instances.
        """
        path = Path(txt_path)
        if not path.is_file():
            return []

        labels: list[LabelItem] = []
        with open(path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                class_id = int(parts[0])
                values = [float(v) for v in parts[1:]]

                if len(values) == 4:
                    # bbox: cx cy w h (normalized)
                    cx, cy, w, h = values
                    x_min = (cx - w / 2.0) * image_width
                    y_min = (cy - h / 2.0) * image_height
                    x_max = (cx + w / 2.0) * image_width
                    y_max = (cy + h / 2.0) * image_height
                    points = [
                        (x_min, y_min),
                        (x_max, y_min),
                        (x_max, y_max),
                        (x_min, y_max),
                    ]
                    label_type = "bbox"
                else:
                    # polygon: x1 y1 x2 y2 ... (normalized)
                    points = []
                    for i in range(0, len(values) - 1, 2):
                        px = values[i] * image_width
                        py = values[i + 1] * image_height
                        points.append((px, py))
                    label_type = "polygon"

                cls_name = class_names.get(class_id, str(class_id))
                labels.append(
                    LabelItem(
                        class_id=class_id,
                        class_name=cls_name,
                        label_type=label_type,
                        points=points,
                        color=_color_for_class(class_id),
                    )
                )

        return labels

    # ------------------------------------------------------------------
    # Binary mask export
    # ------------------------------------------------------------------

    @staticmethod
    def save_binary_mask(
        labels: list[LabelItem],
        image_width: int,
        image_height: int,
        output_path: str,
    ) -> None:
        """Render all labels as a single-channel binary mask and save as PNG.

        All foreground regions (both bboxes and polygons) are drawn as white
        (255) on a black (0) background.

        Args:
            labels: Labels to render.
            image_width: Width of the output mask.
            image_height: Height of the output mask.
            output_path: Destination PNG file path.
        """
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for label in labels:
            pts = np.array(label.points, dtype=np.int32)

            if label.label_type == "bbox":
                xs = pts[:, 0]
                ys = pts[:, 1]
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, thickness=-1)
            elif label.label_type == "polygon":
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 255)

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mask)

    @staticmethod
    def save_semantic_mask(
        labels: list[LabelItem],
        image_width: int,
        image_height: int,
        output_path: str,
        multi_label: bool = True,
    ) -> None:
        """Save segmentation mask as PNG with semantic class encoding.

        Args:
            labels: Labels to render.
            image_width: Width of the output mask.
            image_height: Height of the output mask.
            output_path: Destination PNG file path.
            multi_label: If True, pixel value = class_id + 1 (0 reserved for background).
                        If False, binary mask (0=background, 255=foreground).
        """
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for label in labels:
            # Set pixel value based on mode
            pixel_value = (label.class_id + 1) if multi_label else 255

            if label.label_type == "bbox":
                pts = np.array(label.points, dtype=np.int32)
                xs = pts[:, 0]
                ys = pts[:, 1]
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv2.rectangle(mask, (x1, y1), (x2, y2), pixel_value, thickness=-1)
            elif label.label_type == "polygon":
                pts = np.array(label.points, dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], pixel_value)
            elif label.label_type == "mask" and label.mask_data is not None:
                # Directly use the mask data
                mask_binary = (label.mask_data > 0).astype(np.uint8)
                mask[mask_binary > 0] = pixel_value

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mask)

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------

    @staticmethod
    def save_all_labels(
        label_manager: LabelManager,
        project_manager: ProjectManager,
    ) -> int:
        """Save labels for every image that has annotations.

        Labels are written in YOLO format to the project's label directory.

        Args:
            label_manager: Source of annotation data.
            project_manager: Provides image list and label paths.

        Returns:
            Number of label files written.
        """
        count = 0
        for image_path in project_manager.image_list:
            labels = label_manager.get_labels(image_path)
            if not labels:
                continue

            # Read image dimensions.
            img = cv2.imread(image_path)
            if img is None:
                logger.warning("Could not read image: %s", image_path)
                continue

            h, w = img.shape[:2]
            label_path = project_manager.get_label_path(image_path)
            ExportManager.save_yolo_txt(labels, w, h, label_path)
            count += 1

        logger.info("Saved labels for %d images.", count)
        return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_COLORS: list[str] = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def _color_for_class(class_id: int) -> str:
    """Return a deterministic hex color for a class id."""
    return _DEFAULT_COLORS[class_id % len(_DEFAULT_COLORS)]
