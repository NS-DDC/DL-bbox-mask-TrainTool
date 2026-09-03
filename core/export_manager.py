"""Label export and import utilities for YOLO format and binary masks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.label_manager import LabelItem, LabelManager
from core.project_manager import ProjectManager
from core.image_io import atomic_write_image, atomic_write_text, read_image

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
        class_names: dict[int, str] | None = None,
    ) -> None:
        """Save labels in standard YOLO annotation format.

        Line format (bbox)::

            <class_id> <cx> <cy> <w> <h>

        Line format (polygon/mask)::

            <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>

        Args:
            labels: List of ``LabelItem`` instances.
            image_width: Width of the source image in pixels.
            image_height: Height of the source image in pixels.
            output_path: Destination ``.txt`` file path.
            class_names: Optional mapping of class id to class name.
        """
        if image_width <= 0 or image_height <= 0:
            raise ValueError(f"Invalid source image dimensions: {image_width}x{image_height}")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        lines: list[str] = []
        for label in labels:
            if label.class_id < 0:
                raise ValueError("Class IDs must be nonnegative")
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
                raise ValueError(f"Unknown label type: {label.label_type}")
            if line:
                lines.append(line)

        atomic_write_text(out, "\n".join(lines) + ("\n" if lines else ""))

    @staticmethod
    def _bbox_to_yolo_line(
        label: LabelItem, img_w: int, img_h: int
    ) -> str:
        """Convert a bbox LabelItem to a standard YOLO detection line.

        Standard YOLO format: ``<class_id> <cx> <cy> <w> <h>``
        """
        points = _validated_points(label, img_w, img_h, minimum=2)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        cx = ((x_min + x_max) / 2.0) / img_w
        cy = ((y_min + y_max) / 2.0) / img_h
        w = (x_max - x_min) / img_w
        h = (y_max - y_min) / img_h
        if w <= 0 or h <= 0:
            raise ValueError("A bounding box must have positive width and height")

        return f"{label.class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    @staticmethod
    def _polygon_to_yolo_line(
        label: LabelItem, img_w: int, img_h: int
    ) -> str:
        """Convert a polygon LabelItem to a standard YOLO segmentation line.

        Standard YOLO format: ``<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>``
        """
        coords: list[str] = []
        for x, y in _validated_points(label, img_w, img_h, minimum=3):
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        return f"{label.class_id} " + " ".join(coords)

    @staticmethod
    def _mask_to_yolo_line(
        label: LabelItem, img_w: int, img_h: int
    ) -> str:
        """Convert a mask LabelItem to YOLO segmentation line by finding contours."""
        mask = _validated_mask(label, img_w, img_h)

        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return ""

        # Keep all disconnected regions, not just the largest one. Raster GT
        # remains the lossless representation for holes and tiny components.
        lines = []
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            coords = [f"{value:.6f}" for point in approx
                      for value in (point[0][0] / img_w, point[0][1] / img_h)]
            lines.append(f"{label.class_id} " + " ".join(coords))
        return "\n".join(lines)

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

        Supports both old format (class_id first) and new format
        (class_name class_id first) for backward compatibility.

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
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Source image dimensions must be positive")

        labels: list[LabelItem] = []
        with open(path, "r", encoding="utf-8-sig") as fh:
            for line_number, raw_line in enumerate(fh, 1):
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    raise ValueError(f"Invalid annotation at {path}:{line_number}")

                # Detect format: new format has class_name as first token (non-numeric)
                try:
                    numeric_id = parts[0].lstrip("+-").isdigit()
                    if numeric_id:
                        class_id = int(parts[0])
                        cls_name = class_names.get(class_id, f"class_{class_id}")
                        values = [float(v) for v in parts[1:]]
                    else:
                        cls_name = parts[0]
                        class_id = int(parts[1])
                        values = [float(v) for v in parts[2:]]
                    if class_id < 0 or class_id >= 65535 or not all(np.isfinite(values)):
                        raise ValueError("Invalid class ID or non-finite coordinates")
                    if any(v < 0 or v > 1 for v in values):
                        raise ValueError("Normalized coordinates must be in [0, 1]")
                    if len(values) != 4 and (len(values) < 6 or len(values) % 2):
                        raise ValueError("Polygons need at least three complete points")
                    if len(values) == 4 and (values[2] <= 0 or values[3] <= 0):
                        raise ValueError("Bounding boxes need positive dimensions")
                except (ValueError, IndexError) as exc:
                    raise ValueError(f"Invalid annotation at {path}:{line_number}: {exc}") from exc

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
    # GT mask loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_gt_masks(
        gt_image_dir: str,
        image_path: str,
        image_width: int,
        image_height: int,
        class_names: dict[int, str],
    ) -> list[LabelItem]:
        """Load GT masks from gt_image/<class_name>/ directory structure.

        Args:
            gt_image_dir: Path to the gt_image/ directory.
            image_path: Path to the source image (used for matching filename).
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            class_names: Mapping of class id to class name.

        Returns:
            List of ``LabelItem`` instances with mask_data loaded.
        """
        gt_dir = Path(gt_image_dir)
        if not gt_dir.exists():
            return []

        img_stem = Path(image_path).stem
        labels: list[LabelItem] = []

        # Reverse map: class_name -> class_id
        name_to_id = {name: cid for cid, name in class_names.items()}

        for class_dir in sorted(gt_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            class_id = name_to_id.get(class_name, -1)

            # Try to find matching mask file (any extension)
            # Prefer lossless PNG over legacy JPEG masks if both are present.
            for mask_file in sorted(class_dir.iterdir(), key=lambda p: (p.suffix.lower() != ".png", p.name)):
                if (mask_file.is_file() and mask_file.stem.casefold() == img_stem.casefold()
                        and mask_file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}):
                    # Load mask
                    mask = read_image(mask_file, cv2.IMREAD_UNCHANGED)
                    if mask is None:
                        raise ValueError(f"Could not read GT mask: {mask_file}")
                    if mask.ndim == 3:
                        conversion = cv2.COLOR_BGRA2GRAY if mask.shape[2] == 4 else cv2.COLOR_BGR2GRAY
                        mask = cv2.cvtColor(mask, conversion)

                    if mask.shape != (image_height, image_width):
                        raise ValueError(
                            f"GT mask {mask_file} has dimensions {mask.shape}; "
                            f"source image requires {(image_height, image_width)}. "
                            "Resizing must be an explicit import operation."
                        )

                    # Lossless external masks commonly encode foreground as 1,
                    # including uint16 TIFF. Only JPEG needs noise rejection.
                    foreground = mask >= 128 if mask_file.suffix.lower() in {".jpg", ".jpeg"} else mask > 0
                    mask = foreground.astype(np.uint8) * 255

                    if mask.max() == 0:
                        # An explicit empty PNG clears any older JPEG/TIFF GT.
                        break

                    labels.append(
                        LabelItem(
                            class_id=class_id if class_id >= 0 else 0,
                            class_name=class_name,
                            label_type="mask",
                            points=[],
                            color=_color_for_class(class_id if class_id >= 0 else 0),
                            mask_data=mask,
                        )
                    )
                    break  # Only one mask per class per image

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
        """Render all labels as a single-channel binary mask and save as PNG."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Source image dimensions must be positive")
        mask = np.zeros((image_height, image_width), dtype=np.uint8)

        for label in labels:
            if label.label_type == "mask":
                mask_binary = _validated_mask(label, image_width, image_height)
                mask[mask_binary > 0] = 255
                continue

            pts = np.array(_validated_points(label, image_width, image_height,
                           minimum=2 if label.label_type == "bbox" else 3), dtype=np.int32)

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
        atomic_write_image(out, mask)

    @staticmethod
    def save_semantic_mask(
        labels: list[LabelItem],
        image_width: int,
        image_height: int,
        output_path: str,
        multi_label: bool = True,
    ) -> None:
        """Save segmentation mask as PNG with semantic class encoding."""
        if image_width <= 0 or image_height <= 0:
            raise ValueError("Source image dimensions must be positive")
        max_id = max((label.class_id for label in labels), default=0)
        if any(label.class_id < 0 for label in labels) or (multi_label and max_id >= 65535):
            raise ValueError("Semantic class IDs must be between 0 and 65534")
        dtype = np.uint16 if multi_label and max_id >= 255 else np.uint8
        mask = np.zeros((image_height, image_width), dtype=dtype)

        for label in labels:
            # Set pixel value based on mode
            pixel_value = (label.class_id + 1) if multi_label else 255

            if label.label_type == "bbox":
                pts = np.array(_validated_points(label, image_width, image_height, minimum=2), dtype=np.int32)
                xs = pts[:, 0]
                ys = pts[:, 1]
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                cv2.rectangle(mask, (x1, y1), (x2, y2), pixel_value, thickness=-1)
            elif label.label_type == "polygon":
                pts = np.array(_validated_points(label, image_width, image_height, minimum=3), dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], pixel_value)
            elif label.label_type == "mask":
                # Directly use the mask data
                mask_binary = _validated_mask(label, image_width, image_height)
                mask[mask_binary > 0] = pixel_value
            else:
                raise ValueError(f"Unknown label type: {label.label_type}")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_image(out, mask)

    # ------------------------------------------------------------------
    # Batch save
    # ------------------------------------------------------------------

    @staticmethod
    def save_all_labels(
        label_manager: LabelManager,
        project_manager: ProjectManager,
    ) -> int:
        """Save labels for every image that has annotations."""
        count = 0
        for image_path in project_manager.image_list:
            labels = label_manager.get_labels(image_path)
            if not labels:
                continue

            # Read image dimensions.
            img = read_image(image_path)
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

def _validated_mask(label: LabelItem, width: int, height: int) -> np.ndarray:
    if label.mask_data is None or label.mask_data.shape != (height, width):
        actual = None if label.mask_data is None else label.mask_data.shape
        raise ValueError(f"Mask dimensions {actual} do not match source image {(height, width)}")
    return (label.mask_data > 0).astype(np.uint8) * 255


def _validated_points(label: LabelItem, width: int, height: int, minimum: int) -> np.ndarray:
    points = np.asarray(label.points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum or not np.isfinite(points).all():
        raise ValueError(f"Invalid {label.label_type} points for class {label.class_id}")
    return np.clip(points, (0, 0), (width, height))


_DEFAULT_COLORS: list[str] = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def _color_for_class(class_id: int) -> str:
    """Return a deterministic hex color for a class id."""
    return _DEFAULT_COLORS[class_id % len(_DEFAULT_COLORS)]
