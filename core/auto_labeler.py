"""Background inference and conversion to original-image annotations."""
from __future__ import annotations

import logging
import math
from typing import Optional

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.label_manager import LabelItem
from core.model_manager import ModelManager, DEFAULT_INFER_SIZE

logger = logging.getLogger(__name__)
_DEFAULT_COLORS = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF", "#344593", "#6473FF", "#0018EC",
    "#8438FF", "#520085", "#CB38FF", "#FF95C8", "#FF37C7",
]


def _color_for_class(class_id: int) -> str:
    return _DEFAULT_COLORS[class_id % len(_DEFAULT_COLORS)]


def _as_numpy(value) -> np.ndarray:
    """Accept CPU/GPU tensors and NumPy arrays without importing PyTorch."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _simplify_polygon_pts(pts: np.ndarray, epsilon_ratio: float = 0.002) -> list[tuple[float, float]]:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        return []
    if pts.ndim != 2 or pts.shape[1] != 2 or not np.isfinite(pts).all():
        raise ValueError("Model returned malformed or non-finite polygon coordinates.")
    if len(pts) < 3:
        return []
    contour = pts.reshape(-1, 1, 2)
    epsilon = epsilon_ratio * cv2.arcLength(contour, closed=True)
    approx = cv2.approxPolyDP(contour, epsilon, closed=True)
    if len(approx) < 3 or cv2.contourArea(approx) <= 0:
        return []
    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


def _mask_to_polygons(mask: np.ndarray, epsilon_ratio: float = 0.002) -> list[list[tuple[float, float]]]:
    """Return every external component; raster output is required to retain holes."""
    if mask is None or mask.size == 0:
        return []
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    polygons = [_simplify_polygon_pts(c.reshape(-1, 2), epsilon_ratio) for c in contours]
    return [p for p in polygons if p]


def _mask_to_polygon(mask: np.ndarray, epsilon_ratio: float = 0.002) -> list[tuple[float, float]]:
    """Compatibility helper returning the largest external component."""
    polygons = _mask_to_polygons(mask, epsilon_ratio)
    return max(polygons, key=lambda p: cv2.contourArea(np.asarray(p, np.float32)), default=[])


def _upscale_prob_mask(prob_map: np.ndarray, target_w: int, target_h: int,
                       threshold: float) -> np.ndarray:
    resized = cv2.resize(prob_map.astype(np.float32), (target_w, target_h),
                         interpolation=cv2.INTER_LINEAR)
    return (resized >= threshold).astype(np.uint8) * 255


def _restore_instance_mask(data, orig_shape: tuple[int, int]) -> np.ndarray:
    """Remove YOLO letterbox padding before resizing a raster to original pixels.

    Retina masks already match orig_shape and pass through without resampling.
    The fallback is for older Ultralytics outputs; directly resizing a padded
    square mask would shift/stretch objects on non-square source images.
    """
    mask = _as_numpy(data).astype(np.float32)
    if mask.ndim != 2 or not np.isfinite(mask).all():
        raise ValueError("Model returned an invalid instance mask.")
    orig_h, orig_w = orig_shape
    if mask.shape != (orig_h, orig_w):
        height, width = mask.shape
        gain = min(height / orig_h, width / orig_w)
        pad_x = (width - orig_w * gain) / 2
        pad_y = (height - orig_h * gain) / 2
        left, right = round(pad_x - 0.1), round(width - pad_x + 0.1)
        top, bottom = round(pad_y - 0.1), round(height - pad_y + 0.1)
        mask = mask[top:bottom, left:right]
        if not mask.size:
            raise ValueError("The instance mask contains no image region after removing padding.")
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (mask >= 0.5).astype(np.uint8) * 255


def _xyxy_to_four_corners(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


class AutoLabelWorker(QThread):
    """Generate exactly one chosen annotation representation per instance.

    auto selects polygons for YOLO segmentation, bboxes for detectors, and
    raster masks for Keras semantic segmentation. Failures never emit an empty
    success result, so they cannot clear existing annotations in the caller.
    """

    progress = Signal(int, int)
    image_done = Signal(str, list)
    finished_all = Signal()
    error = Signal(str)

    def __init__(self, model_manager: ModelManager, image_paths: list[str],
                 confidence: float = 0.25, score_threshold: float = 0.25,
                 infer_size: int = DEFAULT_INFER_SIZE, device: str = "",
                 parent: Optional[QThread] = None, *, output_type: str = "auto") -> None:
        super().__init__(parent)
        self._model_manager = model_manager
        self._image_paths = list(dict.fromkeys(image_paths))
        self._confidence = confidence
        self._score_threshold = score_threshold
        self._infer_size = infer_size
        self._device = device
        self._output_type = output_type
        self._abort = False
        self.summary = {"total": len(self._image_paths), "processed": 0, "successful": 0,
                        "failed": 0, "empty": 0, "labels": 0, "cancelled": False,
                        "fatal_error": ""}

    def abort(self) -> None:
        """Stop between images; the caller must retain us until QThread.finished."""
        self._abort = True
        self.requestInterruption()

    def run(self) -> None:
        try:
            if not self._model_manager.is_loaded:
                raise ValueError("No model is loaded.")
            if self._output_type not in {"auto", "bbox", "polygon", "mask"}:
                raise ValueError("Output type must be auto, bbox, polygon or mask.")
            for value in (self._confidence, self._score_threshold):
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Confidence and score thresholds must be between 0 and 1.")
            self._score_threshold = max(self._confidence, self._score_threshold)
            total = len(self._image_paths)
            class_names = self._model_manager.get_class_names()
            for idx, image_path in enumerate(self._image_paths):
                if self._abort or self.isInterruptionRequested():
                    break
                try:
                    labels = self._process_image(image_path, class_names)
                    if self._abort or self.isInterruptionRequested():
                        break
                    if labels is None:
                        raise RuntimeError(self._model_manager.last_error or "The model did not return a prediction.")
                    self.summary["successful"] += 1
                    self.summary["empty"] += int(not labels)
                    self.summary["labels"] += len(labels)
                    self.image_done.emit(image_path, labels)
                except Exception as exc:
                    logger.exception("Auto-label failed for %s", image_path)
                    self.summary["failed"] += 1
                    self.error.emit(f"{image_path}\n{exc}")
                self.summary["processed"] += 1
                self.progress.emit(idx + 1, total)
        except Exception as exc:
            self.summary["fatal_error"] = str(exc)
            self.error.emit(str(exc))
        finally:
            self.summary["cancelled"] = self._abort or self.isInterruptionRequested()
            self.finished_all.emit()

    def _process_image(self, image_path: str, class_names: dict[int, str]) -> Optional[list[LabelItem]]:
        results = self._model_manager.predict(image_path, self._confidence, self._infer_size,
                                              device=self._device)
        if results is None:
            return None
        if isinstance(results, dict) and results.get("model_type") == "KERAS":
            return self._process_keras_results(results, class_names)
        if not isinstance(results, (list, tuple)) or not results:
            raise ValueError("Inference returned no Results object; this is not an empty detection result.")
        labels: list[LabelItem] = []
        for result in results:
            if getattr(result, "probs", None) is not None:
                raise ValueError("Classification scores cannot be converted into object bounding boxes.")
            if getattr(result, "obb", None) is not None or getattr(result, "keypoints", None) is not None:
                raise ValueError("OBB and pose results need a matching annotation workflow.")
            orig_h, orig_w = map(int, result.orig_shape)
            if orig_h <= 0 or orig_w <= 0:
                raise ValueError("Inference returned invalid original image dimensions.")
            boxes = getattr(result, "boxes", None)
            masks = getattr(result, "masks", None)
            task = self._model_manager.get_model_task()
            output = self._output_type
            if output == "auto":
                output = "polygon" if task == "segment" or masks is not None else "bbox"
            if output in {"polygon", "mask"} and task != "segment" and masks is None:
                raise ValueError("This detector only returns bboxes. Select bbox output or load a segmentation model.")
            if boxes is None:
                raise ValueError("The model result has no supported detection boxes/class scores.")
            if not len(boxes):
                continue
            if output in {"polygon", "mask"} and (masks is None or len(masks) != len(boxes)):
                raise ValueError("The segmentation result does not have one mask per detection.")
            xyxy = _as_numpy(boxes.xyxy)
            scores = _as_numpy(boxes.conf).reshape(-1)
            classes = _as_numpy(boxes.cls).reshape(-1)
            if xyxy.shape != (len(boxes), 4) or len(scores) != len(boxes) or len(classes) != len(boxes):
                raise ValueError("Model result box/class/score lengths do not match.")
            if not np.isfinite(xyxy).all() or not np.isfinite(scores).all() or not np.isfinite(classes).all():
                raise ValueError("Model returned non-finite box/class/score values.")
            if np.any(scores < 0) or np.any(scores > 1):
                raise ValueError("Model returned confidence scores outside [0,1].")
            if np.any(classes < 0) or np.any(classes != classes.astype(np.int64)):
                raise ValueError("Model returned invalid class identifiers.")
            polygons = masks.xy if output == "polygon" else None
            for index, score in enumerate(scores):
                if float(score) < self._score_threshold:
                    continue
                cls_id = int(classes[index])
                common = dict(class_id=cls_id, class_name=class_names.get(cls_id, str(cls_id)),
                              color=_color_for_class(cls_id))
                if output == "bbox":
                    x1, y1, x2, y2 = map(float, xyxy[index])
                    x1, x2 = np.clip([x1, x2], 0, orig_w)
                    y1, y2 = np.clip([y1, y2], 0, orig_h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    labels.append(LabelItem(**common, label_type="bbox",
                                            points=_xyxy_to_four_corners(float(x1), float(y1), float(x2), float(y2))))
                elif output == "polygon":
                    # masks.xy already compensates for letterbox and is in original
                    # image space. It does not synthesize missing boundary detail.
                    points = np.asarray(polygons[index], np.float32).copy()
                    if points.size:
                        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                            raise ValueError("Model returned malformed or non-finite polygon coordinates.")
                        points[:, 0] = np.clip(points[:, 0], 0, orig_w)
                        points[:, 1] = np.clip(points[:, 1], 0, orig_h)
                    polygon = _simplify_polygon_pts(points)
                    if polygon:
                        labels.append(LabelItem(**common, label_type="polygon", points=polygon))
                else:
                    mask = _restore_instance_mask(masks.data[index], (orig_h, orig_w))
                    if np.any(mask):
                        labels.append(LabelItem(**common, label_type="mask", mask_data=mask))
        return labels

    def _process_keras_results(self, results: dict, class_names: dict[int, str]) -> list[LabelItem]:
        predictions = np.asarray(results["predictions"])
        if predictions.ndim not in (3, 4) or predictions.shape[0] != 1:
            raise ValueError("Keras requires one segmentation output (1,H,W) or (1,H,W,C); classification cannot localize objects.")
        if not np.isfinite(predictions).all() or np.any(predictions < 0) or np.any(predictions > 1):
            raise ValueError("Keras output must contain finite probabilities in [0,1]. Logits need a model-specific activation adapter.")
        orig_h, orig_w = map(int, results["orig_shape"])
        if orig_h <= 0 or orig_w <= 0 or any(dim <= 0 for dim in predictions.shape):
            raise ValueError("Keras returned invalid image/output dimensions.")
        pred = predictions[0]
        if pred.ndim == 2:
            pred = pred[..., None]
        output = "mask" if self._output_type == "auto" else self._output_type
        labels: list[LabelItem] = []
        for cls_id in range(pred.shape[-1]):
            channel = pred[..., cls_id]
            if float(channel.max()) < self._score_threshold:
                continue
            mask = _upscale_prob_mask(channel, orig_w, orig_h, self._score_threshold)
            if not np.any(mask):
                continue
            common = dict(class_id=cls_id, class_name=class_names.get(cls_id, f"class_{cls_id}"),
                          color=_color_for_class(cls_id))
            if output == "mask":
                labels.append(LabelItem(**common, label_type="mask", mask_data=mask))
            elif output == "bbox":
                count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for x, y, width, height, area in stats[1:count]:
                    if area:
                        labels.append(LabelItem(**common, label_type="bbox", points=
                                                _xyxy_to_four_corners(int(x), int(y), int(x + width), int(y + height))))
            else:
                for polygon in _mask_to_polygons(mask):
                    labels.append(LabelItem(**common, label_type="polygon", points=polygon))
        return labels
