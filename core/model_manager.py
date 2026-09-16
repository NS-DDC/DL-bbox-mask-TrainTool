"""Local checkpoint loading and inference with actionable failure reporting."""
from __future__ import annotations

from contextlib import nullcontext
import logging
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

from core.image_io import read_image

logger = logging.getLogger(__name__)
DEFAULT_INFER_SIZE: int = 640

DINO_SUPPORT_MESSAGE = (
    "DINOv3 backbone weights alone do not produce bounding boxes or masks. "
    "A compatible trained detection/segmentation head, its configuration and "
    "a dedicated inference adapter are required. This release supports local "
    "Ultralytics YOLO/RT-DETR checkpoints and Keras segmentation models. "
    "DINOv3 and Grounding DINO are different model families."
)


def _is_rtdetr(model: Any) -> bool:
    """Inspect the loaded architecture, never the user's checkpoint filename."""
    network = getattr(model, "model", None)
    if "RTDETR" in type(network).__name__.upper():
        return True
    layers = getattr(network, "model", None)
    try:
        return "RTDETR" in type(layers[-1]).__name__.upper()
    except (TypeError, IndexError, KeyError):
        return "RTDETR" in type(model).__name__.upper()


def _normalize_device(device: str) -> str:
    value = str(device or "").strip().lower()
    if value in ("", "auto"):
        return ""
    if value == "cpu":
        return value
    if value == "cuda":
        return "0"
    if value.startswith("cuda:"):
        value = value[5:]
    if value.isdecimal():
        return str(int(value))
    raise ValueError("Choose auto, cpu, or one CUDA GPU index (for example 0 or cuda:0).")


class ModelManager(QObject):
    """Keep the previous working model when a replacement cannot be loaded.

    Only existing local weight files are accepted. AUTO uses the checkpoint's
    architecture, so RT-DETR weights named best.pt work like other names.
    A .pt file must be an Ultralytics checkpoint, not an arbitrary state_dict.
    """

    model_loaded = Signal(str)
    VALID_MODEL_TYPES = {"AUTO", "YOLO", "RT-DETR", "KERAS"}

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._model: Any = None
        self._model_path: Optional[str] = None
        self._model_type: Optional[str] = None
        self._model_task: Optional[str] = None
        self._keras_class_names: Optional[dict[int, str]] = None
        self._lock = RLock()
        self._last_device: Optional[str] = None
        self.last_error = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def last_device(self) -> Optional[str]:
        """Actual selected device from the last inference attempt."""
        return self._last_device

    def load_model(self, path: str, model_type: str = "AUTO") -> bool:
        with self._lock:
            self.last_error = ""
            try:
                requested = str(model_type).strip().upper()
                if requested in {"DINOV3", "DINO-V3", "DONOV3", "DINO"}:
                    raise ValueError(DINO_SUPPORT_MESSAGE)
                if requested not in self.VALID_MODEL_TYPES:
                    raise ValueError(f"Unsupported model type: {model_type}.")
                source = Path(path).expanduser()
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Model file does not exist: {path}. Select downloaded local weights; "
                        "model names and URLs are not downloaded automatically."
                    )
                source = source.resolve()
                suffix = source.suffix.lower()
                resolved_type = requested
                if requested == "AUTO":
                    resolved_type = "KERAS" if suffix in {".h5", ".keras"} else "YOLO"
                if resolved_type == "KERAS":
                    if suffix not in {".h5", ".keras"}:
                        raise ValueError("Keras requires a complete .h5 or .keras model, not weights only.")
                    from keras.models import load_model

                    candidate = load_model(str(source), compile=False)
                    input_shape = getattr(candidate, "input_shape", None)
                    output_shape = getattr(candidate, "output_shape", None)
                    if (not isinstance(input_shape, tuple) or len(input_shape) != 4
                            or input_shape[-1] not in (1, 3)):
                        raise ValueError(
                            "Keras requires one NHWC image input with 1 or 3 channels. "
                            "Input preprocessing is RGB (or grayscale), float32 divided by 255."
                        )
                    if (not isinstance(output_shape, tuple) or len(output_shape) not in (3, 4)
                            or (len(output_shape) == 4 and not output_shape[-1])):
                        raise ValueError(
                            "Keras auto-labeling requires one segmentation probability output "
                            "(N,H,W) or (N,H,W,C). Classification scores do not localize objects."
                        )
                    num_classes = int(output_shape[-1]) if len(output_shape) == 4 else 1
                    keras_names = {i: f"class_{i}" for i in range(num_classes)}
                    task = "segment"
                else:
                    if suffix != ".pt":
                        raise ValueError(
                            "YOLO/RT-DETR requires an Ultralytics .pt checkpoint. "
                            "Paddle, Hugging Face, .pth backbones, ONNX, and plain state_dict "
                            "files need their own adapters. " + DINO_SUPPORT_MESSAGE
                        )
                    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
                    from ultralytics import YOLO, RTDETR

                    candidate = RTDETR(str(source)) if resolved_type == "RT-DETR" else YOLO(str(source))
                    if _is_rtdetr(candidate):
                        # Recent YOLO constructors convert RT-DETR themselves. Older
                        # releases can load the head but select the wrong predictor.
                        if not isinstance(candidate, RTDETR):
                            candidate = RTDETR(str(source))
                        resolved_type = "RT-DETR"
                    elif resolved_type == "RT-DETR":
                        raise ValueError("Selected RT-DETR, but the checkpoint has a different architecture.")
                    task = str(getattr(candidate, "task", ""))
                    if task not in {"detect", "segment"}:
                        raise ValueError(
                            f"The checkpoint task is {task or 'unknown'}. This annotation workflow "
                            "requires detection or instance segmentation, not classification, pose or OBB."
                        )
                    keras_names = None

                # Commit only after every compatibility check has succeeded.
                self._model = candidate
                self._model_path = str(source)
                self._model_type = resolved_type
                self._model_task = task
                self._keras_class_names = keras_names
                self._last_device = None
            except Exception as exc:
                self.last_error = f"Could not load model: {type(exc).__name__}: {exc}"
                logger.exception("%s", self.last_error)
                return False
        self.model_loaded.emit(self._model_path)
        return True

    def get_model(self) -> Any:
        return self._model

    def get_model_type(self) -> Optional[str]:
        return self._model_type

    def get_model_task(self) -> Optional[str]:
        return self._model_task

    def get_model_path(self) -> Optional[str]:
        return self._model_path

    def predict(self, image_path: str, confidence: float = 0.25,
                infer_size: int = DEFAULT_INFER_SIZE, device: str = "") -> Any:
        """Infer the original decoded image, returning original-coordinate results.

        YOLO applies its letterbox; RT-DETR uses its own scale-fill transform.
        Neither changes source pixels or requires caller bbox rescaling.
        None means failure; last_error carries the reason. Empty Results.boxes
        is a successful inference with no detections.
        """
        with self._lock:
            self.last_error = ""
            try:
                if self._model is None:
                    raise RuntimeError("No model is loaded.")
                if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("Confidence must be between 0 and 1.")
                if isinstance(infer_size, bool) or int(infer_size) != infer_size or not 32 <= infer_size <= 4096:
                    raise ValueError("Inference size must be an integer between 32 and 4096 pixels.")
                requested_device = _normalize_device(device)
                image = read_image(image_path)
                if image is None:
                    raise ValueError(f"Cannot decode image: {image_path}")
                if self._model_type in {"YOLO", "RT-DETR"}:
                    import torch

                    selected = requested_device or ("0" if torch.cuda.is_available() else "cpu")
                    if selected != "cpu":
                        if not torch.cuda.is_available() or int(selected) >= torch.cuda.device_count():
                            raise ValueError(
                                f"CUDA GPU {selected} is unavailable in this build/environment. "
                                "Select cpu or install a compatible CUDA PyTorch environment."
                            )
                    if selected != self._last_device:
                        # The cached Ultralytics predictor owns its backend device.
                        self._model.predictor = None
                    self._last_device = selected
                    kwargs = dict(source=image, conf=float(confidence), imgsz=int(infer_size),
                                  device=selected, half=False, verbose=False, save=False,
                                  stream=False)
                    if self._model_task == "segment":
                        kwargs["retina_masks"] = True
                    return self._model.predict(**kwargs)
                if self._model_type == "KERAS":
                    return self._predict_keras(image, infer_size, requested_device)
                raise ValueError(f"Unsupported loaded model type: {self._model_type}")
            except Exception as exc:
                hint = ""
                if "out of memory" in str(exc).lower():
                    hint = " Reduce inference size or select cpu. Original label coordinates are preserved."
                self.last_error = f"Prediction failed: {type(exc).__name__}: {exc}{hint}"
                logger.exception("%s (%s)", self.last_error, image_path)
                return None

    def _predict_keras(self, image: Any, infer_size: int, device: str) -> dict:
        import cv2
        import numpy as np

        orig_h, orig_w = image.shape[:2]
        shape = self._model.input_shape
        target_h, target_w = int(shape[1] or infer_size), int(shape[2] or infer_size)
        resized = cv2.resize(image, (target_w, target_h))
        if shape[-1] == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)[..., None]
        else:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        batch = resized.astype(np.float32)[None, ...] / 255.0
        context = nullcontext()
        if device:
            import keras
            if keras.backend.backend() != "tensorflow":
                raise ValueError("Explicit Keras device selection requires the TensorFlow backend; choose auto.")
            import tensorflow as tf
            if device != "cpu" and int(device) >= len(tf.config.list_physical_devices("GPU")):
                raise ValueError(f"TensorFlow GPU {device} is unavailable; choose cpu or auto.")
            context = tf.device("/CPU:0" if device == "cpu" else f"/GPU:{device}")
        self._last_device = device or "auto"
        with context:
            predictions = self._model.predict(batch, verbose=0)
        if isinstance(predictions, (list, tuple, dict)):
            raise ValueError("Multiple Keras outputs require a model-specific adapter.")
        return {"predictions": np.asarray(predictions), "orig_shape": (orig_h, orig_w),
                "input_shape": (target_h, target_w), "model_type": "KERAS"}

    def get_class_names(self) -> dict[int, str]:
        if self._model is None:
            return {}
        try:
            names = self._keras_class_names if self._model_type == "KERAS" else self._model.names
            pairs = names.items() if isinstance(names, dict) else enumerate(names)
            return {int(key): str(value) for key, value in pairs}
        except (AttributeError, TypeError, ValueError):
            logger.warning("Model class names are unavailable", exc_info=True)
            return {}

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._model_path = self._model_type = self._model_task = None
            self._keras_class_names = None
            self._last_device = None
            self.last_error = ""
