"""Model loading and inference management for YOLO, RT-DETR, and Keras models."""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import QObject, Signal

logger = logging.getLogger(__name__)


class ModelManager(QObject):
    """Loads and manages models for inference.

    Supported model types:
    - ``"YOLO"`` – loaded via ``ultralytics.YOLO`` (.pt files)
    - ``"RT-DETR"`` – loaded via ``ultralytics.RTDETR`` (.pt files)
    - ``"KERAS"`` – loaded via ``tensorflow.keras`` (.h5 files)
    """

    model_loaded = Signal(str)  # emitted with the model file path

    # Valid model type identifiers.
    VALID_MODEL_TYPES: set[str] = {"YOLO", "RT-DETR", "KERAS"}

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._model: Any = None
        self._model_path: Optional[str] = None
        self._model_type: Optional[str] = None
        self._keras_class_names: Optional[dict[int, str]] = None

    # -- Properties ----------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if a model is currently loaded."""
        return self._model is not None

    # -- Public methods ------------------------------------------------------

    def load_model(self, path: str, model_type: str) -> bool:
        """Load a model from *path*.

        Args:
            path: Filesystem path to the model weights file (e.g. ``.pt`` or ``.h5``).
            model_type: One of ``"YOLO"``, ``"RT-DETR"``, or ``"KERAS"``.

        Returns:
            ``True`` if the model was loaded successfully, ``False`` otherwise.
        """
        if model_type not in self.VALID_MODEL_TYPES:
            logger.error("Unsupported model type: %s", model_type)
            return False

        try:
            if model_type == "YOLO":
                from ultralytics import YOLO
                self._model = YOLO(path)
            elif model_type == "RT-DETR":
                from ultralytics import RTDETR
                self._model = RTDETR(path)
            elif model_type == "KERAS":
                try:
                    import tensorflow as tf
                    self._model = tf.keras.models.load_model(path)
                    # Set default class names for Keras models
                    # User can override these via get_class_names()
                    num_classes = self._model.output_shape[-1] if hasattr(self._model, 'output_shape') else 1
                    self._keras_class_names = {i: f"class_{i}" for i in range(num_classes)}
                except ImportError:
                    logger.error("TensorFlow not installed. Cannot load .h5 model.")
                    return False

            self._model_path = path
            self._model_type = model_type
            logger.info("Loaded %s model from %s", model_type, path)
            self.model_loaded.emit(path)
            return True

        except Exception as exc:
            logger.exception("Failed to load model from %s: %s", path, exc)
            self._model = None
            self._model_path = None
            self._model_type = None
            return False

    def get_model(self) -> Any:
        """Return the underlying Ultralytics model object, or ``None``."""
        return self._model

    def get_model_type(self) -> Optional[str]:
        """Return the model type string, or ``None`` if no model is loaded."""
        return self._model_type

    def get_model_path(self) -> Optional[str]:
        """Return the file path of the loaded model, or ``None``."""
        return self._model_path

    def predict(
        self,
        image_path: str,
        confidence: float = 0.25,
    ) -> Any:
        """Run inference on a single image.

        Args:
            image_path: Path to the image file.
            confidence: Minimum confidence threshold for detections.

        Returns:
            The Ultralytics ``Results`` object for YOLO/RT-DETR,
            or numpy array for Keras models, or ``None`` on failure.
        """
        if self._model is None:
            logger.warning("predict() called but no model is loaded.")
            return None

        try:
            if self._model_type in ["YOLO", "RT-DETR"]:
                results = self._model.predict(
                    source=image_path,
                    conf=confidence,
                    verbose=False,
                )
                return results
            elif self._model_type == "KERAS":
                import cv2
                import numpy as np
                import tensorflow as tf

                # Load and preprocess image
                img = cv2.imread(image_path)
                if img is None:
                    return None

                # Resize to model input size (assuming 224x224, adjust as needed)
                input_shape = self._model.input_shape[1:3]
                img_resized = cv2.resize(img, input_shape)
                img_array = np.expand_dims(img_resized, axis=0)
                img_array = img_array / 255.0  # Normalize

                # Predict
                predictions = self._model.predict(img_array, verbose=0)
                return predictions
            else:
                return None
        except Exception as exc:
            logger.exception("Prediction failed for %s: %s", image_path, exc)
            return None

    def get_class_names(self) -> dict[int, str]:
        """Return the model's class-name mapping ``{id: name}``.

        Returns an empty dict if no model is loaded or the mapping is
        unavailable.
        """
        if self._model is None:
            return {}

        try:
            if self._model_type == "KERAS":
                return self._keras_class_names or {}
            else:
                return dict(self._model.names)
        except Exception:
            return {}

    def unload(self) -> None:
        """Unload the current model and free resources."""
        self._model = None
        self._model_path = None
        self._model_type = None
        self._keras_class_names = None
        logger.info("Model unloaded.")
