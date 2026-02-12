"""Training execution via Ultralytics in a background thread."""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import yaml
from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


def generate_data_yaml(
    train_path: str,
    val_path: str,
    class_names: list[str],
    output_path: str,
) -> str:
    """Create a YOLO-format ``data.yaml`` file.

    Args:
        train_path: Absolute path to the training images directory.
        val_path: Absolute path to the validation images directory.
        class_names: Ordered list of class names (index == class id).
        output_path: Where to write the YAML file.

    Returns:
        The absolute path to the written YAML file.
    """
    data: dict[str, Any] = {
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)

    logger.info("data.yaml written to %s", out)
    return str(out.resolve())


class _StdoutCapture(io.TextIOBase):
    """Thin wrapper that intercepts ``write()`` calls and forwards the text
    to a callback while still writing to the original stream."""

    def __init__(self, callback, original_stream):
        super().__init__()
        self._callback = callback
        self._original = original_stream

    def write(self, text: str) -> int:
        if text and text.strip():
            self._callback(text.rstrip("\n"))
        if self._original is not None:
            return self._original.write(text)
        return len(text)

    def flush(self) -> None:
        if self._original is not None:
            self._original.flush()


class TrainWorker(QThread):
    """Background worker that trains a YOLO or RT-DETR model.

    Signals:
        log_message(str): Raw training log lines captured from stdout.
        progress(int, int, float): ``(current_epoch, total_epochs, loss)``
            *Note*: loss may be ``0.0`` if it cannot be parsed from the log.
        finished_training(str): Path to the best model weights file.
        error(str): Emitted when training fails.
    """

    log_message = Signal(str)
    progress = Signal(int, int, float)
    finished_training = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        model_type: str,
        base_model_path: str,
        data_yaml_path: str,
        epochs: int = 100,
        batch_size: int = 16,
        imgsz: int = 640,
        lr: float = 0.01,
        device: str = "",
        parent: Optional[QThread] = None,
    ) -> None:
        super().__init__(parent)
        self._model_type = model_type
        self._base_model_path = base_model_path
        self._data_yaml_path = data_yaml_path
        self._epochs = epochs
        self._batch_size = batch_size
        self._imgsz = imgsz
        self._lr = lr
        self._device = device

    # -- Thread entry --------------------------------------------------------

    def run(self) -> None:  # noqa: D401
        """Instantiate the model and run ``model.train()``."""
        original_stdout = sys.stdout
        capture = _StdoutCapture(self._on_log_line, original_stdout)

        try:
            # Load the model.
            if self._model_type == "RT-DETR":
                from ultralytics import RTDETR
                model = RTDETR(self._base_model_path)
            else:
                from ultralytics import YOLO
                model = YOLO(self._base_model_path)

            # Redirect stdout so we can capture training logs.
            sys.stdout = capture

            train_kwargs: dict[str, Any] = {
                "data": self._data_yaml_path,
                "epochs": self._epochs,
                "batch": self._batch_size,
                "imgsz": self._imgsz,
                "lr0": self._lr,
                "verbose": True,
            }
            if self._device:
                train_kwargs["device"] = self._device

            results = model.train(**train_kwargs)

            # Determine best weights path.
            best_path = self._find_best_weights(model, results)
            self.finished_training.emit(best_path)

        except Exception as exc:
            logger.exception("Training failed: %s", exc)
            self.error.emit(str(exc))

        finally:
            sys.stdout = original_stdout

    # -- Internal helpers ----------------------------------------------------

    def _on_log_line(self, text: str) -> None:
        """Handle a single captured log line."""
        self.log_message.emit(text)
        self._try_parse_progress(text)

    def _try_parse_progress(self, text: str) -> None:
        """Attempt to extract epoch and loss information from a log line.

        Ultralytics log lines during training typically look like:
            ``     1/100   ...  0.1234  ...``
        We try to parse epoch info from this pattern.
        """
        parts = text.split()
        for part in parts:
            if "/" in part:
                tokens = part.split("/")
                if len(tokens) == 2:
                    try:
                        current = int(tokens[0])
                        total = int(tokens[1])
                        # Try to grab a loss value from subsequent numeric tokens.
                        loss = 0.0
                        for p in parts:
                            try:
                                val = float(p)
                                if 0 < val < 100:
                                    loss = val
                                    break
                            except ValueError:
                                continue
                        self.progress.emit(current, total, loss)
                        return
                    except ValueError:
                        continue

    @staticmethod
    def _find_best_weights(model: Any, results: Any) -> str:
        """Resolve the path to the best trained weights.

        Ultralytics typically saves the best weights at
        ``<project>/train/weights/best.pt``.
        """
        try:
            # results may expose save_dir or the trainer has it.
            save_dir = getattr(results, "save_dir", None)
            if save_dir is not None:
                best = Path(save_dir) / "weights" / "best.pt"
                if best.exists():
                    return str(best)
        except Exception:
            pass

        # Fallback: try model.trainer.
        try:
            trainer = getattr(model, "trainer", None)
            if trainer is not None:
                best = Path(trainer.save_dir) / "weights" / "best.pt"
                if best.exists():
                    return str(best)
        except Exception:
            pass

        return ""
