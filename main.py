"""VisionAce - Deep Learning Labeling & Training Tool

Entry point for the application.
"""

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import tempfile
import traceback

APP_VERSION = "1.9.0-improved.1"
APP_NAME = "VisionAce Improved"


class _LogStream:
    """Give console-less library code a usable stream and preserve diagnostics."""

    encoding = "utf-8"

    def __init__(self, level):
        self.level = level

    def write(self, text):
        if text and text.strip():
            logging.getLogger("console").log(self.level, text.rstrip())
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def configure_logging():
    """Use a writable per-user location, even when extracted under Program Files."""
    requested = Path(os.environ.get("VISIONACE_HOME", str(Path.home() / ".visionace-improved")))
    last_error = None
    for base in (requested, Path(tempfile.gettempdir()) / "visionace-improved"):
        try:
            log_dir = base / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "visionace.log"
            handler = RotatingFileHandler(log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8")
            break
        except OSError as exc:
            last_error = exc
    else:
        raise OSError("Cannot create the application log in the user or temporary directory") from last_error
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().setLevel(logging.INFO)
    logging.getLogger().addHandler(handler)
    logging.captureWarnings(True)
    if sys.stdout is None or getattr(sys, "_visionace_redirect_stdio", False):
        sys.stdout = _LogStream(logging.INFO)
    if sys.stderr is None or getattr(sys, "_visionace_redirect_stdio", False):
        sys.stderr = _LogStream(logging.ERROR)
    return log_path


def _write_report(path, report):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _smoke_test(app, window, report_path):
    """CI-only packaging check: no model weights, model inference or downloads."""
    import cv2
    import numpy as np
    import PIL
    import PySide6
    import torch
    import torchvision
    import ultralytics
    from ultralytics import RTDETR, YOLO
    from PySide6.QtCore import QTimer

    # Exercise native binaries; importing alone misses torch/vision DLL mismatches.
    boxes = torch.tensor([[0., 0., 4., 4.], [0., 0., 4., 4.]])
    kept = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8]), 0.5)
    if kept.tolist() != [0]:
        raise RuntimeError("torchvision native NMS check failed")
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    if cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).shape != (8, 12):
        raise RuntimeError("OpenCV native conversion check failed")
    if not callable(YOLO) or not callable(RTDETR):
        raise RuntimeError("Ultralytics model entrypoints unavailable")
    app.processEvents()
    if app.property("visionaceStartupError"):
        raise RuntimeError(app.property("visionaceStartupError"))
    target = Path(report_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    screenshot = target.with_suffix(".png")
    if not window.grab().save(str(screenshot)):
        raise RuntimeError("Qt could not render the main window")
    QTimer.singleShot(100, app.quit)
    exit_code = app.exec()
    if app.property("visionaceStartupError"):
        raise RuntimeError(app.property("visionaceStartupError"))
    if exit_code:
        raise RuntimeError(f"Qt event loop exited with {exit_code}")
    report = {
        "status": "passed", "version": APP_VERSION,
        "frozen": bool(getattr(sys, "frozen", False)),
        "platform": platform.platform(), "python": platform.python_version(),
        "versions": {"PySide6": PySide6.__version__, "torch": torch.__version__,
                     "torchvision": torchvision.__version__, "ultralytics": ultralytics.__version__,
                     "opencv": cv2.__version__, "numpy": np.__version__, "Pillow": PIL.__version__},
        "checks": ["main_window_render", "qt_event_loop", "torchvision_native_nms", "opencv_native_conversion", "yolo_rtdetr_imports"],
        "weights_downloaded": False, "model_inference_tested": False,
        "cuda_runtime": torch.version.cuda, "screenshot": screenshot.name,
    }
    _write_report(target, report)
    logging.info("Packaging smoke check passed: %s", target)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=APP_NAME)
    parser.add_argument("--version", action="version", version=f"{APP_NAME} {APP_VERSION}")
    parser.add_argument("--smoke-test", action="store_true", help="CI startup and runtime packaging check; no model weights")
    parser.add_argument("--report", default="smoke-test.json", help="JSON report path for --smoke-test")
    args = parser.parse_args(argv)
    if args.smoke_test:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    log_path = configure_logging()
    logging.info("Starting %s %s frozen=%s", APP_NAME, APP_VERSION, getattr(sys, "frozen", False))
    app = None
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from config import get_config
        from i18n import set_language
        from ui.main_window import MainWindow

        app = QApplication([sys.argv[0]])
        app.setApplicationName(APP_NAME)
        app.setApplicationVersion(APP_VERSION)
        app.setOrganizationName("VisionAce-Improved")
        app.setStyle("Fusion")
        app.setStyleSheet(_DARK_STYLE)
        set_language(get_config().language)

        def handle_exception(exc_type, exc_value, exc_tb):
            logging.critical("Unhandled application exception", exc_info=(exc_type, exc_value, exc_tb))
            if args.smoke_test:
                app.setProperty("visionaceStartupError", str(exc_value))
                _write_report(args.report, {"status": "failed", "error": str(exc_value), "log": str(log_path)})
                app.exit(1)
            else:
                QMessageBox.critical(None, APP_NAME, f"{exc_value}\n\nDiagnostic log:\n{log_path}")

        sys.excepthook = handle_exception
        window = MainWindow()
        window.setWindowTitle(f"{window.windowTitle()} — Improved {APP_VERSION}")
        window.show()
        if args.smoke_test:
            return _smoke_test(app, window, args.report)
        return app.exec()
    except Exception as exc:
        logging.exception("Application startup failed")
        if args.smoke_test:
            _write_report(args.report, {"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "log": str(log_path)})
        elif app is not None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(None, APP_NAME, f"Startup failed: {exc}\n\nDiagnostic log:\n{log_path}")
        return 1


_DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #2b2b2b;
    color: #cccccc;
}
QMenuBar {
    background-color: #333333;
    color: #cccccc;
}
QMenuBar::item:selected {
    background-color: #4a4a4a;
}
QMenu {
    background-color: #333333;
    color: #cccccc;
    border: 1px solid #555555;
}
QMenu::item:selected {
    background-color: #4a90d9;
}
QToolBar {
    background-color: #333333;
    border: none;
    spacing: 4px;
    padding: 2px;
}
QStatusBar {
    background-color: #333333;
    color: #aaaaaa;
}
QSplitter::handle {
    background-color: #444444;
}
QListWidget {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #444444;
    outline: none;
}
QListWidget::item:selected {
    background-color: #4a90d9;
    color: white;
}
QListWidget::item:hover {
    background-color: #3a3a3a;
}
QGroupBox {
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    background-color: #3a3a3a;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 5px 12px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #666666;
}
QPushButton:pressed {
    background-color: #2a2a2a;
}
QPushButton:disabled {
    color: #666666;
    background-color: #333333;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 4px;
}
QTextEdit {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #555555;
}
QProgressBar {
    border: 1px solid #555555;
    border-radius: 3px;
    text-align: center;
    color: white;
}
QProgressBar::chunk {
    background-color: #4a90d9;
}
QCheckBox, QRadioButton {
    color: #cccccc;
}
QLabel {
    color: #cccccc;
}
QGraphicsView {
    background-color: #1a1a1a;
    border: none;
}
"""


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
