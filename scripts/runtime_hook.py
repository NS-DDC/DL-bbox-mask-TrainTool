"""Prepare a console-less frozen process before third-party runtime hooks."""
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("MPLBACKEND", "Agg")
base = Path(os.environ.get("VISIONACE_HOME", str(Path.home() / ".visionace-improved")))
try:
    base.mkdir(parents=True, exist_ok=True)
except OSError:
    base = Path(tempfile.gettempdir()) / "visionace-improved"
    base.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(base / "ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(base / "matplotlib"))
# Some libraries inspect streams during import, before main configures logging.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
    sys._visionace_redirect_stdio = True
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
    sys._visionace_redirect_stdio = True
