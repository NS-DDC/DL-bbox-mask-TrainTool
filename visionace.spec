# -*- mode: python ; coding: utf-8 -*-
"""Portable CPU build. Run only in a prepared Windows x64 build environment."""
import os

from PyInstaller.utils.hooks import (
    collect_all, collect_data_files, collect_dynamic_libs, copy_metadata,
)

os.environ["YOLO_AUTOINSTALL"] = "false"
os.environ["YOLO_OFFLINE"] = "true"
os.environ["MPLBACKEND"] = "Agg"

# Ultralytics discovers models dynamically; preserve its YAML and Python sources.
ultra_data, ultra_bins, ultra_imports = collect_all("ultralytics", include_py_files=True)
vision_data, vision_bins, vision_imports = collect_all("torchvision", include_py_files=True)
datas = ultra_data + vision_data + copy_metadata("ultralytics", recursive=True)
datas += copy_metadata("PySide6") + copy_metadata("Pillow")
datas += collect_data_files("torch", include_py_files=True)
datas += [(os.path.join(SPECPATH, "assets", "fonts"), "assets/fonts")]
binaries = ultra_bins + vision_bins + collect_dynamic_libs("torch")

a = Analysis(
    ["main.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=ultra_imports + vision_imports + [
        "PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets",
        "cv2", "numpy", "PIL.Image", "torch", "torchvision.ops",
    ],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    runtime_hooks=[os.path.join(SPECPATH, "scripts", "runtime_hook.py")],
    # Keras/TensorFlow remain optional source-environment features.
    excludes=["tensorflow", "keras", "PyQt5", "PyQt6", "PySide2", "IPython", "pytest", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="VisionAce-Improved",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False,
    upx=False,
    name="VisionAce-Improved",
)
