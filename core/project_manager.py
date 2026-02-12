"""Project and folder management for the VisionAce labeling tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


# Supported image extensions (case-insensitive matching is handled at scan time).
SUPPORTED_IMAGE_EXTENSIONS: set[str] = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
}


class ProjectManager(QObject):
    """Manages the currently opened image folder and associated label paths.

    When a folder is opened the manager scans it for images with supported
    extensions and builds an ordered list.  Label files are expected to live
    in a ``labels/`` subdirectory alongside the images, each sharing the
    same stem but with a ``.txt`` extension.
    """

    folder_changed = Signal()
    image_list_updated = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._image_dir: Optional[Path] = None
        self._label_dir: Optional[Path] = None
        self._image_list: list[str] = []

    # -- Properties ----------------------------------------------------------

    @property
    def image_dir(self) -> Optional[Path]:
        """Absolute path to the currently opened image directory."""
        return self._image_dir

    @property
    def label_dir(self) -> Optional[Path]:
        """Absolute path to the labels subdirectory inside the image directory."""
        return self._label_dir

    @property
    def image_list(self) -> list[str]:
        """Sorted list of absolute image file paths found in the current folder."""
        return list(self._image_list)

    @property
    def image_count(self) -> int:
        """Number of images in the current folder."""
        return len(self._image_list)

    # -- Public methods ------------------------------------------------------

    def open_folder(self, path: str) -> bool:
        """Scan *path* for images and set up the project directories.

        Returns ``True`` if the folder was opened successfully (i.e. it exists
        and is a directory), ``False`` otherwise.
        """
        folder = Path(path)
        if not folder.is_dir():
            return False

        self._image_dir = folder.resolve()

        # Create labels subdirectory
        self._label_dir = self._image_dir / "labels"
        self._label_dir.mkdir(exist_ok=True)

        self._scan_images()

        self.folder_changed.emit()
        self.image_list_updated.emit()
        return True

    def get_image_path(self, index: int) -> Optional[str]:
        """Return the absolute image path at *index*, or ``None`` if out of range."""
        if 0 <= index < len(self._image_list):
            return self._image_list[index]
        return None

    def get_label_path(self, image_path: str) -> str:
        """Derive the label ``.txt`` file path for a given image path.

        The label file is placed in the labels/ subdirectory with the same
        stem but with a ``.txt`` extension.
        """
        if self._label_dir is None:
            img = Path(image_path)
            return str(img.with_suffix(".txt"))
        img = Path(image_path)
        return str(self._label_dir / (img.stem + ".txt"))

    def has_labels(self, image_path: str) -> bool:
        """Return ``True`` if a label file already exists for the image."""
        label_path = self.get_label_path(image_path)
        return os.path.isfile(label_path)

    def get_image_index(self, image_path: str) -> int:
        """Return the index of *image_path* in the image list, or -1."""
        try:
            return self._image_list.index(image_path)
        except ValueError:
            return -1

    def refresh(self) -> None:
        """Re-scan the current folder for images."""
        if self._image_dir is not None:
            self._scan_images()
            self.image_list_updated.emit()

    def set_custom_label_dir(self, path: str) -> bool:
        """Set a custom label directory path.

        Args:
            path: Absolute path to the label directory. If empty, uses default labels/ subdirectory.

        Returns:
            True if successful, False if path is invalid.
        """
        if not path:
            # Reset to default labels/ subdirectory
            if self._image_dir:
                self._label_dir = self._image_dir / "labels"
                self._label_dir.mkdir(exist_ok=True)
            return True

        label_dir = Path(path)
        if not label_dir.exists():
            try:
                label_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return False

        if not label_dir.is_dir():
            return False

        self._label_dir = label_dir.resolve()
        return True

    # -- Internal helpers ----------------------------------------------------

    def _scan_images(self) -> None:
        """Populate ``_image_list`` by scanning ``_image_dir``."""
        self._image_list.clear()
        if self._image_dir is None:
            return

        for entry in sorted(self._image_dir.iterdir()):
            if entry.is_file() and entry.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                self._image_list.append(str(entry))
