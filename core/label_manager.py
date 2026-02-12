"""Label data management with undo/redo support via QUndoStack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack, QUndoCommand


@dataclass
class LabelItem:
    """Represents a single annotation label on an image.

    Attributes:
        class_id: Integer class identifier.
        class_name: Human-readable class name.
        label_type: Either "bbox", "polygon", or "mask".
        points: List of (x, y) tuples in absolute pixel coordinates.
                For bbox: exactly 4 corner points (top-left, top-right,
                bottom-right, bottom-left).
                For polygon: N vertices describing the contour.
                For mask: empty list (mask data stored in mask_data).
        color: Display color as a hex string, e.g. "#FF0000".
        mask_data: Optional numpy array for raster-based segmentation.
                   Only used when label_type is "mask".
    """

    class_id: int
    class_name: str
    label_type: str  # "bbox", "polygon", or "mask"
    points: list[tuple[float, float]] = field(default_factory=list)
    color: str = "#00FF00"
    mask_data: Optional[np.ndarray] = None

    def copy(self) -> LabelItem:
        """Return a deep copy of this label item."""
        mask_copy = self.mask_data.copy() if self.mask_data is not None else None
        return LabelItem(
            class_id=self.class_id,
            class_name=self.class_name,
            label_type=self.label_type,
            points=list(self.points),
            color=self.color,
            mask_data=mask_copy,
        )


# ---------------------------------------------------------------------------
# QUndoCommand subclasses
# ---------------------------------------------------------------------------

class AddLabelCommand(QUndoCommand):
    """Undoable command that adds a label to an image."""

    def __init__(
        self,
        manager: LabelManager,
        image_path: str,
        label: LabelItem,
        parent: Optional[QUndoCommand] = None,
    ) -> None:
        super().__init__(parent)
        self.setText(f"Add {label.label_type} label '{label.class_name}'")
        self._manager = manager
        self._image_path = image_path
        self._label = label

    def redo(self) -> None:
        labels = self._manager._labels.setdefault(self._image_path, [])
        labels.append(self._label)
        self._manager.labels_changed.emit(self._image_path)

    def undo(self) -> None:
        labels = self._manager._labels.get(self._image_path, [])
        if self._label in labels:
            labels.remove(self._label)
        self._manager.labels_changed.emit(self._image_path)


class RemoveLabelCommand(QUndoCommand):
    """Undoable command that removes a label from an image."""

    def __init__(
        self,
        manager: LabelManager,
        image_path: str,
        label_index: int,
        parent: Optional[QUndoCommand] = None,
    ) -> None:
        super().__init__(parent)
        self._manager = manager
        self._image_path = image_path
        self._label_index = label_index
        self._label: Optional[LabelItem] = None
        self.setText(f"Remove label at index {label_index}")

    def redo(self) -> None:
        labels = self._manager._labels.get(self._image_path, [])
        if 0 <= self._label_index < len(labels):
            self._label = labels.pop(self._label_index)
        self._manager.labels_changed.emit(self._image_path)

    def undo(self) -> None:
        if self._label is not None:
            labels = self._manager._labels.setdefault(self._image_path, [])
            labels.insert(self._label_index, self._label)
        self._manager.labels_changed.emit(self._image_path)


class UpdateLabelCommand(QUndoCommand):
    """Undoable command that replaces a label at a given index."""

    def __init__(
        self,
        manager: LabelManager,
        image_path: str,
        label_index: int,
        new_label: LabelItem,
        parent: Optional[QUndoCommand] = None,
    ) -> None:
        super().__init__(parent)
        self.setText(f"Update label at index {label_index}")
        self._manager = manager
        self._image_path = image_path
        self._label_index = label_index
        self._new_label = new_label
        self._old_label: Optional[LabelItem] = None

    def redo(self) -> None:
        labels = self._manager._labels.get(self._image_path, [])
        if 0 <= self._label_index < len(labels):
            self._old_label = labels[self._label_index].copy()
            labels[self._label_index] = self._new_label
        self._manager.labels_changed.emit(self._image_path)

    def undo(self) -> None:
        if self._old_label is not None:
            labels = self._manager._labels.get(self._image_path, [])
            if 0 <= self._label_index < len(labels):
                labels[self._label_index] = self._old_label
        self._manager.labels_changed.emit(self._image_path)


class ClearLabelsCommand(QUndoCommand):
    """Undoable command that removes all labels for an image."""

    def __init__(
        self,
        manager: LabelManager,
        image_path: str,
        parent: Optional[QUndoCommand] = None,
    ) -> None:
        super().__init__(parent)
        self.setText(f"Clear labels for image")
        self._manager = manager
        self._image_path = image_path
        self._old_labels: list[LabelItem] = []

    def redo(self) -> None:
        labels = self._manager._labels.get(self._image_path, [])
        self._old_labels = list(labels)
        labels.clear()
        self._manager.labels_changed.emit(self._image_path)

    def undo(self) -> None:
        self._manager._labels[self._image_path] = list(self._old_labels)
        self._manager.labels_changed.emit(self._image_path)


# ---------------------------------------------------------------------------
# LabelManager
# ---------------------------------------------------------------------------

class LabelManager(QObject):
    """Manages per-image annotation labels with full undo/redo support.

    All mutating operations go through QUndoStack so that every change can
    be undone and redone.
    """

    labels_changed = Signal(str)  # emitted with image_path

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._labels: dict[str, list[LabelItem]] = {}
        self._undo_stack = QUndoStack(self)

    # -- Public properties ---------------------------------------------------

    @property
    def undo_stack(self) -> QUndoStack:
        """Expose the undo stack for external binding (e.g. undo/redo actions)."""
        return self._undo_stack

    # -- Query methods -------------------------------------------------------

    def get_labels(self, image_path: str) -> list[LabelItem]:
        """Return a *copy* of the label list for the given image."""
        return list(self._labels.get(image_path, []))

    def get_labels_ref(self, image_path: str) -> list[LabelItem]:
        """Return a direct reference to the internal label list (use with care)."""
        return self._labels.setdefault(image_path, [])

    def label_count(self, image_path: str) -> int:
        """Return the number of labels for an image."""
        return len(self._labels.get(image_path, []))

    # -- Mutating methods (undoable) -----------------------------------------

    def add_label(self, image_path: str, label: LabelItem) -> None:
        """Add a label to the given image (undoable)."""
        cmd = AddLabelCommand(self, image_path, label)
        self._undo_stack.push(cmd)

    def remove_label(self, image_path: str, label_index: int) -> None:
        """Remove a label by index from the given image (undoable)."""
        cmd = RemoveLabelCommand(self, image_path, label_index)
        self._undo_stack.push(cmd)

    def update_label(
        self, image_path: str, label_index: int, new_label: LabelItem
    ) -> None:
        """Replace a label at the given index (undoable)."""
        cmd = UpdateLabelCommand(self, image_path, label_index, new_label)
        self._undo_stack.push(cmd)

    def clear_labels(self, image_path: str) -> None:
        """Remove all labels for the given image (undoable)."""
        cmd = ClearLabelsCommand(self, image_path)
        self._undo_stack.push(cmd)

    # -- Bulk operations (non-undoable, for loading from disk) ---------------

    def set_labels(self, image_path: str, labels: list[LabelItem]) -> None:
        """Directly replace all labels for an image without undo tracking.

        Use this when loading labels from files, not for user edits.
        """
        self._labels[image_path] = list(labels)
        self.labels_changed.emit(image_path)

    def remove_image(self, image_path: str) -> None:
        """Remove all label data for an image path entirely."""
        self._labels.pop(image_path, None)
