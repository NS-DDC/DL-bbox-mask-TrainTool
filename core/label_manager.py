"""Label data management with undo/redo support via QUndoStack."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Optional
import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QUndoStack, QUndoCommand


@dataclass(eq=False)
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
        # Deep copy each label so undo restores correct data even if
        # the original objects are later mutated (e.g. mask_data in-place).
        self._old_labels = [l.copy() for l in labels]
        labels.clear()
        self._manager.labels_changed.emit(self._image_path)

    def undo(self) -> None:
        self._manager._labels[self._image_path] = [l.copy() for l in self._old_labels]
        self._manager.labels_changed.emit(self._image_path)


# ---------------------------------------------------------------------------
# LabelManager
# ---------------------------------------------------------------------------

class ReplaceLabelsCommand(QUndoCommand):
    """Replace annotations as a single reversible edit."""

    def __init__(self, manager, image_path, labels, text="Apply auto labels"):
        super().__init__(text)
        self._manager = manager
        self._path = image_path
        self._old = [label.copy() for label in manager.get_labels(image_path)]
        self._new = [label.copy() for label in labels]
        self._old_forced_dirty = image_path in manager._forced_dirty

    def redo(self):
        # Explicit inference/edit results remain saveable even when empty.
        self._manager._forced_dirty.add(self._path)
        self._manager.set_labels(
            self._path, [label.copy() for label in self._new], mark_clean=False
        )

    def undo(self):
        if self._old_forced_dirty:
            self._manager._forced_dirty.add(self._path)
        else:
            self._manager._forced_dirty.discard(self._path)
        self._manager.set_labels(
            self._path, [label.copy() for label in self._old], mark_clean=False
        )


class LabelManager(QObject):
    """Manages per-image annotation labels with full undo/redo support.

    All mutating operations go through QUndoStack so that every change can
    be undone and redone.
    """

    labels_changed = Signal(str)  # emitted with image_path

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._labels: dict[str, list[LabelItem]] = {}
        # A loaded image is not necessarily edited.  Keep a compact signature
        # of the last disk/save state so browsing images cannot create empty
        # TXT files, while deleting the final real label still counts as edit.
        self._clean_signatures: dict[str, tuple] = {}
        self._forced_dirty: set[str] = set()
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

    def is_image_loaded(self, image_path: str) -> bool:
        """An explicitly empty annotation is distinct from an unread image."""
        return image_path in self._labels

    def is_dirty(self, image_path: str) -> bool:
        """Return whether annotations differ from their loaded/saved state."""
        if image_path not in self._labels:
            return False
        clean = self._clean_signatures.get(image_path, self._annotation_signature([]))
        return (
            image_path in self._forced_dirty
            or self._annotation_signature(self._labels[image_path]) != clean
        )

    def mark_clean(self, image_path: str) -> None:
        """Record the current annotations as successfully persisted."""
        if image_path in self._labels:
            self._clean_signatures[image_path] = self._annotation_signature(
                self._labels[image_path]
            )
            self._forced_dirty.discard(image_path)

    @property
    def loaded_image_paths(self) -> list[str]:
        return list(self._labels)

    def clear(self) -> None:
        self._undo_stack.clear()
        self._labels.clear()
        self._clean_signatures.clear()
        self._forced_dirty.clear()

    def replace_labels(self, image_path: str, labels: list[LabelItem], *, text="Apply auto labels") -> None:
        self._undo_stack.push(ReplaceLabelsCommand(self, image_path, labels, text))

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

    def set_labels(
        self, image_path: str, labels: list[LabelItem], *, mark_clean: bool = True
    ) -> None:
        """Directly replace all labels for an image without undo tracking.

        Disk loads use the default ``mark_clean=True``.  Internal transfers,
        such as temporarily moving a mask into the brush canvas, preserve the
        prior snapshot with ``mark_clean=False``.
        """
        self._labels[image_path] = list(labels)
        if mark_clean:
            self.mark_clean(image_path)
        self.labels_changed.emit(image_path)

    def remove_image(self, image_path: str) -> None:
        """Remove all label data for an image path entirely."""
        self._labels.pop(image_path, None)
        self._clean_signatures.pop(image_path, None)
        self._forced_dirty.discard(image_path)

    @staticmethod
    def _annotation_signature(labels: list[LabelItem]) -> tuple:
        """Build an order-independent signature matching persisted meaning.

        Per-class raster masks are unioned by ``SaveManager``.  Mirroring that
        here keeps an untouched mask clean when it is temporarily loaded into
        the brush and later restored as one combined mask.
        """
        vectors = []
        mask_groups: dict[tuple[int, str], list[np.ndarray | None]] = {}
        for label in labels:
            if label.label_type == "mask":
                mask_groups.setdefault((label.class_id, label.class_name), []).append(
                    label.mask_data
                )
                continue
            vectors.append((
                label.class_id,
                label.class_name,
                label.label_type,
                tuple((float(x), float(y)) for x, y in label.points),
            ))

        masks = []
        for key, arrays in mask_groups.items():
            if any(array is None for array in arrays):
                masks.append((*key, "invalid-none"))
                continue
            shapes = {array.shape for array in arrays if array is not None}
            if len(shapes) != 1:
                parts = sorted(
                    (array.shape, hashlib.sha256(
                        np.ascontiguousarray(array > 0).tobytes()
                    ).digest())
                    for array in arrays if array is not None
                )
                masks.append((*key, "shape-mismatch", tuple(parts)))
                continue
            combined = np.zeros(next(iter(shapes)), dtype=np.bool_)
            for array in arrays:
                combined |= np.asarray(array) > 0
            masks.append((
                *key,
                combined.shape,
                hashlib.sha256(np.ascontiguousarray(combined).tobytes()).digest(),
            ))
        return tuple(sorted(vectors)), tuple(sorted(masks))
