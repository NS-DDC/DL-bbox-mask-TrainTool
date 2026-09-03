"""Original-image-preserving annotation persistence.

Normal saves write annotations only. Optional dataset image copies retain the
source bytes and refuse conflicting destinations. Files are encoded in a staging
directory before replacement; each destination replacement is atomic.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

from core.export_manager import ExportManager
from core.image_io import atomic_write_bytes, copy_original_image, read_image
from core.label_manager import LabelItem

if TYPE_CHECKING:
    from core.label_manager import LabelManager
    from core.project_manager import ProjectManager

MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class SaveManager:
    def __init__(self, label_manager: "LabelManager", project_manager: "ProjectManager") -> None:
        self._labels = label_manager
        self._project = project_manager

    def save_image_labels(
        self,
        image_path: str,
        class_names: dict[int, str],
        image_size: tuple[int, int],
        *,
        copy_original: bool = False,
    ) -> list[str]:
        """Save annotations at source dimensions, including reviewed negatives.

        Multiple masks belonging to one class are unioned. Removed masks are
        replaced with empty PNGs so stale foreground cannot reappear on reload.
        Source images are never modified or re-encoded.
        """
        if not self._project.image_dir:
            return []
        self._project.assert_unique_image_stems()
        w, h = image_size
        if w <= 0 or h <= 0:
            raise ValueError("Source image dimensions must be positive")
        labels = self._labels.get_labels(image_path)
        grouped: dict[str, list[LabelItem]] = defaultdict(list)
        vector_labels = []
        for label in labels:
            if label.class_id < 0 or class_names.get(label.class_id) != label.class_name:
                raise ValueError(f"Class mapping does not match label: {label.class_id} / {label.class_name}")
            if label.label_type == "mask":
                grouped[label.class_name].append(label)
            elif label.label_type in ("bbox", "polygon"):
                vector_labels.append(label)
            else:
                raise ValueError(f"Unsupported label type: {label.label_type}")

        label_path = Path(self._project.get_label_path(image_path))
        gt_dir = Path(self._project.image_dir) / "gt_image"
        stem = Path(image_path).stem
        mask_names = set(grouped)
        if gt_dir.is_dir():
            for class_dir in gt_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                # Existing per-class datasets keep negative masks too. Also
                # clear obsolete class folders if this image had a stale mask.
                has_old_mask = any(p.is_file() and p.stem.casefold() == stem.casefold()
                                   and p.suffix.lower() in MASK_EXTENSIONS
                                   for p in class_dir.iterdir())
                if class_dir.name in class_names.values() or has_old_mask:
                    mask_names.add(class_dir.name)
        mask_paths = {name: self._class_directory(gt_dir, name) / f"{stem}.png"
                      for name in sorted(mask_names)}

        label_path.parent.mkdir(parents=True, exist_ok=True)
        # Complete validation/encoding of every annotation before touching any
        # destination. This is not a multi-file filesystem transaction.
        with tempfile.TemporaryDirectory(prefix=".visionace-save-", dir=label_path.parent) as directory:
            staging = Path(directory)
            pending = []
            staged_txt = staging / "labels.txt"
            ExportManager.save_yolo_txt(vector_labels, w, h, str(staged_txt), class_names)
            for index, (name, destination) in enumerate(mask_paths.items()):
                staged_mask = staging / f"mask-{index}.png"
                ExportManager.save_semantic_mask(grouped.get(name, []), w, h, str(staged_mask), multi_label=False)
                pending.append((staged_mask, destination))
            pending.append((staged_txt, label_path))
            for staged, destination in pending:
                # A custom labels folder can be on a different drive from GT.
                # Publish through a temporary file beside each destination.
                atomic_write_bytes(destination, staged.read_bytes())

        saved = [f"{len(vector_labels)} labels"]
        if mask_paths:
            saved.append(f"GT images ({len(mask_paths)} classes)")
        if copy_original:
            destination = Path(self._project.image_dir) / "images" / Path(image_path).name
            if copy_original_image(image_path, destination):
                saved.append("image")
        return saved

    def save_all_images(
        self, class_names: dict[int, str], *, copy_original: bool = False,
    ) -> tuple[int, int, int]:
        """Save loaded/edited images only; leave never-loaded disk labels intact."""
        if not self._project.image_dir:
            return 0, 0, 0
        self._project.assert_unique_image_stems()
        label_count = gt_count = image_count = 0
        for image_path in self._project.image_list:
            if not self._labels.is_image_loaded(image_path):
                continue
            image = read_image(image_path)
            if image is None:
                raise OSError(f"Could not read original image: {image_path}")
            h, w = image.shape[:2]
            del image
            saved = self.save_image_labels(image_path, class_names, (w, h), copy_original=copy_original)
            label_count += 1
            gt_count += any(item.startswith("GT images") for item in saved)
            image_count += "image" in saved
        return label_count, gt_count, image_count

    def delete_image_labels(self, image_path: str) -> None:
        """Remove annotation artifacts only, never the original image."""
        if not self._project.image_dir:
            return
        self._project.assert_unique_image_stems()
        label_path = Path(self._project.get_label_path(image_path))
        if label_path.is_file():
            label_path.unlink()
        gt_dir = Path(self._project.image_dir) / "gt_image"
        if gt_dir.is_dir():
            stem = Path(image_path).stem.casefold()
            for class_dir in gt_dir.iterdir():
                if not class_dir.is_dir():
                    continue
                self._class_directory(gt_dir, class_dir.name)
                for path in class_dir.iterdir():
                    if path.is_file() and path.stem.casefold() == stem and path.suffix.lower() in MASK_EXTENSIONS:
                        path.unlink()

    def load_labels_from_disk(
        self,
        image_path: str,
        image_size: tuple[int, int],
        class_names: dict[int, str],
        register_class_cb,
    ) -> list[LabelItem]:
        """Load labels without renumbering existing YOLO class IDs."""
        w, h = image_size
        if w <= 0 or h <= 0:
            return []
        class_names = dict(class_names)
        all_labels: list[LabelItem] = []
        label_path = self._project.get_label_path(image_path)
        if Path(label_path).is_file():
            labels = ExportManager.load_yolo_txt(label_path, w, h, class_names)
            names_by_id = {label.class_id: label.class_name for label in labels}
            max_id = max(names_by_id, default=-1)
            # A sparse label ID must not become the next available UI index.
            # Fill earlier absent IDs with stable placeholders before adding it.
            for class_id in range(max_id + 1):
                if class_id not in class_names:
                    name = names_by_id.get(class_id, f"class_{class_id}")
                    registered_id = register_class_cb(name)
                    if registered_id != class_id:
                        raise ValueError(f"Cannot preserve YOLO class ID {class_id} for {name}; load the dataset class mapping first")
                    class_names[class_id] = name
            all_labels.extend(labels)

        if self._project.image_dir:
            gt_dir = Path(self._project.image_dir) / "gt_image"
            if gt_dir.is_dir():
                for class_dir in sorted(gt_dir.iterdir()):
                    if class_dir.is_dir() and class_dir.name not in class_names.values():
                        new_id = register_class_cb(class_dir.name)
                        class_names[new_id] = class_dir.name
                all_labels.extend(ExportManager.load_gt_masks(str(gt_dir), image_path, w, h, class_names))
        return all_labels

    @staticmethod
    def _class_directory(gt_dir: Path, name: str) -> Path:
        """Class names are data, not paths (Windows and Unix validation)."""
        if (not name or name in {".", ".."} or name[-1:] in {".", " "}
                or any(ord(char) < 32 or char in '<>:"/\\|?*' for char in name)):
            raise ValueError(f"Class name cannot be used as a folder name: {name!r}")
        reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
        if name.split(".")[0].upper() in reserved:
            raise ValueError(f"Reserved Windows class folder name: {name!r}")
        directory = gt_dir / name
        if not directory.resolve().is_relative_to(gt_dir.resolve()):
            raise ValueError(f"Class directory escapes the annotation folder: {directory}")
        return directory
