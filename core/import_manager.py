"""Preflighted, non-overwriting annotation import for an existing image folder."""

from __future__ import annotations

import filecmp
from pathlib import Path

import cv2

from core.export_manager import ExportManager
from core.image_io import copy_original_image, read_image
from core.project_metadata import load_classes, save_classes, validate_class_name
from core.save_manager import MASK_EXTENSIONS, SaveManager


def import_annotations(source_dir, project, current_classes):
    """Return (TXT count, GT count, class list) after safe additive import.

    Every file/mapping conflict and GT dimension mismatch is checked before
    writing. Existing files with different bytes are never replaced. Image
    sources are neither copied nor modified. A later I/O failure may leave a
    subset of new files, which can safely be retried.
    """
    project.assert_unique_image_stems()
    source_root = Path(source_dir)
    source_labels, source_gt = source_root / "labels", source_root / "gt_image"
    if not source_labels.is_dir() and not source_gt.is_dir():
        raise ValueError("The selected folder has no labels/ or gt_image/ directory.")
    incoming_classes = load_classes(source_labels)
    classes = [dict(cls) for cls in current_classes]
    incoming_names = [validate_class_name(cls["name"]) for cls in incoming_classes]
    if len({name.casefold() for name in incoming_names}) != len(incoming_names):
        raise ValueError("Imported class mapping contains duplicate names.")
    for class_id, incoming in enumerate(incoming_classes):
        if class_id < len(classes):
            if classes[class_id]["name"].casefold() != incoming["name"].casefold():
                raise ValueError(f"Class ID {class_id} differs between datasets. Import into a separate empty project.")
        else:
            if incoming["name"].casefold() in {cls["name"].casefold() for cls in classes}:
                raise ValueError("Imported class name uses a different ID. Import into a separate empty project.")
            classes.append(dict(incoming))

    images_by_stem = {Path(path).stem.casefold(): path for path in project.image_list}
    pending = []
    parsed_labels = []
    if source_labels.is_dir():
        for source in sorted(source_labels.iterdir()):
            if not source.is_file() or source.suffix.lower() != ".txt" or source.name.casefold() == "classes.txt":
                continue
            image_path = images_by_stem.get(source.stem.casefold())
            if image_path is None:
                continue
            parsed = ExportManager.load_yolo_txt(str(source), 1, 1,
                {i: cls["name"] for i, cls in enumerate(incoming_classes)})
            parsed_labels.extend(parsed)
            pending.append((source, Path(project.get_label_path(image_path)), "txt"))
    if parsed_labels and not incoming_classes:
        if any(label.class_name != f"class_{label.class_id}" for label in parsed_labels):
            raise ValueError("Legacy named annotations require a classes.txt/project class mapping for import.")
        if any(cls["name"] != f"class_{index}" for index, cls in enumerate(classes)):
            raise ValueError("External YOLO labels have no classes.txt/project class mapping. Supply that mapping before importing into a named project.")
        max_id = max(label.class_id for label in parsed_labels)
        for class_id in range(len(classes), max_id + 1):
            classes.append({"name": f"class_{class_id}", "color": "#00aaff"})
    elif any(label.class_id >= len(incoming_classes) for label in parsed_labels):
        raise ValueError("An external YOLO class ID is missing from its class mapping.")
    if incoming_classes and any(label.class_name.casefold() != incoming_classes[label.class_id]["name"].casefold()
                                for label in parsed_labels):
        raise ValueError("An external annotation name disagrees with its class mapping.")

    dimensions = {}
    if source_gt.is_dir():
        for class_dir in sorted(source_gt.iterdir()):
            if not class_dir.is_dir():
                continue
            name = validate_class_name(class_dir.name)
            existing = next((cls["name"] for cls in classes if cls["name"].casefold() == name.casefold()), None)
            if existing is None:
                classes.append({"name": name, "color": "#00aaff"})
            else:
                name = existing
            destination_dir = SaveManager._class_directory(Path(project.image_dir) / "gt_image", name)
            for source in sorted(class_dir.iterdir()):
                if not source.is_file() or source.suffix.lower() not in MASK_EXTENSIONS:
                    continue
                image_path = images_by_stem.get(source.stem.casefold())
                if image_path is None:
                    continue
                if image_path not in dimensions:
                    image = read_image(image_path)
                    if image is None:
                        raise ValueError(f"Cannot read source image: {image_path}")
                    dimensions[image_path] = image.shape[:2]
                    del image
                mask = read_image(source, cv2.IMREAD_UNCHANGED)
                if mask is None or mask.shape[:2] != dimensions[image_path]:
                    raise ValueError(f"Imported GT dimensions do not match the original image: {source}")
                del mask
                # A different extension with the same stem would shadow existing
                # data on reload; treat that as a conflict as well.
                destination = destination_dir / (Path(image_path).stem + source.suffix.lower())
                if destination_dir.is_dir():
                    for other in destination_dir.iterdir():
                        if (other.is_file() and other.stem.casefold() == destination.stem.casefold()
                                and other.suffix.lower() in MASK_EXTENSIONS and other.name.casefold() != destination.name.casefold()):
                            raise FileExistsError(f"Existing GT uses a different format: {other}. Import into a separate empty project.")
                pending.append((source, destination, "gt"))

    destinations = {}
    gt_stems = set()
    for source, destination, _ in pending:
        if _ == "gt":
            gt_key = str(destination.parent.resolve()).casefold(), destination.stem.casefold()
            if gt_key in gt_stems:
                raise ValueError(f"Multiple imported GT files share one class/image: {destination}")
            gt_stems.add(gt_key)
        key = str(destination.resolve()).casefold()
        if key in destinations and not filecmp.cmp(source, destinations[key], shallow=False):
            raise ValueError(f"Two imported files map to the same destination: {destination}")
        destinations[key] = source
        if destination.exists() and not filecmp.cmp(source, destination, shallow=False):
            raise FileExistsError(f"Existing annotation differs: {destination}. Import into a separate empty project.")
    if not pending:
        raise ValueError("No annotations match this project's image filenames.")
    # Mapping is additive and validated before copying; partial I/O failures
    # therefore cannot strand copied labels with an incorrect class mapping.
    save_classes(project.label_dir, classes)
    counts = {"txt": 0, "gt": 0}
    for source, destination, kind in pending:
        copy_original_image(source, destination)
        counts[kind] += 1
    return counts["txt"], counts["gt"], classes
