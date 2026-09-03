"""Regression coverage for source-preserving labeling; executed in CI only."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core.export_manager import ExportManager
from core.image_io import atomic_write_image, read_image
from core.label_manager import LabelItem
from core.project_manager import ProjectManager
from core.save_manager import SaveManager


class Labels:
    def __init__(self):
        self.items = {}

    def get_labels(self, path):
        return self.items.get(path, [])

    def is_image_loaded(self, path):
        return path in self.items


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "원본.tiff"
    pixels = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30) * 100
    atomic_write_image(source, pixels)
    project = ProjectManager()
    project.open_folder(str(tmp_path))
    labels = Labels()
    return source, project, labels, SaveManager(labels, project)


def mask_label(mask, name="defect"):
    return LabelItem(class_id=0, class_name=name, label_type="mask", mask_data=mask)


def test_save_unions_masks_and_preserves_source_bytes(dataset):
    source, project, labels, saver = dataset
    before = source.read_bytes()
    first, second = np.zeros((20, 30), np.uint8), np.zeros((20, 30), np.uint8)
    first[2:5, 3:6] = 255
    second[11:14, 22:25] = 255
    labels.items[str(source)] = [mask_label(first), mask_label(second)]
    saver.save_image_labels(str(source), {0: "defect"}, (30, 20))
    gt = read_image(source.parent / "gt_image" / "defect" / f"{source.stem}.png", cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(gt, np.maximum(first, second))
    assert gt.shape == (20, 30)
    assert source.read_bytes() == before
    assert not (source.parent / "images").exists()


def test_empty_save_clears_masks_and_overrides_stale_legacy_gt(dataset):
    source, project, labels, saver = dataset
    gt_dir = source.parent / "gt_image" / "defect"
    atomic_write_image(gt_dir / f"{source.stem}.jpg", np.full((20, 30), 255, np.uint8))
    labels.items[str(source)] = []
    saver.save_image_labels(str(source), {0: "defect"}, (30, 20))
    assert Path(project.get_label_path(str(source))).read_text() == ""
    assert ExportManager.load_gt_masks(str(gt_dir.parent), str(source), 30, 20, {0: "defect"}) == []


def test_save_all_handles_loaded_empty_but_leaves_unloaded_annotations(dataset):
    source, project, labels, saver = dataset
    other = source.parent / "unloaded.png"
    atomic_write_image(other, np.zeros((20, 30, 3), np.uint8))
    project.refresh()
    first_txt = Path(project.get_label_path(str(source)))
    other_txt = Path(project.get_label_path(str(other)))
    first_txt.write_text("0 0.5 0.5 0.3 0.2\n")
    original = "0 0.2 0.2 0.1 0.1\n"
    other_txt.write_text(original)
    labels.items[str(source)] = []
    assert saver.save_all_images({0: "defect"}) == (1, 0, 0)
    assert first_txt.read_text() == ""
    assert other_txt.read_text() == original


def test_invalid_mask_cannot_partially_replace_existing_txt(dataset):
    source, project, labels, saver = dataset
    text_path = Path(project.get_label_path(str(source)))
    text_path.write_text("original labels")
    labels.items[str(source)] = [mask_label(np.ones((10, 15), np.uint8))]
    with pytest.raises(ValueError, match="dimensions"):
        saver.save_image_labels(str(source), {0: "defect"}, (30, 20))
    assert text_path.read_text() == "original labels"
    assert not (source.parent / "gt_image").exists()


def test_stem_collisions_rejected_without_switching_project(dataset):
    source, project, _, _ = dataset
    collision_dir = source.parent / "collision"
    collision_dir.mkdir()
    (collision_dir / "same.png").write_bytes(b"png")
    (collision_dir / "same.jpg").write_bytes(b"jpg")
    with pytest.raises(ValueError, match="unique filenames"):
        project.open_folder(str(collision_dir))
    assert project.image_dir == source.parent
    assert project.image_list == [str(source)]


def test_class_folder_cannot_escape_gt_root(dataset):
    source, project, labels, saver = dataset
    labels.items[str(source)] = [mask_label(np.ones((20, 30), np.uint8), "../escape")]
    with pytest.raises(ValueError, match="folder name"):
        saver.save_image_labels(str(source), {0: "../escape"}, (30, 20))
    assert not Path(project.get_label_path(str(source))).exists()


def test_sparse_yolo_ids_preserved_on_first_load(dataset):
    source, project, _, saver = dataset
    Path(project.get_label_path(str(source))).write_text("5 0.5 0.5 0.2 0.2\n")
    registered = []

    def register(name):
        registered.append(name)
        return len(registered) - 1

    result = saver.load_labels_from_disk(str(source), (30, 20), {}, register)
    assert registered == [f"class_{index}" for index in range(6)]
    assert result[0].class_id == 5
    assert result[0].class_name == "class_5"


def test_yolo_coordinates_use_original_dimensions(tmp_path):
    label = LabelItem(class_id=0, class_name="defect", label_type="bbox",
                      points=[(100, 50), (500, 50), (500, 250), (100, 250)])
    output = tmp_path / "original.txt"
    ExportManager.save_yolo_txt([label], 2000, 1000, str(output))
    assert output.read_text().strip() == "0 0.150000 0.150000 0.200000 0.200000"
    restored = ExportManager.load_yolo_txt(str(output), 2000, 1000, {0: "defect"})
    np.testing.assert_allclose(restored[0].points, label.points)


@pytest.mark.parametrize("line", ["0 0.1 0.2 nan 0.2", "0 0.1 0.2 0.3 0.4 0.5", "-1 0.5 0.5 0.2 0.2"])
def test_malformed_yolo_is_reported_instead_of_silently_dropped(tmp_path, line):
    output = tmp_path / "bad.txt"
    output.write_text(line)
    with pytest.raises(ValueError, match="bad.txt:1"):
        ExportManager.load_yolo_txt(str(output), 2000, 1000, {0: "defect"})


def test_semantic_mask_supports_more_than_255_classes(tmp_path):
    label = mask_label(np.ones((20, 30), np.uint8))
    label.class_id = 300
    output = tmp_path / "semantic.png"
    ExportManager.save_semantic_mask([label], 30, 20, str(output))
    mask = read_image(output, cv2.IMREAD_UNCHANGED)
    assert mask.dtype == np.uint16
    assert np.all(mask == 301)


def test_external_one_valued_tiff_mask_preserves_foreground(dataset):
    source, _, _, _ = dataset
    mask = np.zeros((20, 30), np.uint16)
    mask[2:5, 4:7] = 1
    gt_root = source.parent / "gt_image"
    atomic_write_image(gt_root / "defect" / f"{source.stem}.tif", mask)
    result = ExportManager.load_gt_masks(str(gt_root), str(source), 30, 20, {0: "defect"})
    assert len(result) == 1
    np.testing.assert_array_equal(result[0].mask_data, (mask > 0).astype(np.uint8) * 255)
