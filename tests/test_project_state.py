"""Project identity and undo regressions; run in GitHub Actions, not locally."""
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from core.label_manager import LabelItem, LabelManager
from core.project_metadata import load_classes, save_classes, validate_class_name


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


def test_project_classes_keep_order_and_unicode(tmp_path):
    classes = [{"name": "파티클", "color": "#123456"}, {"name": "scratch", "color": "#ffffff"}]
    save_classes(tmp_path, classes)
    assert load_classes(tmp_path) == classes


@pytest.mark.parametrize("name", ["../outside", "a/b", "a\\b", "NUL", "COM1.png", "", "bad:"])
def test_classes_cannot_escape_mask_folder(name):
    with pytest.raises(ValueError):
        validate_class_name(name)


def test_corrupt_class_metadata_is_not_silently_discarded(tmp_path):
    (tmp_path / ".visionace-project.json").write_text("broken", encoding="utf-8")
    with pytest.raises(ValueError):
        load_classes(tmp_path)


def test_replace_auto_labels_is_one_undo_and_retains_empty_loaded_state(app):
    manager = LabelManager()
    original = LabelItem(0, "scratch", "bbox", [(0., 0.), (4., 0.), (4., 4.), (0., 4.)])
    manager.set_labels("image", [original])
    manager.replace_labels("image", [])
    assert manager.is_image_loaded("image")
    assert manager.get_labels("image") == []
    manager.undo_stack.undo()
    assert manager.get_labels("image")[0].points == original.points
    manager.undo_stack.redo()
    assert manager.get_labels("image") == []


def test_remove_and_undo_mask_does_not_compare_numpy_arrays(app):
    manager = LabelManager()
    first = LabelItem(0, "mask", "mask", mask_data=np.zeros((3, 4), np.uint8))
    second = LabelItem(0, "mask", "mask", mask_data=np.ones((3, 4), np.uint8))
    manager.add_label("image", first)
    manager.add_label("image", second)
    manager.undo_stack.undo()
    assert len(manager.get_labels("image")) == 1
    assert manager.get_labels("image")[0] is first
    manager.clear()
    assert not manager.is_image_loaded("image")
    assert manager.undo_stack.count() == 0
