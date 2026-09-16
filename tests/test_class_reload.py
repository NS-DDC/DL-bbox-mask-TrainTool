"""Class identity and real Qt save/reopen regressions, run on GitHub only."""

import json
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QPointF, Qt, QEvent
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QMessageBox

import config
from core.class_mapping import plan_model_classes
from core.export_manager import ExportManager
from core.image_io import atomic_write_image, read_image
from core.label_manager import LabelItem
from core.project_metadata import load_classes, save_classes
from ui.file_list_widget import FileListWidget
from ui.main_window import MainWindow
from ui.toolbar_widget import ToolMode


CLASSES = [{"name": name, "color": color} for name, color in
           [("파티클", "#ff0000"), ("scratch", "#00ff00"), ("defect", "#0000ff")]]


@pytest.fixture
def windows(monkeypatch):
    app = QApplication.instance() or QApplication([])
    cfg = config.AppConfig()
    cfg.save = lambda: None
    monkeypatch.setattr(config, "_config", cfg)
    monkeypatch.setattr(FileListWidget, "_start_thumbnail_loader", lambda *args: None)
    errors = []
    for method in ("warning", "critical"):
        monkeypatch.setattr(QMessageBox, method,
                            lambda _parent, title, message: errors.append((title, message)))
    created = []

    def open_window(directory):
        window = MainWindow()
        created.append(window)
        window._open_project(str(directory))
        assert window._current_image_path
        assert window._canvas.isEnabled()
        assert not errors
        return window

    yield open_window
    # Avoid a teardown auto-save concealing an explicit-save regression.
    cfg.auto_save = False
    for window in created:
        window.close()
        window.deleteLater()
    app.processEvents()
    assert not errors


@pytest.fixture
def dataset(tmp_path):
    for name in ("a.png", "b.png"):
        atomic_write_image(tmp_path / name, np.zeros((20, 30, 3), np.uint8))
    save_classes(tmp_path / "labels", CLASSES)
    return tmp_path


def box(class_id=2, name="defect"):
    return LabelItem(class_id, name, "bbox", [(2., 2.), (8., 2.), (8., 8.), (2., 8.)])


def add_disk_mask(directory, image="a", class_id=0):
    mask = np.zeros((20, 30), np.uint8)
    mask[3:9, 4:12] = 255
    path = directory / "gt_image" / CLASSES[class_id]["name"] / f"{image}.png"
    atomic_write_image(path, mask)
    return path, mask


@pytest.mark.parametrize("class_id", [0, 1, 2])
def test_exact_id_selected_in_widget_survives_save_and_fresh_window(windows, dataset, class_id):
    window = windows(dataset)
    window._label_list._class_id_spin.setValue(class_id)
    canvas = window._canvas
    assert canvas._current_class_id == class_id
    assert window._label_list._class_id_spin.text() == str(class_id)
    window._toolbar.set_mode(ToolMode.DETECTION)
    canvas._polygon_points = [QPointF(2, 2), QPointF(8, 2), QPointF(8, 8)]
    canvas._on_key_press(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Return,
                                  Qt.KeyboardModifier.NoModifier))
    assert window._save_current_labels()
    txt = Path(window._project.get_label_path(window._current_image_path))
    assert txt.read_text().split()[0] == str(class_id)

    reopened = windows(dataset)
    labels = reopened._labels.get_labels(reopened._current_image_path)
    assert [(item.class_id, item.class_name) for item in labels] == [(class_id, CLASSES[class_id]["name"])]
    assert f"[{class_id}] {CLASSES[class_id]['name']}" in reopened._label_list._instance_list.item(0).text()
    assert reopened._label_list.get_classes() == CLASSES


def test_sparse_disk_ids_register_without_changing_selected_drawing_class(windows, tmp_path):
    atomic_write_image(tmp_path / "a.png", np.zeros((20, 30, 3), np.uint8))
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    window = windows(tmp_path)
    assert window._labels.get_labels(window._current_image_path)[0].class_id == 2
    assert window._label_list.selected_class_id() == 0
    assert window._canvas._current_class_id == 0
    reopened = windows(tmp_path)
    assert [c["name"] for c in reopened._label_list.get_classes()] == ["class_0", "class_1", "class_2"]


def test_browsing_masks_preserves_selected_class_and_does_not_rewrite(windows, dataset):
    path, pixels = add_disk_mask(dataset, "a", 0)
    add_disk_mask(dataset, "b", 0)
    before = path.read_bytes(), path.stat().st_mtime_ns
    window = windows(dataset)
    window._toolbar.set_mode(ToolMode.SEGMENTATION)
    assert window._canvas.has_unfinished_mask()
    window._file_list.select_image(1)
    assert window._labels.undo_stack.count() == 0
    assert all(not window._labels.is_dirty(p) for p in window._labels.loaded_image_paths)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
    assert not (dataset / "labels" / "a.txt").exists()
    window._label_list.select_class(2)
    window._file_list.select_image(0)
    assert window._label_list.selected_class_id() == 2
    assert window._canvas._current_class_id == 2
    assert not window._canvas.has_unfinished_mask()
    window._canvas._current_mask[12:15, 20:24] = 255
    assert window._save_current_labels()
    reopened = windows(dataset)
    labels = reopened._labels.get_labels(reopened._current_image_path)
    assert {label.class_id for label in labels} == {0, 2}
    np.testing.assert_array_equal(next(l.mask_data for l in labels if l.class_id == 0), pixels)


def test_erasing_entire_existing_mask_saves_deletion_and_one_undo_restores(windows, dataset):
    mask_path, pixels = add_disk_mask(dataset)
    window = windows(dataset)
    window._toolbar.set_mode(ToolMode.SEGMENTATION)
    window._canvas._current_mask[:] = 0
    assert window._save_current_labels()
    assert not read_image(mask_path).any()
    assert window._labels.get_labels(window._current_image_path) == []
    window._on_undo()
    assert window._labels.undo_stack.index() == 0
    restored = window._labels.get_labels(window._current_image_path)
    assert len(restored) == 1
    np.testing.assert_array_equal(restored[0].mask_data, pixels)
    assert window._save_current_labels()
    reopened = windows(dataset)
    np.testing.assert_array_equal(reopened._labels.get_labels(reopened._current_image_path)[0].mask_data, pixels)


def test_mask_edit_is_one_undo_and_redo_keeps_original_class(windows, dataset):
    _, pixels = add_disk_mask(dataset, class_id=2)
    window = windows(dataset)
    window._on_edit_mask_requested(0)
    assert window._label_list.selected_class_id() == 2
    window._canvas._current_mask[15:18, 21:25] = 255
    edited = window._canvas._current_mask.copy()
    window._on_undo()
    assert window._labels.undo_stack.count() == 1
    np.testing.assert_array_equal(window._canvas._current_mask, pixels)
    window._on_redo()
    np.testing.assert_array_equal(window._canvas._current_mask, edited)
    assert window._canvas._current_class_id == 2


@pytest.mark.parametrize("mode", [ToolMode.DETECTION, ToolMode.SEGMENTATION])
def test_switch_class_finishes_pending_polygon_with_its_original_id(windows, dataset, mode):
    window = windows(dataset)
    window._toolbar.set_mode(mode)
    window._label_list.select_class(1)
    window._canvas._polygon_points = [QPointF(2, 2), QPointF(8, 2), QPointF(8, 8)]
    window._label_list.select_class(2)
    labels = window._labels.get_labels(window._current_image_path)
    assert [(item.class_id, item.class_name) for item in labels] == [(1, "scratch")]
    assert window._canvas._current_class_id == 2


def test_class_list_refresh_and_existing_registration_preserve_selection(windows, dataset):
    window = windows(dataset)
    panel = window._label_list
    panel.select_class(2)
    panel._refresh_class_list()
    panel.add_class("scratch", select=False)
    panel.add_class("another", select=False)
    assert panel.selected_class_id() == window._canvas._current_class_id == 2
    assert panel._class_list.item(2).text() == "[2] defect"


def test_visibility_is_reset_for_next_image_even_when_label_count_matches(windows, dataset):
    for name in ("a", "b"):
        (dataset / "labels" / f"{name}.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    window = windows(dataset)
    window._label_list._on_hide_all()
    assert window._label_list.get_visibility() == [False]
    window._file_list.select_image(1)
    assert window._label_list.get_visibility() == [True]
    assert window._canvas._label_items[0].graphics_item.isVisible()


def test_empty_class_list_invalidates_canvas_and_blocks_new_drawing(windows, dataset):
    window = windows(dataset)
    window._label_list.set_classes([])
    assert window._canvas._current_class_id == -1
    window._toolbar.set_mode(ToolMode.DETECTION)
    window._canvas._on_mouse_press(window._canvas._view.mapFromScene(QPointF(3, 3)))
    assert not window._canvas._drawing


def test_auto_label_first_detection_does_not_assign_its_order_as_class_id(windows, tmp_path, monkeypatch):
    atomic_write_image(tmp_path / "a.png", np.zeros((20, 30, 3), np.uint8))
    window = windows(tmp_path)
    monkeypatch.setattr(window._model, "get_class_names", lambda: {0: "파티클", 1: "scratch", 2: "defect"})
    window._apply_auto_labels(window._current_image_path, [box()], "append")
    assert window._labels.get_labels(window._current_image_path)[0].class_id == 2
    assert [cls["name"] for cls in window._label_list.get_classes()] == [cls["name"] for cls in CLASSES]
    assert window._save_current_labels()
    reopened = windows(tmp_path)
    assert reopened._labels.get_labels(reopened._current_image_path)[0].class_id == 2


def test_auto_label_keeps_existing_project_ids_and_selected_class(windows, dataset, monkeypatch):
    window = windows(dataset)
    window._label_list.select_class(1)
    monkeypatch.setattr(window._model, "get_class_names", lambda: {0: "defect", 1: "new class"})
    window._apply_auto_labels(window._current_image_path, [box(0)], "append")
    assert window._labels.get_labels(window._current_image_path)[0].class_id == 2
    assert window._canvas._current_class_id == 1
    assert window._label_list.get_class_name(3) == "new class"


def test_auto_label_rejects_inconsistent_result_without_changing_classes(windows, dataset, monkeypatch):
    window = windows(dataset)
    monkeypatch.setattr(window._model, "get_class_names", lambda: {0: "scratch"})
    with pytest.raises(ValueError, match="does not match"):
        window._apply_auto_labels(window._current_image_path, [box(0)], "append")
    assert window._labels.get_labels(window._current_image_path) == []
    assert window._label_list.get_classes() == CLASSES


def test_explicit_metadata_ids_override_record_order_and_are_validated(tmp_path):
    save_classes(tmp_path, CLASSES)
    path = tmp_path / ".visionace-project.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["classes"].reverse()
    path.write_text(json.dumps(data), encoding="utf-8")
    assert load_classes(tmp_path) == CLASSES
    data["classes"][0]["id"] = 0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="unique and contiguous"):
        load_classes(tmp_path)


def test_sparse_model_ids_and_mapping_are_independent_of_iteration_order():
    planned, mapping = plan_model_classes([], {2: "defect", 0: "scratch"})
    assert [cls["name"] for cls in planned] == ["scratch", "class_1", "defect"]
    assert mapping == {0: 0, 2: 2}


def test_legacy_named_annotation_cannot_override_project_id(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("defect 0 0.5 0.5 0.2 0.2\n")
    with pytest.raises(ValueError, match="conflicts with the project"):
        ExportManager.load_yolo_txt(str(path), 30, 20, {0: "scratch"})
