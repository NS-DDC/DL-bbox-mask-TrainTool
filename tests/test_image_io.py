"""Run in GitHub Actions; do not execute on the user's workstation."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from core import image_io


def test_unicode_tiff_retains_original_bit_depth_and_dimensions(tmp_path):
    path = tmp_path / "원본 웨이퍼.tif"
    pixels = np.arange(19 * 31, dtype=np.uint16).reshape(19, 31) * 101
    image_io.atomic_write_image(path, pixels)
    restored = image_io.read_image(path, cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(restored, pixels)
    assert restored.dtype == np.uint16
    assert image_io.read_image(path).shape == (19, 31, 3)


def test_exif_orientation_is_not_applied(tmp_path):
    from PIL import Image

    path = tmp_path / "회전.jpg"
    pixels = np.zeros((13, 29, 3), dtype=np.uint8)
    pixels[:, :9] = [255, 0, 0]
    image = Image.fromarray(pixels)
    exif = Image.Exif()
    exif[274] = 6  # Camera display metadata asks for a 90-degree rotation.
    image.save(path, exif=exif)
    decoded = image_io.read_image(path)
    assert decoded.shape == (13, 29, 3)
    assert decoded[6, 3, 2] > 200  # Raw red region stays on the left.


def test_copy_is_exact_and_never_replaces_a_different_original(tmp_path):
    source = tmp_path / "원본.tiff"
    source.write_bytes(b"opaque-original-with-metadata\x00\xff")
    destination = tmp_path / "export" / source.name
    assert image_io.copy_original_image(source, destination)
    assert destination.read_bytes() == source.read_bytes()
    assert not image_io.copy_original_image(source, destination)
    assert not image_io.copy_original_image(source, source)
    destination.write_bytes(b"another-image")
    with pytest.raises(FileExistsError):
        image_io.copy_original_image(source, destination)
    assert destination.read_bytes() == b"another-image"


def test_atomic_write_keeps_previous_file_on_replace_failure(tmp_path, monkeypatch):
    destination = tmp_path / "labels.txt"
    destination.write_text("original labels", encoding="utf-8")

    def fail_replace(*_):
        raise OSError("simulated locked destination")

    monkeypatch.setattr(image_io.os, "replace", fail_replace)
    with pytest.raises(OSError, match="locked"):
        image_io.atomic_write_text(destination, "replacement")
    assert destination.read_text(encoding="utf-8") == "original labels"
    assert list(tmp_path.iterdir()) == [destination]


def test_unreadable_image_is_explicit_failure(tmp_path):
    path = tmp_path / "손상.png"
    path.write_bytes(b"not an image")
    assert image_io.read_image(path) is None


def test_canvas_keeps_source_coordinates_at_fit_and_actual_size(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QPoint, QPointF
    from ui.canvas_widget import CanvasWidget

    app = QApplication.instance() or QApplication([])
    path = tmp_path / "원본.tif"
    image_io.atomic_write_image(path, np.zeros((257, 1025, 3), dtype=np.uint8))
    canvas = CanvasWidget()
    canvas.resize(500, 350)
    canvas.show()
    app.processEvents()
    assert canvas.load_image(str(path))
    assert canvas.get_image_size() == (1025, 257)
    assert canvas._scene.sceneRect().width() == 1025
    canvas.fit_to_window()
    assert canvas._view.transform().m11() < 1
    canvas.actual_size()
    assert canvas._view.transform().m11() == 1
    assert canvas.get_image_size() == (1025, 257)
    corner = canvas._view.mapFromScene(QPointF(1025, 257))
    assert canvas._scene_pos(corner) == QPointF(1025, 257)
    assert not canvas.load_image(str(tmp_path / "missing.png"))
    assert canvas.get_image_size() == (1025, 257)
    canvas.clear_canvas()
    canvas.close()
