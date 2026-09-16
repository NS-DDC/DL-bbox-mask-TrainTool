"""Canvas navigation regressions; executed on the GitHub Qt runner only."""

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QGraphicsScene

from ui.canvas_widget import _GraphicsView


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance() or QApplication([])
    yield instance


@pytest.fixture
def large_view(app):
    scene = QGraphicsScene()
    scene.setSceneRect(0, 0, 2000, 2000)
    view = _GraphicsView(scene)
    view.resize(320, 240)
    view.show()
    app.processEvents()
    vertical = view.verticalScrollBar()
    horizontal = view.horizontalScrollBar()
    vertical.setValue(vertical.maximum() // 2)
    horizontal.setValue(horizontal.maximum() // 2)
    yield view
    view.close()
    view.deleteLater()
    app.processEvents()


def _drag(view, button, start, end):
    QTest.mousePress(view.viewport(), button, Qt.KeyboardModifier.NoModifier, start)
    QTest.mouseMove(view.viewport(), end, 1)
    QTest.mouseRelease(view.viewport(), button, Qt.KeyboardModifier.NoModifier, end)


def test_right_drag_pans_when_selection_mode_enables_it(large_view):
    view = large_view
    forwarded = []
    view.mouse_pressed.connect(lambda *args: forwarded.append(args))
    before = view.verticalScrollBar().value()

    view.set_right_drag_pan_enabled(True)
    _drag(view, Qt.MouseButton.RightButton, QPoint(150, 150), QPoint(150, 90))

    assert view.verticalScrollBar().value() > before
    assert forwarded == []
    assert not view._panning


def test_right_button_keeps_drawing_behavior_when_pan_is_disabled(large_view):
    view = large_view
    forwarded = []
    view.mouse_pressed.connect(lambda _pos, button: forwarded.append(button))

    view.set_right_drag_pan_enabled(False)
    QTest.mouseClick(view.viewport(), Qt.MouseButton.RightButton)

    assert forwarded == [Qt.MouseButton.RightButton]


def test_space_left_drag_pans_in_drawing_modes(large_view):
    view = large_view
    forwarded = []
    view.mouse_pressed.connect(lambda *args: forwarded.append(args))
    before = view.verticalScrollBar().value()
    view.set_right_drag_pan_enabled(False)

    QTest.keyPress(view, Qt.Key.Key_Space)
    _drag(view, Qt.MouseButton.LeftButton, QPoint(150, 150), QPoint(150, 90))
    QTest.keyRelease(view, Qt.Key.Key_Space)

    assert view.verticalScrollBar().value() > before
    assert forwarded == []
    assert not view._space_pan


def test_space_double_click_does_not_finish_annotation(large_view):
    view = large_view
    finished = []
    view.mouse_double_clicked.connect(lambda *args: finished.append(args))
    view.set_right_drag_pan_enabled(False)
    QTest.keyPress(view, Qt.Key.Key_Space)
    QTest.mouseDClick(view.viewport(), Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier, QPoint(150, 100))
    QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton)
    QTest.keyRelease(view, Qt.Key.Key_Space)
    assert finished == []
