"""VisionAce - Deep Learning Labeling & Training Tool

Entry point for the application.
"""

import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from config import get_config
from i18n import set_language
from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VisionAce")
    app.setOrganizationName("VisionAce")

    # Apply dark style
    app.setStyle("Fusion")
    app.setStyleSheet(_DARK_STYLE)

    # Load language setting
    config = get_config()
    set_language(config.language)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


_DARK_STYLE = """
QMainWindow, QDialog {
    background-color: #2b2b2b;
    color: #cccccc;
}
QMenuBar {
    background-color: #333333;
    color: #cccccc;
}
QMenuBar::item:selected {
    background-color: #4a4a4a;
}
QMenu {
    background-color: #333333;
    color: #cccccc;
    border: 1px solid #555555;
}
QMenu::item:selected {
    background-color: #4a90d9;
}
QToolBar {
    background-color: #333333;
    border: none;
    spacing: 4px;
    padding: 2px;
}
QStatusBar {
    background-color: #333333;
    color: #aaaaaa;
}
QSplitter::handle {
    background-color: #444444;
}
QListWidget {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #444444;
    outline: none;
}
QListWidget::item:selected {
    background-color: #4a90d9;
    color: white;
}
QListWidget::item:hover {
    background-color: #3a3a3a;
}
QGroupBox {
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
}
QPushButton {
    background-color: #3a3a3a;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 5px 12px;
    min-height: 20px;
}
QPushButton:hover {
    background-color: #4a4a4a;
    border-color: #666666;
}
QPushButton:pressed {
    background-color: #2a2a2a;
}
QPushButton:disabled {
    color: #666666;
    background-color: #333333;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #555555;
    border-radius: 3px;
    padding: 4px;
}
QTextEdit {
    background-color: #1e1e1e;
    color: #cccccc;
    border: 1px solid #555555;
}
QProgressBar {
    border: 1px solid #555555;
    border-radius: 3px;
    text-align: center;
    color: white;
}
QProgressBar::chunk {
    background-color: #4a90d9;
}
QCheckBox, QRadioButton {
    color: #cccccc;
}
QLabel {
    color: #cccccc;
}
QGraphicsView {
    background-color: #1a1a1a;
    border: none;
}
"""


if __name__ == "__main__":
    main()
