"""Help dialog for VisionAce."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QScrollArea, QWidget,
    QFrame,
)
from PySide6.QtCore import Qt

from i18n import tr


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("help_title"))
        self.setMinimumSize(480, 520)
        self.resize(520, 600)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Set dialog background
        self.setStyleSheet("QDialog { background-color: #1e1e1e; }")

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #1e1e1e; border: none; }")

        content = QWidget()
        content.setStyleSheet("QWidget { background-color: #1e1e1e; }")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 16, 20, 16)
        content_layout.setSpacing(12)

        # --- Keyboard Shortcuts ---
        content_layout.addWidget(self._section_title(tr("help_shortcuts_title")))
        shortcuts = [
            "help_shortcut_open",
            "help_shortcut_save",
            "help_shortcut_undo",
            "help_shortcut_redo",
            "help_shortcut_delete",
            "help_shortcut_skip",
            "help_shortcut_enter",
            "help_shortcut_esc",
            "help_shortcut_f1",
            "help_shortcut_quit",
            "",
            "help_shortcut_select",
            "help_shortcut_bbox",
            "help_shortcut_segmentation",
        ]
        for key in shortcuts:
            if key == "":
                content_layout.addWidget(self._separator())
            else:
                content_layout.addWidget(self._shortcut_row(tr(key)))

        content_layout.addSpacing(8)

        # --- Tools ---
        content_layout.addWidget(self._section_title(tr("help_tools_title")))
        tools = [
            "help_tool_select",
            "help_tool_bbox",
            "help_tool_segmentation",
        ]
        for key in tools:
            content_layout.addWidget(self._desc_label(tr(key)))

        content_layout.addSpacing(8)

        # --- Tips ---
        content_layout.addWidget(self._section_title(tr("help_tips_title")))
        tips = [
            "help_tip_zoom",
            "help_tip_pan",
            "help_tip_brush",
            "help_tip_bbox_mode",
            "help_tip_class_color",
            "help_tip_right_click",
            "help_tip_autosave",
        ]
        for key in tips:
            content_layout.addWidget(self._tip_label(tr(key)))

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        # Close button
        btn_container = QWidget()
        btn_container.setStyleSheet("background-color: #2b2b2b;")
        btn_layout = QVBoxLayout(btn_container)
        btn_layout.setContentsMargins(20, 8, 20, 12)
        close_btn = QPushButton(tr("help_close"))
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addWidget(btn_container)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #4a90d9; "
            "padding: 4px 0; border-bottom: 1px solid #444;"
        )
        return label

    @staticmethod
    def _shortcut_row(text: str) -> QLabel:
        # Split on first " : "
        if " : " in text:
            key, desc = text.split(" : ", 1)
            html = (
                f'<span style="color:#ffd966; font-family:monospace; font-size:12px; font-weight:bold;">'
                f'{key}</span>'
                f'<span style="color:#888;"> &mdash; </span>'
                f'<span style="color:#ffffff;">{desc}</span>'
            )
        else:
            html = f'<span style="color:#ffffff;">{text}</span>'
        label = QLabel(html)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet("padding: 2px 8px; background-color: #1e1e1e;")
        return label

    @staticmethod
    def _desc_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #ffffff; padding: 3px 8px; font-size: 12px; background-color: #1e1e1e;")
        return label

    @staticmethod
    def _tip_label(text: str) -> QLabel:
        label = QLabel(f"• {text}")
        label.setWordWrap(True)
        label.setStyleSheet("color: #f0f0f0; padding: 2px 8px; font-size: 12px; background-color: #1e1e1e;")
        return label

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444; margin: 4px 8px;")
        return line
