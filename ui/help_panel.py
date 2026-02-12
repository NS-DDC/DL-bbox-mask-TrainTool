"""Help panel for VisionAce - dockable widget version."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt

from i18n import tr


class HelpPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(10)

        # --- Workflow ---
        content_layout.addWidget(self._section_title(tr("help_workflow_title")))
        workflow_steps = [
            "help_workflow_step1",
            "help_workflow_step2",
            "help_workflow_step3",
            "help_workflow_step4",
            "help_workflow_step4_bbox",
            "help_workflow_step4_seg",
            "help_workflow_step5",
            "help_workflow_step6",
            "",
            "help_workflow_result",
            "help_workflow_result_images",
            "help_workflow_result_labels",
            "help_workflow_result_gtimage",
        ]
        for key in workflow_steps:
            if key == "":
                content_layout.addSpacing(3)
            else:
                content_layout.addWidget(self._desc_label(tr(key)))

        content_layout.addSpacing(10)

        # --- Keyboard Shortcuts ---
        content_layout.addWidget(self._section_title(tr("help_shortcuts_title")))
        shortcuts = [
            "help_shortcut_open",
            "help_shortcut_save",
            "help_shortcut_undo",
            "help_shortcut_redo",
            "help_shortcut_delete",
            "help_shortcut_prev",
            "help_shortcut_next_save",
            "help_shortcut_next_no_save",
            "help_shortcut_exclude",
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

        content_layout.addSpacing(6)

        # --- Tools ---
        content_layout.addWidget(self._section_title(tr("help_tools_title")))
        tools = [
            "help_tool_select",
            "help_tool_bbox",
            "help_tool_segmentation",
        ]
        for key in tools:
            content_layout.addWidget(self._desc_label(tr(key)))

        content_layout.addSpacing(6)

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
            "help_tip_save_extension",
            "help_tip_recent_folders",
            "help_tip_exclude",
        ]
        for key in tips:
            content_layout.addWidget(self._tip_label(tr(key)))

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #5eb3f6; "
            "padding: 4px 0; border-bottom: 2px solid #5eb3f6;"
        )
        return label

    @staticmethod
    def _shortcut_row(text: str) -> QLabel:
        # Split on first " : "
        if " : " in text:
            key, desc = text.split(" : ", 1)
            html = (
                f'<span style="color:#ffd966; font-family:monospace; font-size:12px; font-weight:bold;">'
                f'{key}</span><br>'
                f'<span style="color:#e0e0e0; font-size:11px;">{desc}</span>'
            )
        else:
            html = f'<span style="color:#e8e8e8; font-size:12px;">{text}</span>'
        label = QLabel(html)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet("padding: 3px 6px;")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _desc_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #e8e8e8; padding: 3px 6px; font-size: 12px;")
        return label

    @staticmethod
    def _tip_label(text: str) -> QLabel:
        label = QLabel(f"• {text}")
        label.setWordWrap(True)
        label.setStyleSheet("color: #d0d0d0; padding: 3px 6px; font-size: 12px;")
        return label

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444; margin: 3px 4px;")
        return line

    def retranslate(self):
        """Re-create UI with updated translations."""
        # Clear all child widgets
        layout = self.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        # Rebuild UI
        self._setup_ui()
