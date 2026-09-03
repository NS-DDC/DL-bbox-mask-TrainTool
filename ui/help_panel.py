"""Help panel for VisionAce - dockable widget version."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QScrollArea, QFrame,
)
from PySide6.QtCore import Qt

from i18n import tr


class HelpPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._scroll = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        self._scroll = self._build_scroll_content()
        layout.addWidget(self._scroll)

    def _build_scroll_content(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background-color: #1e1e1e; }")

        content = QWidget()
        content.setStyleSheet("QWidget { background-color: #1e1e1e; }")
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
            "",
            "help_shortcut_class_number",
            "",
            "help_shortcut_brush_plus",
            "help_shortcut_brush_minus",
            "help_shortcut_zoom_in",
            "help_shortcut_zoom_out",
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
            "help_tip_mask_edit",
            "help_tip_import_labels",
            "help_tip_label_format",
            "help_tip_resume_work",
        ]
        for key in tips:
            content_layout.addWidget(self._tip_label(tr(key)))

        content_layout.addSpacing(10)

        # --- Data Formats & Import ---
        content_layout.addWidget(self._section_title(tr("help_formats_title")))

        format_items = [
            ("help_fmt_folder_title",  "help_fmt_folder_body"),
            ("help_fmt_yolo_title",    "help_fmt_yolo_body"),
            ("help_fmt_gtmask_title",  "help_fmt_gtmask_body"),
            ("help_fmt_import_title",  "help_fmt_import_body"),
            ("help_fmt_resume_title",  "help_fmt_resume_body"),
        ]
        for title_key, body_key in format_items:
            content_layout.addWidget(self._fmt_title_label(tr(title_key)))
            content_layout.addWidget(self._fmt_body_label(tr(body_key)))

        content_layout.addStretch()
        scroll.setWidget(content)
        return scroll

    @staticmethod
    def _section_title(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
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
                f'<span style="color:#ffffff; font-size:11px;">{desc}</span>'
            )
        else:
            html = f'<span style="color:#ffffff; font-size:12px;">{text}</span>'
        label = QLabel(html)
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setStyleSheet("padding: 3px 6px; background-color: #1e1e1e;")
        label.setWordWrap(True)
        return label

    @staticmethod
    def _desc_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("color: #ffffff; padding: 3px 6px; font-size: 12px; background-color: #1e1e1e;")
        return label

    @staticmethod
    def _tip_label(text: str) -> QLabel:
        label = QLabel(f"• {text}")
        label.setWordWrap(True)
        label.setStyleSheet("color: #f0f0f0; padding: 3px 6px; font-size: 12px; background-color: #1e1e1e;")
        return label

    @staticmethod
    def _fmt_title_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet(
            "color: #ffd966; font-size: 12px; font-weight: bold; "
            "padding: 6px 6px 2px 6px; background-color: #1e1e1e;"
        )
        return label

    @staticmethod
    def _fmt_body_label(text: str) -> QLabel:
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setStyleSheet(
            "color: #d0d0d0; font-size: 11px; font-family: monospace; "
            "padding: 2px 6px 8px 12px; background-color: #252525; "
            "border-left: 2px solid #444;"
        )
        return label

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #444; margin: 3px 4px;")
        return line

    def retranslate(self):
        """Re-create content with updated translations without recreating layout."""
        layout = self.layout()
        if layout and self._scroll:
            # Remove old scroll area
            layout.removeWidget(self._scroll)
            self._scroll.deleteLater()
            # Build new content
            self._scroll = self._build_scroll_content()
            layout.addWidget(self._scroll)
