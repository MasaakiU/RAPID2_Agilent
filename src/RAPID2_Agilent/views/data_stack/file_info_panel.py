import math
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFontDatabase, QTextOption
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QTextEdit, QVBoxLayout, QWidget

from RAPID2_Agilent.views.widgets import make_icon_button


class FileInfoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FileInfoPanel")
        self._file_path_text = ""

        self.file_name_edit = self._make_text_edit("FileNameTextEdit")
        self.scan_type_edit = self._make_text_edit("ScanTypeTextEdit")
        self.button_layout = self._make_button_layout()
        self.move_to_top_button = make_icon_button("angles-up-solid", "Move to top")
        self.move_up_button = make_icon_button("angle-up-solid", "Move up")
        self.move_down_button = make_icon_button("angle-down-solid", "Move down")
        self.move_to_bottom_button = make_icon_button("angles-down-solid", "Move to bottom")
        self.close_button = make_icon_button("xmark-solid", "Close")
        self.button_layout.addStretch(1)
        self.add_button(self.move_to_top_button)
        self.add_button(self.move_up_button)
        self.add_button(self.move_down_button)
        self.add_button(self.move_to_bottom_button)
        self.add_button(self.close_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(10)
        layout.addLayout(self.button_layout, 0)
        layout.addWidget(self.file_name_edit)
        layout.addWidget(self.scan_type_edit)
        layout.addStretch(1)

    @staticmethod
    def _make_button_layout() -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        return layout

    def add_button(self, button: QWidget) -> None:
        self.button_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignTop)

    @staticmethod
    def _make_text_edit(object_name: str) -> QTextEdit:
        text_edit = QTextEdit()
        text_edit.setObjectName(object_name)
        text_edit.setReadOnly(True)
        text_edit.setFrameShape(QFrame.Shape.NoFrame)
        text_edit.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        text_edit.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        text_edit.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        text_edit.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        text_edit.setStyleSheet("QTextEdit { background: transparent; }")
        text_edit.document().setDocumentMargin(0)
        return text_edit

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_text_edit_heights()

    def set_file_info(self, file_path, scan_type_summary: list[tuple[str, int]]):
        path = None if file_path is None else Path(file_path)
        self._file_path_text = "" if path is None else str(path)
        self.file_name_edit.setPlainText("" if path is None else path.name)
        self.scan_type_edit.setPlainText(
            "\n".join(f"{label} ({count})" for label, count in scan_type_summary)
        )
        self.setToolTip("")
        self.file_name_edit.setToolTip(self._file_path_text)
        self.scan_type_edit.setToolTip("")
        self._fit_text_edit_heights()

    def set_file_path(self, file_path):
        self.set_file_info(file_path, [])

    def toPlainText(self) -> str:
        return "\n".join(
            text
            for text in (
                self.file_name_edit.toPlainText(),
                self.scan_type_edit.toPlainText(),
            )
            if text
        )

    def _fit_text_edit_heights(self):
        for text_edit in (self.file_name_edit, self.scan_type_edit):
            text = text_edit.toPlainText()
            if not text:
                text_edit.setFixedHeight(0)
                continue
            text_width = self._available_text_width(text_edit)
            text_edit.document().setTextWidth(text_width)
            line_count = self._wrapped_line_count(text, text_width, text_edit)
            text_edit.setFixedHeight(line_count * text_edit.fontMetrics().lineSpacing())

    def _available_text_width(self, text_edit: QTextEdit) -> int:
        layout_margins = self.layout().contentsMargins()
        fallback_width = self.width() - layout_margins.left() - layout_margins.right()
        if fallback_width > 0:
            return fallback_width
        return max(1, text_edit.viewport().width())

    @staticmethod
    def _wrapped_line_count(text: str, text_width: int, text_edit: QTextEdit) -> int:
        font_metrics = text_edit.fontMetrics()
        line_count = 0
        for line in text.splitlines() or [""]:
            line_width = font_metrics.horizontalAdvance(line)
            line_count += max(1, math.ceil(line_width / text_width))
        return line_count
