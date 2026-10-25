import math
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt, pyqtSignal
from PyQt6.QtGui import QFontDatabase, QPainter, QPalette, QTextOption
from PyQt6.QtWidgets import QFrame, QSizePolicy, QTextEdit, QToolTip, QVBoxLayout, QWidget

from RAPID2_Agilent.views.layout_metrics import SMALL, MEDIUM


class FileInfoPanel(QWidget):
    clicked = pyqtSignal(object)  # keyboard modifiers

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("FileInfoPanel")
        self._selected = False
        self._hovered = False
        self._file_path_text = ""

        self.file_name_edit = self._make_text_edit("FileNameTextEdit")
        self.scan_type_edit = self._make_text_edit("ScanTypeTextEdit")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(SMALL, SMALL, SMALL, SMALL)
        layout.setSpacing(MEDIUM)
        layout.addWidget(self.file_name_edit)
        layout.addWidget(self.scan_type_edit)
        layout.addStretch(1)

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
        text_edit.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        text_edit.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        text_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        # transparent without a style sheet: one would keep the text colour from following Light/Dark changes
        text_edit.viewport().setAutoFillBackground(False)
        text_edit.document().setDocumentMargin(0)
        return text_edit

    def is_selected(self) -> bool:
        return self._selected

    def set_selected(self, selected: bool) -> None:
        if selected == self._selected:
            return
        self._selected = selected
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(event.modifiers())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        event.accept()

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def event(self, event):
        # the text edits take no mouse events, so their tooltips are shown from here
        if event.type() == QEvent.Type.ToolTip:
            for text_edit in (self.file_name_edit, self.scan_type_edit):
                if text_edit.toolTip() and text_edit.geometry().contains(event.pos()):
                    QToolTip.showText(event.globalPos(), text_edit.toolTip(), self)
                    return True
            QToolTip.hideText()
            event.ignore()
            return True
        return super().event(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        if self._selected:
            color = self.palette().color(QPalette.ColorRole.Highlight)
            color.setAlpha(80)
            painter.fillRect(self.rect(), color)
        if self._hovered:
            # the text colour, faint: a darker grey in Light mode, a lighter one in Dark
            color = self.palette().color(QPalette.ColorRole.WindowText)
            color.setAlpha(25)
            painter.fillRect(self.rect(), color)

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
