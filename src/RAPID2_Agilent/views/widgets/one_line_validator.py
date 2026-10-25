from PyQt6.QtGui import QValidator

# a tab, a line feed, a carriage return: each made a space
_SPACES = {ord("\t"): " ", ord("\n"): " ", ord("\r"): " "}


class OneLineValidator(QValidator):
    """Text of one line and no tabs, as the .rpd.csv export writes it: a tab or line break typed or pasted becomes a
    space as it comes, rather than an error at the export. One for one, so that the cursor stays where it was."""

    def validate(self, text: str, pos: int):
        return QValidator.State.Acceptable, text.translate(_SPACES), pos
