"""The progress of opening files: shown as they are read, by the scans of each profile, and stopped by Cancel."""

from PyQt6.QtWidgets import QProgressDialog

from tests.data_paths import require
from tests.test_target_panel import COPY_DATA_PATH, DATA_PATH, build_app


def _recording(monkeypatch, presenter) -> list:
    """What the presenter's progress dialogs are told: (label, value, maximum, shown then)."""
    told = []
    progress_of = presenter._progress

    def recorded(title, label):
        dialog, on_progress = progress_of(title, label)

        def on(what, done, total):
            on_progress(what, done, total)
            told.append((what, done, total, dialog.isVisible()))

        return dialog, on

    monkeypatch.setattr(presenter, "_progress", recorded)
    return told


def test_folders_are_read_with_their_progress_shown_by_file_and_scan(monkeypatch):
    _app, _view, _model, presenter = build_app(())
    told = _recording(monkeypatch, presenter)
    presenter.open_d_folders_clicked([str(require(DATA_PATH)), str(require(COPY_DATA_PATH))])
    assert [model.file_path for model in presenter.data_stack_presenter.models] == [DATA_PATH, COPY_DATA_PATH]
    assert told[0][:3] == ("Opening dummy_RP_2min_neg.d (1 of 2)", 0, 200)
    assert told[-1][:3] == ("Opening dummy_RP_2min_neg_copy.d (2 of 2)", 200, 200)
    values = [done for _what, done, _total, _shown in told]
    # along the scans of each, not only at its ends, and shown at once, closing itself at the end
    assert values == sorted(values) and len(set(values)) > 50
    assert [shown for *_rest, shown in told] == [True] * (len(told) - 1) + [False]


def test_a_cancel_keeps_the_files_read_and_reads_no_more(monkeypatch):
    _app, _view, _model, presenter = build_app(())
    _recording(monkeypatch, presenter)
    monkeypatch.setattr(QProgressDialog, "wasCanceled", lambda dialog: "(2 of 2)" in dialog.labelText())
    presenter.open_d_folders_clicked([str(require(DATA_PATH)), str(require(COPY_DATA_PATH))])
    assert [model.file_path for model in presenter.data_stack_presenter.models] == [DATA_PATH]
