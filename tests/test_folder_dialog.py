"""The macOS folder dialog (osascript's choose folder): opened where it can be, and its failures told of."""

import subprocess
from types import SimpleNamespace

from RAPID2_Agilent.presenters import main_presenter
from RAPID2_Agilent.presenters.main_presenter import MainPresenter


def test_a_folder_gone_opens_its_nearest_folder_and_a_failure_is_told_but_not_a_cancel(monkeypatch, tmp_path):
    runs, warnings = [], []
    stderr = [""]

    def run(command, **_kwargs):
        runs.append(command)
        return subprocess.CompletedProcess(command, 1, "", stderr[0])

    monkeypatch.setattr(main_presenter.subprocess, "run", run)
    monkeypatch.setattr(main_presenter.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    presenter = SimpleNamespace(view=None)

    def choose(start_dir):
        return MainPresenter._open_native_macos_folders(presenter, "Select", start_dir, allow_multiple=True)

    # a folder gone (a server unmounted): the dialog opens at its nearest folder that is
    stderr[0] = "execution error: User canceled. (-128)"
    assert choose(str(tmp_path / "gone" / "deeper")) == []
    assert runs[-1][-1] == str(tmp_path)
    # a cancel is not told of
    assert warnings == []
    # another failure is, rather than nothing happening
    stderr[0] = "execution error: Can't make file into type alias. (-1700)"
    assert choose(str(tmp_path)) == []
    assert len(warnings) == 1 and "(-1700)" in warnings[0]
