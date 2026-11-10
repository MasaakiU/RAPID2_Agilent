"""The MCP server of a real RAPID2 window, reached over HTTP the way the relay reaches it."""

import os
import threading
import time
from dataclasses import replace

import pytest
from PyQt6.QtCore import QCoreApplication, QEventLoop
from PyQt6.QtWidgets import QApplication, QDialog

from RAPID2_Agilent.mcp import catalog as catalog_file
from RAPID2_Agilent.mcp import endpoint as endpoint_file
from RAPID2_Agilent.mcp import registry
from RAPID2_Agilent.mcp.controller import ACCESS_REVOKED_MESSAGE, _AccessSession
from RAPID2_Agilent.mcp.protocol import ToolError
from RAPID2_Agilent.mcp.relay.upstream import HttpUpstream
from RAPID2_Agilent.mcp.tools import ViewerContext
from RAPID2_Agilent.models.main_model import MainModel
from RAPID2_Agilent.models.target import MSLevel
from RAPID2_Agilent.presenters.main_presenter import MainPresenter
from RAPID2_Agilent.views.main_window import MainWindow
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"
MSMS_DATA_PATH = DATA_DIR / "MSMS_data" / "9_PRJ_0-1-2_PA34-1_1uM_n1.d"


def pump_until(predicate, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    return predicate()


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    view = MainWindow()
    presenter = MainPresenter(view, MainModel())
    yield app, view, presenter
    presenter.shutdown()
    view.close()
    view.deleteLater()
    app.processEvents()


@pytest.fixture
def serving(window):
    app, view, presenter = window
    controller = presenter.mcp_controller
    assert controller.start(), controller.last_error
    return app, view, presenter, controller


def tool_call(name, request_id=1, **arguments):
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


def send_while_pumping(message):
    """POST from a worker thread, as the relay does, while the main thread runs events."""

    upstream = HttpUpstream(first_contact_wait=0.0)
    outcome = {}

    def worker():
        try:
            outcome["reply"] = upstream.send(message)
        except BaseException as exc:  # noqa: BLE001 - reported to the test
            outcome["error"] = exc

    thread = threading.Thread(target=worker)
    thread.start()
    assert pump_until(lambda: not thread.is_alive()), "no reply"
    thread.join(2)
    if "error" in outcome:
        raise outcome["error"]
    return outcome["reply"]["result"]


def test_a_first_start_turns_access_on_and_a_choice_to_turn_it_off_is_kept(window):
    # On until the user turns it off: the first start writes the launcher a plugin runs, with nothing to switch on.
    _, view, presenter = window
    settings = presenter.model.settings_model
    settings.settings.remove("mcp_server_enabled")
    presenter.apply_ai_assistant_setting()
    assert presenter.mcp_controller.running and view.ai_assistant_access_action.isChecked()
    assert os.access(presenter.mcp_controller.launcher_path, os.X_OK)
    presenter.set_ai_assistant_access(False)
    assert settings.mcp_server_enabled is False
    presenter.apply_ai_assistant_setting()
    assert not presenter.mcp_controller.running
    settings.settings.remove("mcp_server_enabled")


def test_starting_publishes_the_address_and_shows_the_letter(serving):
    _, view, _, controller = serving
    document = endpoint_file.read_endpoint(registry.document_path(controller.label))
    assert document["url"] == controller.url and document["token"] == controller.token
    assert view.windowTitle() == f"RAPID2 Agilent — {controller.label}"
    assert view.ai_assistant_access_action.isChecked()
    assert controller.client_command().startswith("claude mcp add rapid2-agilent -- ")
    assert controller.client_command("codex").startswith("codex mcp add rapid2-agilent -- ")
    assert os.access(controller.launcher_path, os.X_OK)
    catalog = catalog_file.read_catalog()
    names = [tool["name"] for tool in catalog["tools"]]
    assert "rapid2_get_status" in names
    # the relay runs these itself while no window is listening; a viewer tool needs one
    analysis = catalog_file.analysis_tool_names(catalog)
    assert "rapid2_extract_xic" in analysis and "rapid2_get_status" not in analysis


def test_a_viewer_tool_reaches_the_window(serving):
    _, _, _, controller = serving
    result = send_while_pumping(tool_call("rapid2_get_status"))
    assert not result["isError"]
    assert result["structuredContent"]["files"] == []
    assert result["structuredContent"]["instance"] == controller.label


def test_files_opened_and_windows_set_through_mcp_show_in_the_viewer(serving):
    _, _, presenter, _ = serving
    path = str(require(DATA_PATH))
    opened = send_while_pumping(tool_call("rapid2_open_files", paths=[path]))["structuredContent"]
    assert opened["opened"] == [path]
    assert [str(model.file_path) for model in presenter.data_stack_presenter.models] == [path]

    shown = send_while_pumping(tool_call("rapid2_set_view", rt_min=1.8, rt_max=1.95, mz_min=150.0, mz_max=151.0))
    assert not shown["isError"]
    assert presenter.data_stack_presenter.current_rt_window == (1.8, 1.95)
    assert presenter.data_stack_presenter.current_mz_window == (150.0, 151.0)

    captured = send_while_pumping(tool_call("rapid2_capture", part="plots"))
    images = [block for block in captured["content"] if block["type"] == "image"]
    assert len(images) == 1 and images[0]["mimeType"] == "image/png"
    assert "_images" not in captured["structuredContent"]


def test_set_view_shows_the_ms2_scans_of_a_precursor_in_the_plots_and_the_panel(window):
    app, view, presenter = window
    presenter.open_d_folders_clicked([str(require(MSMS_DATA_PATH))])
    panel = view.target_panel
    # the file's scans are negative ions
    panel.adduct_edit.setText("[M-H]-")
    panel.adduct_edit.setModified(True)
    panel.adduct_edit.editingFinished.emit()
    target_presenter = presenter.target_panel_presenter
    target = target_presenter.add_target_from_view()
    context = ViewerContext(presenter)
    status = context.status()
    assert (status["n_ms2_precursors"], status["ms2"]) == (5, None)

    with pytest.raises(ToolError, match="Give collision_energy"):
        context.set_view(precursor_mz=673.48)
    with pytest.raises(ToolError, match="MS level 1, 2"):
        context.set_view(ms_level=3)
    assert target_presenter.current_id == target.id  # a refused call changes nothing
    shown = context.set_view(precursor_mz=673.48, collision_energy=20, mz_window=(415.2, 415.3))

    assert (shown["ms_level"], shown["ms2"]["collision_energy"]) == (2, 20.0)
    child = presenter.data_stack_presenter.child_presenters[0]
    assert len(child.view.chromatogram_window.plot_items[0].xData) == 205
    # as a click on the m/z x RT map: the target is left as it was, and the panel's draft shows the view
    assert target_presenter.current_id is None
    assert presenter.model.target_list_model.get(target.id) == target
    assert (panel.ms_level(), panel.precursor_button.text()) == (MSLevel.MS2, "673.4814, CE 20")
    assert panel.mz_values()[:2] == pytest.approx((415.25, 0.05))

    context.set_view(ms_level=1)
    assert (panel.ms_level(), presenter.data_stack_presenter.ms_level) == (MSLevel.MS1, 1)
    context.set_view(ms_level=2)  # the draft's precursor and CE again
    assert panel.precursor_button.text() == "673.4814, CE 20"


def test_an_analysis_tool_does_not_wait_for_the_event_loop(serving):
    # Nothing pumps events here: a call marshalled onto the main thread would time out.
    reply = HttpUpstream(first_contact_wait=0.0).send(tool_call("rapid2_calculate_ion_mz", formula="C6H12O6", adduct="[M-H]-"))
    assert not reply["result"]["isError"]


def test_stopping_withdraws_the_address_and_keeps_the_letter(serving):
    _, view, _, controller = serving
    label = controller.label
    controller.stop()
    assert not registry.document_path(label).exists()
    assert view.windowTitle() == "RAPID2 Agilent"
    assert not view.ai_assistant_access_action.isChecked()
    assert controller.start()
    assert controller.label == label


def test_a_call_queued_before_access_went_off_is_refused(serving):
    _, _, _, controller = serving
    revoked = _AccessSession()
    revoked.revoke()
    ran = []
    with pytest.raises(ToolError, match="turned off"):
        controller._run_on_main_thread(lambda: ran.append(True), revoked)
    with pytest.raises(ToolError) as caught:
        controller._guarded(lambda: ran.append(True), revoked)
    assert str(caught.value) == ACCESS_REVOKED_MESSAGE
    assert ran == []


def test_a_modal_dialog_turns_viewer_tools_away(serving):
    _, view, _, controller = serving
    dialog = QDialog(view)
    dialog.setModal(True)
    dialog.show()
    try:
        if QApplication.activeModalWidget() is None:
            pytest.skip("this platform plugin does not activate modal dialogs")
        with pytest.raises(ToolError, match="dialog"):
            controller._guarded(lambda: None, None)
    finally:
        dialog.close()


def test_releasing_gives_the_letter_back(window):
    _, _, presenter = window
    controller = presenter.mcp_controller
    assert controller.start()
    label = controller.label
    controller.release()
    claim = registry.claim_label()
    try:
        assert claim.label == label
    finally:
        claim.release()


def test_targets_are_added_shown_imported_and_taken_for_export_in_the_window(window, tmp_path):
    from RAPID2_Agilent.services import rpd_csv

    _app, view, presenter = window
    path = str(require(DATA_PATH))
    presenter.open_d_folders_clicked([path])
    context = ViewerContext(presenter)
    targets = presenter.model.target_list_model
    target_presenter = presenter.target_panel_presenter

    added = context.add_target({"formula": "C6H12O6", "adduct": "[M-H]-", "rt": 1.0, "rt_tolerance": 0.5})["added"]
    assert added["index"] == 1 and added["mz"] == pytest.approx(added["calculated_mz"])
    assert target_presenter.current_id == targets.targets[0].id
    assert presenter.data_stack_presenter.current_rt_window == (0.5, 1.5)
    assert context.list_targets()["selected_target"] == context.status()["selected_target"] == 1
    with pytest.raises(ToolError, match="No target #2"):
        context.select_target(2)

    # a file of two targets, read in after the one there, and undone in one step
    file_path = tmp_path / "two.rpd.csv"
    rpd_csv.write_rpd_csv(file_path, rpd_csv.export_rows([*targets.targets, replace(targets.targets[0], index=2, name="copy")], [])[0])
    with pytest.raises(ToolError, match="give mode"):
        context.import_targets(str(file_path), None)
    imported = context.import_targets(str(file_path), "append")
    assert (imported["imported"], imported["targets"]) == (2, 3)
    assert [target.index for target in targets.targets] == [1, 2, 3] and target_presenter.current_id is None
    targets.undo()
    assert len(targets.targets) == 1
    assert context.import_targets(str(file_path), "replace")["replaced"]
    assert [target.name for target in targets.targets] == ["C6H12O6", "copy"]
    assert context.select_target(2)["selected"]["name"] == "copy"

    snapshot = context.target_snapshot()
    assert snapshot["paths"] == [path] and snapshot["targets"] == targets.targets and snapshot["left_out"] == []


def test_a_rpd_file_is_opened_through_mcp_of_the_spectra_it_holds(window, tmp_path):
    from RAPID2_Agilent.io.rpd_file import write_rpd
    from RAPID2_Agilent.models.rpd_model import RPDFileModel
    from tests.test_rpd_file import _data

    _app, _view, presenter = window
    context = ViewerContext(presenter)
    write_rpd(tmp_path / "x.rpd", _data("discrete"))
    write_rpd(tmp_path / "y.rpd", _data("continuous"))
    status = context.open_files([str(tmp_path / "x.rpd")])
    assert [(file["name"], file["spectra"]) for file in status["files"]] == [("x.rpd", "centroid")]
    assert [type(model) for model in presenter.data_stack_presenter.models] == [RPDFileModel]
    # an export reads it too, rather than leaving it out
    snapshot = context.target_snapshot()
    assert len(snapshot["paths"]) == 1 and snapshot["spectra"] == ["centroid"] and snapshot["left_out"] == []
    with pytest.raises(ToolError, match="y.rpd has no centroid spectra"):
        context.open_files([str(tmp_path / "y.rpd")], "centroid")
    with pytest.raises(ToolError, match="No .rpd file"):
        context.open_files([str(tmp_path / "missing.rpd")])


def test_files_are_opened_through_mcp_of_the_spectra_asked_for(window, tmp_path):
    from RAPID2_Agilent.models.rpd_model import AgilentCentroidModel, AgilentDModel

    _app, _view, presenter = window
    context = ViewerContext(presenter)
    path, copy = str(require(DATA_PATH)), str(require(DATA_DIR / "dummy_RP_2min_neg_copy.d"))
    assert context.open_files([path], "centroid")["opened"] == [path]
    status = context.open_files([copy])
    assert [(file["name"], file["spectra"]) for file in status["files"]] == [
        ("dummy_RP_2min_neg.d", "centroid"), ("dummy_RP_2min_neg_copy.d", "profile"),
    ]
    assert [type(model) for model in presenter.data_stack_presenter.models] == [AgilentCentroidModel, AgilentDModel]
    # an export reads each as the window has it
    assert context.target_snapshot()["spectra"] == ["centroid", "profile"]

    # of a run without centroid spectra, none is opened, and no dialog stops the call
    (tmp_path / "x.d" / "AcqData").mkdir(parents=True)
    (tmp_path / "x.d" / "AcqData" / "MSProfile.bin").write_bytes(b"x")
    with pytest.raises(ToolError, match="x.d has no centroid spectra"):
        context.open_files([str(tmp_path / "x.d")], "centroid")
    # one that cannot be read is told of, not shown in a dialog
    with pytest.raises(ToolError, match="x.d could not be opened"):
        context.open_files([str(tmp_path / "x.d")])
    assert len(presenter.data_stack_presenter.models) == 2
