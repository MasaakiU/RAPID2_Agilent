from dataclasses import replace
from pathlib import Path

import pytest

from RAPID2_Agilent.io import run_hashes
from RAPID2_Agilent.models.run_peaks import Peak, RunPeaks, RunPeaksStore
from RAPID2_Agilent.models.target import MSLevel, Target, ToleranceUnit, TraceType
from RAPID2_Agilent.services import RunCache, peak_detection, rpd_csv
from RAPID2_Agilent.services.peak_detection import find_run_peaks
from RAPID2_Agilent.services.quantification import quantify_target
from tests.data_paths import DATA_DIR, require


DATA_PATH = DATA_DIR / "dummy_RP_2min_neg.d"  # negative ions only
# as docs/rpd_csv_format.md gives them
DATA_HASHES = (
    "e5081b949ae2413040809130c36e917e97576ed1ff7f0522cb4555d02a5cf7b4",
    "120744d611b758a1fe085d423e7568176b642d99ef9695ca09de2e51b1cc715c",
)


def _target(index, **fields):
    values = dict(
        index=index, name=f"target {index}", formula="", adduct="[M-H]-", trace_type=TraceType.XIC,
        ms_level=MSLevel.MS1, rt=1.0, rt_tolerance=0.5, mz=150.5, mz_tolerance=10.0,
        mz_tolerance_unit=ToleranceUnit.PPM,
    )
    return Target(**(values | fields))


@pytest.fixture
def run():
    cache = RunCache()
    path = require(DATA_PATH)
    yield rpd_csv.Run(cache.get(path), path)
    cache.close()


@pytest.mark.parametrize(
    ("given", "written"),
    [("a/x", "a/x.rpd.csv"), ("a/x.csv", "a/x.rpd.csv"), ("a/x.rpd.csv", "a/x.rpd.csv"), ("a/x.d", "a/x.d.rpd.csv")],
)
def test_the_file_name_ends_in_rpd_csv(given, written):
    assert rpd_csv.rpd_csv_path(given) == Path(written)


def test_the_hashes_are_those_of_the_spec_and_read_once(run, monkeypatch):
    monkeypatch.setattr(run_hashes, "_hashes", {})  # as if no test before had hashed the run
    read = []
    assert run_hashes.run_hashes(run.path, read.append) == DATA_HASHES
    assert run_hashes.is_known(run.path)
    assert run_hashes.run_hashes(run.path, read.append) == DATA_HASHES
    # all the data the first time, nothing the second
    assert sum(read) == run_hashes.data_size(run.path)
    assert run_hashes.data_sha256(run.path) == DATA_HASHES[1]


def test_a_row_for_each_target_and_run_with_its_columns(run, tmp_path):
    xic = _target(2)
    tic = _target(1, trace_type=TraceType.TIC, rt_tolerance=0.25, note="by hand")
    positive = _target(3, adduct="[M+H]+", mz_tolerance=0.01, mz_tolerance_unit=ToleranceUnit.MZ)

    rows, notes = rpd_csv.export_rows([xic, tic, positive], [run])

    assert [row["#"] for row in rows] == ["1", "2", "3"]
    tic_row, xic_row, positive_row = rows
    assert (xic_row["data_hash"], xic_row["scan_sha256"]) == (DATA_HASHES[1], DATA_HASHES[0])
    assert (xic_row["file_path"], xic_row["file_name"]) == (str(run.path), "dummy_RP_2min_neg.d")
    assert (xic_row["scan_settings"], xic_row["trace_type"], xic_row["MS_level"]) == ("ESI- Scan", "XIC", "MS1")
    # a ppm tolerance is a whole number; the windows in m/z and RT
    assert (xic_row["m/z"], xic_row["m/z_range"], xic_row["m/z_range_unit"]) == ("150.5", "10", "ppm")
    assert float(xic_row["m/z_btm"]) == pytest.approx(150.5 * (1 - 10e-6))
    assert (xic_row["RT_btm"], xic_row["RT_top"]) == ("0.5", "1.5")
    quantity = quantify_target(run.data, xic)
    assert float(xic_row["area"]) == quantity.area
    assert xic_row["N_scans (value used for calculation)"] == str(quantity.n_scans)
    assert (xic_row["precursor_m/z"], xic_row["collision_energy"], xic_row["note"]) == ("", "", "")
    # no m/z for TIC and BPC
    assert (tic_row["m/z"], tic_row["m/z_range"], tic_row["m/z_range_unit"], tic_row["m/z_btm"]) == ("", "", "", "")
    assert tic_row["note"] == "by hand"
    assert float(tic_row["area"]) > 0
    # no scans of the adduct's polarity: no values, and a note
    assert positive_row["m/z_range_unit"] == "abs"
    assert (positive_row["N_scans (value used for calculation)"], positive_row["area"], positive_row["scan_settings"]) == ("0", "", "")
    assert notes == ["#3 target 3, dummy_RP_2min_neg.d: 0 scans to use in the RT window, so no values"]

    path = tmp_path / "out.rpd.csv"
    rpd_csv.write_rpd_csv(path, rows)
    data = path.read_bytes()
    assert data.startswith(b"\xef\xbb\xbf") and b"\r" not in data
    lines = data.decode("utf-8-sig").split("\n")
    assert lines[0].split("\t") == list(rpd_csv.COLUMNS)
    assert lines[-1] == "" and len(lines) == 1 + 3 + 1
    assert all(line.count("\t") == len(rpd_csv.COLUMNS) - 1 for line in lines[:-1])


def test_the_values_are_in_the_windows_of_the_run_found_where_it_has_none(run, monkeypatch):
    # the scipy stand-in, which finds a peak of the TIC in the target's window
    monkeypatch.setattr(peak_detection, "DETECTOR", "scipy")
    in_peak, no_peak, found = _target(1), _target(2), _target(3, trace_type=TraceType.TIC)
    run_peaks = RunPeaksStore()
    run_peaks.set(in_peak.id, run.path, RunPeaks(in_peak.mz_window(), (0.8, 1.2)))
    run_peaks.set(no_peak.id, run.path, RunPeaks(no_peak.mz_window(), None, manual=True))

    rows, notes = rpd_csv.export_rows([in_peak, no_peak, found], [run], run_peaks=run_peaks)

    in_peak_row, no_peak_row, found_row = rows
    in_run = quantify_target(run.data, in_peak, run_peaks=run_peaks.get(in_peak.id, run.path))
    assert float(in_peak_row["area"]) == in_run.area != quantify_target(run.data, in_peak).area
    assert 0.8 <= float(in_peak_row["RT_btm (value used for calculation)"]) < float(
        in_peak_row["RT_top (value used for calculation)"]
    ) <= 1.2
    # the target's window, the one the peak is looked for in
    assert (in_peak_row["RT_btm"], in_peak_row["RT_top"]) == ("0.5", "1.5")
    assert (no_peak_row["N_scans (value used for calculation)"], no_peak_row["area"]) == ("0", "")
    assert notes == ["#2 target 2, dummy_RP_2min_neg.d: no peak taken for the target, so no values"]
    # the peaks of the TIC found, and kept; the values in the window of the one taken
    kept = run_peaks.get(found.id, run.path)
    assert kept == find_run_peaks(run.data, found) and kept.rt_window is not None
    assert float(found_row["area"]) == quantify_target(run.data, found, run_peaks=kept).area
    assert float(found_row["RT_btm (value used for calculation)"]) >= kept.rt_window[0]


def test_the_deconvolved_area_of_the_peak_taken_is_written_beside_the_windows_values(run):
    target = _target(1)
    run_peaks = RunPeaksStore()
    peak = Peak(0.8, 1.2, 1.0, area=12.5, area_sigma=0.5, area_interval=1.25)
    run_peaks.set(target.id, run.path, RunPeaks(target.mz_window(), (0.8, 1.2), (peak,)))

    (row,), _notes = rpd_csv.export_rows([target], [run], run_peaks=run_peaks)

    assert (row["area (deconvolved)"], row["area (deconvolved) sigma"], row["area (deconvolved) 95% interval"]) == (
        "12.5", "0.5", "1.25"
    )
    assert row["area"] != ""
    (row,), _notes = rpd_csv.export_rows([target], [run])  # of the target's window, of no peak found
    assert row["area (deconvolved)"] == "" and row["area"] != ""


def test_without_runs_a_row_for_each_target():
    rows, notes = rpd_csv.export_rows([_target(1), _target(2, trace_type=TraceType.BPC)], [])

    assert [row["#"] for row in rows] == ["1", "2"] and notes == []
    assert rows[0]["m/z"] == "150.5" and rows[0]["file_name"] == rows[0]["area"] == ""
    assert all(list(row) == list(rpd_csv.COLUMNS) for row in rows)


def test_numbers_have_all_their_digits_and_no_exponent():
    row = rpd_csv.export_rows([_target(1, rt=0.00001, rt_tolerance=3.0, mz=104.10699049531094)], [])[0][0]
    assert (row["RT"], row["RT_range"], row["m/z"]) == ("0.00001", "3", "104.10699049531094")


def test_the_progress_callback_can_stop_the_export(run):
    def cancel(what, done, total):
        raise rpd_csv.Cancelled

    with pytest.raises(rpd_csv.Cancelled):
        rpd_csv.export_rows([_target(1)], [run], on_progress=cancel)


def test_the_main_window_exports_the_targets_in_the_open_files(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from tests.test_target_panel import build_app, type_into

    app, view, model, presenter = build_app((DATA_PATH,))
    shown = []
    monkeypatch.setattr(QMessageBox, "information", lambda parent, title, text: shown.append(text))
    presenter.export_targets_clicked(str(tmp_path / "none"))
    assert shown == ["There are no targets to export."] and not (tmp_path / "none.rpd.csv").exists()

    type_into(view.target_panel, "adduct", "[M-H]-")
    presenter.target_panel_presenter.add_target_from_view()
    presenter.export_targets_clicked(str(tmp_path / "out"))

    lines = (tmp_path / "out.rpd.csv").read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 2
    row = dict(zip(rpd_csv.COLUMNS, lines[1].split("\t")))
    assert (row["file_name"], row["adduct"], row["data_hash"]) == ("dummy_RP_2min_neg.d", "[M-H]-", DATA_HASHES[1])


# -- import ------------------------------------------------------------------
def _exported(tmp_path, targets, name="targets.rpd.csv"):
    path = tmp_path / name
    rpd_csv.write_rpd_csv(path, rpd_csv.export_rows(targets, [])[0])
    return path


def _write(tmp_path, lines, name="targets.rpd.csv"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _same_target(imported, original):
    """Equal but for the id."""
    return imported == replace(original, id=imported.id)


def test_exported_targets_are_imported_as_they_were(tmp_path):
    targets = [
        _target(1, formula="C6H12O6", note="by hand"),
        _target(2, mz_tolerance=0.01, mz_tolerance_unit=ToleranceUnit.MZ, ms_level=MSLevel.ALL),
        _target(3, trace_type=TraceType.TIC, mz=rpd_csv.TIC_MZ, mz_tolerance=rpd_csv.TIC_MZ_TOLERANCE, mz_tolerance_unit=ToleranceUnit.MZ),
        _target(4, trace_type=TraceType.BPC, adduct="[M+H]+", mz=rpd_csv.TIC_MZ, mz_tolerance=rpd_csv.TIC_MZ_TOLERANCE, mz_tolerance_unit=ToleranceUnit.MZ),
        _target(6, ms_level=MSLevel.MS2, mz=89.0, precursor_mz=179.0561, collision_energy=20.0, isolation_width=1.3),
        _target(7, ms_level=MSLevel.MS2, mz=89.0),
    ]

    groups = rpd_csv.read_targets_file(_exported(tmp_path, targets))

    assert [group.number for group in groups] == [1, 2, 3, 4, 6, 7]
    assert all(len(group.variants) == 1 for group in groups)
    assert [group.lines for group in groups] == [((2,),), ((3,),), ((4,),), ((5,),), ((6,),), ((7,),)]
    assert all(_same_target(group.variants[0], target) for group, target in zip(groups, targets))
    # numbered again, with new ids
    imported = rpd_csv.resolve_groups(groups)
    assert [target.index for target in imported] == [1, 2, 3, 4, 5, 6]
    assert not {target.id for target in imported} & {target.id for target in targets}
    assert [target.index for target in rpd_csv.resolve_groups(groups, first_index=8)] == [8, 9, 10, 11, 12, 13]


def test_a_target_of_several_rows_is_one_and_rows_may_come_in_any_order(run, tmp_path):
    targets = [_target(2), _target(1, trace_type=TraceType.TIC)]
    rows, _notes = rpd_csv.export_rows(targets, [run, run])
    path = tmp_path / "out.rpd.csv"
    rpd_csv.write_rpd_csv(path, rows[::-1])
    # CRLF and no BOM read as well
    path.write_bytes(path.read_bytes().removeprefix(b"\xef\xbb\xbf").replace(b"\n", b"\r\n"))

    groups = rpd_csv.read_targets_file(path)

    assert [(group.number, group.lines) for group in groups] == [(1, ((4, 5),)), (2, ((2, 3),))]


def test_rows_of_one_number_holding_different_targets_are_variants(tmp_path):
    path = _exported(tmp_path, [_target(1), _target(2)])
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = lines[0].split("\t")
    changed = dict(zip(header, lines[1].split("\t"))) | {"RT": "1.5", "note": "moved"}
    again = dict(zip(header, lines[1].split("\t")))
    path = _write(tmp_path, [lines[0], lines[1], lines[2], "\t".join(changed[name] for name in header), "\t".join(again[name] for name in header)])

    first, second = rpd_csv.read_targets_file(path)

    assert first.number == 1 and first.lines == ((2, 5), (4,))
    assert [variant.rt for variant in first.variants] == [1.0, 1.5]
    assert len(second.variants) == 1
    # kept apart unless a variant is chosen, and numbered in order
    assert [(target.index, target.rt) for target in rpd_csv.resolve_groups([first, second])] == [(1, 1.0), (2, 1.5), (3, 1.0)]
    assert [(target.index, target.note) for target in rpd_csv.resolve_groups([second, first], {1: 1})] == [(1, "moved"), (2, "")]


def test_a_header_with_columns_wrong_is_refused_naming_each(tmp_path):
    header = [name for name in rpd_csv.COLUMNS if name not in ("RT", "note")] + ["adduct", "area2"]
    with pytest.raises(rpd_csv.TargetFileError) as raised:
        rpd_csv.read_targets_file(_write(tmp_path, ["\t".join(header)]))
    assert raised.value.problems == [
        "line 1, adduct: the column is there more than once",
        "line 1, area2: not a column of the format",
        "line 1, RT: the column is missing",
        "line 1, note: the column is missing",
    ]
    with pytest.raises(rpd_csv.TargetFileError, match="not a file of targets"):
        rpd_csv.read_targets_file(_write(tmp_path, ["name\tmz"]))


def test_every_problem_of_the_rows_is_listed_and_nothing_imported(tmp_path):
    path = _exported(tmp_path, [_target(1), _target(2), _target(3)])
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = lines[0].split("\t")

    def changed(line, **values):
        row = dict(zip(header, line.split("\t"))) | values
        return "\t".join(row[name] for name in header)

    lines[1] = changed(lines[1], RT="1,5", trace_type="xic")
    lines[2] = changed(lines[2], adduct="[M-H]")
    lines[3] = lines[3] + "\textra"
    with pytest.raises(rpd_csv.TargetFileError) as raised:
        rpd_csv.read_targets_file(_write(tmp_path, lines))
    assert raised.value.problems == [
        "line 2, trace_type: one of XIC, TIC, BPC, not 'xic'",
        "line 2, RT: not a number: '1,5'",
        "line 3: Invalid adduct: '[M-H]' (expected e.g. [M+H]+, [M-H]-, [M+2H]2+)",
        f"line 4: {len(header) + 1} fields, where the header has {len(header)}",
    ]


def test_a_file_of_rapid2_is_read_as_its_load_targets_read_it(tmp_path):
    header = ["", "compound_name", "formula", "is_TIC", "m/z", "m/z_range", "RT", "RT_range", "file_path", "RT_btm (value used for calculation)", "RT_btm (value used for calculation)", "scan_settings"]
    path = _write(tmp_path, [
        "\t".join(header),
        "0\tPC 30:0 [M+Cl]-\tC38H76NO8PCl-\tFalse\t740.4997\t0.01\t5.2\t0.2\ta.d\t5.0\t5.0\tESI- QuadrupoleTimeOfFlight SCAN",
        "1\tPC 30:0 [M+Cl]-\tC38H76NO8PCl-\tFalse\t740.4997\t0.01\t5.2\t0.2\tb.d\t5.0\t5.0\tESI- QuadrupoleTimeOfFlight SCAN",
        "2\tTIC\t\tTrue\t-1\t-1\t3.0\t3.0\ta.d\t0.0\t0.0\tESI- QuadrupoleTimeOfFlight SCAN",
        "3\tPC 32:0\tC40H81NO8PH+\tFalse\t734.5694\t0.01\t6.1\t0.2\ta.d\t5.9\t5.9\tESI+ QuadrupoleTimeOfFlight SCAN",
        "4\tglucose [M-H]-\tC6H11O6-\tFalse\t179.0561\t0.02\t1.0\t0.1\ta.d\t0.9\t0.9\tESI- QuadrupoleTimeOfFlight SCAN",
        # the name's adduct does not make the ion of the formula
        "5\tcholine [M+Na]+\tC5H14NO+\tFalse\t104.1070\t0.01\t3.4\t0.1\ta.d\t3.3\t3.3\tESI+ QuadrupoleTimeOfFlight SCAN",
    ])

    groups = rpd_csv.read_targets_file(path)

    assert [group.number for group in groups] == [1, 2, 3, 4, 5]
    assert groups[0].lines == ((2, 3),)
    pc, tic, pc32, glucose, choline = (group.variants[0] for group in groups)
    assert (pc.name, pc.formula, pc.adduct, pc.trace_type, pc.ms_level) == ("PC 30:0", "C38H76NO8P", "[M+Cl]-", TraceType.XIC, MSLevel.MS1)
    assert (pc.mz, pc.mz_tolerance, pc.mz_tolerance_unit, pc.rt, pc.rt_tolerance) == (740.4997, 0.01, ToleranceUnit.MZ, 5.2, 0.2)
    # no formula: the adduct of the scans' polarity; the m/z of RAPID2's -1 the plots' initial one
    assert (tic.trace_type, tic.formula, tic.adduct) == (TraceType.TIC, "", "[M-H]-")
    assert (tic.mz, tic.mz_tolerance) == (rpd_csv.TIC_MZ, rpd_csv.TIC_MZ_TOLERANCE)
    # no adduct in the name: the charge taken off, as [M]+
    assert (pc32.name, pc32.formula, pc32.adduct) == ("PC 32:0", "C40H82NO8P", "[M]+")
    assert (glucose.name, glucose.formula, glucose.adduct) == ("glucose", "C6H12O6", "[M-H]-")
    assert (choline.name, choline.formula, choline.adduct) == ("choline [M+Na]+", "C5H14NO", "[M]+")


@pytest.mark.parametrize(
    ("name", "formula", "scan_settings", "expected"),
    [
        ("x [2M+Na]+", "C12H24O12Na+", "", ("x", "C6H12O6", "[2M+Na]+")),
        ("x [M+FA-H]-", "C7H13O8-", "", ("x", "C6H12O6", "[M+FA-H]-")),
        ("x [M+2H]2+", "C6H14O62+", "", ("x [M+2H]2+", "C6H14O62", "[M]+")),
        ("x [M+2H]2+", "C6H14O6+2", "", ("x", "C6H12O6", "[M+2H]2+")),
        ("blank", "", "ESI+ Scan", ("blank", "", "[M+H]+")),
        ("blank", "", "IonMode 32- Scan", ("blank", "", "[M-H]-")),
        ("blank", "", "ESI+ Scan, ESI- Scan", ("blank", "", "[M+H]+")),
        ("blank [M+Na]+", "", "ESI- Scan", ("blank", "", "[M+Na]+")),
        ("salt", "NaCl", "", ("salt", "ClNa", "[M+H]+")),
    ],
)
def test_a_rapid2_ion_formula_is_made_neutral(name, formula, scan_settings, expected):
    assert rpd_csv._neutral(name, formula, scan_settings) == expected


def test_the_main_window_loads_targets_asking_what_to_do(tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox

    from tests.test_target_panel import build_app

    path = _exported(tmp_path, [_target(1), _target(2, trace_type=TraceType.TIC)])
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header = lines[0].split("\t")
    moved = dict(zip(header, lines[1].split("\t"))) | {"RT": "1.5"}
    path = _write(tmp_path, [*lines, "\t".join(moved[name] for name in header)])

    app, view, model, presenter = build_app((DATA_PATH,))
    asked, answers = [], []

    def answer(box):
        asked.append(box.text())
        text = answers.pop(0)
        next(button for button in box.buttons() if button.text() == text).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", answer)
    targets = model.target_list_model
    # into an empty list without asking to replace; the variants of #1 kept apart
    answers[:] = ["Keep as separate targets"]
    presenter.load_targets_clicked(str(path))
    assert [(target.index, target.rt, target.trace_type) for target in targets.targets] == [
        (1, 1.0, TraceType.XIC), (2, 1.5, TraceType.XIC), (3, 1.0, TraceType.TIC),
    ]
    assert "#1 hold 2 different targets, differing in rt" in asked[0]

    # appended, one of the variants used; undone at once
    asked.clear()
    answers[:] = ["Use values of line 4", "Append"]
    presenter.load_targets_clicked(str(path))
    assert len(asked) == 2
    assert [(target.index, target.rt) for target in targets.targets][3:] == [(4, 1.5), (5, 1.0)]
    targets.undo()
    assert len(targets.targets) == 3

    answers[:] = ["Use values of line 2", "Replace"]
    presenter.load_targets_clicked(str(path))
    assert [(target.index, target.rt) for target in targets.targets] == [(1, 1.0), (2, 1.0)]
    assert presenter.target_panel_presenter.current_id is None

    # cancelled: nothing changes
    answers[:] = ["Cancel"]
    presenter.load_targets_clicked(str(path))
    assert len(targets.targets) == 2

    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: warned.append(text))
    presenter.load_targets_clicked(str(_write(tmp_path, ["\t".join(header), "x" + lines[2]], name="bad.rpd.csv")))
    assert warned and "line 2, #: not a positive whole number" in warned[0]
    assert len(targets.targets) == 2
