"""The jobs and the option table every alignment front end shares."""

import sys

import pytest

from RAPID2_Agilent.alignment import AlignmentConfig, with_options
from RAPID2_Agilent.alignment.__main__ import _configuration, build_parser
from RAPID2_Agilent.alignment.runner import RESULT_FILES, check_output_dir, save_figures


def test_options_set_their_fields_and_none_leaves_them():
    config = with_options(AlignmentConfig(), ppm=20.0, scan_stride=3, anchor_stride=5, scale=None)
    assert config.binning.ppm == 20.0
    assert config.selection.stride == 3 and config.anchors.stride == 5
    assert config.transform.scale is None


def test_an_unknown_option_is_refused():
    with pytest.raises(ValueError, match="Unknown alignment option"):
        with_options(AlignmentConfig(), stride=3)


@pytest.mark.parametrize("command", [["align", "a.d", "b.d", "--output", "out"], ["synthetic", "--output", "out"], ["scales", "a.d"]])
def test_every_cli_command_has_a_flag_for_every_option(command):
    # _configuration reads one attribute per OPTIONS key, so a flag missing from
    # the parser fails here.
    args = build_parser().parse_args(command)
    assert _configuration(args) == AlignmentConfig()
    assert _configuration(build_parser().parse_args([*command, "--scan-stride", "4"])).selection.stride == 4


def test_an_existing_result_is_not_overwritten(tmp_path):
    assert check_output_dir(tmp_path) == tmp_path
    (tmp_path / RESULT_FILES[0]).write_text("{}")
    with pytest.raises(FileExistsError):
        check_output_dir(tmp_path)


def test_without_matplotlib_no_figures_are_drawn(tmp_path, monkeypatch):
    # The "diagnostics" extra, and the standalone build, which leaves it out.
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert save_figures(None, tmp_path) == []
    assert list(tmp_path.iterdir()) == []
