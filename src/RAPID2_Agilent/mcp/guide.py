# -*- coding: utf-8 -*-

"""What an assistant is told about RAPID2 before it calls anything. No Qt.

Sent in the handshake (`instructions`), by a window and by the relay alike,
so it is written once here. Kept short: it is a map of the tools, and the
details each tool needs are in its own description.
"""

from __future__ import annotations


SERVER_INSTRUCTIONS = """\
RAPID2 reads Agilent LC-MS .d folders directly, without vendor software.

Two kinds of tools:
- Analysis (rapid2_find_runs, rapid2_describe_run, rapid2_list_precursors, rapid2_extract_xic,
  rapid2_extract_trace, rapid2_extract_spectrum, rapid2_calculate_ion_mz, rapid2_quantify_targets,
  rapid2_map_spots, rapid2_compare_runs, rapid2_suggest_alignment_scales, rapid2_align_runs) read
  any .d folder by its path, open in the window or not, and leave the window as it is. They work
  while RAPID2 is closed too.
- Viewer (rapid2_get_status, rapid2_open_files, rapid2_set_view, rapid2_capture,
  rapid2_list_targets, rapid2_add_target, rapid2_select_target, rapid2_import_targets,
  rapid2_export_targets) read or change what the RAPID2 window shows the user, and its targets.

RTs are in minutes, except the alignment results, which are in seconds. Intensities are raw
profile counts. Long traces come back reduced to max_points points that keep the maxima.
An XIC takes an m/z, or a formula and an adduct; the adduct's charge chooses the scans of its
polarity, and a run of both polarities needs one (polarity). A scan whose data points miss the
m/z window is left out of an XIC as not measured (n_missing), as in the window's plots.
MS2 (MS/MS) scans are chosen by precursor and collision energy: take them from
rapid2_list_precursors, or for the window from rapid2_get_status, and pass precursor_mz and
collision_energy; the m/z of an XIC is then a product ion.
To compare compounds across runs, rapid2_quantify_targets gives the area, height and peak top
of each target in each run. For a first look at a run, rapid2_map_spots; for what differs
between two, rapid2_compare_runs.
For rapid2_align_runs, first call rapid2_suggest_alignment_scales and choose a scale.

If a viewer tool answers that RAPID2 is not reachable, call rapid2_launch. With several RAPID2
windows open, viewer tools need `instance`: the letter in the window title.
"""


def server_instructions():
    return SERVER_INSTRUCTIONS
