# -*- coding: utf-8 -*-

"""What an assistant is told about RAPID2 before it calls anything. No Qt.

Sent in the handshake (`instructions`), by a window and by the relay alike,
so it is written once here. Kept short: it is a map of the tools, and the
details each tool needs are in its own description.
"""

from __future__ import annotations


SERVER_INSTRUCTIONS = """\
RAPID2 reads Agilent LC-MS .d folders directly, without vendor software, and RAPID2's own .rpd
files (MS1 scans, converted from mzData or by RAPID2), each a run.

Two kinds of tools:
- Analysis (rapid2_find_runs, rapid2_describe_run, rapid2_list_precursors, rapid2_extract_xic,
  rapid2_extract_trace, rapid2_extract_spectrum, rapid2_calculate_ion_mz,
  rapid2_calculate_isotope_pattern, rapid2_quantify_targets, rapid2_map_spots, rapid2_compare_runs,
  rapid2_suggest_alignment_scales, rapid2_align_runs) read
  any run by its path, open in the window or not, and leave the window as it is. They work
  while RAPID2 is closed too. The alignment reads .d folders only.
- Viewer (rapid2_get_status, rapid2_open_files, rapid2_set_view, rapid2_capture,
  rapid2_list_targets, rapid2_add_target, rapid2_select_target, rapid2_import_targets,
  rapid2_export_targets) read or change what the RAPID2 window shows the user, and its targets.

RTs are in minutes, except the alignment results, which are in seconds. Intensities are raw
profile counts. A run can be read of its centroid spectra instead (spectra: centroid), about 1/17
of the data and much quicker from a network drive; their intensities are peak heights, an XIC or
area about 0.15 to 0.4 of the profile's, so compare only values read of the same spectra.
A .rpd file holds one of the two and is read of it (rapid2_describe_run says which).
Choose by where a .d folder is: on a network drive or server (a share mounted under /Volumes on
macOS, a \\\\server\\share path or a mapped network drive on Windows), read its centroid spectra; on
a local disk, its profile. Read runs to be compared of one kind: centroid for all if any of them is
on a server, unless the user copies them to a local disk.
A run's centroid spectra are read whole, once, at the first call that names it (slow from a network
drive) and kept in memory with the last few runs used, so later calls on it are quick. Profile and
centroid are kept apart: stay with one kind for a run rather than alternating.
Long traces come back reduced to max_points points that keep the maxima.
An XIC takes an m/z, or a formula and an adduct; the adduct's charge chooses the scans of its
polarity, and a run of both polarities needs one (polarity). A scan whose data points miss the
m/z window is left out of an XIC as not measured (n_missing), as in the window's plots.
MS2 (MS/MS) scans are chosen by precursor and collision energy: take them from
rapid2_list_precursors, or for the window from rapid2_get_status, and pass precursor_mz and
collision_energy; the m/z of an XIC is then a product ion.
rapid2_find_runs gives each .d folder's sample name, type and acquisition time (order: acquired
for the injection order); rapid2_describe_run all its sample information.
To compare compounds across runs, rapid2_quantify_targets gives the area, height and peak top
of each target in each run, the values RAPID2's Export Targets writes; keep them in a .rpd.csv
file (rpd_csv) rather than calling it again. Its flagged rows are the ones to look at again,
of their XIC (rapid2_extract_xic), before trusting them. For a first look at a run,
rapid2_map_spots; for what differs between two, rapid2_compare_runs.
For rapid2_align_runs, first call rapid2_suggest_alignment_scales and choose a scale.

If a viewer tool answers that RAPID2 is not reachable, call rapid2_launch. With several RAPID2
windows open, viewer tools need `instance`: the letter in the window title.
"""


def server_instructions():
    return SERVER_INSTRUCTIONS
