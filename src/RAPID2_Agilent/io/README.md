# Agilent raw-data reader

`AgilentReader` owns the `.d` metadata, calibration and memory-mapped profile.
It has no Qt or application-model dependencies. `models.rpd_model.AgilentDModel`
provides the existing GUI API by delegating to one reader instance.

```python
from pathlib import Path
from RAPID2_Agilent.io import AgilentReader

with AgilentReader() as reader:
    reader.load(Path("run.d"))
    ms1_scans = reader.scan_indices_for_ms_level(1)
    scan_index = int(ms1_scans[0])
    rt = reader.RT_list[scan_index]
    mz, intensity = reader.extract_spectrum(scan_index)
```

The scan indices refer to the original acquisition order. `RT_list` keeps the
existing reader's `ScanTime` values without a unit conversion. For choosing MS2
scans, `precursor_mz_list`, `isolation_width_list` and `collision_energy_list`
hold each scan's `MzOfInterest`, `MzIsolationWidth` and `CollisionEnergy` (0 for
MS1 scans); `services.operations.precursors` lists them for the GUI. The reader also
supports XIC, TIC, BPC and averaged-spectrum extraction, with optional scan
selection. The reading and numerical algorithms are shared with the GUI.

Use `close()` or a context manager to release the mapped profile. `close()` clears
the loaded metadata; calling `load()` again first closes the previous run.

`MSCalibInfo.poly` lazily stores one read-only float64 polynomial correction
array per loaded file. At load time, the reader checks that all scans have the
same time grid, clipping limits, polynomial coefficients and flags. If they do,
single and averaged spectra share this array while applying each scan's own
`coeff` and `base`. If they do not, corrections are calculated per scan without
retaining a shared array. Reloading creates a new `MSCalibInfo` with `poly=None`.
The Python reference calibration and scalar XIC boundary search remain usable
without creating the full correction array.

Older acquisitions (e.g. of 2022) differ in three ways. Their MSScan.xsd is older:
one SpectrumParamValues (profile only), and no MzOfInterest and MzIsolationWidth,
which are then 0. Their calibration polynomials go up to t^11; orders above 7 are
evaluated with NumPy instead of the native `calculate_poly`. Their MSProfile.bin
records are LZF compressed, to start_t, delta_t and an int32 intensity per data
point. A record is taken as LZF when its header is not that of the token format
(`\x90` and its point count). At load, `MSProfile.rewrite_lzf_records` writes such
records again in memory in the token format (`_native/lzf_profile`), so all
decoders read them unchanged. A 15 min run takes about 1 s and 100 MB, 0.6 of the
LZF bytes.

`run_hashes` gives the two SHA-256 of a run that the .rpd.csv export writes:
`scan_sha256` of `AcqData/MSScan.bin`, quick, to look for a run by, and
`data_sha256` of its data files, to tell it is the very run
(docs/rpd_csv_format.md, section 3.3). `run_hashes()` keeps them for the
session, and reads a run again only once one of its files has changed. Of a
.rpd file it gives `""` and the SHA-256 the file holds, the data_hash RAPID2
exported.

## RAPID2 .rpd files

`RPDReader` reads the .rpd files of RAPID2 (versions 2.2 and 2.3) and answers
as `AgilentReader` does (`MSData`); `models.rpd_model.RPDFileModel` delegates
to it. A .rpd file holds MS1 scans of one setting, converted from mzData, a row
a scan padded to the longest: all of it is read into memory (about 1.5 GB and
4 s for an 18 min run of profile data). Of profile data ("continuous") the
extraction is that of `AgilentReader`, nearest-bin averaging included. Of
centroid data ("discrete") a scan with no peak in an m/z window has 0 there:
`mz_window_gaps` leaves no scan out. The extraction is `ScanPointsReader`'s,
shared with `AgilentCentroidReader`: the points of all the scans in one array,
each scan's from its start (of a .rpd file, its padded rows as they are).

`rpd_file` reads and writes the format itself (layout in its docstring). The
m/z and intensities are summed back from their differences in the order RAPID2
sums them, so the values are its own, read without a second copy of the
arrays. Older 2.2 files that did not keep where the scans end are cut where
their m/z first falls or is NaN (that oldest form lost the m/z beyond, as in
RAPID2). The pickles inside are read with only RAPID2's two classes and numpy's
arrays allowed.

`mzdata.convert` converts an mzData file (`x.mzdata.xml`) to `x.rpd` next to
it, as RAPID2 does: the same arrays, differences and sections, so that RAPID2
(numpy 1) reads the files too; their arrays are pickled as `numpy.array` of
their values, since numpy 2's pickles name `numpy._core`. The scans must all be
MS1 of one scan mode, as the format holds the settings once (the first scan's).
Their polarities may differ, as in a run of polarity segments: the polarity of
each is written beside RAPID2's attributes (`polarity_list`), which RAPID2
leaves unread, taking all the scans for the first's as with its own files
(RAPID2's own file of such a run says one polarity for all its scans, and is
read so). The XML is read a spectrum at a time; writing a 12 min run takes about
15 s, mostly compression.

## Centroid spectra of .d runs

`AgilentCentroidReader` reads the centroid spectra MassHunter stored at
acquisition (`AcqData/MSPeak.bin`), not the profile, and answers as the others
do (`MSData`); `models.rpd_model.AgilentCentroidModel` delegates to it. It is
for runs on a server: `AgilentReader.load` reads all of MSProfile.bin to index
its records, while MSPeak.bin is about 1/17 of it (50 MB against 906 MB for a
30 min run), read once whole; the run is then read from memory (12 bytes a
centroid). The older acquisitions (e.g. of 2022) have no MSPeak.bin.

A scan's centroid spectrum is the entry of its `SpectrumParamValues` of
`SpectrumFormatID` 2 (1 is the profile's): at its `SpectrumOffset`,
`PointCount` flight times (float64), then their intensities (float32). Their
m/z are of the scan's calibration, as the profile's (`MSCalibInfo.mz_at_times`);
a strong peak's is within about 1 ppm of the profile's apex, its intensity
about the profile's height. The first scan of a run has none. The scan records
(`read_scan_records`, `scan_lists`) are those `AgilentReader` reads, the TIC and
BPC theirs too, of the profile. An XIC sums the centroids' heights in its
window, not the profile's points: about 0.15 to 0.4 of the profile's XIC, so
the values of the two are not to be compared; the shapes of strong peaks
follow it. A scan holds at most the centroids the acquisition kept (6000 in
the runs seen), so weak ions may be missing.

The main window asks, of each .d folder chosen with both spectra, which to
read (with "Apply to all"); folders given are read of their profile, or of
their centroid spectra with `--centroid` (the command line) or `spectra:
centroid` (the AI assistant), and one of centroid spectra only of them. The
CLI and the MCP analysis tools take `spectra` too (`services.open_run`), all
but the alignment, which reads the profile. Every reader says which it reads
(`MSData.spectra`), and the answers of the tools say it of their values.

## Native extensions

The existing Cython/C++ sources and local binaries are now under `io/_native/`.
Compiled binaries are specific to the OS, CPU architecture and Python ABI;
rebuild them on a different execution environment. Runtime dependencies of this
reader are NumPy, lxml and these native extensions. Building from `.pyx` also
requires Cython and a C++ compiler compatible with the flags in those files.

Installing the project builds them in place (`setup.py` at the project root):

```sh
python -m pip install -e .
# After editing the .pyx/.cpp/.hpp sources, rebuild all modules in place:
python setup.py build_ext --inplace --force
```

Rebuild both `calibrate_mz.pyx` and `average_spectrum.pyx` after changes to their
shared polynomial-array interface. The native calibration now takes a required
`poly` array; `calculate_poly` builds it separately using the native arithmetic.
`setup.py` resolves the C++ sources named in the `.pyx` directives relative to
the native directory, so the build runs from the project root.

The GUI dependencies are the optional `gui` extra, so the reader can be installed
without Qt. A separate GPU environment's installation and build workflow remains
to be configured when that environment is selected; no GPU alignment
implementation is included here.
