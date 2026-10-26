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
session, and reads a run again only once one of its files has changed.

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
