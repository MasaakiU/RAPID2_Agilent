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
existing reader's `ScanTime` values without a unit conversion. The reader also
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

## Native extensions

The existing Cython/C++ sources and local binaries are now under `io/_native/`.
Compiled binaries are specific to the OS, CPU architecture and Python ABI;
rebuild them on a different execution environment. Runtime dependencies of this
reader are NumPy, lxml and these native extensions. Building from `.pyx` also
requires Cython and a C++ compiler compatible with the flags in those files.

Build from the project root with the project Python environment:

```sh
python src/RAPID2_Agilent/io/_native/build_extensions.py
# Or rebuild only the polynomial-related modules:
python src/RAPID2_Agilent/io/_native/build_extensions.py calibrate_mz average_spectrum
```

Rebuild both `calibrate_mz.pyx` and `average_spectrum.pyx` after changes to their
shared polynomial-array interface. The native calibration now takes a required
`poly` array; `calculate_poly` builds it separately using the native arithmetic.
The build helper keeps C++ source paths relative to the native directory even
when Cython recognizes the surrounding Python package.

The top-level project still declares the GUI dependencies, and installing it does
not currently build these native extensions automatically. A separate GPU
environment's installation and build workflow remains to be configured when that
environment is selected; no GPU alignment implementation is included here.
