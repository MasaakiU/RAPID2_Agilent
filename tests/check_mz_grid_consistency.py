"""Inspect every profile m/z array using the project's AgilentReader.

Datasets containing MS/MS scans (MSLevel > 1) are excluded by default.

Run from the project environment:
    python tests/check_mz_grid_consistency.py
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from RAPID2_Agilent.io.agilent_reader import AgilentReader


class GridSummary:
    def __init__(self):
        self.reference = None
        self.result = {
            "scans": 0,
            "exact_matches_to_reference_including_reference": 0,
            "max_abs_delta_mz_vs_reference_for_equal_lengths": 0.0,
            "max_abs_delta_ppm_vs_reference_for_equal_lengths": 0.0,
            "first_mismatch": None,
        }
        self.point_counts = Counter()
        self.digests = set()
        self.raw_grids = set()
        self.calibrations = set()
        self.calibration_ids = set()
        self.calibration_rows = []

    def add(self, scan_index, mz, raw_grid, calibration, calibration_id, digest):
        result = self.result
        result["scans"] += 1
        self.point_counts[len(mz)] += 1
        self.digests.add(digest)
        self.raw_grids.add(raw_grid)
        self.calibrations.add(tuple(calibration))
        self.calibration_ids.add(calibration_id)
        self.calibration_rows.append(calibration)
        if self.reference is None:
            self.reference = mz.copy()
            result["reference_scan_index_zero_based"] = scan_index
            result["reference_mz_endpoints"] = [float(mz[0]), float(mz[-1])]
        same = np.array_equal(mz, self.reference)
        result["exact_matches_to_reference_including_reference"] += int(same)
        if len(mz) == len(self.reference):
            delta = np.abs(mz - self.reference)
            max_index = int(np.argmax(delta))
            if delta[max_index] > result["max_abs_delta_mz_vs_reference_for_equal_lengths"]:
                result["max_abs_delta_mz_vs_reference_for_equal_lengths"] = float(delta[max_index])
                result["max_abs_delta_location"] = {
                    "scan_index_zero_based": scan_index,
                    "point_index_zero_based": max_index,
                    "reference_mz": float(self.reference[max_index]),
                    "scan_mz": float(mz[max_index]),
                }
            result["max_abs_delta_ppm_vs_reference_for_equal_lengths"] = max(
                result["max_abs_delta_ppm_vs_reference_for_equal_lengths"],
                float(np.max(delta / np.abs(self.reference)) * 1e6),
            )
        if not same and result["first_mismatch"] is None:
            example = {"scan_index_zero_based": scan_index, "point_count": len(mz)}
            if len(mz) == len(self.reference):
                index = int(np.argmin(np.abs(self.reference - 500.0)))
                example.update({
                    "unequal_points": int(np.count_nonzero(mz != self.reference)),
                    "example_point_index_near_mz_500": index,
                    "reference_mz": float(self.reference[index]),
                    "scan_mz": float(mz[index]),
                    "delta_mz_at_example": float(mz[index] - self.reference[index]),
                })
            result["first_mismatch"] = example

    def finish(self):
        result = self.result
        result["all_mz_arrays_exactly_equal"] = (
            result["exact_matches_to_reference_including_reference"] == result["scans"]
        )
        result["point_count_histogram"] = dict(sorted(self.point_counts.items()))
        result["unique_mz_arrays_sha256"] = len(self.digests)
        result["unique_raw_time_grids"] = len(self.raw_grids)
        result["raw_time_grids_start_delta_count"] = sorted(self.raw_grids)
        result["unique_calibration_rows"] = len(self.calibrations)
        result["calibration_ids"] = sorted(self.calibration_ids)
        names = ["coeff", "base", "left", "right", "a2", "b2", "c2", "d2", "e2", "f2"]
        rows = np.asarray(self.calibration_rows)
        result["varying_calibration_fields"] = [
            name for j, name in enumerate(names) if np.any(rows[:, j] != rows[0, j])
        ]
        return result


def inspect_file(path, include_msms=False):
    model = AgilentReader()
    model.load(path)
    if not model.N_scans:
        raise ValueError(f"No scans: {path}")
    info = model.ms_scan_info
    ms_levels = sorted({
        info.get_scan_info(scan_index, "MSLevel")
        for scan_index in range(model.N_scans)
    })
    if not include_msms and any(level > 1 for level in ms_levels):
        return {
            "path": str(path),
            "excluded": True,
            "reason": "Dataset contains MS/MS scans (MSLevel > 1); the entire .d dataset is excluded.",
            "ms_levels": ms_levels,
        }
    overall = GridSummary()
    groups = {}
    for scan_index in range(model.N_scans):
        offset = info.get_spectrum_info(scan_index, model.spectrum_param_index, "SpectrumOffset")
        start_t, delta_t, header_count, _ = model.ms_profile.decode_header(offset)
        mz, intensity = model.extract_spectrum(scan_index)
        point_count = info.get_spectrum_info(scan_index, model.spectrum_param_index, "PointCount")
        if not (len(mz) == len(intensity) == header_count == point_count):
            raise ValueError(f"Inconsistent point counts: {path}, scan {scan_index}")
        if not (np.all(np.isfinite(mz)) and np.all(mz > 0) and np.all(np.diff(mz) > 0)):
            raise ValueError(f"Invalid or non-increasing m/z array: {path}, scan {scan_index}")
        polarity = info.get_scan_info(scan_index, "IonPolarity")
        level = info.get_scan_info(scan_index, "MSLevel")
        label = f"{model._format_ion_polarity(polarity)} MS{level}"
        if label not in groups:
            groups[label] = GridSummary()
        arguments = (
            scan_index, mz, (start_t, delta_t, header_count),
            model.ms_calib_info.calib_info[scan_index],
            info.get_scan_info(scan_index, "CalibrationID"),
            hashlib.sha256(mz.tobytes()).hexdigest(),
        )
        overall.add(*arguments)
        groups[label].add(*arguments)
    return {
        "path": str(path),
        "overall": overall.finish(),
        "by_scan_type": {name: summary.finish() for name, summary in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", nargs="?", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("mz_grid_consistency_report.json"))
    parser.add_argument("--include-msms", action="store_true", help="Include datasets containing MS/MS scans.")
    args = parser.parse_args()
    paths = sorted(path for path in args.data.rglob("*.d") if path.is_dir())
    if not paths:
        parser.error(f"No .d directories in {args.data}")
    results = []
    excluded_files = []
    for path in paths:
        result = inspect_file(path, include_msms=args.include_msms)
        if result.get("excluded"):
            excluded_files.append(result)
            print(f"Excluded {path}: {result['reason']} MS levels: {result['ms_levels']}", flush=True)
            continue
        results.append(result)
        summary = result["overall"]
        print(f"{path.name}: {json.dumps(summary, ensure_ascii=False)}", flush=True)
    report = {
        "method": "AgilentReader.load + extract_spectrum, spectrum_param_index=0 (profile); all scans and all points, including zero-intensity bins; exact np.array_equal comparison to the first scan in each group. Delta/ppm statistics compare equal-length arrays by point index to that group's first scan; ppm denominator is its m/z.",
        "include_msms": args.include_msms,
        "excluded_files": excluded_files,
        "files": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
