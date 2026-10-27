"""Write XICs of .d runs to label the peaks of (dataset.py): of the targets of a .rpd.csv file, of the strongest spots
of each run's m/z-RT map, and of m/z at random, each over the whole run.

    python tools/peak_labeling/extract_xics.py RUN.d [RUN.d ...] --out xics.npz [--targets targets.rpd.csv]
        [--spots 15] [--random 5] [--ppm 10] [--seed 0] [--keep-calibration]

Scans whose mass calibration jumps away from that of the scans about them are put back in line first
(calibration.py), unless --keep-calibration; those of each run are in the dataset's "runs".
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import repair_calibration_jumps  # noqa: E402
from dataset import Trace, write_dataset  # noqa: E402

from RAPID2_Agilent.models.chemistry import parse_adduct  # noqa: E402
from RAPID2_Agilent.models.target import TraceType  # noqa: E402
from RAPID2_Agilent.services import RunCache, maps, operations, rpd_csv  # noqa: E402
from RAPID2_Agilent.services.quantification import target_trace  # noqa: E402

MS1 = 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help=".d folders")
    parser.add_argument("--out", required=True, help="the dataset to write, .npz")
    parser.add_argument("--targets", help="a .rpd.csv file (or RAPID2's) of targets, traced in every run")
    parser.add_argument("--spots", type=int, default=15, help="strongest spots of each run's m/z-RT map")
    parser.add_argument("--random", type=int, default=5, help="m/z at random in each run")
    parser.add_argument("--ppm", type=float, default=10.0, help="half the m/z window of a spot's or a random XIC")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keep-calibration", action="store_true",
                        help="each scan's calibration as stored, those jumping away from the others' too")
    args = parser.parse_args(argv)

    targets = []
    if args.targets:
        targets = rpd_csv.resolve_groups(rpd_csv.read_targets_file(args.targets))
    rng = np.random.default_rng(args.seed)
    runs, traces, done = RunCache(max_runs=1), [], {}
    try:
        for run in args.runs:
            data = runs.get(run)
            run_name = Path(run).name
            print(f"{run_name}: ", end="", flush=True)
            key = str(Path(run).resolve())
            repaired = [] if args.keep_calibration or key in done else repair_calibration_jumps(data)
            done.setdefault(key, {"calibration_repaired": repaired})
            if repaired:
                print(f"calibration of {len(repaired)} scans repaired, ", end="", flush=True)
            for target in targets:
                rt, intensity = target_trace(data, target)
                window = target.mz_window() if target.trace_type is TraceType.XIC else None
                polarity = 1 if parse_adduct(target.adduct).charge > 0 else -1
                _add(traces, run, "target", f"#{target.index} {target.name}", window, polarity, target.ms_level.number,
                     rt, intensity)
            for spot in maps.strongest_spots(maps.built_map(data), top_n=args.spots) if args.spots else []:
                polarity = int(data.polarity_list[spot.scan])
                window = _ppm_window(spot.mz, args.ppm)
                _add(traces, run, "spot", f"m/z {spot.mz:.4f} (spot at {spot.rt:.2f} min)", window, polarity, MS1,
                     *_xic(data, window, polarity))
            scans = operations.trace_scans(data, MS1)
            low, high = data.profile_mz_range(scans[:: max(1, len(scans) // 20)])
            polarities, counts = np.unique(data.polarity_list[scans], return_counts=True)
            polarity = int(polarities[np.argmax(counts)])
            for mz in rng.uniform(low, high, args.random):
                window = _ppm_window(float(mz), args.ppm)
                _add(traces, run, "random", f"m/z {mz:.4f} (random)", window, polarity, MS1,
                     *_xic(data, window, polarity))
            print(f"{sum(trace.run == str(Path(run).resolve()) for trace in traces)} traces")
    finally:
        runs.close()
    write_dataset(args.out, traces, done)
    print(f"{len(traces)} traces written to {args.out}")
    return 0


def _ppm_window(mz: float, ppm: float) -> tuple[float, float]:
    return mz * (1 - ppm * 1e-6), mz * (1 + ppm * 1e-6)


def _xic(data, window: tuple[float, float], polarity: int) -> tuple[np.ndarray, np.ndarray]:
    """The XIC of the MS1 scans of the polarity, those whose data points the window misses left out, as the plots
    draw it."""
    scans = operations.trace_scans(data, MS1, polarity=polarity)
    if len(scans):
        scans = scans[~operations.mz_window_misses(data, window, scans).missing]
    if len(scans) == 0:
        return np.array([]), np.array([])
    rt, intensity = operations.extract_trace(data, TraceType.XIC, mz_window=window, scan_indices=scans)
    return np.asarray(rt, dtype=np.float64), np.asarray(intensity, dtype=np.float64)


def _add(traces: list[Trace], run, kind, name, window, polarity, ms_level, rt, intensity) -> None:
    if len(rt) < 3:
        return  # nothing to label
    traces.append(Trace(f"{len(traces):04d}", str(Path(run).resolve()), kind, name, window, polarity, ms_level,
                        rt, intensity))


if __name__ == "__main__":
    raise SystemExit(main())
