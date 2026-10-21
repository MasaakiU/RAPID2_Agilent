"""Run with python -m RAPID2_Agilent.alignment."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from .config import AlignmentConfig, config_from_dict
from .diagnostics import save_diagnostics, save_xic_overlay
from .io import source_from_reader
from .pipeline import align_agilent, align_sources
from .representation import build_profile_matrix
from .result_io import save_result
from .synthetic import generate_synthetic_pair, mapping_errors
from .transforms import suggest_scales


def _configuration(args: argparse.Namespace) -> AlignmentConfig:
    config = config_from_dict(json.loads(args.config.read_text())) if args.config else AlignmentConfig()
    for group, options in (
        ("binning", {"ppm": args.ppm, "aggregation": args.aggregation}),
        ("transform", {"scale": args.scale, "method": args.transform}),
        ("compute", {"device": args.device, "metric": args.metric, "chunk_size": args.chunk_size}),
        ("selection", {"max_scans": args.max_scans, "stride": args.scan_stride, "polarity": args.polarity,
                       "reader_rt_unit": args.reader_rt_unit}),
        ("dtw", {"rt_band_seconds": args.rt_band_seconds}),
        ("anchors", {"stride": args.anchor_stride}),
    ):
        changes = {name: value for name, value in options.items() if value is not None}
        config = replace(config, **{group: replace(getattr(config, group), **changes)})
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Raw-scan LC-QTOF alignment; RT mapping direction is query → reference.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    align = subparsers.add_parser("align", help="Align two Agilent .d directories directly")
    align.add_argument("reference", type=Path)
    align.add_argument("query", type=Path)
    align.add_argument("--xic", nargs=2, type=float, action="append", metavar=("LOW_MZ", "HIGH_MZ"))
    synthetic = subparsers.add_parser("synthetic", help="Run known-drift validation including strong run-specific ions")
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--n-reference", type=int, default=101)
    synthetic.add_argument("--n-query", type=int, default=117)
    synthetic.add_argument("--strong-amplitude", type=float, default=1e6)
    scales = subparsers.add_parser("scales", help="Inspect candidate asinh scales without choosing one")
    scales.add_argument("run", type=Path)
    for command in (align, synthetic, scales):
        command.add_argument("--config", type=Path)
        command.add_argument("--ppm", type=float)
        command.add_argument("--aggregation", choices=("sum", "mean", "max"))
        command.add_argument("--transform", choices=("none", "sqrt", "log1p", "asinh"))
        command.add_argument("--scale", type=float)
        command.add_argument("--device", choices=("auto", "cpu", "torch_cpu", "cuda"))
        command.add_argument("--metric", choices=("cosine", "pearson"))
        command.add_argument("--chunk-size", type=int)
        command.add_argument("--max-scans", type=int)
        command.add_argument("--scan-stride", type=int)
        command.add_argument("--polarity", choices=("positive", "negative"))
        command.add_argument("--reader-rt-unit", choices=("minutes", "seconds"))
        command.add_argument("--rt-band-seconds", type=float)
        command.add_argument("--anchor-stride", type=int)
    for command in (align, synthetic):
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = _configuration(args)
        if args.command == "scales":
            from ..io import AgilentReader
            with AgilentReader() as reader:
                reader.load(args.run)
                source = source_from_reader(reader, config.selection)
                profile = build_profile_matrix(source, config.binning)
                print(json.dumps({"run": source.metadata, "scale_candidates": suggest_scales(profile.values)}, indent=2))
            return
        config.transform.validate_for_use()
        if any((args.output / name).exists() for name in ("mapping.json", "metadata.json", "diagnostics.npz")):
            raise FileExistsError("Choose a new output directory; an alignment result already exists there")

        def progress(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

        truth = None
        if args.command == "synthetic":
            pair = generate_synthetic_pair(n_reference=args.n_reference, n_query=args.n_query,
                                           strong_amplitude=args.strong_amplitude, seed=args.seed)
            result = align_sources(pair.reference, pair.query, config, progress=progress)
            truth = pair.query_to_reference_truth
            errors = mapping_errors(result.mapping.transform(pair.query.rt_seconds), truth)
            result.metadata["synthetic_validation"] = errors
            print(json.dumps(errors, indent=2))
        else:
            result = align_agilent(args.reference, args.query, config, progress=progress)
        save_result(result, args.output)
        save_diagnostics(result, args.output / "diagnostics.png", truth=truth)
        if args.command == "synthetic":
            (args.output / "truth.json").write_text(json.dumps({
                "query_seconds": pair.query.rt_seconds.tolist(),
                "reference_seconds": truth.tolist(), "errors": errors,
            }, indent=2) + "\n")
        elif args.xic:
            from ..io import AgilentReader
            with AgilentReader() as a, AgilentReader() as b:
                a.load(args.reference)
                b.load(args.query)
                save_xic_overlay(source_from_reader(a, config.selection), source_from_reader(b, config.selection),
                                 result.mapping, args.xic, args.output / "xic_overlay.png")
        print(f"Saved {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError, NotImplementedError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
