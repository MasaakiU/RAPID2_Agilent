"""Run with python -m RAPID2_Agilent.alignment.

The full-option alignment CLI: every alignment setting is a flag here, and it
has the synthetic validation. `rapid2-agilent-cli align-runs` and the MCP tool
take fewer options. All of them run the same jobs in `runner`, and their
options become a configuration through the same `config.OPTIONS` table, so
nothing about an alignment is decided in more than one place.
"""

import argparse
import json
from pathlib import Path
import sys

from .config import OPTIONS, AlignmentConfig, config_from_dict, with_options
from .pipeline import align_sources
from .result_io import save_result
from .runner import align_and_save, check_output_dir, save_figures, scale_candidates
from .synthetic import generate_synthetic_pair, mapping_errors


def _configuration(args: argparse.Namespace) -> AlignmentConfig:
    config = config_from_dict(json.loads(args.config.read_text())) if args.config else AlignmentConfig()
    return with_options(config, **{name: getattr(args, name) for name in OPTIONS})


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _configuration(args)
        if args.command == "scales":
            from ..io import AgilentReader
            with AgilentReader() as reader:
                reader.load(args.run)
                print(json.dumps(scale_candidates(reader, config), indent=2))
            return

        def progress(message: str) -> None:
            print(message, file=sys.stderr, flush=True)

        if args.command == "align":
            align_and_save(args.reference, args.query, args.output, config, xic_windows=args.xic or (), progress=progress)
        else:
            config.transform.validate_for_use()
            check_output_dir(args.output)
            pair = generate_synthetic_pair(n_reference=args.n_reference, n_query=args.n_query,
                                           strong_amplitude=args.strong_amplitude, seed=args.seed)
            result = align_sources(pair.reference, pair.query, config, progress=progress)
            truth = pair.query_to_reference_truth
            errors = mapping_errors(result.mapping.transform(pair.query.rt_seconds), truth)
            result.metadata["synthetic_validation"] = errors
            print(json.dumps(errors, indent=2))
            save_result(result, args.output)
            save_figures(result, args.output, truth=truth)
            (args.output / "truth.json").write_text(json.dumps({
                "query_seconds": pair.query.rt_seconds.tolist(),
                "reference_seconds": truth.tolist(), "errors": errors,
            }, indent=2) + "\n")
        print(f"Saved {args.output}")
    except (ValueError, FileNotFoundError, FileExistsError, NotImplementedError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
