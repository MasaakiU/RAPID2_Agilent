"""Command line over the analysis tools, printing their JSON answers.

    rapid2-agilent-cli extract-xic --path run.d --mz 150.5 --tolerance 0.5 --unit m/z
    rapid2-agilent-cli --format tsv -o xic.tsv extract-xic --path run.d --formula C5H13NO --adduct "[M+H]+" --max-points all
    python -m RAPID2_Agilent.cli --help

Generated from the MCP analysis tools (`mcp.tools.analysis_tools`): each tool is
a command named after it (rapid2_extract_xic -> extract-xic), each property of
its input schema an option, and the answer is the tool's own. The arguments are
checked by the validator the MCP server uses. So a tool added there is a
command here with nothing to write. An object argument is written as JSON.

--format tsv writes the answer's table instead (see `table`), and --output
writes to a file instead of stdout.
"""

import argparse
import json
import sys

from .mcp.protocol import ToolError, ToolRefusal, validate_arguments
from .mcp.tools import AnalysisSession, analysis_tools


PREFIX = "rapid2_"
# What a nullable option takes for null, e.g. `--ms-level all` for every scan.
NULL_WORDS = frozenset({"all", "none", "null"})


def command_name(tool_name):
    return tool_name.removeprefix(PREFIX).replace("_", "-")


def _converter(schema):
    """Text to the JSON type a property declares."""

    declared = schema.get("type") or "string"
    types = [declared] if isinstance(declared, str) else list(declared)
    nullable = "null" in types
    base = next((kind for kind in types if kind != "null"), "string")
    convert = {"number": float, "integer": int, "object": json.loads}.get(base, str)

    def parse(text):
        if nullable and text.lower() in NULL_WORDS:
            return None
        return convert(text)

    # argparse names the type in its error: "invalid integer value: 'x'".
    parse.__name__ = "JSON object" if base == "object" else base
    return parse


def _help(schema):
    text = schema.get("description") or ""
    types = schema.get("type")
    if isinstance(types, list) and "null" in types:
        text += f" ({' / '.join(sorted(NULL_WORDS))} for null)"
    items = schema.get("items") or {}
    if items.get("type") == "object":
        text += f" (each a JSON object of {', '.join(items.get('properties', {}))})"
    if "default" in schema:
        text += f" (default: {schema['default']})"
    return text.strip() or None


def build_parser(tools):
    parser = argparse.ArgumentParser(
        prog="rapid2-agilent-cli",
        description="RAPID2's analysis tools, as commands. Each prints its answer as JSON.",
    )
    parser.add_argument(
        "--format", dest="output_format", choices=("json", "tsv"), default="json",
        help="tsv: the answer's table only, a row per point, peak or row (default: json)",
    )
    parser.add_argument("-o", "--output", dest="output_file", metavar="FILE", help="write the answer to FILE instead of stdout")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    for tool in tools:
        command = commands.add_parser(command_name(tool.name), help=tool.title, description=tool.description)
        command.set_defaults(tool=tool)
        required = set(tool.input_schema.get("required", ()))
        for name, schema in tool.input_schema.get("properties", {}).items():
            # Absent stays absent, so the tool's own default applies; an
            # explicit null (`--ms-level all`) still reaches it.
            options = {"dest": name, "required": name in required, "help": _help(schema), "default": argparse.SUPPRESS}
            if schema.get("type") == "array":
                options.update(nargs="+", type=_converter(schema.get("items") or {}))
            else:
                options["type"] = _converter(schema)
                if "enum" in schema:
                    options["choices"] = schema["enum"]
            command.add_argument("--" + name.replace("_", "-"), **options)
    return parser


def table(payload):
    """(columns, rows) of an answer as TSV: its trace (rt, intensity), or its list of records (peaks, rows, ...);
    several lists are one table, with the list each row is of as its first column. None if it has none."""

    if isinstance(payload.get("rt"), list) and isinstance(payload.get("intensity"), list):
        return ["rt", "intensity"], list(zip(payload["rt"], payload["intensity"]))
    lists = {
        name: value
        for name, value in payload.items()
        if isinstance(value, list) and value and all(isinstance(item, dict) for item in value)
    }
    if not lists:
        return None
    columns = list(dict.fromkeys(column for records in lists.values() for record in records for column in record))
    rows = [
        ([name] if len(lists) > 1 else []) + [record.get(column) for column in columns]
        for name, records in lists.items()
        for record in records
    ]
    return (["list"] if len(lists) > 1 else []) + columns, rows


def _tsv(columns, rows):
    def cell(value):
        if value is None:
            return ""
        if isinstance(value, list):
            return ",".join(cell(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        # a tab or a line break would end the cell
        return " ".join(str(value).split()) if isinstance(value, str) else str(value)

    return "".join("\t".join(cell(value) for value in row) + "\n" for row in [columns, *rows])


def main(argv=None):
    analysis = AnalysisSession()
    try:
        parser = build_parser(analysis_tools(analysis))
        args = parser.parse_args(argv)
        tool = args.tool
        arguments = {name: value for name, value in vars(args).items() if name in tool.input_schema["properties"]}
        problem = validate_arguments(tool.input_schema, arguments)
        if problem is not None:
            parser.error(problem)
        try:
            payload = tool.handler(arguments)
        except ToolRefusal as exc:
            print(json.dumps(exc.payload, indent=2, ensure_ascii=False, default=str), file=sys.stderr)
            return 1
        except ToolError as exc:
            print(f"{parser.prog} {args.command}: {exc}", file=sys.stderr)
            return 1
        if args.output_format == "tsv":
            found = table(payload)
            if found is None:
                print(f"{parser.prog} {args.command}: the answer has no table; use --format json", file=sys.stderr)
                return 1
            text = _tsv(*found)
        else:
            text = json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n"
        if args.output_file is None:
            sys.stdout.write(text)
        else:
            with open(args.output_file, "w", encoding="utf-8", newline="") as f:
                f.write(text)
        return 0
    finally:
        analysis.close()


if __name__ == "__main__":
    sys.exit(main())
