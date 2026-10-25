"""What the GUI, the CLI and the MCP tools do to runs, written once. No Qt.

Where new code goes, so that it is written in one place:

``operations``  An operation on one run, returning numpy arrays: which scans,
                which extraction. GUI presenters plot these; `reports` summarizes
                them. Add here first, then call it from the front ends.
``reports``     The same operations as JSON-serializable dicts, for the CLI and
                the MCP tools: reductions for text (max_points, top_n) only.
``runs``        Finding and opening .d folders by path.
``quantification``  The area and the other values of a target's trace in a run.
``peak_detection``  The peaks of a target's trace in a run, and the one taken for the target's.
``rpd_csv``     Those values of the targets in the runs, written as a .rpd.csv file; and
                targets read back from one.
``maps``        The strongest spots of a run's m/z x RT map, and where two runs' maps differ.

The front ends only translate. A GUI presenter turns widget state into the
arguments of an operation and plots the result; an MCP analysis tool
(`mcp.tools`) declares a schema and calls a report; `rapid2-agilent-cli` is
generated from those MCP tools (`RAPID2_Agilent.cli`), so a tool added once is
a command too. Alignment jobs live in `alignment.runner` and options become a
configuration through `alignment.config.OPTIONS`, for the same reason.
"""

# `reports` is imported by whoever needs it (`from RAPID2_Agilent.services import reports`):
# it brings in the alignment package and scipy, which the GUI loads only once
# the B - A map asks for them.
from . import operations
from .runs import RunCache, find_runs, run_path

__all__ = ["RunCache", "find_runs", "operations", "run_path"]
