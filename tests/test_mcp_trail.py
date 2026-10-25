# -*- coding: utf-8 -*-

"""The tool-call trail, as a reader of a crash would meet it.

What is being pinned is not the wording but the properties a reproduction
depends on: that the call is recorded *before* it runs, that one call is one
line, and that nothing a tool returns can make the file unreadable or
unbounded.
"""

import logging
import os
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from RAPID2_Agilent.mcp import trail
from RAPID2_Agilent.mcp.protocol import ToolError, ToolRefusal


class TrailTests(unittest.TestCase):
    def lines(self, handler, arguments, *, expect=None):
        """Run one traced call and return what it wrote."""

        traced = trail.traced(handler, "rapid2_demo")
        with self.assertLogs(trail.log, level=logging.INFO) as caught:
            if expect is None:
                traced(arguments)
            else:
                with self.assertRaises(expect):
                    traced(arguments)
        return [record.getMessage() for record in caught.records]

    # -- what a reproduction needs --------------------------------------
    def test_the_call_is_recorded_before_it_runs(self):
        # The whole point. A crash inside the handler must still leave the
        # call that caused it behind, and a line written afterwards would be
        # missing for exactly that call.
        written = []

        def handler(_arguments):
            written.append(self.lines is not None)
            raise MemoryError("stands in for a process that dies here")

        traced = trail.traced(handler, "rapid2_demo")
        with self.assertLogs(trail.log, level=logging.INFO) as caught:
            with self.assertRaises(MemoryError):
                traced({"pane": "blot"})

        self.assertTrue(written, "the handler never ran")
        self.assertIn("rapid2_demo <- pane=blot", caught.records[0].getMessage())

    def test_a_call_and_its_answer_are_two_lines(self):
        lines = self.lines(lambda _arguments: {"status": "ok"}, {"x": 1})

        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "rapid2_demo <- x=1")
        self.assertTrue(lines[1].startswith("rapid2_demo -> ok in "))

    def test_arguments_are_ordered_so_two_runs_compare(self):
        lines = self.lines(
            lambda _arguments: {}, {"width": 5, "height": 2, "pane": "blot"}
        )

        self.assertEqual(lines[0], "rapid2_demo <- height=2 pane=blot width=5")

    def test_a_call_with_no_arguments_still_reads_as_a_call(self):
        lines = self.lines(lambda _arguments: {}, {})

        self.assertEqual(lines[0], "rapid2_demo <- -")

    # -- what must never reach the file ---------------------------------
    def test_the_stamp_is_left_out(self):
        # Long, and reconstructible from the read that produced it.
        lines = self.lines(
            lambda _arguments: {}, {"expected_stamp": "x" * 200, "pane": "blot"}
        )

        self.assertEqual(lines[0], "rapid2_demo <- pane=blot")

    def test_a_long_value_is_cut_and_says_so(self):
        lines = self.lines(lambda _arguments: {}, {"text": "a" * 500})

        self.assertIn("a" * trail.MAX_VALUE_CHARS, lines[0])
        self.assertIn("[500 chars]", lines[0])
        self.assertLess(len(lines[0]), 200)

    def test_a_newline_in_an_argument_cannot_split_the_line(self):
        # One line is one event, or the file cannot be read as a sequence.
        lines = self.lines(lambda _arguments: {}, {"text": "top\nbottom"})

        self.assertEqual(lines[0], "rapid2_demo <- text=top bottom")
        self.assertNotIn("\n", lines[0])

    def test_the_answer_is_never_written_out(self):
        # A capture is megabytes of base64; the log keeps the verdict only.
        payload = {"status": "ok", "data": "Q" * 5000}
        lines = self.lines(lambda _arguments: payload, {})

        self.assertNotIn("Q" * 100, " ".join(lines))

    # -- outcomes --------------------------------------------------------
    def test_a_refusal_records_which_kind_it_was(self):
        # "told no" and "told no for a reason nobody expected" are different
        # things to find in a log.
        def handler(_arguments):
            raise ToolRefusal({"status": "work_cancelled"})

        lines = self.lines(handler, {}, expect=ToolRefusal)

        self.assertIn("refused work_cancelled", lines[1])

    def test_a_failure_records_its_reason(self):
        def handler(_arguments):
            raise ToolError("No RAPID2 document is open.")

        lines = self.lines(handler, {}, expect=ToolError)

        self.assertIn("failed: No RAPID2 document is open.", lines[1])

    def test_an_unplanned_error_keeps_its_stack(self):
        # The one case that is a bug in RAPID2 rather than an answer.
        def handler(_arguments):
            raise ZeroDivisionError("division by zero")

        traced = trail.traced(handler, "rapid2_demo")
        with self.assertLogs(trail.log, level=logging.INFO) as caught:
            with self.assertRaises(ZeroDivisionError):
                traced({})

        self.assertIsNotNone(caught.records[1].exc_info)

    def test_an_answer_that_is_not_a_mapping_is_still_recorded(self):
        lines = self.lines(lambda _arguments: "text", {})

        self.assertIn("-> ok", lines[1])

    def test_the_answer_is_handed_back_unchanged(self):
        payload = {"status": "ok", "value": 7}
        traced = trail.traced(lambda _arguments: payload, "rapid2_demo")

        with self.assertLogs(trail.log, level=logging.INFO):
            self.assertIs(traced({}), payload)


if __name__ == "__main__":
    unittest.main()
