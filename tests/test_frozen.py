"""The standalone build's programs, told apart by their executables' names."""

import sys
import unittest
from pathlib import Path
from unittest import mock

from RAPID2_Agilent import frozen


def running_as(executable, platform="darwin"):
    return mock.patch.multiple(sys, executable=executable, platform=platform)


class ProgramTests(unittest.TestCase):
    def test_each_executable_runs_the_program_it_is_named_after(self):
        for program in frozen.PROGRAMS:
            with self.subTest(program=program), running_as(f"/app/Contents/MacOS/{program}"):
                self.assertEqual(frozen.program_name(), program)

    def test_a_windows_executable_is_known_without_its_suffix_or_case(self):
        with running_as("/RAPID2_Agilent/RAPID2-Agilent-CLI.exe"):
            self.assertEqual(frozen.program_name(), frozen.CLI)

    def test_an_executable_of_another_name_is_the_window(self):
        with running_as("/app/Contents/MacOS/RAPID2"):
            self.assertEqual(frozen.program_name(), frozen.GUI)

    def test_the_programs_sit_beside_the_running_one(self):
        with running_as("/app/Contents/MacOS/rapid2-agilent"):
            self.assertEqual(frozen.executable(frozen.RELAY), Path("/app/Contents/MacOS/rapid2-agilent-mcp"))
        self.assertEqual(
            frozen.executable(frozen.CLI, platform="win32").name, "rapid2-agilent-cli.exe"
        )

    def test_run_hands_the_arguments_to_that_programs_main(self):
        with running_as("/app/Contents/MacOS/rapid2-agilent-cli"), \
                mock.patch("RAPID2_Agilent.cli.main", return_value=3) as cli_main:
            self.assertEqual(frozen.run(["--help"]), 3)
        cli_main.assert_called_once_with(["--help"])

        with running_as("/app/Contents/MacOS/rapid2-agilent-mcp"), \
                mock.patch("RAPID2_Agilent.mcp.relay.__main__.main", return_value=0) as relay_main:
            self.assertEqual(frozen.run([]), 0)
        relay_main.assert_called_once_with([])


class EnvironmentTests(unittest.TestCase):
    def test_from_source_a_child_inherits_the_environment(self):
        self.assertFalse(frozen.is_frozen())
        self.assertIsNone(frozen.child_environment())

    def test_a_frozen_programs_child_starts_as_an_application_of_its_own(self):
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.dict("os.environ", {"_PYI_PARENT_PROCESS_LEVEL": "1"}):
            environment = frozen.child_environment()

        self.assertEqual(environment[frozen.RESET_ENVIRONMENT], "1")
        self.assertEqual(environment["_PYI_PARENT_PROCESS_LEVEL"], "1")  # the bootloader clears it


if __name__ == "__main__":
    unittest.main()
