# -*- coding: utf-8 -*-

"""The instance registry: which windows exist, and which letter each holds.

The letter is claimed with an exclusive lock rather than a look-then-write, so
the interesting test is not that one process can claim one -- it is that any
number of processes starting at the same instant come away with different ones.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from RAPID2_Agilent.mcp import endpoint as endpoint_file
from RAPID2_Agilent.mcp import registry


CHILD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from RAPID2_Agilent.mcp import registry
claim = registry.claim_label(sys.argv[2])
print(claim.label, flush=True)
# Hold it while the other processes are claiming theirs.
time.sleep(1.0)
"""


class ClaimTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def claim(self, **kwargs):
        claim = registry.claim_label(self.root, **kwargs)
        self.addCleanup(claim.release)
        return claim

    def test_letters_are_handed_out_in_order(self):
        self.assertEqual(
            [self.claim().label for _ in range(3)], ["A", "B", "C"]
        )

    def test_a_released_letter_is_handed_out_again(self):
        first = registry.claim_label(self.root)
        self.assertEqual(first.label, "A")
        first.release()

        self.assertEqual(self.claim().label, "A")

    def test_running_out_of_letters_is_an_error(self):
        self.claim(labels="AB")
        self.claim(labels="AB")

        with self.assertRaises(registry.NoLabelAvailable):
            self.claim(labels="AB")

    def test_the_claim_is_atomic_across_processes(self):
        # The constraint that matters: several RAPID2 processes launched at once
        # must not agree on a letter.
        children = [
            subprocess.Popen(
                [sys.executable, "-c", CHILD, str(ROOT / "src"), str(self.root)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(6)
        ]
        labels = []
        for child in children:
            out, err = child.communicate(timeout=60)
            self.assertEqual(child.returncode, 0, err)
            labels.append(out.strip())

        self.assertEqual(len(labels), 6)
        self.assertEqual(sorted(labels), sorted(set(labels)), labels)
        self.assertEqual(sorted(labels), list("ABCDEF"))

    def test_started_at_is_recorded_once(self):
        claim = self.claim()

        self.assertTrue(claim.started_at)
        self.assertEqual(claim.started_at, claim.started_at)


class StaleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def publish(self, label, *, token="tok", port=1):
        return endpoint_file.write_endpoint(
            "127.0.0.1",
            port,
            token,
            path=registry.document_path(label, self.root),
            instance_id=f"id-{label}",
            label=label,
            started_at="2026-09-04T00:00:00+00:00",
        )

    def test_an_address_with_no_owner_is_swept(self):
        # Nobody holds the lock, so the process that wrote this is gone.
        path = self.publish("A")

        self.assertEqual(registry.sweep_stale(self.root), ["A"])
        self.assertFalse(path.exists())

    def test_a_live_instance_is_left_alone(self):
        claim = registry.claim_label(self.root)
        self.addCleanup(claim.release)
        path = self.publish(claim.label)

        self.assertEqual(registry.sweep_stale(self.root), [])
        self.assertTrue(path.exists())

    def test_sweeping_does_not_reserve_anything(self):
        self.publish("A")
        registry.sweep_stale(self.root)

        claim = registry.claim_label(self.root)
        self.addCleanup(claim.release)
        self.assertEqual(claim.label, "A")

    def test_claiming_discards_what_the_previous_owner_left(self):
        path = self.publish("A")

        claim = registry.claim_label(self.root)
        self.addCleanup(claim.release)

        self.assertEqual(claim.label, "A")
        self.assertFalse(path.exists())


class ReadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def publish(self, label, **kwargs):
        fields = {"instance_id": f"id-{label}", "label": label}
        fields.update(kwargs)
        endpoint_file.write_endpoint(
            "127.0.0.1",
            1,
            "tok",
            path=registry.document_path(label, self.root),
            **fields,
        )

    def test_instances_come_back_in_letter_order(self):
        self.publish("C")
        self.publish("A")

        self.assertEqual(
            [d["label"] for d in registry.read_instances(self.root)], ["A", "C"]
        )

    def test_an_unusable_address_is_skipped(self):
        registry.document_path("A", self.root).parent.mkdir(
            parents=True, exist_ok=True
        )
        registry.document_path("A", self.root).write_text("{}", encoding="utf-8")
        self.publish("B")

        self.assertEqual(
            [d["label"] for d in registry.read_instances(self.root)], ["B"]
        )

    def test_the_registry_carries_addresses_not_document_state(self):
        # The role split, pinned: a reader must not start depending on this
        # file to know what a window has open.
        self.publish("A")

        document = registry.read_instances(self.root)[0]

        self.assertEqual(
            sorted(document),
            sorted(
                [
                    "schema_version",
                    "url",
                    "host",
                    "port",
                    "token",
                    "pid",
                    "instance_id",
                    "app_version",
                    "label",
                    "started_at",
                    "toolset_version",
                ]
            ),
        )


class NamingTests(unittest.TestCase):
    DOCUMENT = {"label": "A", "instance_id": "8f3a1c2d4e5f6071"}

    def test_a_letter_names_a_window_either_case(self):
        self.assertTrue(registry.names_instance(self.DOCUMENT, "A"))
        self.assertTrue(registry.names_instance(self.DOCUMENT, "a"))

    def test_a_full_id_names_a_window(self):
        self.assertTrue(
            registry.names_instance(self.DOCUMENT, "8f3a1c2d4e5f6071")
        )

    def test_a_long_enough_prefix_names_a_window(self):
        self.assertTrue(registry.names_instance(self.DOCUMENT, "8f3a"))

    def test_a_short_prefix_does_not(self):
        # Too short to be a deliberate abbreviation.
        self.assertFalse(registry.names_instance(self.DOCUMENT, "8f"))

    def test_nothing_names_nothing(self):
        self.assertFalse(registry.names_instance(self.DOCUMENT, ""))
        self.assertFalse(registry.names_instance(self.DOCUMENT, "   "))

    def test_a_prefix_shared_by_two_windows_matches_neither(self):
        instances = [
            {"label": "A", "instance_id": "8f3a0000"},
            {"label": "B", "instance_id": "8f3a1111"},
        ]

        self.assertIsNone(registry.match_instance(instances, "8f3a"))
        self.assertEqual(registry.match_instance(instances, "B")["label"], "B")


if __name__ == "__main__":
    unittest.main()
