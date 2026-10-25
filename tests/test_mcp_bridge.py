# -*- coding: utf-8 -*-

import os
import threading
import time
import unittest
from unittest import mock


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QCoreApplication, QEventLoop, QThread
from PyQt6.QtWidgets import QApplication

from RAPID2_Agilent.mcp import bridge as bridge_module
from RAPID2_Agilent.mcp.bridge import (
    MainThreadBridge,
    MainThreadTimeout,
    _Call,
)


def pump_until(predicate, *, timeout=5.0):
    """Run the main-thread event loop until a worker reports it is done."""

    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
    return predicate()


class MainThreadBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.bridge = MainThreadBridge()
        self.main_thread = QThread.currentThread()

    def tearDown(self):
        self.bridge.deleteLater()
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)

    def run_in_worker(self, fn):
        """Call ``fn`` on a worker thread, pumping the main loop meanwhile."""

        outcome = {}

        def worker():
            try:
                outcome["value"] = fn()
            except BaseException as exc:  # noqa: BLE001 - reported to the test
                outcome["error"] = exc

        thread = threading.Thread(target=worker, name="test-worker")
        thread.start()
        pump_until(lambda: not thread.is_alive())
        thread.join(2)
        self.assertFalse(thread.is_alive(), "worker thread did not finish")
        return outcome

    # -- the core guarantee ----------------------------------------------
    def test_callable_runs_on_the_main_thread(self):
        # Every viewer tool touches Qt widgets, so this is the property the
        # whole design rests on.
        observed = {}

        def work():
            observed["thread"] = QThread.currentThread()
            return 42

        outcome = self.run_in_worker(lambda: self.bridge.invoke(work, timeout=5))

        self.assertEqual(outcome.get("value"), 42)
        self.assertIs(observed["thread"], self.main_thread)

    def test_exception_is_reraised_on_the_calling_thread(self):
        def work():
            raise ValueError("from the main thread")

        outcome = self.run_in_worker(lambda: self.bridge.invoke(work, timeout=5))

        self.assertIsInstance(outcome.get("error"), ValueError)
        self.assertIn("from the main thread", str(outcome["error"]))

    def test_calls_are_serialised_with_each_other(self):
        # Two concurrent requests must not interleave inside the window.
        order = []

        def slow(tag):
            order.append(f"enter-{tag}")
            time.sleep(0.05)
            order.append(f"exit-{tag}")
            return tag

        outcomes = []
        threads = []
        for tag in ("a", "b"):
            thread = threading.Thread(
                target=lambda t=tag: outcomes.append(
                    self.bridge.invoke(lambda: slow(t), timeout=5)
                )
            )
            threads.append(thread)
            thread.start()
        pump_until(lambda: not any(t.is_alive() for t in threads))
        for thread in threads:
            thread.join(2)

        self.assertEqual(sorted(outcomes), ["a", "b"])
        # enter/exit never nest: each call completed before the next started.
        for index in range(0, len(order), 2):
            self.assertTrue(order[index].startswith("enter-"))
            self.assertEqual(order[index + 1], order[index].replace("enter-", "exit-"))

    # -- timeouts --------------------------------------------------------
    def test_timeout_when_the_main_thread_never_runs_the_call(self):
        # Stands in for a busy main thread: nothing pumps the event loop, so
        # the queued call is never delivered.
        ran = []
        outcome = {}

        def worker():
            try:
                self.bridge.invoke(lambda: ran.append(True), timeout=0.2)
            except BaseException as exc:  # noqa: BLE001
                outcome["error"] = exc

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join(3)

        self.assertIsInstance(outcome.get("error"), MainThreadTimeout)
        self.assertFalse(thread.is_alive())

        # An abandoned call must not run later and change the window behind
        # a caller that already gave up.
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 50)
        self.assertEqual(ran, [])

    # The claim/abandon protocol is asserted directly. Reaching the
    # "already started" branch through `invoke` needs the main thread to
    # deliver the queued call before the deadline, and how long that takes is
    # not something a test can pin down -- the timing-based version of this
    # passed alone and failed under the load of the full suite.
    def test_a_claimed_call_can_no_longer_be_abandoned(self):
        call = _Call(lambda: "x")

        self.assertTrue(call.claim())
        self.assertFalse(call.claim(), "a call must run at most once")
        self.assertFalse(
            call.abandon(), "a running call must never be abandoned"
        )

    def test_an_abandoned_call_never_runs(self):
        call = _Call(lambda: "x")

        self.assertTrue(call.abandon())
        self.assertFalse(
            call.claim(), "a call the caller gave up on must not run later"
        )

    def test_invoke_waits_out_a_call_it_could_not_abandon(self):
        """The grace period, without depending on scheduler latency.

        A call is only abandoned if the main thread has not claimed it yet.
        Standing in a _Call that refuses to be abandoned puts invoke() on the
        grace-period branch deterministically -- the earlier version of this
        test raced the main thread for it and, under the load of the full
        suite, never reached the branch at all.
        """

        finished = []

        def slow():
            time.sleep(0.2)
            finished.append(True)
            return "done"

        class AlreadyClaimed(_Call):
            def abandon(self):
                return False

        with mock.patch.object(bridge_module, "_Call", AlreadyClaimed):
            outcome = self.run_in_worker(
                lambda: self.bridge.invoke(slow, timeout=0.01)
            )

        # The deadline passed, the call could not be abandoned, so it was
        # waited out rather than left half applied.
        self.assertEqual(outcome.get("value"), "done")
        self.assertNotIn("error", outcome)
        self.assertEqual(finished, [True])

    def test_invoke_gives_up_on_a_call_that_never_starts(self):
        """The other side of the same branch: nothing claimed it, so it dies."""

        ran = []

        class NeverClaimed(_Call):
            def claim(self):
                return False

        with mock.patch.object(bridge_module, "_Call", NeverClaimed):
            outcome = self.run_in_worker(
                lambda: self.bridge.invoke(lambda: ran.append(True), timeout=0.05)
            )

        self.assertIsInstance(outcome.get("error"), MainThreadTimeout)
        self.assertEqual(ran, [])

    # -- re-entrancy -----------------------------------------------------
    def test_invoking_from_the_main_thread_does_not_deadlock(self):
        self.assertTrue(self.bridge.on_main_thread())
        self.assertEqual(self.bridge.invoke(lambda: "direct", timeout=0.1), "direct")


if __name__ == "__main__":
    unittest.main()
