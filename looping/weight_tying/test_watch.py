"""Result collection must not confuse a read failure with a training failure."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from checkpoint_utils import atomic_json_save
from looping.weight_tying.common import protocol_sha256

HAS_MODAL = importlib.util.find_spec("modal") is not None
if HAS_MODAL:
    import modal
    from looping.weight_tying.watch import wait_for_results


@unittest.skipUnless(HAS_MODAL, "Modal client is only needed for monitoring tests")
class WatchTests(unittest.TestCase):
    def registry(self, root):
        path = root / "jobs.json"
        atomic_json_save({"protocol_sha256": protocol_sha256(), "jobs": [
            {"architecture": "tied", "regime": "early", "seed": 20260902, "call_id": "fixture-call"}
        ]}, path)
        return path

    def test_read_failures_are_retried_without_replacing_workers(self):
        result = {"config": {"protocol_sha256": protocol_sha256(), "architecture": "tied",
                             "regime": "early", "seed": 20260902}, "status": "complete", "updates": 20000}
        call = Mock()
        call.get.side_effect = [modal.exception.ConnectionError("Deadline exceeded"), TimeoutError(), result]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.registry(root)
            with patch("modal.FunctionCall.from_id", return_value=call), patch("looping.weight_tying.watch.time.sleep"):
                snapshot = wait_for_results(registry, root / "received.json", 5)
            self.assertEqual(snapshot["errors"], {})
            self.assertEqual(len(snapshot["results"]), 1)
            self.assertEqual(call.get.call_count, 3)
            call.get.assert_called_with(timeout=1)

    def test_worker_failure_is_recorded_and_reported(self):
        call = Mock()
        call.get.side_effect = modal.exception.RemoteError("worker failed")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.registry(root)
            with patch("modal.FunctionCall.from_id", return_value=call):
                with self.assertRaises(modal.exception.RemoteError):
                    wait_for_results(registry, root / "received.json", 5)
            saved = json.loads((root / "received.json").read_text())
            self.assertEqual(saved["results"], {})
            self.assertIn("tied_early_seed20260902", saved["errors"])
