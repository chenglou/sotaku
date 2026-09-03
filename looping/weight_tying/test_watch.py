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
    from grpclib.exceptions import StreamTerminatedError
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
        call.get.side_effect = [modal.exception.ConnectionError("Deadline exceeded"),
                                StreamTerminatedError("Connection lost"), TimeoutError(), result]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.registry(root)
            with patch("modal.FunctionCall.from_id", return_value=call), patch("looping.weight_tying.watch.time.sleep"):
                snapshot = wait_for_results(registry, root / "received.json", 5)
            self.assertEqual(snapshot["errors"], {})
            self.assertEqual(len(snapshot["results"]), 1)
            self.assertEqual(call.get.call_count, 4)
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

    def test_evaluation_selections_are_collected_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.registry(root)
            entries = json.loads(registry.read_text())
            entries["action"] = "evaluate"
            base = entries["jobs"][0]
            entries["jobs"] = [dict(base, selection=selection, call_id=selection)
                               for selection in ("final", "best_validation")]
            atomic_json_save(entries, registry)
            calls = {}
            for selection in ("final", "best_validation"):
                calls[selection] = Mock()
                calls[selection].get.return_value = {
                    "identity": {"protocol_sha256": protocol_sha256(),
                                 "run_name": "tied_early_seed20260902", "selection": selection},
                    "elapsed_seconds": 1.0,
                }
            with patch("modal.FunctionCall.from_id", side_effect=calls.__getitem__):
                snapshot = wait_for_results(registry, root / "received.json", 5)
            self.assertEqual(set(snapshot["results"]), {
                "tied_early_seed20260902/final", "tied_early_seed20260902/best_validation"})
            with patch("modal.FunctionCall.from_id") as lookup:
                self.assertEqual(wait_for_results(registry, root / "received.json", 5), snapshot)
                lookup.assert_not_called()

    def test_evaluation_rejects_the_wrong_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.registry(root)
            entries = json.loads(registry.read_text())
            entries["action"] = "evaluate"
            entries["jobs"][0]["selection"] = "final"
            atomic_json_save(entries, registry)
            call = Mock()
            call.get.return_value = {"identity": {"protocol_sha256": protocol_sha256(),
                                                  "run_name": "tied_early_seed20260902",
                                                  "selection": "best_validation"}}
            with patch("modal.FunctionCall.from_id", return_value=call):
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    wait_for_results(registry, root / "received.json", 5)
            self.assertFalse((root / "received.json").exists())

    def test_duplicate_jobs_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            registry = self.registry(root)
            entries = json.loads(registry.read_text())
            entries["jobs"].append(dict(entries["jobs"][0]))
            atomic_json_save(entries, registry)
            with patch("modal.FunctionCall.from_id") as lookup:
                with self.assertRaisesRegex(ValueError, "Duplicate job"):
                    wait_for_results(registry, root / "received.json", 5)
                lookup.assert_not_called()
