"""Check cohort locking and outcome accounting independently of GPU runs."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from checkpoint_utils import atomic_json_save, atomic_torch_save
from dataset_utils import DATASET_REVISION
from looping.weight_tying.analyze import paired_puzzle_counts, reliability_summary, training_summary, verify_scores
from looping.weight_tying.common import atomic_npz, protocol, protocol_sha256, run_config, run_name
from looping.weight_tying.evaluate import evaluate_arrays, seal_cohort
from looping.weight_tying.model import StudyTransformer
from looping.weight_tying.test_study import ANSWER, QUESTION
from looping.weight_tying.data import encode_rows
from runtime_utils import file_sha256


def fake_completed_cohort(root):
    data = root / "data"
    data.mkdir()
    atomic_npz(data / "holdout.npz", placeholder=np.zeros(1))
    atomic_json_save({"protocol_sha256": protocol_sha256(), "dataset_revision": DATASET_REVISION,
                      "files": {"holdout.npz": file_sha256(data / "holdout.npz")}}, data / "manifest.json")
    data_digest = file_sha256(data / "manifest.json")
    for architecture in protocol()["architectures"]:
        for regime in protocol()["regimes"]:
            for seed in protocol()["seeds"]:
                directory = root / "runs" / run_name(architecture, regime, seed)
                directory.mkdir(parents=True)
                config = run_config(architecture, regime, seed)
                result = {"config": config, "status": "complete", "updates": 20000,
                          "data_manifest_sha256": data_digest, "sample_digest": f"{regime}/{seed}",
                          "best_validation": {"updates": 18000}}
                atomic_json_save(result, directory / "result.json")
                for selection, updates in (("final", 20000), ("best_validation", 18000)):
                    path = directory / f"{selection}.pt"
                    atomic_torch_save({"test": torch.zeros(1)}, path)
                    atomic_json_save({"config": config, "updates": updates,
                                      "data_manifest_sha256": data_digest, "weights_sha256": file_sha256(path)},
                                     directory / f"{selection}.pt.json")


class StudyAnalysisTests(unittest.TestCase):
    def test_paired_puzzle_counts_separate_disagreements(self):
        counts = paired_puzzle_counts(np.array([True, True, False, False]),
                                     np.array([True, False, True, False]))
        self.assertEqual(counts, {"both_solved": 1, "tied_only": 1, "untied_only": 1,
                                  "neither_solved": 1, "total": 4})
        with self.assertRaisesRegex(ValueError, "boolean vectors"):
            paired_puzzle_counts(np.array([1]), np.array([True]))

    def test_failures_stay_in_reliability_denominator(self):
        scores = {"1024": {"accuracy": 0.95}, "4096": {"accuracy": 0.90}}
        results = {"tied_late_seed20260902": {"status": "complete", "evaluations": {"final": {"holdout": scores}}},
                   "tied_late_seed20260903": {"status": "numerical_failure"}}
        counts = reliability_summary(results, "tied", "late")
        self.assertEqual(counts["planned"], 3)
        self.assertEqual(counts["numerical_failures"], 1)
        self.assertEqual(counts["pending_training"], 1)
        self.assertEqual(counts["datasets"]["holdout"]["healthy"], 1)
        self.assertEqual(counts["datasets"]["development"]["pending_evaluation"], 1)

    def test_saved_predictions_reproduce_scores_and_detect_corruption(self):
        digits, targets = encode_rows([QUESTION], [ANSWER])
        labels = np.array([0], dtype=np.uint8)
        data = {"digits": digits, "targets": targets, "labels": labels}
        predictions = {"predictions_16": targets.copy(), "finite_16": np.array([True]),
                       "solved_16": np.array([True]), "labels": labels}
        scores = {"16": {"solved": 1, "total": 1, "accuracy": 1.0, "nonfinite": 0,
                          "difficulty": {"0": {"solved": 1, "total": 1}}}}
        verify_scores(data, predictions, scores, [16])
        scores["16"]["accuracy"] = 0.0
        with self.assertRaisesRegex(ValueError, "Config mismatch"):
            verify_scores(data, predictions, scores, [16])

    def test_lock_is_stable_and_detects_changed_exports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_completed_cohort(root)
            locked = seal_cohort(root)
            self.assertEqual(len(locked["identity"]["runs"]), 18)
            self.assertEqual(locked, seal_cohort(root))
            path = root / "runs/tied_early_seed20260902/final.pt"
            atomic_torch_save({"test": torch.ones(1)}, path)
            with self.assertRaisesRegex(ValueError, "Unverified checkpoint"):
                seal_cohort(root)

    def test_lock_rejects_wrong_seed_or_unpaired_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_completed_cohort(root)
            path = root / "runs/untied_compute_early_seed20260902/result.json"
            result = json.loads(path.read_text())
            result["sample_digest"] = "different sequence"
            atomic_json_save(result, path)
            with self.assertRaisesRegex(ValueError, "not paired"):
                seal_cohort(root)
            result["sample_digest"] = "early/20260902"
            result["config"]["seed"] = 0
            atomic_json_save(result, path)
            with self.assertRaisesRegex(ValueError, "Config mismatch"):
                seal_cohort(root)

    def test_nonfinite_predictions_cannot_count_as_solved(self):
        torch.set_num_threads(1)
        model = StudyTransformer(width=16, feedforward_width=32).eval()
        with torch.no_grad():
            model.output_head.weight.fill_(float("nan"))
        digits, targets = encode_rows([QUESTION], [ANSWER])
        scores, arrays = evaluate_arrays(model, digits, targets, [1, 2])
        self.assertEqual(scores["2"]["solved"], 0)
        self.assertEqual(scores["2"]["nonfinite"], 1)
        self.assertFalse(arrays["finite_2"][0])

    def test_late_floor_uses_registered_training_interval(self):
        result = {"config": {"regime": "late"}, "status": "complete", "updates": 20000,
                  "parameters": 1, "best_validation": {}, "timings_seconds": {},
                  "estimated_model_flops": 1, "processed_puzzles": 1, "sample_digest": "x",
                  "history": [{"updates": step, "scores": {"1024": {"accuracy": score}}}
                              for step, score in ((1000, 0.0), (12000, 0.9), (20000, 0.98))]}
        summary = training_summary(result)
        self.assertAlmostEqual(summary["late_validation_mean"], 0.94)
        self.assertEqual(summary["late_validation_minimum"], 0.9)
