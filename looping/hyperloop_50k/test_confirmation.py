"""50K protocol, exact resumption, checkpoint selection, and unchanged-code checks."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from checkpoint_utils import atomic_json_save, atomic_torch_save
from dataset_utils import RATING_BUCKETS, phase_bucket_keys
from looping.exp_stay_solved import FULL_50K_SCHEDULE
from looping.hyperloop.common import protocol as original_protocol
from looping.hyperloop.evaluate import load_export as load_original_export
from looping.hyperloop_50k.analyze import analyze, compare
from looping.hyperloop_50k.common import SOURCE_PATHS, build_model, protocol, run_config, state_sha256, verify_inherited_preflight
from looping.hyperloop_50k.evaluate import evaluate_run, load_export
from looping.hyperloop_50k.train import train_run
from looping.weight_tying.common import atomic_npz
from looping.weight_tying.test_study import fixture, make_smoke_data
from looping.weight_tying.train import PairedSampler, learning_rate
from model_io import load_model
from runtime_utils import file_sha256, runtime_manifest
from looping.hyperloop_50k.evaluate import export_model


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_full_schedule_matches_effective_public_curriculum(self):
        spec, original = protocol(), original_protocol()
        settings = spec["training"]
        self.assertEqual(settings["steps"], FULL_50K_SCHEDULE["total_steps"])
        self.assertEqual(settings["warmup_steps"], FULL_50K_SCHEDULE["warmup_steps"])
        keys = [(lower, upper) for lower, upper, _ in RATING_BUCKETS]
        for actual, (begin, end, minimum, _) in zip(settings["phases"], FULL_50K_SCHEDULE["phases"]):
            # The public trainer selects whole rating buckets, not raw rating >= 21 or 6.
            effective_minimum = min(lower for lower, _ in phase_bucket_keys(keys, minimum))
            self.assertEqual(actual, [begin, end, effective_minimum])
        self.assertEqual(spec["arms"], {"baseline": 0, "gated_four": 4})
        self.assertTrue(set(spec["seeds"]).isdisjoint(original["seeds"]))
        for key in ("model", "window_length", "burnin_iterations", "burnin_probability", "data_files", "evaluation"):
            self.assertEqual(spec[key], original[key])
        for phase, short in zip(settings["phases"], original["training"]["phases"]):
            self.assertEqual(phase, [int(short[0] * 2.5), int(short[1] * 2.5), short[2]])
        self.assertEqual(len(settings["phases"]), 4)
        for step in (0, 559, 560, 4000, 12000, 19999):
            self.assertGreater(learning_rate(step, settings), 0)
        self.assertGreater(learning_rate(20000, settings), learning_rate(20000, original["training"]))
        for invalid in (("gated_one", spec["seeds"][0]), ("baseline", original["seeds"][0])):
            with self.assertRaises(ValueError):
                run_config(*invalid)

    def test_inherited_full_batch_evidence_and_identity_checks(self):
        report = verify_inherited_preflight()
        self.assertEqual(report["batch_size"], 2048)
        self.assertTrue(all(row["populated_optimizer_resume_exact"] for row in report["arms"]))
        changed = copy.deepcopy(protocol())
        changed["training"]["batch_size"] = 1024
        with patch("looping.hyperloop_50k.common.protocol", return_value=changed):
            with self.assertRaisesRegex(ValueError, "batch_size"):
                verify_inherited_preflight()
        changed = copy.deepcopy(protocol())
        changed["model"]["width"] = 96
        with patch("looping.hyperloop_50k.common.protocol", return_value=changed):
            with self.assertRaisesRegex(ValueError, "model"):
                verify_inherited_preflight()

    def test_paired_batches_and_resume_across_all_curriculum_transitions(self):
        spec = protocol()
        ratings = np.array([0, 1, 2, 3, 10, 11, 50, 51, 100], dtype=np.int16)
        regime = {key: spec[key] for key in ("burnin_probability", "burnin_iterations")}
        for seed in spec["seeds"]:
            left = PairedSampler(ratings, seed, regime, spec["training"])
            right = PairedSampler(ratings, seed, regime, spec["training"])
            for step in (0, 9999, 10000, 19999, 20000, 29999, 30000, 49999):
                indices, horizon = left.sample(step, 2048)
                other_indices, other_horizon = right.sample(step, 2048)
                np.testing.assert_array_equal(indices, other_indices)
                self.assertEqual(horizon, other_horizon)
                minimum = next(minimum for begin, end, minimum in spec["training"]["phases"] if begin <= step < end)
                self.assertTrue(np.all(ratings[indices] >= minimum))
                saved = left.state_dict()
                expected = left.sample(step, 2048)
                left.load_state_dict(saved)
                actual = left.sample(step, 2048)
                np.testing.assert_array_equal(expected[0], actual[0])
                self.assertEqual(expected[1], actual[1])
                right.load_state_dict(left.state_dict())

    def test_resume_is_exact_for_all_arms_and_rejects_wrong_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            results = []
            for arm in protocol()["arms"]:
                seed = protocol()["seeds"][0]
                directory = root / arm
                train_run(root / "data", directory, arm, seed, smoke=True, stop_after=2, device="cpu")
                # Simulate preemption between the selected export and checkpoint publication.
                atomic_torch_save({}, directory / "best_validation.pt")
                resumed = train_run(root / "data", directory, arm, seed, smoke=True, device="cpu")
                whole = train_run(root / "data", root / (arm + "_whole"), arm, seed, smoke=True, device="cpu")
                results.append(resumed)
                self.assertEqual(resumed["sample_digest"], whole["sample_digest"])
                self.assertEqual(state_sha256(torch.load(directory / "final.pt", weights_only=True)),
                                 state_sha256(torch.load(root / (arm + "_whole") / "final.pt", weights_only=True)))
                loaded, manifest = load_export(directory / "final.pt")
                _, selected = load_export(directory / "best_validation.pt")
                self.assertEqual(selected["updates"], resumed["best_validation"]["updates"])
                self.assertEqual(loaded.streams, protocol()["arms"][arm])
                self.assertEqual(manifest["updates"], 4)
                with self.assertRaises(ValueError):
                    load_original_export(directory / "final.pt")
                with self.assertRaises(ValueError):
                    load_model(directory / "final.pt")
                with self.assertRaises(ValueError):
                    train_run(root / "data", directory, arm, seed + 1, smoke=True, device="cpu")
                altered = torch.load(directory / "final.pt", weights_only=True)
                altered["initial_encoder.bias"][0] += 1
                atomic_torch_save(altered, directory / "final.pt")
                with self.assertRaises(ValueError):
                    load_export(directory / "final.pt")
            for result in results[1:]:
                for key in ("sample_digest", "base_initial_state_sha256", "work_counts"):
                    self.assertEqual(result[key], results[0][key])

    def test_failed_seeds_cannot_disappear_from_decision(self):
        self.assertEqual(compare({})["status"], "pending")
        profiles = {(arm, seed): {"1024": 0.94, "4096": 0.9}
                    for arm in protocol()["arms"] for seed in protocol()["seeds"]}
        self.assertEqual(compare(profiles)["status"], "not_promising")
        for seed in protocol()["seeds"]:
            profiles["gated_four", seed] = {"1024": 0.945, "4096": 0.90}
        self.assertEqual(compare(profiles)["status"], "promising")
        profiles["gated_four", protocol()["seeds"][0]] = None
        self.assertEqual(compare(profiles)["status"], "not_promising")

    def test_full_evaluation_metadata_selection_reuse_and_independent_scoring(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            data_dir.mkdir()
            spec = protocol()
            benchmark = json.loads((Path(__file__).parents[2] / spec["evaluation"]["benchmark"]).read_text())
            inputs, answers = fixture()
            digits = np.repeat(inputs.argmax(-1).numpy(), 25000, axis=0)
            targets = np.repeat(answers.numpy(), 25000, axis=0)
            labels, indices = np.array(benchmark["bucket_names"]), np.array(benchmark["indices"])
            atomic_npz(data_dir / "development.npz", digits=digits, targets=targets, labels=labels, indices=indices)
            data_identity = {"development.npz": file_sha256(data_dir / "development.npz")}
            config = run_config("baseline", spec["seeds"][0])
            source_identity = runtime_manifest(SOURCE_PATHS)["source_sha256"]
            directory = root / "runs" / f"baseline_seed{spec['seeds'][0]}"
            directory.mkdir(parents=True)
            model = build_model(config)
            for selection, step in (("final", 50000), ("best_validation", 49000)):
                export_model(model, directory / f"{selection}.pt", config, step, data_identity, source_identity)
            training = {"config": config, "status": "complete", "updates": 50000,
                        "source_sha256": source_identity, "data_sha256": data_identity,
                        "best_validation": {"updates": 49000}, "parameters": 796937,
                        "base_initial_state_sha256": "fixture", "sample_digest": "fixture", "work_counts": {},
                        "timings_seconds": {}, "history": [{"updates": 50000, "scores": {"1024": {"accuracy": 1.0}}}]}
            atomic_json_save(training, directory / "result.json")
            scores, predictions = {}, {}
            for horizon in spec["evaluation"]["iterations"]:
                predictions[f"predictions_{horizon}"] = targets.astype(np.uint8)
                predictions[f"solved_{horizon}"] = np.ones(25000, dtype=bool)
                predictions[f"finite_{horizon}"] = np.ones(25000, dtype=bool)
                scores[str(horizon)] = {"accuracy": 1.0, "solved": 25000, "total": 25000, "nonfinite": 0}
            with patch("looping.hyperloop_50k.evaluate.validate_data", return_value=data_identity), \
                 patch("looping.hyperloop_50k.common.validate_data", return_value=data_identity), \
                 patch("looping.hyperloop_50k.evaluate.evaluate_arrays", return_value=(scores, predictions, {})) as evaluator:
                for selection in ("final", "best_validation"):
                    result = evaluate_run(data_dir, directory, selection, device="cpu")
                    self.assertEqual(result["identity"]["checkpoint_selection"], selection)
                    self.assertEqual(result["identity"]["selection"], spec["evaluation"]["selection"])
                    self.assertEqual(evaluate_run(data_dir, directory, selection, device="cpu"), result)
                self.assertEqual(evaluator.call_count, 2)
                report = analyze(root, data_dir)
                self.assertEqual(report["versus_baseline"]["status"], "pending")
                output = directory / "evaluations/final"
                predictions["predictions_1024"] = targets.copy()
                predictions["predictions_1024"][0, 0] = (targets[0, 0] + 1) % 9
                atomic_npz(output / "predictions.npz", **predictions)
                changed = json.loads((output / "result.json").read_text())
                changed["predictions_sha256"] = file_sha256(output / "predictions.npz")
                atomic_json_save(changed, output / "result.json")
                with self.assertRaises(AssertionError):
                    analyze(root, data_dir)


if __name__ == "__main__":
    unittest.main()
