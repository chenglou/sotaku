"""50K schedule, frozen controls, exact resumption, and evaluation identity."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from checkpoint_utils import atomic_json_save, atomic_torch_save
from looping.hyperloop_50k.common import protocol as control_protocol
from looping.weight_tying.common import atomic_npz
from looping.weight_tying.test_study import fixture, make_smoke_data
from looping.weight_tying.train import PairedSampler, learning_rate
from looping.width.common import protocol as short_protocol
from looping.width.evaluate import load_export as load_short_export
from looping.width.model import WidthTransformer
from looping.width_50k.common import (
    SOURCE_PATHS, build_model, protocol, run_config, state_sha256,
    verify_completed_pair, verify_inherited_preflight, verify_reference,
)
from looping.width_50k.evaluate import evaluate_run, export_model, load_export
from looping.width_50k.train import train_run
from runtime_utils import file_sha256, runtime_manifest


class ConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_schedule_and_model_match_the_intended_comparison(self):
        spec, controls, short = protocol(), control_protocol(), short_protocol()
        self.assertEqual(spec["training"], controls["training"])
        self.assertEqual(spec["seeds"], controls["seeds"])
        self.assertTrue(set(spec["seeds"]).isdisjoint(short["seeds"]))
        self.assertEqual(spec["arms"], {"width160": 160})
        self.assertEqual(spec["training"]["steps"], 50000)
        self.assertEqual(spec["training"]["warmup_steps"], 1400)
        for phase, earlier in zip(spec["training"]["phases"], short["training"]["phases"]):
            self.assertEqual(phase, [int(earlier[0] * 2.5), int(earlier[1] * 2.5), earlier[2]])
        self.assertGreater(learning_rate(20000, spec["training"]), learning_rate(20000, short["training"]))
        torch.manual_seed(123)
        expected = WidthTransformer(160)
        torch.manual_seed(123)
        actual = build_model(run_config("width160", spec["seeds"][0]))
        self.assertEqual(state_sha256(actual.state_dict()), state_sha256(expected.state_dict()))
        self.assertEqual(sum(p.numel() for p in actual.parameters()), 1241929)
        for arm, seed in (("width192", spec["seeds"][0]), ("width160", short["seeds"][0])):
            with self.assertRaises(ValueError):
                run_config(arm, seed)

    def test_inherited_full_batch_and_frozen_control_checks(self):
        report = verify_inherited_preflight()
        self.assertEqual(report["batch_size"], 2048)
        self.assertTrue(report["compiled"])
        self.assertEqual(report["arms"][0]["width"], 160)
        for field in ("batch_size", "learning_rate"):
            changed = copy.deepcopy(protocol())
            changed["training"][field] *= 2
            with patch("looping.width_50k.common.protocol", return_value=changed):
                with self.assertRaises(ValueError):
                    verify_inherited_preflight()
        changed = copy.deepcopy(protocol())
        changed["arms"]["width160"] = 192
        with patch("looping.width_50k.common.protocol", return_value=changed):
            with self.assertRaises(ValueError):
                verify_inherited_preflight()
        reference = verify_reference()
        for seed in protocol()["seeds"]:
            result = copy.deepcopy(reference["runs"][str(seed)])
            result["config"] = run_config("width160", seed)
            verify_completed_pair(result)
            result["sample_digest"] = "wrong"
            with self.assertRaises(ValueError):
                verify_completed_pair(result)

    def test_paired_sampling_and_restore_across_all_50k_transitions(self):
        spec = protocol()
        ratings = np.array([0, 1, 2, 3, 10, 11, 50, 51, 100], dtype=np.int16)
        regime = {key: spec[key] for key in ("burnin_probability", "burnin_iterations")}
        for seed in spec["seeds"]:
            baseline = PairedSampler(ratings, seed, regime, control_protocol()["training"])
            actual = PairedSampler(ratings, seed, regime, spec["training"])
            for step in (0, 9999, 10000, 19999, 20000, 29999, 30000, 49999):
                expected_indices, expected_horizon = baseline.sample(step, 2048)
                indices, horizon = actual.sample(step, 2048)
                np.testing.assert_array_equal(indices, expected_indices)
                self.assertEqual(horizon, expected_horizon)
                minimum = next(low for begin, end, low in spec["training"]["phases"] if begin <= step < end)
                self.assertTrue(np.all(ratings[indices] >= minimum))
                saved = actual.state_dict()
                expected = actual.sample(step, 2048)
                actual.load_state_dict(saved)
                restored = actual.sample(step, 2048)
                np.testing.assert_array_equal(restored[0], expected[0])
                self.assertEqual(restored[1], expected[1])
                baseline.load_state_dict(actual.state_dict())

    def test_resume_selected_recovery_and_wrong_recipe_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_smoke_data(root / "data")
            seed = protocol()["seeds"][0]
            directory = root / "resumed"
            train_run(root / "data", directory, "width160", seed, smoke=True, stop_after=2, device="cpu")
            atomic_torch_save({}, directory / "best_validation.pt")
            resumed = train_run(root / "data", directory, "width160", seed, smoke=True, device="cpu")
            whole = train_run(root / "data", root / "whole", "width160", seed, smoke=True, device="cpu")
            for key in ("sample_digest", "horizon_counts", "work_counts", "best_validation"):
                self.assertEqual(resumed[key], whole[key])
            self.assertEqual(state_sha256(torch.load(directory / "final.pt", weights_only=True)),
                             state_sha256(torch.load(root / "whole/final.pt", weights_only=True)))
            model, manifest = load_export(directory / "final.pt")
            _, selected = load_export(directory / "best_validation.pt")
            self.assertEqual(model.width, 160)
            self.assertEqual(manifest["updates"], 4)
            self.assertEqual(selected["updates"], resumed["best_validation"]["updates"])
            with self.assertRaises(ValueError):
                load_short_export(directory / "final.pt")
            with self.assertRaises(ValueError):
                train_run(root / "data", directory, "width160", seed + 1, smoke=True, device="cpu")
            with self.assertRaises(ValueError):
                evaluate_run(root / "data", directory, "final", device="cpu")
            manifest["config"]["protocol"]["training"]["steps"] = 20000
            atomic_json_save(manifest, str(directory / "final.pt") + ".json")
            with self.assertRaises(ValueError):
                load_export(directory / "final.pt")

    def test_full_evaluation_checkpoint_identity_and_cached_result_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            spec = protocol()
            benchmark = json.loads((Path(__file__).parents[2] / spec["evaluation"]["benchmark"]).read_text())
            inputs, answers = fixture()
            digits = np.repeat(inputs.argmax(-1).numpy(), 25000, axis=0)
            targets = np.repeat(answers.numpy(), 25000, axis=0)
            atomic_npz(data / "development.npz", digits=digits, targets=targets,
                       labels=np.array(benchmark["bucket_names"]), indices=np.array(benchmark["indices"]))
            identity = {"development.npz": file_sha256(data / "development.npz")}
            config = run_config("width160", spec["seeds"][0])
            source = runtime_manifest(SOURCE_PATHS)["source_sha256"]
            directory = root / "run"
            directory.mkdir()
            model = build_model(config)
            for selection, step in (("final", 50000), ("best_validation", 49000)):
                export_model(model, directory / f"{selection}.pt", config, step, identity, source)
            atomic_json_save({"config": config, "status": "complete", "updates": 50000,
                              "source_sha256": source, "data_sha256": identity,
                              "best_validation": {"updates": 49000}}, directory / "result.json")
            scores, predictions = {}, {}
            for horizon in spec["evaluation"]["iterations"]:
                predictions[f"predictions_{horizon}"] = targets.astype(np.uint8)
                predictions[f"solved_{horizon}"] = np.ones(25000, dtype=bool)
                predictions[f"finite_{horizon}"] = np.ones(25000, dtype=bool)
                scores[str(horizon)] = {"accuracy": 1.0, "solved": 25000, "total": 25000, "nonfinite": 0}
            with patch("looping.width_50k.evaluate.validate_data", return_value=identity), \
                 patch("looping.width_50k.evaluate.evaluate_arrays", return_value=(scores, predictions, {})) as evaluator:
                for selection, step in (("final", 50000), ("best_validation", 49000)):
                    result = evaluate_run(data, directory, selection, device="cpu")
                    self.assertEqual(result["identity"]["updates"], step)
                    self.assertEqual(result["identity"]["checkpoint_selection"], selection)
                    self.assertEqual(evaluate_run(data, directory, selection, device="cpu"), result)
                self.assertEqual(evaluator.call_count, 2)
                result_path = directory / "evaluations/final/result.json"
                result = json.loads(result_path.read_text())
                result["predictions_sha256"] = "wrong"
                atomic_json_save(result, result_path)
                with self.assertRaises(ValueError):
                    evaluate_run(data, directory, "final", device="cpu")


if __name__ == "__main__":
    unittest.main()
