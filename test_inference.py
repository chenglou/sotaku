import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from datasets import Dataset

from checkpoint_utils import atomic_json_save, atomic_torch_save
from dataset_utils import benchmark_manifest, validate_benchmark
from inference import RecurrentRunner
from model_io import load_model, write_model_manifest
from stabilize.exp_testbed_20k import SudokuTransformer


class InferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.inputs = torch.zeros(2, 81, 10)
        self.inputs[:, :, 0] = 1

    def test_recurrence_matches_forward_including_non_weight_settings(self):
        variants = (
            {}, {"outer_state_norm": True}, {"outer_state_rms_cap": 1.0},
            {"layer_schedule": [0, 0, 1], "residual_scale": 0.5, "feedback_scale": 0.25},
        )
        for settings in variants:
            model = SudokuTransformer(unique_layers=2, training_iterations=5, **settings).eval()
            with torch.no_grad():
                expected = model(self.inputs, return_all=True)
                actual, _ = RecurrentRunner(model, chunk_size=2).run_batch(self.inputs, [1, 3, 5])
            for horizon in (1, 3, 5):
                torch.testing.assert_close(actual[horizon], expected[horizon - 1], rtol=0, atol=0)

    def test_manifest_roundtrip_preserves_normalization_and_rejects_bad_hash(self):
        model = SudokuTransformer(unique_layers=2, outer_state_norm=True, training_iterations=2).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_torch_save(model.state_dict(), path)
            write_model_manifest(path, model)
            restored, manifest = load_model(path)
            self.assertTrue(restored.outer_state_norm)
            self.assertEqual(manifest["model"]["unique_layers"], 2)
            with torch.no_grad():
                torch.testing.assert_close(restored(self.inputs), model(self.inputs), rtol=0, atol=0)
            metadata_path = Path(str(path) + ".json")
            broken = json.loads(metadata_path.read_text())
            broken["weights"]["sha256"] = "0" * 64
            atomic_json_save(broken, metadata_path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_model(path)

    def test_unlabelled_weights_cannot_silently_drop_model_settings(self):
        model = SudokuTransformer(outer_state_norm=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_torch_save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "Missing inference manifest"):
                load_model(path, legacy_exp="stabilize.exp_testbed_20k")

    def test_training_bundle_is_not_accepted_as_public_weights(self):
        model = SudokuTransformer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            atomic_torch_save({"model_state_dict": model.state_dict(), "step": 2}, path)
            write_model_manifest(path, model)
            with self.assertRaisesRegex(ValueError, "state_dict"):
                load_model(path)

    def test_public_weights_load_in_fp32_even_with_a_different_default_dtype(self):
        model = SudokuTransformer(unique_layers=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_torch_save(model.state_dict(), path)
            write_model_manifest(path, model)
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float64)
                restored, _ = load_model(path)
                self.assertTrue(all(parameter.dtype == torch.float32 for parameter in restored.parameters()))
            finally:
                torch.set_default_dtype(previous)

    def test_solution_tracking_has_exact_iteration_numbers(self):
        class PredictableModel(torch.nn.Module):
            def initial_encoder(self, inputs):
                return torch.zeros(inputs.shape[0], 81, 1)

            def recurrent_step(self, hidden, predictions, cos, sin):
                return hidden + 1

            def output_head(self, hidden):
                logits = torch.zeros(hidden.shape[0], 81, 9)
                digit = (hidden[..., 0] == 3).long()
                return logits.scatter(-1, digit.unsqueeze(-1), 1)

        outputs, diagnostics = RecurrentRunner(PredictableModel(), track_solutions=True, chunk_size=2).run_batch(
            self.inputs, [1, 4], targets=torch.zeros(2, 81, dtype=torch.long),
            empty_mask=torch.ones(2, 81, dtype=torch.bool),
        )
        self.assertEqual(diagnostics["first_solved_iteration"].tolist(), [1, 1])
        self.assertEqual(diagnostics["regression_count"].tolist(), [1, 1])
        self.assertEqual(diagnostics["last_regression_iteration"].tolist(), [3, 3])
        self.assertEqual(diagnostics["stayed_solved_after_first"].tolist(), [False, False])

    def test_frozen_benchmark_verifies_row_content(self):
        rows = {"question": ["." * 81] * 10, "answer": ["1" * 81] * 10,
                "rating": [1, 3, 51, 11, 0] * 2}
        dataset = Dataset.from_dict(rows)
        manifest = benchmark_manifest(dataset, per_bucket=1)
        self.assertEqual(len(validate_benchmark(dataset, manifest)), 5)
        changed = Dataset.from_dict({**rows, "answer": ["2" * 81] * 10})
        with self.assertRaisesRegex(ValueError, "content"):
            validate_benchmark(changed, manifest)

    def test_evaluator_saves_per_puzzle_results_and_rejects_output_reuse(self):
        from iters.eval_more_iters import evaluate
        model = SudokuTransformer(unique_layers=1, training_iterations=2).eval()
        dataset = Dataset.from_dict({
            "question": ["." * 81] * 5, "answer": ["1" * 81] * 5,
            "rating": [0, 1, 3, 11, 51],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            atomic_torch_save(model.state_dict(), path)
            write_model_manifest(path, model)
            settings = dict(device="cpu", iter_counts=(1, 2), batch_size=3,
                            output_dir=Path(directory) / "evaluation", dataset=dataset)
            original_run = RecurrentRunner.run_batch

            def require_full_precision(runner, *args, **kwargs):
                self.assertFalse(torch.is_autocast_enabled("cpu"))
                return original_run(runner, *args, **kwargs)

            with torch.autocast("cpu", dtype=torch.bfloat16), patch.object(RecurrentRunner, "run_batch", require_full_precision):
                result = evaluate(path, **settings)
            self.assertEqual(result["scores"]["2"]["total"], 5)
            self.assertEqual(result["identity"]["precision"], "fp32")
            self.assertEqual(result["identity"]["matmul_precision"], "highest")
            from release_tools.verify_records import verify_evaluation
            verified = verify_evaluation(settings["output_dir"], dataset)
            self.assertEqual(verified["total"], 5)
            from release_tools.prepare import package_records
            atomic_json_save({"verified": {"evaluation": verified}}, Path(directory) / "verified_records.json")
            archive_path = Path(directory) / "records.zip"
            checksums = package_records(path, directory, settings["output_dir"] / "benchmark.json", archive_path)
            self.assertIn("records.zip", checksums)
            with self.assertRaises(FileExistsError):
                package_records(path, directory, settings["output_dir"] / "benchmark.json", archive_path)
            self.assertEqual(evaluate(path, **settings), result)
            with self.assertRaisesRegex(ValueError, "batch_size"):
                evaluate(path, **{**settings, "batch_size": 2})
            result["identity"]["precision"] = "bf16"
            atomic_json_save(result, settings["output_dir"] / "result.json")
            with self.assertRaisesRegex(ValueError, "precision"):
                evaluate(path, **settings)

    def test_reference_export_preserves_bytes_and_records_the_selected_step(self):
        from release_tools.prepare import export_weights
        from runtime_utils import file_sha256
        from model_io import model_settings
        model = SudokuTransformer(unique_layers=1, training_iterations=2).eval()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pt"
            resume = Path(directory) / "resume.pt"
            destination = Path(directory) / "export.pt"
            atomic_torch_save(model.state_dict(), source)
            atomic_torch_save({
                "model_state_dict": model.state_dict(), "model_settings": model_settings(model),
                "config": {"random_seed": 7}, "step": 42,
            }, resume)
            with self.assertRaisesRegex(ValueError, "explicit trust"):
                export_weights(source, resume, destination)
            manifest = export_weights(source, resume, destination, trust_checkpoint=True)
            self.assertEqual(file_sha256(source), file_sha256(destination))
            self.assertEqual(manifest["training"]["last_step"], 42)
            self.assertEqual(manifest["training"]["optimizer_updates"], 43)
            load_model(destination)
            with self.assertRaises(FileExistsError):
                export_weights(source, resume, destination, trust_checkpoint=True)

    def test_public_training_presets_and_explicit_seed(self):
        import train
        with patch("sys.argv", ["train.py", "--run-name", "development_fixture"]), patch.object(train, "train_development") as trainer:
            train.main()
            trainer.assert_called_once_with(output_dir="runs/training", arm="late_state_ce",
                                            run_name="development_fixture", random_seed=20260730)
        with patch("sys.argv", ["train.py", "--preset", "reference", "--seed", "13", "--run-name", "reference_fixture"]), patch.object(train, "train_reference") as trainer:
            train.main()
            trainer.assert_called_once_with(output_dir="runs/training", arm="late_state_ce",
                                            run_name="reference_fixture", random_seed=13)


if __name__ == "__main__":
    unittest.main()
