import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from modal.file_pattern_matcher import FilePatternMatcher

from inference import RecurrentRunner
from modal_config import PROJECT_IGNORE
from stabilize.exp_testbed_20k import SudokuTransformer
from viz.visualize import get_attention_weights, main

ROOT = Path(__file__).resolve().parent


class VisualizationTest(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.zeros(1, 81, 10)
        self.inputs[:, :, 0] = 1

    def test_attention_capture_preserves_actual_recurrence(self):
        variants = (
            {}, {"outer_state_norm": True}, {"outer_state_rms_cap": 1.0},
            {"layer_schedule": [1, 0, 1], "residual_scale": 0.5, "feedback_scale": 0.25},
        )
        for settings in variants:
            with self.subTest(settings=settings):
                model = SudokuTransformer(unique_layers=2, **settings).eval()
                expected, _ = RecurrentRunner(model).run_batch(self.inputs, [1, 2, 3])
                attention, logits = get_attention_weights(model, self.inputs, 3)
                self.assertEqual(len(attention), 3)
                for iteration in range(3):
                    torch.testing.assert_close(logits[iteration], expected[iteration + 1], rtol=0, atol=0)
                    self.assertEqual(len(attention[iteration]), len(model.layer_schedule))
                    for weights in attention[iteration]:
                        self.assertEqual(weights.shape, (1, 4, 81, 81))
                        self.assertFalse(weights.requires_grad)
                        torch.testing.assert_close(weights.sum(-1), torch.ones(1, 4, 81))
                self.assertTrue(all(not layer.norm1._forward_hooks for layer in model.layers))

    def test_hooks_removed_after_failed_inference(self):
        model = SudokuTransformer(unique_layers=1)
        with patch.object(model, "recurrent_step", side_effect=RuntimeError("failed step")):
            with self.assertRaisesRegex(RuntimeError, "failed step"):
                get_attention_weights(model, self.inputs, 2)
        self.assertFalse(model.layers[0].norm1._forward_hooks)

    def test_cli_rejects_unlabelled_weights_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save(SudokuTransformer().state_dict(), path)
            with patch("sys.argv", ["visualize", str(path), "--output-dir", directory]):
                with patch("viz.visualize.load_dataset") as dataset:
                    with self.assertRaisesRegex(ValueError, "Missing inference manifest"):
                        main()
                    dataset.assert_not_called()


class ModalScriptsTest(unittest.TestCase):
    def test_whole_project_uploads_use_shared_exclusions(self):
        uploads = 0
        paths = list(ROOT.glob("*.py"))
        for directory in ("es", "iters", "looping", "release_tools", "stabilize", "viz"):
            paths.extend((ROOT / directory).rglob("*.py"))
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "add_local_dir" and node.args
                        and isinstance(node.args[0], ast.Constant) and node.args[0].value == "."):
                    continue
                uploads += 1
                exclusions = next(keyword.value for keyword in node.keywords if keyword.arg == "ignore")
                names = {child.id for child in ast.walk(exclusions) if isinstance(child, ast.Name)}
                self.assertIn("PROJECT_IGNORE", names, str(path.relative_to(ROOT)))
        self.assertGreater(uploads, 0)
        self.assertTrue({".git/", ".claude/", ".codex/", ".env", "temp-side-convo.txt",
                         "**/*.pt", "release/v2/*.zip"}.issubset(PROJECT_IGNORE))

    def test_upload_exclusions_cover_nested_artifacts_but_retain_source(self):
        excluded = FilePatternMatcher(*PROJECT_IGNORE)
        for path in (".git", ".git/config", ".claude/session.json", ".codex/config.toml",
                     ".env", "temp-side-convo.txt", "release/v2/bundle.zip", "model.pt",
                     "release/v2/model.pt", "looping/run.log", "looping/__pycache__/x.pyc",
                     "venv/lib/site-packages/module.py"):
            self.assertTrue(excluded(path), path)
        for path in ("modal_config.py", "model_io.py", "looping/modal_stay_solved.py",
                     "release_tools/modal_checks.py"):
            self.assertFalse(excluded(path), path)

    def test_long_analysis_jobs_spawn_once_and_commit_on_failure(self):
        for filename in ("iters/modal_eval_interventions.py", "iters/modal_spectral_stable.py",
                         "modal_spectral_viridian.py", "viz/modal_viz.py"):
            with self.subTest(filename=filename):
                tree = ast.parse((ROOT / filename).read_text())
                main_function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                     and node.name == "main")
                calls = [node.func.attr for node in ast.walk(main_function)
                         if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
                self.assertEqual(calls.count("spawn"), 1)
                self.assertNotIn("remote", calls)
                commits = [node for block in ast.walk(tree) if isinstance(block, ast.Try)
                           for statement in block.finalbody for node in ast.walk(statement)
                           if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                           and isinstance(node.func.value, ast.Name)
                           and node.func.value.id == "outputs_volume" and node.func.attr == "commit"]
                self.assertEqual(len(commits), 1)


if __name__ == "__main__":
    unittest.main()
