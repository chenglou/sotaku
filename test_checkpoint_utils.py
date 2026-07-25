import tempfile
import unittest
from pathlib import Path

import torch

from checkpoint_utils import atomic_torch_save, load_branch_checkpoint


class BranchCheckpointTest(unittest.TestCase):
    def test_branch_allows_only_declared_config_changes(self):
        source_model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "source.pt"
            atomic_torch_save(
                {
                    "model_state_dict": source_model.state_dict(),
                    "config": {
                        "d_model": 128,
                        "run_name": "source",
                        "objective": "control",
                    },
                },
                checkpoint_path,
            )
            target_model = torch.nn.Linear(2, 1)
            checkpoint = load_branch_checkpoint(
                checkpoint_path,
                target_model,
                {
                    "d_model": 128,
                    "run_name": "branch",
                    "objective": "consistency",
                },
                {"run_name", "objective"},
            )
            self.assertEqual(checkpoint["config"]["run_name"], "source")
            for source, target in zip(
                source_model.parameters(),
                target_model.parameters(),
            ):
                torch.testing.assert_close(source, target)

    def test_branch_rejects_undeclared_or_removed_settings(self):
        model = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "source.pt"
            atomic_torch_save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "d_model": 128,
                        "schedule": "50k",
                    },
                },
                checkpoint_path,
            )
            with self.assertRaisesRegex(
                ValueError,
                "Branch config mismatch.*schedule",
            ):
                load_branch_checkpoint(
                    checkpoint_path,
                    torch.nn.Linear(2, 1),
                    {"d_model": 128},
                    set(),
                )


if __name__ == "__main__":
    unittest.main()
