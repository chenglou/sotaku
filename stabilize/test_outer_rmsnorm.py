import unittest
from unittest.mock import patch

import torch

import stabilize.exp_testbed_20k as testbed_module
from iters.eval_state_rms_cap import evaluate_trajectory
from iters.state_norm import per_token_rms
from stabilize.eval_testbed_outer_cap1 import evaluate as evaluate_cap1_full
from stabilize.eval_lr2e3_outer_cap1 import (
    BEST_MODEL_CONFIGS,
    evaluate as evaluate_cap1_50k_best,
)
from stabilize.eval_lr2e3_outer_rmsnorm import (
    BEST_MODEL_CONFIGS as RMSNORM_BEST_MODEL_CONFIGS,
    evaluate as evaluate_rmsnorm_50k_best,
)
from stabilize.exp_lr2e3_outer_cap1 import (
    FULL_50K_SCHEDULE,
    train as train_cap1_50k,
)
from stabilize.exp_lr2e3_outer_rmsnorm import train as train_rmsnorm_50k
from stabilize.exp_testbed_20k import SudokuTransformer, resolve_schedule
from stabilize.exp_testbed_outer_cap1 import train as train_cap1
from stabilize.exp_testbed_outer_rmsnorm import train


class OuterRMSNormTest(unittest.TestCase):
    def test_outer_norm_adds_no_parameters_or_state_dict_entries(self):
        baseline = SudokuTransformer(outer_state_norm=False)
        normalized = SudokuTransformer(outer_state_norm=True)
        self.assertEqual(baseline.state_dict().keys(), normalized.state_dict().keys())
        self.assertEqual(
            sum(parameter.numel() for parameter in baseline.parameters()),
            sum(parameter.numel() for parameter in normalized.parameters()),
        )

    def test_model_normalizes_the_carried_state_per_token(self):
        model = SudokuTransformer(outer_state_norm=True)
        hidden_state = torch.randn(3, 81, 128) * 40
        normalized = model.normalize_outer_state(hidden_state)
        torch.testing.assert_close(
            per_token_rms(normalized),
            torch.ones(3, 81),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_disabled_outer_norm_is_exact_identity(self):
        model = SudokuTransformer(outer_state_norm=False)
        hidden_state = torch.randn(2, 81, 128)
        self.assertIs(model.normalize_outer_state(hidden_state), hidden_state)

    def test_outer_cap_only_shrinks_tokens_above_one(self):
        model = SudokuTransformer(outer_state_rms_cap=1.0)
        hidden_state = torch.randn(2, 81, 128)
        hidden_state[:, :40] *= 0.1
        constrained = model.normalize_outer_state(hidden_state)
        torch.testing.assert_close(constrained[:, :40], hidden_state[:, :40])
        self.assertLessEqual(float(per_token_rms(constrained).max()), 1.001)

    def test_outer_norm_and_cap_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            SudokuTransformer(outer_state_norm=True, outer_state_rms_cap=1.0)

    def test_full_horizon_evaluator_applies_model_constraint_each_iteration(self):
        model = SudokuTransformer(outer_state_norm=True).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        targets = torch.zeros(1, 81, dtype=torch.long)
        empty_mask = torch.ones(1, 81, dtype=torch.bool)

        with patch.object(
            model,
            "normalize_outer_state",
            wraps=model.normalize_outer_state,
        ) as normalize_outer_state:
            results = evaluate_trajectory(
                model=model,
                module=testbed_module,
                inputs=inputs,
                targets=targets,
                empty_mask=empty_mask,
                bucket_names=["0"],
                checkpoints=(2,),
                maximum_rms=None,
                batch_size=1,
                device=torch.device("cpu"),
            )

        self.assertEqual(normalize_outer_state.call_count, 2)
        self.assertAlmostEqual(
            results["2"]["post_cap_token_rms"]["mean"],
            1.0,
            places=5,
        )

    @patch("stabilize.exp_testbed_outer_rmsnorm.train_testbed")
    def test_experiment_wrapper_passes_the_clean_intervention(self, train_testbed):
        train_testbed.return_value = {"ok": True}
        result = train("/tmp/output", run_name="trial_name", random_seed=123)
        self.assertEqual(result, {"ok": True})
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_testbed_outer_rmsnorm",
            run_name="trial_name",
            outer_state_norm=True,
            random_seed=123,
            checkpoint_on_probe=True,
        )

    @patch("stabilize.exp_testbed_outer_cap1.train_testbed")
    def test_cap1_wrapper_passes_the_clean_intervention(self, train_testbed):
        train_testbed.return_value = {"ok": True}
        result = train_cap1("/tmp/output", run_name="trial_name", random_seed=123)
        self.assertEqual(result, {"ok": True})
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_testbed_outer_cap1",
            run_name="trial_name",
            outer_state_rms_cap=1.0,
            random_seed=123,
            checkpoint_on_probe=True,
        )

    @patch("stabilize.eval_testbed_outer_cap1.evaluate_rms_cap")
    def test_full_cap1_eval_uses_training_recurrence_on_full_test_set(self, evaluate):
        evaluate.return_value = {"ok": True}
        result = evaluate_cap1_full(
            output_dir="/tmp/output",
            output_prefix="full_eval",
        )
        self.assertEqual(result, {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=((
                "cap1_trial0",
                "/outputs/model_testbed_outer_cap1_trial0.pt",
            ),),
            experiment_module="stabilize.exp_testbed_20k",
            caps=(1.0,),
            checkpoints=(16, 128, 1024, 2048),
            examples_per_bucket=5000,
            batch_size=250,
            seed=42,
            device="cuda",
            output_dir="/tmp/output",
            output_prefix="full_eval",
        )

    def test_full_schedule_matches_the_original_50k_recipe(self):
        schedule = resolve_schedule(FULL_50K_SCHEDULE)
        self.assertEqual(schedule['warmup_steps'], 1400)
        self.assertEqual(schedule['total_steps'], 50000)
        self.assertEqual(schedule['eval_every'], 5000)
        self.assertEqual(schedule['probe_every'], 2000)
        self.assertEqual(schedule['phases'][-1][1], 50000)

    @patch("stabilize.eval_lr2e3_outer_cap1.evaluate_rms_cap")
    def test_full_50k_eval_uses_all_harvested_checkpoints(self, evaluate):
        evaluate.return_value = {"ok": True}
        result = evaluate_cap1_50k_best(
            output_dir="/tmp/output",
            output_prefix="full_eval",
        )
        self.assertEqual(result, {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=BEST_MODEL_CONFIGS,
            experiment_module="stabilize.exp_testbed_20k",
            caps=(1.0,),
            checkpoints=(16, 128, 1024, 2048),
            examples_per_bucket=5000,
            batch_size=250,
            seed=42,
            device="cuda",
            output_dir="/tmp/output",
            output_prefix="full_eval",
        )

    @patch("stabilize.exp_lr2e3_outer_cap1.train_scheduled")
    def test_full_cap1_wrapper_passes_only_cap_and_schedule_changes(self, train):
        train.return_value = {"ok": True}
        result = train_cap1_50k(
            output_dir="/tmp/output",
            run_name="trial_name",
            random_seed=123,
        )
        self.assertEqual(result, {"ok": True})
        train.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_lr2e3_outer_cap1",
            run_name="trial_name",
            outer_state_rms_cap=1.0,
            random_seed=123,
            checkpoint_on_probe=True,
            schedule=FULL_50K_SCHEDULE,
        )

    @patch("stabilize.exp_lr2e3_outer_rmsnorm.train_scheduled")
    def test_full_rmsnorm_wrapper_passes_only_norm_and_schedule_changes(self, train):
        train.return_value = {"ok": True}
        result = train_rmsnorm_50k(
            output_dir="/tmp/output",
            run_name="trial_name",
            random_seed=123,
        )
        self.assertEqual(result, {"ok": True})
        train.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_lr2e3_outer_rmsnorm",
            run_name="trial_name",
            outer_state_norm=True,
            random_seed=123,
            checkpoint_on_probe=True,
            schedule=FULL_50K_SCHEDULE,
        )

    @patch("stabilize.eval_lr2e3_outer_rmsnorm.evaluate_outer_state")
    def test_full_rmsnorm_eval_reapplies_the_training_recurrence(self, evaluate):
        evaluate.return_value = {"ok": True}
        result = evaluate_rmsnorm_50k_best(
            output_dir="/tmp/output",
            output_prefix="full_eval",
        )
        self.assertEqual(result, {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=RMSNORM_BEST_MODEL_CONFIGS,
            experiment_module="stabilize.exp_testbed_20k",
            caps=(None,),
            checkpoints=(16, 128, 1024, 2048),
            examples_per_bucket=5000,
            batch_size=250,
            seed=42,
            device="cuda",
            output_dir="/tmp/output",
            output_prefix="full_eval",
            model_kwargs={"outer_state_norm": True},
        )


if __name__ == "__main__":
    unittest.main()
