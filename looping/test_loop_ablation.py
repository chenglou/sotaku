import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from iters.exp_baseline_lr2e3 import SudokuTransformer as OriginalSudokuTransformer
from looping.exp_loop_ablation import (
    ARM_CONFIGS,
    LINEAR_LOOP_SCALE,
    SQRT_LOOP_SCALE,
    get_arm_config,
    train,
)
from looping.exp_late_supervision import (
    LATE_SUPERVISION_CONFIGS,
    get_late_supervision_config,
    train as train_late_supervision,
)
from looping.exp_scaled_n64 import (
    LINEAR_SCALE as N64_LINEAR_SCALE,
    MICROBATCH_SIZE as N64_MICROBATCH_SIZE,
    RUN_BATCH_SIZE as N64_RUN_BATCH_SIZE,
    TRAINING_ITERATIONS as N64_TRAINING_ITERATIONS,
    train as train_scaled_n64,
)
from looping.eval_horizon_damping import relaxed_recurrent_step
from looping.eval_late_supervision_full import evaluate_run as evaluate_late_run
from looping.eval_late_recipe import (
    RECOMMENDED_POLICY,
    evaluate_checkpoint as evaluate_late_recipe,
    make_damping_policy,
)
from looping.eval_scaled_n64_full import evaluate_run as evaluate_scaled_n64
from looping.eval_delayed_damping_full import (
    DELAYED_LATE_CHECKPOINT_FULL_POLICIES,
    LATE_CHECKPOINT_FULL_POLICIES,
    evaluate_model as evaluate_damping_full,
)
from looping.eval_damping_robustness import (
    DELAYED_LATE_CHECKPOINT_MODEL,
    LATE_CHECKPOINT_TRIAL1_MODEL,
    LATE_THROUGH_1024_CHECKPOINT_MODEL,
    ES_BOUNDARY_MODELS,
    LATE_CHECKPOINT_MODEL,
    LATE_CHECKPOINT_POLICIES,
    POLICY_GRID,
    RESCUE_POLICIES,
    STEP35_POLICY_GRID,
    evaluate_es_boundary,
    evaluate_es_boundary_strong,
    evaluate_delayed_late_checkpoint,
    evaluate_late_checkpoint,
    evaluate_late_checkpoint_trial1,
    evaluate_late_checkpoint_through_1024,
)
from looping.eval_loop_diagnostics import (
    CLEAN_A_TRAJECTORY_MODELS,
    MODEL_PRESETS,
    _matrix_summary,
    linear_cka,
    one_step_digit_lens,
    parameter_gradient_conflict,
    recurrent_stages,
)
from looping.modal_loop_ablation import ARM_NAMES
from looping.modal_loop_diagnostics import PRESET_NAMES
from looping.modal_late_supervision import ARM_NAMES as LATE_ARM_NAMES
from looping.modal_damping_robustness import normalize_experiment_name
import stabilize.exp_testbed_20k as testbed_module
from stabilize.exp_testbed_20k import (
    SudokuTransformer,
    resolve_late_supervision,
)


class _AddOne(nn.Module):
    def forward(self, hidden_state, rope_cos, rope_sin):
        return hidden_state + 1


class _Double(nn.Module):
    def forward(self, hidden_state, rope_cos, rope_sin):
        return hidden_state * 2


class _ProposePlusTwo:
    def apply_recurrent_updates(
        self,
        hidden_state,
        predictions,
        rope_cos,
        rope_sin,
    ):
        return hidden_state + 2

    def normalize_outer_state(self, hidden_state):
        return hidden_state


class LoopAblationTest(unittest.TestCase):
    def test_damping_modal_normalizes_cli_hyphens(self):
        self.assertEqual(
            normalize_experiment_name("delayed-late-checkpoint"),
            "delayed_late_checkpoint",
        )

    def test_modal_arm_names_match_training_presets(self):
        self.assertEqual(set(ARM_NAMES), set(ARM_CONFIGS))

    def test_modal_diagnostic_names_match_model_presets(self):
        self.assertEqual(set(PRESET_NAMES), set(MODEL_PRESETS))
        self.assertEqual(
            [config["name"] for config in CLEAN_A_TRAJECTORY_MODELS],
            [
                "clean_a_step_30000",
                "clean_a_step_35000",
                "clean_a_step_40000",
                "clean_a_step_45000",
                "clean_a_step_49999",
            ],
        )

    def test_modal_late_supervision_names_match_training_presets(self):
        self.assertEqual(set(LATE_ARM_NAMES), set(LATE_SUPERVISION_CONFIGS))

    def test_default_model_matches_the_original_sota_forward(self):
        torch.manual_seed(7)
        original = OriginalSudokuTransformer().eval()
        refactored = SudokuTransformer().eval()
        refactored.load_state_dict(original.state_dict())
        inputs = torch.zeros(2, 81, 10)
        inputs[:, :, 0] = 1
        with torch.no_grad():
            expected = original(inputs)
            actual = refactored(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_forward_can_continue_from_a_detached_recurrent_state(self):
        torch.manual_seed(11)
        model = SudokuTransformer(unique_layers=1).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        rope_cos = testbed_module.ROPE_COS
        rope_sin = testbed_module.ROPE_SIN
        hidden_state = model.initial_encoder(inputs)
        predictions = torch.zeros(1, 81, 9)
        with torch.no_grad():
            for _ in range(4):
                hidden_state = model.recurrent_step(
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                )
                predictions = torch.softmax(
                    model.output_head(hidden_state),
                    dim=-1,
                )
            initial_state = (hidden_state.detach(), predictions.detach())
            actual = model(inputs, initial_state=initial_state)
            for _ in range(16):
                hidden_state = model.recurrent_step(
                    hidden_state,
                    predictions,
                    rope_cos,
                    rope_sin,
                )
                predictions = torch.softmax(
                    model.output_head(hidden_state),
                    dim=-1,
                )
            expected = model.output_head(hidden_state)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_training_iteration_count_controls_the_supervised_unroll(self):
        model = SudokuTransformer(unique_layers=1, training_iterations=3).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        with torch.no_grad():
            logits = model(inputs, return_all=True)
        self.assertEqual(len(logits), 3)

    def test_late_supervision_settings_are_validated_together(self):
        self.assertEqual(
            resolve_late_supervision((32, 128), 0.2, 0.5),
            (32, 128),
        )
        with self.assertRaises(ValueError):
            resolve_late_supervision((31,), 0.2, 0.5)
        with self.assertRaises(ValueError):
            resolve_late_supervision((32,), 0.0, 0.5)
        with self.assertRaises(ValueError):
            resolve_late_supervision((), 0.2, 0.5)

    def test_horizon_damping_interpolates_the_complete_recurrent_update(self):
        hidden_state = torch.zeros(1, 1, 1)
        result = relaxed_recurrent_step(
            _ProposePlusTwo(),
            hidden_state,
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            0.25,
        )
        torch.testing.assert_close(result, torch.full_like(result, 0.5))

    def test_damping_robustness_matrix_keeps_the_confirmed_policy(self):
        confirmed_policy = {
            "name": "warm128_a025",
            "alpha": 0.25,
            "warmup_iterations": 128,
        }
        self.assertIn(confirmed_policy, RESCUE_POLICIES)
        self.assertIn(confirmed_policy, POLICY_GRID)
        self.assertEqual(len({policy["name"] for policy in POLICY_GRID}), 7)
        self.assertEqual(
            len({policy["name"] for policy in STEP35_POLICY_GRID}),
            7,
        )
        self.assertEqual(
            [model["name"] for model in ES_BOUNDARY_MODELS],
            [
                "cohort_a_step_40000",
                "cohort_d_step_45000",
                "cohort_g_step_40000",
                "clean_b_step_35000",
            ],
        )

    @patch("looping.eval_damping_robustness.evaluate")
    def test_es_boundary_helpers_return_their_evaluations(self, evaluate):
        evaluate.side_effect = ({"standard": True}, {"strong": True})
        self.assertEqual(evaluate_es_boundary(), {"standard": True})
        self.assertEqual(evaluate_es_boundary_strong(), {"strong": True})
        self.assertEqual(
            [call.kwargs["output_prefix"] for call in evaluate.call_args_list],
            [
                "delayed-damping-es-boundary-n1000",
                "delayed-damping-es-boundary-strong-n1000",
            ],
        )

    @patch("looping.eval_damping_robustness.evaluate")
    def test_late_checkpoint_grid_waits_until_the_late_horizon(self, evaluate):
        evaluate.return_value = {"ok": True}
        self.assertEqual(evaluate_late_checkpoint(), {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=(LATE_CHECKPOINT_MODEL,),
            policies=LATE_CHECKPOINT_POLICIES,
            examples_per_bucket=200,
            horizons=(128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix="delayed-damping-late-checkpoint-n1000",
        )

    @patch("looping.eval_damping_robustness.evaluate")
    def test_delayed_late_checkpoint_uses_the_same_policy_grid(self, evaluate):
        evaluate.return_value = {"ok": True}
        self.assertEqual(evaluate_delayed_late_checkpoint(), {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=(DELAYED_LATE_CHECKPOINT_MODEL,),
            policies=LATE_CHECKPOINT_POLICIES,
            examples_per_bucket=200,
            horizons=(128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix=(
                "delayed-damping-late-after2k-checkpoint-n1000"
            ),
        )

    @patch("looping.eval_damping_robustness.evaluate")
    def test_late_checkpoint_replicate_uses_the_same_policy_grid(self, evaluate):
        evaluate.return_value = {"ok": True}
        self.assertEqual(evaluate_late_checkpoint_trial1(), {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=(LATE_CHECKPOINT_TRIAL1_MODEL,),
            policies=LATE_CHECKPOINT_POLICIES,
            examples_per_bucket=200,
            horizons=(128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix="delayed-damping-late-trial1-checkpoint-n1000",
        )

    @patch("looping.eval_damping_robustness.evaluate")
    def test_through_1024_checkpoint_compares_damping_start_times(self, evaluate):
        evaluate.return_value = {"ok": True}
        self.assertEqual(evaluate_late_checkpoint_through_1024(), {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=(LATE_THROUGH_1024_CHECKPOINT_MODEL,),
            policies=LATE_CHECKPOINT_POLICIES,
            examples_per_bucket=200,
            horizons=(128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix="delayed-damping-late-through1024-checkpoint-n1000",
        )

    @patch("looping.eval_delayed_damping_full.evaluate")
    def test_late_checkpoint_full_eval_uses_the_selected_late_policy(
        self,
        evaluate,
    ):
        evaluate.return_value = {"ok": True}
        model_name = LATE_CHECKPOINT_MODEL["name"]
        self.assertEqual(evaluate_damping_full(model_name), {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=(LATE_CHECKPOINT_MODEL,),
            policies=LATE_CHECKPOINT_FULL_POLICIES,
            examples_per_bucket=5000,
            batch_size=250,
            horizons=(16, 128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix=f"delayed-damping-full-{model_name}",
        )

    @patch("looping.eval_delayed_damping_full.evaluate")
    def test_delayed_late_checkpoint_full_eval_skips_repeated_undamped_run(
        self,
        evaluate,
    ):
        evaluate.return_value = {"ok": True}
        model_name = DELAYED_LATE_CHECKPOINT_MODEL["name"]
        self.assertEqual(evaluate_damping_full(model_name), {"ok": True})
        evaluate.assert_called_once_with(
            model_configs=(DELAYED_LATE_CHECKPOINT_MODEL,),
            policies=DELAYED_LATE_CHECKPOINT_FULL_POLICIES,
            examples_per_bucket=5000,
            batch_size=250,
            horizons=(16, 128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix=f"delayed-damping-full-{model_name}",
        )

    @patch("looping.eval_delayed_damping_full.evaluate")
    def test_late_replicate_full_eval_skips_repeated_undamped_run(
        self,
        evaluate,
    ):
        model_name = LATE_CHECKPOINT_TRIAL1_MODEL["name"]
        evaluate_damping_full(model_name)
        evaluate.assert_called_once_with(
            model_configs=(LATE_CHECKPOINT_TRIAL1_MODEL,),
            policies=DELAYED_LATE_CHECKPOINT_FULL_POLICIES,
            examples_per_bucket=5000,
            batch_size=250,
            horizons=(16, 128, 512, 1024, 2048, 4096),
            output_dir=".",
            output_prefix=f"delayed-damping-full-{model_name}",
        )

    def test_matched_schedules_reuse_the_same_two_layers_in_different_orders(self):
        inputs = torch.zeros(1, 1, 128)
        predictions = torch.zeros(1, 1, 9)
        rope = torch.empty(0)

        abab = SudokuTransformer(
            unique_layers=2,
            layer_schedule=(0, 1, 0, 1),
            feedback_scale=0,
        )
        aabb = SudokuTransformer(
            unique_layers=2,
            layer_schedule=(0, 0, 1, 1),
            feedback_scale=0,
        )
        abab.layers = nn.ModuleList([_AddOne(), _Double()])
        aabb.layers = nn.ModuleList([_AddOne(), _Double()])

        abab_output = abab.apply_recurrent_updates(inputs, predictions, rope, rope)
        aabb_output = aabb.apply_recurrent_updates(inputs, predictions, rope, rope)
        torch.testing.assert_close(abab_output, torch.full_like(inputs, 6))
        torch.testing.assert_close(aabb_output, torch.full_like(inputs, 8))

    def test_schedule_pair_has_identical_parameter_count_and_state_dict_shape(self):
        abab = SudokuTransformer(unique_layers=2, layer_schedule=(0, 1, 0, 1))
        aabb = SudokuTransformer(unique_layers=2, layer_schedule=(0, 0, 1, 1))
        self.assertEqual(
            sum(parameter.numel() for parameter in abab.parameters()),
            sum(parameter.numel() for parameter in aabb.parameters()),
        )
        self.assertEqual(abab.state_dict().keys(), aabb.state_dict().keys())

    def test_scaling_constants_follow_the_two_paper_controls(self):
        self.assertEqual(LINEAR_LOOP_SCALE, 1 / 16)
        self.assertEqual(SQRT_LOOP_SCALE, 1 / 4)
        linear = SudokuTransformer(residual_scale=LINEAR_LOOP_SCALE)
        self.assertTrue(all(
            layer.residual_scale == LINEAR_LOOP_SCALE
            for layer in linear.layers
        ))

    def test_arm_configs_are_copied_before_returning(self):
        first = get_arm_config("schedule_abab")
        first["layer_schedule"] = ()
        self.assertEqual(
            get_arm_config("schedule_abab")["layer_schedule"],
            (0, 1, 0, 1),
        )

    def test_invalid_schedule_is_rejected(self):
        with self.assertRaises(ValueError):
            SudokuTransformer(unique_layers=2, layer_schedule=(0, 2))

    def test_recurrent_stage_names_preserve_repeated_layer_order(self):
        model = SudokuTransformer(
            outer_state_norm=True,
            unique_layers=2,
            layer_schedule=(0, 0, 1, 1),
        ).eval()
        hidden_state = torch.zeros(1, 81, 128)
        predictions = torch.zeros(1, 81, 9)
        rope_cos = torch.ones(81, 16)
        rope_sin = torch.zeros(81, 16)
        names, _, _ = recurrent_stages(
            model,
            hidden_state,
            predictions,
            rope_cos,
            rope_sin,
        )
        self.assertEqual(names, [
            "input_state",
            "after_feedback",
            "after_A_1",
            "after_A_2",
            "after_B_1",
            "after_B_2",
            "after_outer_state",
        ])

    def test_gradient_summary_detects_exact_cancellation(self):
        summary = _matrix_summary([
            torch.tensor([1.0, 0.0]),
            torch.tensor([-1.0, 0.0]),
        ])
        self.assertAlmostEqual(summary["minimum_off_diagonal_cosine"], -1.0)
        self.assertAlmostEqual(summary["cancellation_ratio"], 0.0)

    def test_linear_cka_is_invariant_to_orthogonal_coordinate_rotation(self):
        first = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
        ])
        rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
        self.assertAlmostEqual(linear_cka(first, first @ rotation), 1.0, places=6)
        self.assertAlmostEqual(
            linear_cka(first * 1e-4, (first @ rotation) * 1e-4),
            1.0,
            places=6,
        )

    def test_digit_lens_smoke_has_nine_finite_vectors_per_stage(self):
        model = SudokuTransformer(unique_layers=1).eval()
        model.requires_grad_(False)
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        targets = torch.zeros(1, 81, dtype=torch.long)
        empty_mask = torch.ones(1, 81, dtype=torch.bool)
        hidden_state = model.initial_encoder(inputs).detach()
        result, lens_matrices, _ = one_step_digit_lens(
            model,
            hidden_state,
            targets,
            empty_mask,
            0,
            testbed_module.ROPE_COS,
            testbed_module.ROPE_SIN,
        )
        self.assertTrue(torch.isfinite(torch.tensor(result["next_iteration_loss"])))
        for lens_matrix in lens_matrices.values():
            self.assertEqual(lens_matrix.shape, (9, 128))
            self.assertTrue(torch.isfinite(lens_matrix).all())

    def test_parameter_gradient_conflict_smoke_covers_all_iterations(self):
        model = SudokuTransformer(unique_layers=1).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        targets = torch.zeros(1, 81, dtype=torch.long)
        empty_mask = torch.ones(1, 81, dtype=torch.bool)
        result = parameter_gradient_conflict(
            model,
            inputs,
            targets,
            empty_mask,
        )
        self.assertEqual(len(result["losses"]), 16)
        self.assertEqual(
            len(result["groups"]["all"]["cosine_matrix"]),
            16,
        )

    @patch("looping.exp_loop_ablation.train_testbed")
    def test_training_wrapper_passes_only_the_selected_arm(self, train_testbed):
        train_testbed.return_value = {"ok": True}
        result = train(
            "/tmp/output",
            arm="schedule_aabb",
            run_name="matched_trial",
            random_seed=123,
        )
        self.assertEqual(result, {"ok": True})
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_loop_schedule_aabb",
            run_name="matched_trial",
            random_seed=123,
            checkpoint_on_probe=True,
            outer_state_norm=True,
            unique_layers=2,
            layer_schedule=(0, 0, 1, 1),
        )

    @patch("looping.exp_late_supervision.train_testbed")
    def test_late_supervision_wrapper_passes_the_selected_arm(self, train_testbed):
        train_testbed.return_value = {"ok": True}
        result = train_late_supervision(
            "/tmp/output",
            arm="random_aux",
            run_name="late_trial",
            random_seed=456,
        )
        self.assertEqual(result, {"ok": True})
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_loop_late_random_aux",
            run_name="late_trial",
            random_seed=456,
            checkpoint_on_probe=True,
            late_supervision_horizons=(32, 64, 128, 256, 512),
            late_supervision_probability=0.2,
            late_supervision_mix=0.5,
        )

    @patch("looping.exp_late_supervision.train_testbed")
    def test_delayed_late_supervision_arm_records_its_start_step(
        self,
        train_testbed,
    ):
        train_late_supervision(
            "/tmp/output",
            arm="random_aux_after_2k",
            run_name="delayed_late_trial",
            random_seed=457,
        )
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_loop_late_random_aux_after_2k",
            run_name="delayed_late_trial",
            random_seed=457,
            checkpoint_on_probe=True,
            late_supervision_horizons=(32, 64, 128, 256, 512),
            late_supervision_probability=0.2,
            late_supervision_mix=0.5,
            late_supervision_start_step=2000,
        )

    @patch("looping.exp_late_supervision.train_testbed")
    def test_delayed_replacement_arm_uses_only_the_late_loss_on_sampled_steps(
        self,
        train_testbed,
    ):
        train_late_supervision(
            "/tmp/output",
            arm="random_replace_after_2k",
            run_name="delayed_replace_trial",
            random_seed=458,
        )
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_loop_late_random_replace_after_2k",
            run_name="delayed_replace_trial",
            random_seed=458,
            checkpoint_on_probe=True,
            late_supervision_horizons=(32, 64, 128, 256, 512),
            late_supervision_probability=0.2,
            late_supervision_mix=1.0,
            late_supervision_start_step=2000,
        )

    @patch("looping.exp_late_supervision.train_testbed")
    def test_extreme_timing_arms_use_the_same_every_batch_auxiliary_loss(
        self,
        train_testbed,
    ):
        train_late_supervision(
            "/tmp/output",
            arm="random_aux_p100",
            run_name="early_extreme",
            random_seed=459,
        )
        early_settings = train_testbed.call_args.kwargs

        train_testbed.reset_mock()
        train_late_supervision(
            "/tmp/output",
            arm="random_aux_p100_after_10k",
            run_name="late_extreme",
            random_seed=459,
        )
        late_settings = train_testbed.call_args.kwargs

        shared_keys = (
            "late_supervision_horizons",
            "late_supervision_probability",
            "late_supervision_mix",
        )
        self.assertEqual(
            {key: early_settings[key] for key in shared_keys},
            {key: late_settings[key] for key in shared_keys},
        )
        self.assertEqual(early_settings["late_supervision_probability"], 1.0)
        self.assertEqual(early_settings["late_supervision_mix"], 0.5)
        self.assertNotIn("late_supervision_start_step", early_settings)
        self.assertEqual(late_settings["late_supervision_start_step"], 10000)

    def test_late_supervision_configs_are_copied_before_returning(self):
        first = get_late_supervision_config("random_aux")
        first["late_supervision_horizons"] = ()
        self.assertEqual(
            get_late_supervision_config("random_aux")[
                "late_supervision_horizons"
            ],
            (32, 64, 128, 256, 512),
        )

    def test_late_replacement_rmsnorm_combines_the_two_stabilizers(self):
        config = get_late_supervision_config("random_replace_rmsnorm")
        self.assertTrue(config["outer_state_norm"])
        self.assertEqual(config["late_supervision_mix"], 1.0)
        self.assertEqual(config.get("late_supervision_start_step", 0), 0)

    def test_dense_late_replacement_changes_only_the_sampling_rate(self):
        baseline = get_late_supervision_config("random_replace")
        dense = get_late_supervision_config("random_replace_p50")
        self.assertEqual(dense["late_supervision_probability"], 0.5)
        self.assertEqual(
            {
                key: value
                for key, value in dense.items()
                if key != "late_supervision_probability"
            },
            {
                key: value
                for key, value in baseline.items()
                if key != "late_supervision_probability"
            },
        )

    def test_aabb_late_replacement_combines_schedule_and_late_states(self):
        config = get_late_supervision_config("random_replace_aabb")
        self.assertTrue(config["outer_state_norm"])
        self.assertEqual(config["unique_layers"], 2)
        self.assertEqual(config["layer_schedule"], (0, 0, 1, 1))
        self.assertEqual(config["late_supervision_probability"], 0.2)

    def test_fixed_late_replacement_targets_the_healthy_boundary(self):
        config = get_late_supervision_config("fixed128_replace")
        self.assertEqual(config["late_supervision_horizons"], (128,))
        self.assertEqual(config["late_supervision_probability"], 0.2)
        self.assertEqual(config["late_supervision_mix"], 1.0)

    def test_long_late_replacement_extends_the_sampled_state_support(self):
        config = get_late_supervision_config("random_replace_through_1024")
        self.assertEqual(
            config["late_supervision_horizons"],
            (32, 64, 128, 256, 512, 1024),
        )
        self.assertEqual(config["late_supervision_probability"], 0.2)
        self.assertEqual(config["late_supervision_mix"], 1.0)

    @patch("looping.eval_late_supervision_full.evaluate")
    def test_late_full_evaluation_uses_the_best_probe_checkpoint(self, evaluate):
        evaluate_late_run(
            "late_trial",
            outer_state_norm=True,
            examples_per_bucket=12,
            output_dir="/tmp/output",
        )
        evaluate.assert_called_once_with(
            model_configs=((
                "late_trial_best",
                "/outputs/looping/model_late_trial_best_probe.pt",
            ),),
            experiment_module="stabilize.exp_testbed_20k",
            caps=(None,),
            checkpoints=(16, 128, 1024, 2048),
            examples_per_bucket=12,
            batch_size=250,
            seed=42,
            output_dir="/tmp/output",
            output_prefix="late_trial_best_full_horizon",
            model_kwargs={"outer_state_norm": True},
        )

    @patch("looping.eval_late_supervision_full.evaluate")
    def test_late_full_evaluation_restores_arm_architecture(self, evaluate):
        evaluate_late_run(
            "late_aabb_trial",
            arm="random_replace_aabb",
            examples_per_bucket=12,
            output_dir="/tmp/output",
        )
        self.assertEqual(
            evaluate.call_args.kwargs["model_kwargs"],
            {
                "outer_state_norm": True,
                "unique_layers": 2,
                "layer_schedule": (0, 0, 1, 1),
            },
        )

    @patch("looping.eval_late_recipe.evaluate")
    def test_recommended_late_recipe_uses_confirmed_damping_policy(self, evaluate):
        evaluate_late_recipe(
            "/tmp/model.pt",
            model_name="recommended_trial",
            examples_per_bucket=12,
            batch_size=7,
            horizons=(128, 1024),
            output_dir="/tmp/output",
            output_prefix="recommended_eval",
        )
        evaluate.assert_called_once_with(
            model_configs=({
                "name": "recommended_trial",
                "path": "/tmp/model.pt",
                "model_kwargs": {},
            },),
            policies=RECOMMENDED_POLICY,
            examples_per_bucket=12,
            batch_size=7,
            horizons=(128, 1024),
            output_dir="/tmp/output",
            output_prefix="recommended_eval",
        )

    def test_late_recipe_can_name_a_nearby_damping_policy(self):
        self.assertEqual(
            make_damping_policy(alpha=0.5, warmup_iterations=512),
            {
                "name": "warm512_a05",
                "alpha": 0.5,
                "warmup_iterations": 512,
            },
        )
        with self.assertRaises(ValueError):
            make_damping_policy(alpha=0, warmup_iterations=512)

    @patch("looping.exp_scaled_n64.train_testbed")
    def test_scaled_n64_wrapper_matches_its_training_loop_count(self, train_testbed):
        train_scaled_n64(
            "/tmp/output",
            run_name="n64_trial",
            random_seed=789,
        )
        self.assertEqual(N64_TRAINING_ITERATIONS, 64)
        self.assertEqual(N64_LINEAR_SCALE, 1 / 64)
        self.assertEqual(N64_RUN_BATCH_SIZE, 2048)
        self.assertEqual(N64_MICROBATCH_SIZE, 1024)
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_loop_scaled_n64",
            run_name="n64_trial",
            random_seed=789,
            checkpoint_on_probe=True,
            training_iterations=64,
            run_batch_size=2048,
            microbatch_size=1024,
            residual_scale=1 / 64,
            feedback_scale=1 / 64,
        )

    @patch("looping.eval_scaled_n64_full.evaluate")
    def test_scaled_n64_full_eval_restores_the_scaled_recurrence(self, evaluate):
        evaluate_scaled_n64(
            "n64_trial",
            examples_per_bucket=12,
            output_dir="/tmp/output",
        )
        evaluate.assert_called_once_with(
            model_configs=((
                "n64_trial_best",
                "/outputs/looping/model_n64_trial_best_probe.pt",
            ),),
            experiment_module="stabilize.exp_testbed_20k",
            caps=(None,),
            checkpoints=(16, 64, 128, 1024, 2048),
            examples_per_bucket=12,
            batch_size=250,
            seed=42,
            output_dir="/tmp/output",
            output_prefix="n64_trial_best_full_horizon",
            model_kwargs={
                "training_iterations": 64,
                "residual_scale": 1 / 64,
                "feedback_scale": 1 / 64,
            },
        )


if __name__ == "__main__":
    unittest.main()
