import unittest
from unittest.mock import patch

import torch

from looping.exp_stay_solved import (
    CLEAN_REPLACEMENT,
    FULL_50K_SCHEDULE,
    FULL_50K_SUFFIX as EXPERIMENT_FULL_50K_SUFFIX,
    HORIZON_START_STEPS,
    LATE_HORIZONS,
    RECHECK_GAPS,
    SCREEN_SCHEDULE,
    SCREEN_SUFFIX as EXPERIMENT_SCREEN_SUFFIX,
    STAGED_RAMP_STEPS,
    STAGED_START_STEP,
    STAY_SOLVED_CONFIGS,
    get_stay_solved_config,
    train as train_stay_solved,
)
from looping.modal_stay_solved import (
    ARM_NAMES,
    FULL_50K_SUFFIX as MODAL_FULL_50K_SUFFIX,
    SCREEN_SUFFIX as MODAL_SCREEN_SUFFIX,
)
from looping.modal_late_switch import (
    MODES as LATE_SWITCH_MODES,
    SOURCE_CHECKPOINT as LATE_SWITCH_SOURCE,
    SOURCE_STEP as LATE_SWITCH_SOURCE_STEP,
)
from stabilize.exp_testbed_20k import (
    SudokuTransformer,
    correct_prediction_consistency_loss,
    solved_puzzle_margin_floor_loss,
    late_supervision_plan_for_step,
    resolve_late_recheck,
    resolve_late_supervision_timing,
)


class StaySolvedTest(unittest.TestCase):
    def test_forward_can_return_the_exact_final_recurrent_state(self):
        torch.manual_seed(17)
        model = SudokuTransformer(unique_layers=1, training_iterations=3).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        with torch.no_grad():
            logits, (hidden_state, predictions) = model(
                inputs,
                return_all=True,
                return_state=True,
            )
        self.assertEqual(len(logits), 3)
        torch.testing.assert_close(
            model.output_head(hidden_state),
            logits[-1],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            predictions,
            torch.softmax(logits[-1], dim=-1),
            rtol=0,
            atol=0,
        )

    def test_staged_plan_ramps_probability_and_unlocks_horizons(self):
        self.assertEqual(
            late_supervision_plan_for_step(
                1999,
                LATE_HORIZONS,
                HORIZON_START_STEPS,
                0.2,
                STAGED_START_STEP,
                STAGED_RAMP_STEPS,
            ),
            ((), 0.0),
        )
        horizons, probability = late_supervision_plan_for_step(
            3000,
            LATE_HORIZONS,
            HORIZON_START_STEPS,
            0.2,
            STAGED_START_STEP,
            STAGED_RAMP_STEPS,
        )
        self.assertEqual(horizons, (32, 64))
        self.assertAlmostEqual(probability, 0.05)
        horizons, probability = late_supervision_plan_for_step(
            6000,
            LATE_HORIZONS,
            HORIZON_START_STEPS,
            0.2,
            STAGED_START_STEP,
            STAGED_RAMP_STEPS,
        )
        self.assertEqual(horizons, LATE_HORIZONS)
        self.assertAlmostEqual(probability, 0.2)

    def test_screen_preserves_the_healthy_delayed_start_window(self):
        first_phase = SCREEN_SCHEDULE["phases"][0]
        self.assertEqual(SCREEN_SCHEDULE["warmup_steps"], 560)
        self.assertLess(STAGED_START_STEP, first_phase[1])
        self.assertEqual(first_phase[:3], (0, 4000, 21))

    def test_late_timing_validation_rejects_misaligned_metadata(self):
        self.assertEqual(
            resolve_late_supervision_timing(
                LATE_HORIZONS,
                STAGED_START_STEP,
                STAGED_RAMP_STEPS,
                HORIZON_START_STEPS,
            ),
            HORIZON_START_STEPS,
        )
        with self.assertRaises(ValueError):
            resolve_late_supervision_timing(
                LATE_HORIZONS,
                STAGED_START_STEP,
                STAGED_RAMP_STEPS,
                HORIZON_START_STEPS[:-1],
            )
        with self.assertRaises(ValueError):
            resolve_late_supervision_timing(
                LATE_HORIZONS,
                STAGED_START_STEP,
                STAGED_RAMP_STEPS,
                tuple(reversed(HORIZON_START_STEPS)),
            )

    def test_recheck_settings_are_validated_together(self):
        self.assertEqual(
            resolve_late_recheck(RECHECK_GAPS, 0.5, 0.1, True),
            RECHECK_GAPS,
        )
        self.assertEqual(
            resolve_late_recheck(RECHECK_GAPS, 0.0, 0.1, True),
            RECHECK_GAPS,
        )
        self.assertEqual(
            resolve_late_recheck(
                RECHECK_GAPS,
                0.0,
                0.0,
                True,
                margin_floor_weight=0.1,
                margin_floor=5.0,
            ),
            RECHECK_GAPS,
        )
        with self.assertRaises(ValueError):
            resolve_late_recheck((), 0.0, 0.1, True)
        with self.assertRaises(ValueError):
            resolve_late_recheck(RECHECK_GAPS, 0.0, 0.0, True)
        with self.assertRaises(ValueError):
            resolve_late_recheck((17,), 0.5, 0.0, True)
        with self.assertRaises(ValueError):
            resolve_late_recheck(RECHECK_GAPS, 0.5, 0.0, False)

    def test_consistency_only_anchors_currently_correct_empty_cells(self):
        anchor_logits = torch.zeros(1, 2, 9)
        anchor_logits[0, 0, 0] = 5
        anchor_logits[0, 1, 1] = 5
        targets = torch.zeros(1, 2, dtype=torch.long)
        mask = torch.ones(1, 2)

        unchanged = anchor_logits.clone().requires_grad_()
        unchanged_loss = correct_prediction_consistency_loss(
            anchor_logits,
            unchanged,
            targets,
            mask,
        )
        self.assertAlmostEqual(unchanged_loss.item(), 0.0, places=6)

        changed = anchor_logits.clone()
        changed[0, 0, 0] = 0
        changed[0, 0, 2] = 5
        changed.requires_grad_()
        changed_loss = correct_prediction_consistency_loss(
            anchor_logits,
            changed,
            targets,
            mask,
        )
        self.assertGreater(changed_loss.item(), 1.0)
        changed_loss.backward()
        self.assertGreater(changed.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(changed.grad[0, 1].abs().sum().item(), 0)

    def test_margin_floor_uses_weakest_cell_of_solved_anchor_puzzles(self):
        anchor_logits = torch.zeros(2, 2, 9)
        anchor_logits[0, :, 0] = 5
        anchor_logits[1, 0, 0] = 5
        anchor_logits[1, 1, 1] = 5
        targets = torch.zeros(2, 2, dtype=torch.long)
        mask = torch.ones(2, 2)
        future_logits = anchor_logits.clone()
        future_logits[0, 1, 0] = 0.25
        future_logits.requires_grad_()

        loss = solved_puzzle_margin_floor_loss(
            anchor_logits,
            future_logits,
            targets,
            mask,
            margin_floor=1.0,
        )
        self.assertAlmostEqual(loss.item(), 0.75, places=6)
        loss.backward()
        self.assertGreater(
            future_logits.grad[0, 1].abs().sum().item(),
            0,
        )
        self.assertEqual(
            future_logits.grad[0, 0].abs().sum().item(),
            0,
        )
        self.assertEqual(
            future_logits.grad[1].abs().sum().item(),
            0,
        )

    def test_detached_recheck_window_backpropagates_without_the_first_graph(self):
        torch.manual_seed(23)
        model = SudokuTransformer(unique_layers=1, training_iterations=2).eval()
        inputs = torch.zeros(1, 81, 10)
        inputs[:, :, 0] = 1
        targets = torch.zeros(1, 81, dtype=torch.long)
        first_logits, first_state = model(
            inputs,
            return_all=True,
            return_state=True,
        )
        detached_state = tuple(value.detach() for value in first_state)
        recheck_logits = model(
            inputs,
            return_all=True,
            initial_state=detached_state,
        )
        recheck_ce = torch.nn.functional.cross_entropy(
            recheck_logits[-1].reshape(-1, 9),
            targets.reshape(-1),
        )
        consistency = correct_prediction_consistency_loss(
            first_logits[-1],
            recheck_logits[-1],
            targets,
            inputs[:, :, 0],
        )
        (recheck_ce + 0.1 * consistency).backward()
        gradient_norm = sum(
            parameter.grad.abs().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(gradient_norm, 0)
        self.assertFalse(detached_state[0].requires_grad)

    def test_experiment_presets_keep_controls_separate(self):
        self.assertEqual(set(ARM_NAMES), set(STAY_SOLVED_CONFIGS))
        self.assertEqual(EXPERIMENT_SCREEN_SUFFIX, MODAL_SCREEN_SUFFIX)
        self.assertEqual(
            EXPERIMENT_FULL_50K_SUFFIX,
            MODAL_FULL_50K_SUFFIX,
        )
        control = get_stay_solved_config("control")
        late_state_ce = get_stay_solved_config("late_state_ce")
        clean = get_stay_solved_config("clean_curriculum")
        stay = get_stay_solved_config("stay_consistency")
        margin = get_stay_solved_config("stay_margin_floor")
        self.assertNotIn("late_recheck_gaps", control)
        self.assertEqual(late_state_ce, control)
        self.assertEqual(
            clean["late_supervision_horizon_start_steps"],
            HORIZON_START_STEPS,
        )
        self.assertEqual(
            clean["late_supervision_probability_ramp_steps"],
            STAGED_RAMP_STEPS,
        )
        self.assertEqual(stay["late_recheck_gaps"], RECHECK_GAPS)
        self.assertEqual(stay["late_consistency_weight"], 0.1)
        self.assertEqual(margin["late_margin_floor_weight"], 0.1)
        self.assertEqual(margin["late_margin_floor"], 5.0)

    def test_configs_are_copied_before_returning(self):
        config = get_stay_solved_config("clean_curriculum")
        config["late_supervision_horizons"] = ()
        self.assertEqual(
            get_stay_solved_config("clean_curriculum")[
                "late_supervision_horizons"
            ],
            LATE_HORIZONS,
        )

    @patch("looping.exp_stay_solved.train_testbed")
    def test_screen_wrapper_passes_the_short_matched_schedule(self, train_testbed):
        train_testbed.return_value = {"ok": True}
        result = train_stay_solved(
            "/tmp/output",
            arm="clean_curriculum",
            run_name="clean_screen",
            random_seed=123,
            screen=True,
        )
        self.assertEqual(result, {"ok": True})
        expected_settings = dict(CLEAN_REPLACEMENT)
        expected_settings["schedule"] = SCREEN_SCHEDULE
        train_testbed.assert_called_once_with(
            output_dir="/tmp/output",
            experiment_name="exp_loop_stay_clean_curriculum",
            run_name="clean_screen",
            random_seed=123,
            checkpoint_on_probe=True,
            **expected_settings,
        )

    @patch("looping.exp_stay_solved.train_testbed")
    def test_corrected_screen_uses_a_fresh_run_name(self, train_testbed):
        train_stay_solved(
            "/tmp/output",
            arm="control",
            screen=True,
        )
        self.assertEqual(
            train_testbed.call_args.kwargs["run_name"],
            "loop_stay_control_healthy_screen_trial0",
        )

    @patch("looping.exp_stay_solved.train_testbed")
    def test_full_wrapper_does_not_override_the_20k_schedule(self, train_testbed):
        train_stay_solved(
            "/tmp/output",
            arm="stay_recheck",
            run_name="stay_full",
            random_seed=124,
            screen=False,
        )
        self.assertNotIn("schedule", train_testbed.call_args.kwargs)
        self.assertEqual(
            train_testbed.call_args.kwargs["late_recheck_gaps"],
            RECHECK_GAPS,
        )

    @patch("looping.exp_stay_solved.train_testbed")
    def test_50k_wrapper_passes_the_historical_schedule(self, train_testbed):
        train_stay_solved(
            "/tmp/output",
            arm="stay_consistency",
            random_seed=125,
            full_50k=True,
        )
        call_settings = train_testbed.call_args.kwargs
        self.assertEqual(call_settings["schedule"], FULL_50K_SCHEDULE)
        self.assertEqual(
            call_settings["run_name"],
            "loop_stay_stay_consistency_50k_trial0",
        )
        self.assertEqual(FULL_50K_SCHEDULE["warmup_steps"], 1400)
        self.assertEqual(FULL_50K_SCHEDULE["total_steps"], 50000)
        self.assertEqual(
            tuple((start, end, rating) for start, end, rating, _ in
                  FULL_50K_SCHEDULE["phases"]),
            (
                (0, 10000, 21),
                (10000, 20000, 6),
                (20000, 30000, 1),
                (30000, 50000, 0),
            ),
        )

    def test_screen_and_50k_modes_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "cannot both"):
            train_stay_solved(
                "/tmp/output",
                screen=True,
                full_50k=True,
            )

    @patch("looping.exp_stay_solved.train_testbed")
    def test_branch_wrapper_forwards_the_source_checkpoint(self, train_testbed):
        train_stay_solved(
            "/tmp/output",
            arm="stay_consistency",
            run_name="late_switch",
            full_50k=True,
            branch_checkpoint_path="/tmp/source_step39000.pt",
        )
        call_settings = train_testbed.call_args.kwargs
        self.assertEqual(
            call_settings["branch_checkpoint_path"],
            "/tmp/source_step39000.pt",
        )
        self.assertEqual(call_settings["schedule"], FULL_50K_SCHEDULE)

    def test_late_switch_uses_the_selected_control_checkpoint(self):
        self.assertEqual(
            set(LATE_SWITCH_MODES),
            {"plain", "consistency", "margin_floor5"},
        )
        self.assertEqual(LATE_SWITCH_SOURCE_STEP, 39000)
        self.assertTrue(
            LATE_SWITCH_SOURCE.endswith(
                "loop_stay_control_50k_trial0_checkpoint_step39000.pt"
            )
        )


if __name__ == "__main__":
    unittest.main()
