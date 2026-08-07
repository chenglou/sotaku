import unittest

from looping.exp_health_methods import (
    FUTURE_AUXILIARY,
    HEALTH_METHOD_CONFIGS,
    MARGIN_CAP_SPECS,
    get_health_method_config,
)
from looping.exp_stay_solved import LATE_HORIZONS, RECHECK_GAPS
from looping.modal_health_methods import ARM_NAMES


class HealthMethodsTest(unittest.TestCase):
    def test_matrix_contains_one_mechanism_per_arm(self):
        self.assertEqual(
            set(HEALTH_METHOD_CONFIGS),
            {
                "vanilla",
                "rmsnorm",
                "late_state_ce",
                "consistency_only",
                "margin_only",
                "margin_cap80",
                "margin_cap128",
                "margin_cap192",
            },
        )
        self.assertEqual(set(ARM_NAMES), set(HEALTH_METHOD_CONFIGS))
        self.assertEqual(HEALTH_METHOD_CONFIGS["vanilla"], {})
        self.assertEqual(
            HEALTH_METHOD_CONFIGS["rmsnorm"],
            {"outer_state_norm": True},
        )

    def test_auxiliary_arms_keep_late_cross_entropy_disabled(self):
        self.assertEqual(
            FUTURE_AUXILIARY["late_supervision_horizons"],
            LATE_HORIZONS,
        )
        self.assertEqual(
            FUTURE_AUXILIARY["late_recheck_gaps"],
            RECHECK_GAPS,
        )
        self.assertTrue(FUTURE_AUXILIARY["late_auxiliary_only"])
        self.assertEqual(
            FUTURE_AUXILIARY["late_recheck_loss_weight"],
            0.0,
        )

        consistency = get_health_method_config("consistency_only")
        margin = get_health_method_config("margin_only")
        self.assertEqual(consistency["late_consistency_weight"], 0.1)
        self.assertNotIn("late_margin_floor_weight", consistency)
        self.assertEqual(margin["late_margin_floor_weight"], 0.1)
        self.assertNotIn("late_consistency_weight", margin)

    def test_margin_caps_limit_the_latest_trained_state(self):
        expected_caps = {
            "margin_cap80": 80,
            "margin_cap128": 128,
            "margin_cap192": 192,
        }
        for arm, expected_cap in expected_caps.items():
            spec = MARGIN_CAP_SPECS[arm]
            actual_cap = (
                max(spec["late_supervision_horizons"])
                + 16
                + max(spec["late_recheck_gaps"])
                + 16
            )
            self.assertEqual(actual_cap, expected_cap)

            config = get_health_method_config(arm)
            self.assertEqual(
                config["late_supervision_horizons"],
                spec["late_supervision_horizons"],
            )
            self.assertEqual(
                config["late_recheck_gaps"],
                spec["late_recheck_gaps"],
            )
            self.assertTrue(config["late_auxiliary_only"])
            self.assertEqual(config["late_recheck_loss_weight"], 0.0)
            self.assertEqual(config["late_margin_floor_weight"], 0.1)

    def test_configs_are_copied(self):
        config = get_health_method_config("margin_only")
        config["late_margin_floor"] = 99
        self.assertEqual(
            get_health_method_config("margin_only")["late_margin_floor"],
            5.0,
        )

    def test_unknown_arm_is_rejected(self):
        with self.assertRaises(ValueError):
            get_health_method_config("unknown")

if __name__ == "__main__":
    unittest.main()
