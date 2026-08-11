import importlib.util
import json
import math
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).with_name("causal_axes.py")
SPEC = importlib.util.spec_from_file_location("causal_axes", MODULE_PATH)
causal_axes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(causal_axes)


class CausalAxesTests(unittest.TestCase):
    def test_balanced_split_assignment(self):
        buckets = []
        for _, _, name in causal_axes.RATING_BUCKETS:
            buckets.extend([name] * 12)
        assignments = causal_axes.assign_balanced_splits(buckets, 4)
        for split_name in causal_axes.SPLIT_NAMES:
            selected = [
                bucket for bucket, split in zip(buckets, assignments)
                if split == split_name
            ]
            self.assertEqual(len(selected), 20)
            self.assertEqual({name: selected.count(name) for name in set(selected)}, {
                name: 4 for _, _, name in causal_axes.RATING_BUCKETS
            })

    def test_ridge_axis_recovers_group_residual_signal(self):
        generator = torch.Generator().manual_seed(7)
        true_axis = torch.randn(12, generator=generator)
        true_axis = true_axis / true_axis.norm()
        features = torch.randn(400, 12, generator=generator)
        groups = torch.arange(4).repeat_interleave(100)
        group_offsets = torch.tensor([-3.0, -1.0, 2.0, 5.0])
        labels = features @ true_axis + group_offsets[groups]
        fitted = causal_axes.fit_ridge_axis(features, labels, groups, 0.01)
        self.assertGreater(float(fitted @ true_axis), 0.98)

    def test_answer_evidence_direction_increases_its_margin(self):
        generator = torch.Generator().manual_seed(11)
        weights = torch.randn(9, 16, generator=generator)
        directions = causal_axes.answer_evidence_axes(weights)
        for digit in range(9):
            others = torch.cat((weights[:digit], weights[digit + 1:])).mean(0)
            gain = (weights[digit] - others) @ directions[digit]
            self.assertGreater(float(gain), 0.0)

    def test_perturbation_is_exactly_norm_matched_and_signed(self):
        generator = torch.Generator().manual_seed(13)
        update = torch.randn(3, 5, 7, generator=generator)
        mask = torch.tensor([
            [1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1],
            [0, 1, 1, 1, 1],
        ], dtype=torch.bool)
        direction = torch.randn(7, generator=generator)
        direction = direction / direction.norm()
        positive = causal_axes.make_perturbation(update, mask, direction, 0.25)
        negative = causal_axes.make_perturbation(update, mask, direction, -0.25)
        update_norm = (update * mask[:, :, None]).flatten(1).norm(dim=1)
        perturb_norm = positive.flatten(1).norm(dim=1)
        self.assertTrue(torch.allclose(perturb_norm, 0.25 * update_norm, atol=1e-6))
        self.assertTrue(torch.allclose(positive, -negative, atol=1e-7))
        self.assertTrue((positive[~mask] == 0).all())

    def test_random_controls_are_orthogonal_and_deterministic(self):
        real = torch.arange(1, 9, dtype=torch.float32)
        real = real / real.norm()
        first = causal_axes.random_global_axes(real, 5, 17)
        second = causal_axes.random_global_axes(real, 5, 17)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.allclose(first @ real, torch.zeros(5), atol=1e-6))
        self.assertTrue(torch.allclose(first.norm(dim=1), torch.ones(5), atol=1e-6))

    def test_completed_artifacts_cover_the_frozen_protocol(self):
        metrics_path = MODULE_PATH.with_name("final_metrics.json")
        if not metrics_path.exists():
            self.skipTest("full local analysis has not been run")
        with open(metrics_path) as metrics_file:
            metrics = json.load(metrics_file)
        self.assertEqual(set(metrics["models"]), {
            "stable_plain", "collapsed_plain", "late_state_ce", "combined_margin"
        })
        self.assertEqual(metrics["split_audit"]["final_count"], 20)
        self.assertTrue(metrics["split_audit"]["selection_frozen_before_final"])
        self.assertEqual(metrics["config"]["random_directions"], 8)
        self.assertEqual(metrics["config"]["doses"], list(causal_axes.DOSES))
        for model in metrics["models"].values():
            for window_name in causal_axes.INTERVENTION_WINDOWS:
                for axis_name in ("answer_evidence", "solvedness_progress"):
                    records = model["interventions"][window_name][axis_name]
                    self.assertEqual(set(records), {str(dose) for dose in causal_axes.DOSES})
                    for record in records.values():
                        controls = record["random_controls"]["endpoint"]["solved"]
                        self.assertEqual(len(controls["direction_mean_effects"]), 8)

        def assert_finite(value):
            if isinstance(value, dict):
                for child in value.values():
                    assert_finite(child)
            elif isinstance(value, list):
                for child in value:
                    assert_finite(child)
            elif isinstance(value, float):
                self.assertTrue(math.isfinite(value))

        assert_finite(metrics)

    def test_completed_pngs_are_nonempty(self):
        for filename in (
            "dose_response.png", "immediate_semantics.png", "heldout_probes.png"
        ):
            path = MODULE_PATH.with_name(filename)
            if not path.exists():
                self.skipTest("plots have not been generated")
            self.assertGreater(path.stat().st_size, 50_000)
            self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
