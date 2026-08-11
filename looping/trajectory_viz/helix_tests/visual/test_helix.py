import math
import unittest

import torch

from looping.trajectory_viz.helix_tests.visual.analyze_helix import (
    flatten_blank_split,
)
from looping.trajectory_viz.helix_tests.visual.helix_geometry import (
    NATURAL_ORDER,
    analyze_periodic_readout,
    decoder_row_basis,
    fit_pca,
    intrinsic_digit_geometry,
    periodic_code,
    remove_subspace,
    stratified_split_indices,
    unique_cycle_orders,
)


class HelixGeometryTest(unittest.TestCase):
    def test_unique_cycles_are_rotation_and_reflection_reduced(self):
        cycles = unique_cycle_orders()
        self.assertEqual(len(cycles), math.factorial(8) // 2)
        self.assertIn(NATURAL_ORDER, cycles)
        self.assertEqual(len(set(cycles)), len(cycles))

    def test_periodic_code_follows_requested_cycle(self):
        order = (0, 2, 4, 6, 8, 7, 5, 3, 1)
        code = periodic_code(order)
        for position, digit in enumerate(order):
            expected = torch.tensor(
                [
                    math.cos(2 * math.pi * position / 9),
                    math.sin(2 * math.pi * position / 9),
                ]
            )
            self.assertTrue(torch.allclose(code[digit], expected, atol=1e-6))

    def test_stratified_split_keeps_every_bucket_in_each_split(self):
        buckets = [bucket for bucket in ("easy", "hard") for _ in range(8)]
        splits = stratified_split_indices(buckets, seed=7)
        for split in splits.values():
            present = {buckets[index] for index in split}
            self.assertEqual(present, {"easy", "hard"})
        combined = sorted(index for split in splits.values() for index in split)
        self.assertEqual(combined, list(range(16)))

    def test_pca_fit_is_applied_without_refitting(self):
        generator = torch.Generator().manual_seed(4)
        plane = torch.randn(200, 2, generator=generator)
        train = torch.cat((plane, torch.zeros(200, 3)), dim=1)
        test = train + torch.tensor([1.0, -2.0, 0.0, 0.0, 0.0])
        projection = fit_pca(train, component_count=3)
        self.assertGreater(projection.captured_fraction(test, 2), 0.999)
        self.assertEqual(projection.project(test).shape, (200, 3))

    def test_output_head_span_removal_is_orthogonal(self):
        generator = torch.Generator().manual_seed(3)
        output_weight = torch.randn(9, 16, generator=generator)
        basis = decoder_row_basis(output_weight)
        values = torch.randn(40, 16, generator=generator)
        residual = remove_subspace(values, basis)
        self.assertLess((residual @ basis).abs().max().item(), 2e-5)

    def test_periodic_readout_generalizes_linear_digit_code(self):
        generator = torch.Generator().manual_seed(5)
        labels = torch.arange(9).repeat_interleave(30)
        prototypes = torch.randn(9, 18, generator=generator)

        def sample(noise_seed):
            local = torch.Generator().manual_seed(noise_seed)
            return prototypes[labels] + 0.04 * torch.randn(
                len(labels), 18, generator=local
            )

        train = sample(10)
        validation = sample(11)
        test = sample(12)
        geometry = intrinsic_digit_geometry(train, labels, test, labels)
        shortest = tuple(digit - 1 for digit in geometry["train_shortest_cycle"])
        summary, _, _ = analyze_periodic_readout(
            train,
            labels,
            validation,
            labels,
            test,
            labels,
            torch.arange(len(labels)) // 30,
            torch.zeros(len(labels), dtype=torch.long),
            shortest,
            seed=9,
        )
        self.assertGreater(
            summary["test_natural_cycle"]["sector_accuracy"], 0.98
        )

    def test_ideal_circle_has_a_short_natural_cycle(self):
        code = periodic_code()
        labels = torch.arange(9)
        geometry = intrinsic_digit_geometry(code, labels, code, labels)
        self.assertLessEqual(geometry["natural_cycle_shorter_percentile"], 0.001)

    def test_flattening_keeps_only_originally_blank_cells(self):
        states = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4).float()
        updates = torch.ones_like(states)
        logits = torch.zeros(2, 2, 3, 9)
        logits[..., 0] = 1
        capture = {
            "states": states,
            "updates": updates,
            "logits": logits,
            "iterations": torch.tensor([0, 1]),
        }
        targets = torch.zeros(2, 3, dtype=torch.long)
        empty = torch.tensor([[True, False, True], [False, False, True]])
        flattened = flatten_blank_split(capture, targets, empty, [0, 1])
        self.assertEqual(flattened["state"].shape, (6, 4))
        self.assertTrue(torch.equal(flattened["cell"], torch.tensor([0, 2, 0, 2, 2, 2])))


if __name__ == "__main__":
    unittest.main()
