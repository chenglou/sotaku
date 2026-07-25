import unittest

import torch

from es.cdrge_9x9 import apply_rademacher_, cdrge_step


class CDRGETest(unittest.TestCase):
    def test_antithetic_probes_restore_parameters(self):
        parameter = torch.nn.Parameter(torch.tensor([0.25, -0.5, 1.0]))
        original = parameter.detach().clone()

        apply_rademacher_([parameter], base_seed=17, scale=0.1)
        apply_rademacher_([parameter], base_seed=17, scale=-0.2)
        apply_rademacher_([parameter], base_seed=17, scale=0.1)

        torch.testing.assert_close(parameter, original, rtol=0, atol=1e-7)

    def test_tied_radius_step_matches_one_dimensional_quadratic(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0]))

        def loss():
            return float((0.5 * parameter.square()).item())

        losses_plus, losses_minus, update_norm = cdrge_step(
            [parameter],
            loss,
            direction_seeds=[9],
            epsilon=0.1,
        )

        self.assertAlmostEqual(abs(losses_plus[0] - losses_minus[0]), 0.4, places=5)
        self.assertAlmostEqual(update_norm, 0.2, places=5)
        torch.testing.assert_close(parameter, torch.tensor([1.8]), rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
