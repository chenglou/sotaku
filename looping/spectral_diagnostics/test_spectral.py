import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence
from torch.nn.attention import SDPBackend, sdpa_kernel

from looping.spectral_diagnostics.core import (
    JacobianOperator, derivative_checks, estimate_eigenvalues, finite_difference_power, unit_directions,
)


class SpectralTests(unittest.TestCase):
    def test_known_diagonal(self):
        matrix = torch.diag(torch.linspace(0.1, 2, 24, dtype=torch.float64))
        op = JacobianOperator(lambda x: matrix @ x, torch.zeros(24, dtype=torch.float64))
        result = estimate_eigenvalues(op, seed=1)
        self.assertTrue(result["validated"])
        self.assertAlmostEqual(result["radius"], 2, places=8)

    def test_complex_dominant_eigenvalues(self):
        matrix = torch.diag(torch.linspace(0.1, 0.5, 24, dtype=torch.float64))
        matrix[:2, :2] = torch.tensor([[0.8, -1.7], [1.7, 0.8]])
        op = JacobianOperator(lambda x: matrix @ x, torch.zeros(24, dtype=torch.float64))
        result = estimate_eigenvalues(op, seed=2)
        expected = float(torch.linalg.eigvals(matrix).abs().max())
        self.assertTrue(result["validated"])
        self.assertAlmostEqual(result["radius"], expected, places=8)

    def test_nonnormal_radius_is_not_one_step_norm(self):
        matrix = torch.diag(torch.linspace(0.1, 0.9, 24, dtype=torch.float64))
        matrix[0, 1] = 20
        op = JacobianOperator(lambda x: matrix @ x, torch.zeros(24, dtype=torch.float64))
        result = estimate_eigenvalues(op, seed=3)
        self.assertTrue(result["validated"])
        self.assertAlmostEqual(result["radius"], 0.9, places=7)
        self.assertGreater(float(torch.linalg.matrix_norm(matrix, 2)), 20)

    def test_incomplete_solver_does_not_report_radius(self):
        op = JacobianOperator(lambda x: x, torch.zeros(24, dtype=torch.float64))
        error = ArpackNoConvergence("test", np.asarray([1.0]), np.eye(24)[:, :1])
        with patch("looping.spectral_diagnostics.core.eigs", side_effect=error):
            result = estimate_eigenvalues(op, seed=4)
        self.assertIsNone(result["radius"])
        self.assertFalse(result["validated"])

    def test_bad_residual_is_rejected(self):
        op = JacobianOperator(lambda x: x, torch.zeros(24, dtype=torch.float64))
        with patch("looping.spectral_diagnostics.core.eigs", return_value=(np.arange(2, 6), np.eye(24)[:, :4])):
            result = estimate_eigenvalues(op, seed=5)
        self.assertIsNone(result["radius"])

    def test_forward_ad_matches_explicit_nonlinear_jacobian(self):
        state = torch.linspace(-1, 1, 24, dtype=torch.float64)
        function = lambda x: x + 0.2 * x.sin()
        op = JacobianOperator(function, state)
        direction = unit_directions(state, 1, 6)[0]
        exact = torch.autograd.functional.jacobian(function, state) @ direction
        torch.testing.assert_close(op.apply(direction), exact)
        checks = derivative_checks(function, state, [direction], [1e-3, 1e-4])
        self.assertLess(checks[-1]["central_relative_error"], 1e-8)

    def test_finite_difference_power_linear(self):
        state = torch.ones(24, dtype=torch.float64)
        result = finite_difference_power(lambda x: 2 * x, state, seed=7)
        self.assertAlmostEqual(result["last_gain"], 2, places=8)

    def test_fp32_rejected_for_primary_operator(self):
        with self.assertRaises(ValueError):
            JacobianOperator(lambda x: x, torch.zeros(24))

    def test_tiny_perturbation_rounding_is_detected(self):
        state = torch.full((24,), 1e8, dtype=torch.float32)
        checks = derivative_checks(lambda x: x, state, unit_directions(state, 1, 8), [1e-3])
        self.assertEqual(checks[0]["unchanged_input_fraction"], 1.0)
        self.assertEqual(checks[0]["forward_relative_error"], 1.0)

    def test_model_recurrence_and_jvp(self):
        from model_io import DEFAULT_SETTINGS, build_model
        from looping.spectral_diagnostics.run import recurrent_function
        from iters.exp_baseline_lr2e3 import SudokuTransformer, ROPE_COS, ROPE_SIN
        torch.set_num_threads(2)
        torch.manual_seed(19)
        modern = build_model(DEFAULT_SETTINGS).double().eval().requires_grad_(False)
        legacy = SudokuTransformer().double().eval().requires_grad_(False)
        legacy.load_state_dict(modern.state_dict())
        state = torch.randn(1, 81, 128, dtype=torch.float64)
        function = recurrent_function(modern, torch.float64, "cpu")
        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                old = state + legacy.pred_proj(legacy.output_head(state).softmax(-1))
                for layer in legacy.layers:
                    old = layer(old, ROPE_COS.double(), ROPE_SIN.double())
            torch.testing.assert_close(function(state), old, rtol=0, atol=0)
            direction = unit_directions(state, 1, 10)[0]
            checks = derivative_checks(function, state, [direction], [1e-4, 1e-5])
        self.assertLess(min(row["central_relative_error"] for row in checks), 1e-5)


if __name__ == "__main__":
    unittest.main()
