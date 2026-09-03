import unittest

import torch

from looping.trajectory_viz.helix_tests.analyze_helix import (
    centroid_fourier_fraction,
    fit_digit_probes,
    ridge_fit,
    ridge_predict,
)


def load_tests(loader, standard_tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_ridge_recovers_linear_target, test_cyclic_probe_recovers_embedded_digits,
        test_first_harmonic_distinguishes_ring_from_one_hot,
    ))


def test_ridge_recovers_linear_target():
    torch.manual_seed(1)
    features = torch.randn(200, 5)
    targets = features @ torch.randn(5, 2) + 0.3
    prediction = ridge_predict(features, ridge_fit(features, targets))
    assert torch.mean((prediction - targets.double()).square()) < 1e-7


def test_cyclic_probe_recovers_embedded_digits():
    digits = torch.arange(9).repeat(40)
    angle = 2 * torch.pi * digits / 9
    features = torch.stack([torch.cos(angle), torch.sin(angle)], 1)
    score, _ = fit_digit_probes(features[:180], digits[:180], features[180:], digits[180:])
    assert score["cyclic_r2"] > 0.999
    assert score["cyclic_nearest_digit_accuracy"] == 1


def test_first_harmonic_distinguishes_ring_from_one_hot():
    digits = torch.arange(9).repeat_interleave(20)
    angle = 2 * torch.pi * digits / 9
    ring = torch.stack([torch.cos(angle), torch.sin(angle)], 1)
    assert centroid_fourier_fraction(ring, digits) > 0.999
    assert centroid_fourier_fraction(torch.nn.functional.one_hot(digits, 9), digits) < 0.3
