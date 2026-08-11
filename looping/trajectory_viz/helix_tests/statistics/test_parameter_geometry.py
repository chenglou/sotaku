import json
import unittest

import torch
import torch.nn as nn

from looping.trajectory_viz.helix_tests.statistics.parameter_geometry import (
    analyze_model_parameter_geometry,
    analyze_parameter_geometries,
    extract_digit_parameter_vectors,
    fit_parameter_geometry,
    fixed_digit_codes,
)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_encoder = nn.Linear(10, 4, bias=False)
        self.pred_proj = nn.Linear(9, 5, bias=False)
        self.output_head = nn.Linear(6, 9, bias=False)


class ParameterGeometryTests(unittest.TestCase):
    def test_extracts_digit_columns_and_output_rows(self):
        model = _ToyModel()
        with torch.no_grad():
            model.initial_encoder.weight.copy_(torch.arange(40).reshape(4, 10))
            model.pred_proj.weight.copy_(torch.arange(45).reshape(5, 9))
            model.output_head.weight.copy_(torch.arange(54).reshape(9, 6))

        vectors = extract_digit_parameter_vectors(model)

        self.assertEqual(
            tuple(vectors),
            (
                "input_encoder_digit_columns",
                "prediction_feedback_columns",
                "output_head_rows",
            ),
        )
        torch.testing.assert_close(
            vectors["input_encoder_digit_columns"],
            model.initial_encoder.weight[:, 1:10].T.double(),
        )
        torch.testing.assert_close(
            vectors["prediction_feedback_columns"],
            model.pred_proj.weight.T.double(),
        )
        torch.testing.assert_close(
            vectors["output_head_rows"], model.output_head.weight.double()
        )
        self.assertTrue(
            all(not value.requires_grad for value in vectors.values())
        )

    def test_fixed_fits_recover_an_orthogonal_ideal_helix(self):
        codes = fixed_digit_codes()
        linear_axis = torch.tensor([0.0, 0.0, 1.5], dtype=torch.float64)
        cosine_axis = torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64)
        sine_axis = torch.tensor([0.0, 2.0, 0.0], dtype=torch.float64)
        vectors = (
            codes["linear"] @ linear_axis.unsqueeze(0)
            + codes["cyclic"]
            @ torch.stack((cosine_axis, sine_axis), dim=0)
        )

        result = fit_parameter_geometry(vectors)

        self.assertAlmostEqual(result["fits"]["helix"]["r2"], 1.0, 12)
        self.assertEqual(result["fits"]["categorical"]["r2"], 1.0)
        self.assertAlmostEqual(
            result["helix_to_categorical_fraction"], 1.0, 12
        )
        geometry = result["axis_geometry"]
        self.assertGreater(geometry["cyclic_singular_value_ratio"], 0.1)
        self.assertLess(abs(geometry["cyclic_axis_cosine"]), 0.4)
        self.assertAlmostEqual(
            geometry["helix_axis_leakage_into_cyclic_plane"], 0.0, 12
        )

    def test_ring_axes_have_equal_singular_values_and_zero_cosine(self):
        cyclic = fixed_digit_codes()["cyclic"]
        vectors = torch.cat((2.5 * cyclic, torch.zeros(9, 2)), dim=1)

        geometry = fit_parameter_geometry(vectors)["axis_geometry"]

        self.assertAlmostEqual(
            geometry["cyclic_singular_value_ratio"], 1.0, 12
        )
        self.assertAlmostEqual(geometry["cyclic_axis_cosine"], 0.0, 12)

    def test_leakage_detects_a_helix_axis_in_the_cyclic_plane(self):
        codes = fixed_digit_codes()
        axes = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
            ],
            dtype=torch.float64,
        )
        vectors = codes["helix"] @ axes

        leakage = fit_parameter_geometry(vectors)["axis_geometry"][
            "helix_axis_leakage_into_cyclic_plane"
        ]

        self.assertAlmostEqual(leakage, 1.0, 12)

    def test_global_permutation_test_is_reproducible_and_json_serializable(
        self,
    ):
        cyclic = fixed_digit_codes()["cyclic"]
        source_vectors = {
            "first": cyclic
            @ torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
            "second": cyclic
            @ torch.tensor([[1.0, 1.0], [-1.0, 1.0]], dtype=torch.float64),
        }

        first = analyze_parameter_geometries(
            source_vectors, permutations=63, seed=31415
        )
        second = analyze_parameter_geometries(
            source_vectors, permutations=63, seed=31415
        )

        self.assertEqual(first, second)
        cyclic_test = first["sources"]["first"]["fits"]["cyclic"][
            "permutation_test"
        ]
        self.assertLessEqual(cyclic_test["p_value"], 0.05)
        self.assertEqual(cyclic_test["count"], 63)
        self.assertEqual(
            first["sources"]["first"]["fits"]["categorical"][
                "permutation_test"
            ]["p_value"],
            1.0,
        )
        json.dumps(first, allow_nan=False)

    def test_model_convenience_function_returns_all_sources(self):
        result = analyze_model_parameter_geometry(
            _ToyModel(), permutations=0, seed=7
        )

        self.assertEqual(
            tuple(result["sources"]),
            (
                "input_encoder_digit_columns",
                "prediction_feedback_columns",
                "output_head_rows",
            ),
        )
        for source in result["sources"].values():
            for fit in source["fits"].values():
                self.assertIsNone(fit["permutation_test"]["p_value"])

    def test_rejects_invalid_class_vectors(self):
        bad_values = (
            torch.zeros(8, 3),
            torch.zeros(9),
            torch.full((9, 2), float("nan")),
        )
        for bad_vectors in bad_values:
            with self.subTest(shape=tuple(bad_vectors.shape)):
                with self.assertRaises(ValueError):
                    fit_parameter_geometry(bad_vectors)


if __name__ == "__main__":
    unittest.main()
