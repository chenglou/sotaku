import math
import unittest

import numpy as np

from looping.trajectory_viz.helix_tests.statistics.statistics_core import (
    bootstrap_geometry_intervals,
    digit_code_matrix,
    evaluate_digit_geometry,
    fit_weighted_multivariate_ols,
    global_digit_order_permutation_test,
    make_stratified_puzzle_folds,
)


def _synthetic_helix(seed=7, puzzle_count=30, noise=0.03):
    generator = np.random.default_rng(seed)
    digits = np.tile(np.arange(9), puzzle_count)
    puzzle_ids = np.repeat(np.arange(puzzle_count), 9)
    strata = np.repeat(np.arange(puzzle_count) % 3, 9)
    helix = digit_code_matrix(digits, "helix")
    # Orthogonal axes make the coefficient descriptors interpretable.
    axes = np.zeros((3, 7))
    axes[0, 0] = 1.4
    axes[1, 1] = 2.0
    axes[2, 2] = 2.0
    puzzle_nuisance = generator.normal(size=(puzzle_count, 1))
    base = puzzle_nuisance[puzzle_ids]
    nuisance_axis = np.array([[0.0, 0.0, 0.0, 1.2, 0.0, 0.0, 0.0]])
    # Accelerate-backed NumPy can report stale LAPACK floating-point flags on
    # a later, perfectly finite matmul.  The explicit finite-valued assertions
    # below exercise the result rather than those stale flags.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        response = helix @ axes + base @ nuisance_axis
    response += generator.normal(scale=noise, size=response.shape)
    return response, digits, puzzle_ids, strata, base


def test_digit_codes_have_prespecified_geometry():
    digits_zero_based = np.arange(9)
    digits_one_based = np.arange(1, 10)
    linear = digit_code_matrix(digits_zero_based, "linear")
    cyclic = digit_code_matrix(digits_zero_based, "cyclic")
    assert np.allclose(linear, digit_code_matrix(digits_one_based, "linear", digit_offset=1))
    assert abs(linear.mean()) < 1e-15
    assert np.allclose(np.mean(linear**2), 1.0)
    assert cyclic.shape == (9, 2)
    assert np.allclose(np.linalg.norm(cyclic, axis=1), 1.0)
    assert np.allclose(digit_code_matrix(digits_zero_based, "helix"),
                       np.column_stack((linear, cyclic)))
    categorical = digit_code_matrix(digits_zero_based, "categorical")
    assert categorical.shape == (9, 8)
    assert np.allclose(categorical[-1], 0.0)


def test_weighted_multivariate_ols_matches_integer_replication():
    design = np.array([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0]])
    response = np.column_stack((design @ np.array([0.2, 1.7]),
                                design @ np.array([-0.4, 0.3])))
    response[1] += np.array([0.1, -0.2])
    weights = np.array([1, 3, 2])
    weighted = fit_weighted_multivariate_ols(design, response, weights)
    repeated_rows = np.repeat(np.arange(3), weights)
    repeated = fit_weighted_multivariate_ols(
        design[repeated_rows], response[repeated_rows]
    )
    assert np.allclose(weighted.coefficients, repeated.coefficients)


def test_stratified_folds_keep_puzzles_whole_and_balance_each_stratum():
    puzzle_ids = np.repeat(np.arange(23), np.arange(23) % 4 + 1)
    puzzle_strata = np.repeat(np.arange(23) % 3, np.arange(23) % 4 + 1)
    folds = make_stratified_puzzle_folds(
        puzzle_ids, puzzle_strata, n_folds=5, seed=4
    )
    for puzzle in np.unique(puzzle_ids):
        assert len(np.unique(folds[puzzle_ids == puzzle])) == 1
    for stratum in np.unique(puzzle_strata):
        puzzle_folds = [
            folds[np.flatnonzero(puzzle_ids == puzzle)[0]]
            for puzzle in np.unique(puzzle_ids[puzzle_strata == stratum])
        ]
        counts = np.bincount(puzzle_folds, minlength=5)
        assert counts.max() - counts.min() <= 1


def test_cross_validation_recovers_helix_and_reports_cluster_residuals():
    response, digits, puzzle_ids, strata, base = _synthetic_helix()
    weights = np.repeat(np.linspace(0.5, 1.5, 30), 9)
    result = evaluate_digit_geometry(
        response,
        digits,
        puzzle_ids,
        base_design=base,
        sample_weight=weights,
        strata=strata,
        n_folds=5,
        seed=11,
        bootstrap_replicates=100,
    )
    partial = result["partial_r2"]
    assert partial["helix_over_base"] > 0.998
    assert partial["helix_over_linear"] > 0.99
    assert partial["helix_over_cyclic"] > 0.9
    assert result["helix_categorical_gain_fraction"] > 0.995
    assert len(result["per_puzzle"]) == 30
    for model in result["models"]:
        clustered = sum(record["sse"][model] for record in result["per_puzzle"])
        assert np.isclose(clustered, result["models"][model]["sse"])

    descriptors = result["coefficient_geometry"]
    for fold in descriptors["fold_descriptors"]:
        assert fold["cyclic_singular_value_ratio"] > 0.97
        assert fold["cyclic_axis_absolute_cosine"] < 0.03
        assert fold["linear_axis_leakage_into_cyclic_span"] < 0.03
    assert descriptors["cyclic_subspace_stability"]["mean"] > 0.995
    assert descriptors["helix_subspace_stability"]["mean"] > 0.995
    interval = result["bootstrap"]["intervals"]["partial_r2.helix_over_base"]
    assert interval["lower"] <= interval["median"] <= interval["upper"]
    assert interval["finite_replicates"] == 100


def test_nested_metrics_can_be_negative_on_held_out_puzzles():
    generator = np.random.default_rng(19)
    puzzle_count = 12
    digits = np.tile(np.arange(9), puzzle_count)
    puzzle_ids = np.repeat(np.arange(puzzle_count), 9)
    # Each puzzle gets an unrelated random digit lookup.  A categorical model
    # can overfit training puzzle averages and lose to the held-out intercept.
    response = np.concatenate([
        generator.normal(size=(9, 1)) for _ in range(puzzle_count)
    ])
    result = evaluate_digit_geometry(
        response, digits, puzzle_ids, n_folds=6, seed=3
    )
    assert any(value < 0 for value in result["partial_r2"].values())


def test_bootstrap_resamples_puzzles_within_strata_reproducibly():
    records = []
    for puzzle in range(8):
        records.append({
            "puzzle_id": puzzle,
            "stratum": puzzle % 2,
            "sse": {
                "null": 12.0 + puzzle,
                "base": 10.0 + puzzle,
                "helix": 5.0 + 0.5 * puzzle,
                "categorical": 4.0 + 0.5 * puzzle,
            },
        })
    first = bootstrap_geometry_intervals(records, replicates=40, seed=2)
    second = bootstrap_geometry_intervals(records, replicates=40, seed=2)
    assert first == second
    assert first["intervals"]["partial_r2.helix_over_base"]["finite_replicates"] == 40


def test_global_permutation_uses_plus_one_p_value_and_detects_order():
    response, digits, puzzle_ids, strata, base = _synthetic_helix(
        puzzle_count=24, noise=0.01
    )
    test = global_digit_order_permutation_test(
        response,
        digits,
        puzzle_ids,
        base_design=base,
        strata=strata,
        n_folds=4,
        permutations=31,
        statistic="helix_partial_r2",
        seed=23,
    )
    assert test["p_value"] == (1 + test["exceedances"]) / 32
    assert test["p_value"] <= 2 / 32
    assert len(test["null_values"]) == 31
    assert all(sorted(mapping) == list(range(9)) for mapping in test["label_permutations"])


def test_fwl_permutation_scores_match_brute_force_cross_validation():
    response, digits, puzzle_ids, strata, base = _synthetic_helix(
        puzzle_count=16, noise=0.05
    )
    weights = np.repeat(np.linspace(0.7, 1.3, 16), 9)
    folds = make_stratified_puzzle_folds(
        puzzle_ids, strata, n_folds=4, seed=13
    )
    statistic_names = (
        "linear_partial_r2",
        "cyclic_partial_r2",
        "helix_partial_r2",
        "cyclic_given_linear_partial_r2",
        "linear_given_cyclic_partial_r2",
        "helix_categorical_gain_fraction",
    )
    fast = global_digit_order_permutation_test(
        response,
        digits,
        puzzle_ids,
        base_design=base,
        sample_weight=weights,
        strata=strata,
        fold_ids=folds,
        permutations=5,
        statistic=statistic_names,
        seed=31,
    )

    def brute_statistics(labels):
        result = evaluate_digit_geometry(
            response,
            labels,
            puzzle_ids,
            base_design=base,
            sample_weight=weights,
            strata=strata,
            fold_ids=folds,
            models=("linear", "cyclic", "helix", "categorical"),
        )
        return {
            "linear_partial_r2": result["partial_r2"]["linear_over_base"],
            "cyclic_partial_r2": result["partial_r2"]["cyclic_over_base"],
            "helix_partial_r2": result["partial_r2"]["helix_over_base"],
            "cyclic_given_linear_partial_r2": result["partial_r2"]
            ["helix_over_linear"],
            "linear_given_cyclic_partial_r2": result["partial_r2"]
            ["helix_over_cyclic"],
            "helix_categorical_gain_fraction": result[
                "helix_categorical_gain_fraction"
            ],
        }

    observed = brute_statistics(digits)
    for name in statistic_names:
        assert np.isclose(fast["statistics"][name]["observed"], observed[name],
                          rtol=1e-11, atol=1e-12)
    for permutation_index, mapping in enumerate(fast["label_permutations"]):
        brute = brute_statistics(np.asarray(mapping)[digits])
        for name in statistic_names:
            fast_value = fast["statistics"][name]["null_values"][permutation_index]
            assert np.isclose(fast_value, brute[name], rtol=1e-11, atol=1e-12)


def test_fwl_matches_direct_cv_with_nearly_redundant_nuisance_columns():
    response, digits, puzzle_ids, strata, base = _synthetic_helix(
        puzzle_count=20, noise=0.04
    )
    generator = np.random.default_rng(83)
    nearly_duplicate = base + generator.normal(scale=1e-11, size=base.shape)
    nuisance = np.column_stack((base, nearly_duplicate, 2.0 * base))
    folds = make_stratified_puzzle_folds(
        puzzle_ids, strata, n_folds=5, seed=17
    )
    statistic_names = (
        "helix_partial_r2",
        "cyclic_given_linear_partial_r2",
        "linear_given_cyclic_partial_r2",
    )
    fast = global_digit_order_permutation_test(
        response,
        digits,
        puzzle_ids,
        base_design=nuisance,
        strata=strata,
        fold_ids=folds,
        permutations=3,
        statistic=statistic_names,
        seed=91,
    )

    def brute(labels):
        result = evaluate_digit_geometry(
            response,
            labels,
            puzzle_ids,
            base_design=nuisance,
            strata=strata,
            fold_ids=folds,
        )
        return {
            "helix_partial_r2": result["partial_r2"]["helix_over_base"],
            "cyclic_given_linear_partial_r2": result["partial_r2"]
            ["helix_over_linear"],
            "linear_given_cyclic_partial_r2": result["partial_r2"]
            ["helix_over_cyclic"],
        }

    expected = brute(digits)
    for name in statistic_names:
        assert np.isclose(
            fast["statistics"][name]["observed"], expected[name],
            rtol=1e-9, atol=1e-10,
        )
    for permutation_index, mapping in enumerate(fast["label_permutations"]):
        expected = brute(np.asarray(mapping)[digits])
        for name in statistic_names:
            assert np.isclose(
                fast["statistics"][name]["null_values"][permutation_index],
                expected[name], rtol=1e-9, atol=1e-10,
            )


def test_invalid_within_puzzle_fold_assignment_is_rejected():
    response, digits, puzzle_ids, _, _ = _synthetic_helix(puzzle_count=4)
    folds = np.repeat([0, 1, 0, 1], 9)
    folds[1] = 1 - folds[1]
    try:
        evaluate_digit_geometry(response, digits, puzzle_ids, fold_ids=folds)
    except ValueError as error:
        assert "constant within puzzle" in str(error)
    else:
        raise AssertionError("expected a within-puzzle fold assignment error")


def test_gain_fraction_is_nan_when_categorical_has_no_positive_gain():
    records = [
        {"stratum": 0, "sse": {"null": 2.0, "base": 1.0,
                                 "helix": 0.9, "categorical": 1.1}},
        {"stratum": 0, "sse": {"null": 2.0, "base": 1.0,
                                 "helix": 0.9, "categorical": 1.1}},
    ]
    result = bootstrap_geometry_intervals(records, replicates=5, seed=0)
    interval = result["intervals"]["helix_categorical_gain_fraction"]
    assert math.isnan(interval["lower"])
    assert interval["finite_replicates"] == 0


class StatisticsCoreTests(unittest.TestCase):
    """Standard-library runner for environments where pytest is unavailable."""

    def test_digit_codes_have_prespecified_geometry(self):
        test_digit_codes_have_prespecified_geometry()

    def test_weighted_multivariate_ols_matches_integer_replication(self):
        test_weighted_multivariate_ols_matches_integer_replication()

    def test_stratified_folds_keep_puzzles_whole_and_balance_each_stratum(self):
        test_stratified_folds_keep_puzzles_whole_and_balance_each_stratum()

    def test_cross_validation_recovers_helix_and_reports_cluster_residuals(self):
        test_cross_validation_recovers_helix_and_reports_cluster_residuals()

    def test_nested_metrics_can_be_negative_on_held_out_puzzles(self):
        test_nested_metrics_can_be_negative_on_held_out_puzzles()

    def test_bootstrap_resamples_puzzles_within_strata_reproducibly(self):
        test_bootstrap_resamples_puzzles_within_strata_reproducibly()

    def test_global_permutation_uses_plus_one_p_value_and_detects_order(self):
        test_global_permutation_uses_plus_one_p_value_and_detects_order()

    def test_fwl_permutation_scores_match_brute_force_cross_validation(self):
        test_fwl_permutation_scores_match_brute_force_cross_validation()

    def test_fwl_matches_direct_cv_with_nearly_redundant_nuisance_columns(self):
        test_fwl_matches_direct_cv_with_nearly_redundant_nuisance_columns()

    def test_invalid_within_puzzle_fold_assignment_is_rejected(self):
        test_invalid_within_puzzle_fold_assignment_is_rejected()

    def test_gain_fraction_is_nan_when_categorical_has_no_positive_gain(self):
        test_gain_fraction_is_nan_when_categorical_has_no_positive_gain()


if __name__ == "__main__":
    unittest.main()
