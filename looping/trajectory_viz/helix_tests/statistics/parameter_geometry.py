"""Measure fixed digit-order geometry in the model's learned parameters.

The three parameter families considered here each provide one vector per Sudoku
digit.  Every fit is performed after subtracting the mean of the nine vectors,
so its denominator is exactly the between-digit parameter energy.  The
categorical code spans all centered functions of digit identity and therefore
explains 100% of that energy.
"""

import math
from collections.abc import Mapping

import torch


DIGIT_COUNT = 9
PARAMETER_SOURCES = (
    "input_encoder_digit_columns",
    "prediction_feedback_columns",
    "output_head_rows",
)


def _as_centered_class_vectors(vectors):
    vectors = torch.as_tensor(vectors).detach().to(device="cpu", dtype=torch.float64)
    if vectors.ndim != 2 or vectors.shape[0] != DIGIT_COUNT:
        raise ValueError(
            f"expected a [9, dimension] matrix, received {tuple(vectors.shape)}"
        )
    if vectors.shape[1] == 0:
        raise ValueError("class vectors must have at least one feature")
    if not bool(torch.isfinite(vectors).all()):
        raise ValueError("class vectors must be finite")
    return vectors - vectors.mean(dim=0, keepdim=True)


def fixed_digit_codes():
    """Return the centered, fixed-order codes used by every fit."""

    digit_index = torch.arange(DIGIT_COUNT, dtype=torch.float64)
    angle = 2 * math.pi * digit_index / DIGIT_COUNT
    linear = (digit_index - digit_index.mean()).unsqueeze(1)
    cyclic = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    helix = torch.cat((linear, cyclic), dim=1)
    categorical = torch.eye(DIGIT_COUNT, dtype=torch.float64)
    codes = {
        "linear": linear,
        "cyclic": cyclic,
        "helix": helix,
        "categorical": categorical,
    }
    return {
        name: code - code.mean(dim=0, keepdim=True)
        for name, code in codes.items()
    }


def _projection_matrix(code):
    # pinv also handles the centered categorical code, whose nine columns have
    # rank eight.
    return code @ torch.linalg.pinv(code)


def _clamp_unit(value):
    return min(1.0, max(0.0, float(value)))


def _fit_with_projection(centered_vectors, projection):
    reconstruction = projection @ centered_vectors
    residual_energy = (centered_vectors - reconstruction).square().sum()
    total_energy = centered_vectors.square().sum()
    if total_energy <= torch.finfo(torch.float64).eps:
        return {
            "r2": 0.0,
            "explained_energy": 0.0,
            "residual_energy": 0.0,
        }
    r2 = 1 - residual_energy / total_energy
    return {
        "r2": _clamp_unit(r2.item()),
        "explained_energy": max(
            0.0, float((total_energy - residual_energy).item())
        ),
        "residual_energy": max(0.0, float(residual_energy.item())),
    }


def _axis_cosine(first, second):
    denominator = first.norm() * second.norm()
    if denominator <= torch.finfo(torch.float64).eps:
        return 0.0
    return float(torch.dot(first, second) / denominator)


def _cyclic_singular_value_ratio(cyclic_axes):
    singular_values = torch.linalg.svdvals(cyclic_axes)
    if singular_values.numel() < 2 or singular_values[0] <= torch.finfo(
        torch.float64
    ).eps:
        return 0.0
    return _clamp_unit((singular_values[1] / singular_values[0]).item())


def _axis_leakage_into_plane(axis, plane_axes):
    """Return the fraction of axis energy contained in the plane's row span."""

    axis_energy = axis.square().sum()
    if axis_energy <= torch.finfo(torch.float64).eps:
        return 0.0
    projection = plane_axes.T @ torch.linalg.pinv(plane_axes.T) @ axis
    return _clamp_unit((projection.square().sum() / axis_energy).item())


def fit_parameter_geometry(vectors):
    """Fit the four fixed digit codes to one ``[9, dimension]`` matrix.

    The result contains only JSON-serializable values.  The helix/categorical
    fraction is the energy explained by the three-dimensional helix divided by
    the categorical fit's explained energy.  Because the categorical code is
    saturated, the fraction is equal to the helix R2 whenever the centered
    vectors have nonzero energy.
    """

    centered_vectors = _as_centered_class_vectors(vectors)
    codes = fixed_digit_codes()
    projections = {
        name: _projection_matrix(code) for name, code in codes.items()
    }
    fits = {
        name: {
            **_fit_with_projection(centered_vectors, projections[name]),
            "code_rank": int(torch.linalg.matrix_rank(code).item()),
        }
        for name, code in codes.items()
    }

    total_energy = float(centered_vectors.square().sum().item())
    # State the saturated reference exactly rather than exposing a value such
    # as 0.9999999999999998 from the pseudoinverse.
    fits["categorical"].update(
        r2=1.0,
        explained_energy=total_energy,
        residual_energy=0.0,
    )

    helix_axes = torch.linalg.lstsq(
        codes["helix"], centered_vectors
    ).solution
    # Use the jointly fitted helix coefficients for the axis descriptors.  A
    # separately fitted cyclic model can absorb part of the numeric ramp
    # because a nine-point ramp is not orthogonal to its first Fourier mode.
    cyclic_axes = helix_axes[1:]
    cyclic_axis_cosine = _axis_cosine(cyclic_axes[0], cyclic_axes[1])
    if total_energy == 0:
        helix_fraction = 0.0
    else:
        helix_fraction = fits["helix"]["explained_energy"] / total_energy

    return {
        "class_count": DIGIT_COUNT,
        "vector_dimension": int(centered_vectors.shape[1]),
        "between_class_energy": total_energy,
        "fits": fits,
        "helix_to_categorical_fraction": _clamp_unit(helix_fraction),
        "axis_geometry": {
            "cyclic_singular_value_ratio": _cyclic_singular_value_ratio(
                cyclic_axes
            ),
            "cyclic_axis_cosine": cyclic_axis_cosine,
            "cyclic_axis_absolute_cosine": abs(cyclic_axis_cosine),
            "helix_axis_leakage_into_cyclic_plane": _axis_leakage_into_plane(
                helix_axes[0], helix_axes[1:]
            ),
        },
    }


def _global_digit_permutations(permutations, seed):
    if not isinstance(permutations, int) or isinstance(permutations, bool):
        raise TypeError("permutations must be an integer")
    if permutations < 0:
        raise ValueError("permutations must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return [
        torch.randperm(DIGIT_COUNT, generator=generator)
        for _ in range(permutations)
    ]


def _null_summary(observed_r2, null_r2):
    count = len(null_r2)
    if count == 0:
        return {
            "p_value": None,
            "exceedances": 0,
            "count": 0,
            "null_mean_r2": None,
            "null_standard_deviation_r2": None,
            "null_p95_r2": None,
            "null_max_r2": None,
            "null_r2": [],
        }
    values = torch.tensor(null_r2, dtype=torch.float64)
    tolerance = 1e-12 * max(1.0, abs(observed_r2))
    exceedances = sum(
        score >= observed_r2 - tolerance for score in null_r2
    )
    return {
        # The add-one correction gives a valid Monte Carlo permutation p-value
        # even when no sampled ordering matches the observed score.
        "p_value": float((exceedances + 1) / (count + 1)),
        "exceedances": int(exceedances),
        "count": count,
        "null_mean_r2": float(values.mean().item()),
        "null_standard_deviation_r2": float(
            values.std(unbiased=False).item()
        ),
        "null_p95_r2": float(torch.quantile(values, 0.95).item()),
        "null_max_r2": float(values.max().item()),
        "null_r2": [float(value) for value in null_r2],
    }


def analyze_parameter_geometries(source_vectors, permutations=999, seed=42):
    """Analyze several parameter sources with one shared permutation pool.

    Each null sample applies the same global permutation of the nine digit
    labels to every source.  This preserves the learned vectors and tests only
    whether their association with the fixed numeric digit order is unusually
    strong.  P-values use the upper tail of R2 with the standard add-one Monte
    Carlo correction.
    """

    if not isinstance(source_vectors, Mapping) or not source_vectors:
        raise ValueError("source_vectors must be a non-empty mapping")
    global_permutations = _global_digit_permutations(permutations, seed)
    codes = fixed_digit_codes()
    projections = {
        name: _projection_matrix(code) for name, code in codes.items()
    }
    results = {}
    for source_name, vectors in source_vectors.items():
        if not isinstance(source_name, str):
            raise TypeError("parameter source names must be strings")
        centered_vectors = _as_centered_class_vectors(vectors)
        source_result = fit_parameter_geometry(centered_vectors)
        for fit_name, fit_result in source_result["fits"].items():
            null_r2 = [
                _fit_with_projection(
                    centered_vectors[digit_permutation],
                    projections[fit_name],
                )["r2"]
                for digit_permutation in global_permutations
            ]
            fit_result["permutation_test"] = _null_summary(
                fit_result["r2"], null_r2
            )
        results[source_name] = source_result

    return {
        "config": {
            "digit_count": DIGIT_COUNT,
            "permutations": permutations,
            "seed": int(seed),
            "permutation_scheme": (
                "one shared global permutation of the nine digit labels is "
                "applied to every parameter source"
            ),
            "p_value_correction": "(exceedances + 1) / (permutations + 1)",
        },
        "global_digit_permutations": [
            permutation.tolist() for permutation in global_permutations
        ],
        "sources": results,
    }


def extract_digit_parameter_vectors(model):
    """Extract the nine learned vectors from each requested parameter family."""

    model = getattr(model, "_orig_mod", model)
    try:
        input_weight = model.initial_encoder.weight
        feedback_weight = model.pred_proj.weight
        output_weight = model.output_head.weight
    except AttributeError as error:
        raise ValueError(
            "model must expose initial_encoder, pred_proj, and output_head weights"
        ) from error

    if input_weight.ndim != 2 or input_weight.shape[1] < DIGIT_COUNT + 1:
        raise ValueError("initial_encoder.weight must have at least 10 columns")
    if feedback_weight.ndim != 2 or feedback_weight.shape[1] != DIGIT_COUNT:
        raise ValueError("pred_proj.weight must have exactly 9 columns")
    if output_weight.ndim != 2 or output_weight.shape[0] != DIGIT_COUNT:
        raise ValueError("output_head.weight must have exactly 9 rows")

    def copy_to_cpu(parameter):
        return parameter.detach().to(device="cpu", dtype=torch.float64).clone()

    return {
        # Input index zero represents an empty cell; indices one through nine
        # are the given Sudoku digits.
        "input_encoder_digit_columns": copy_to_cpu(
            input_weight[:, 1 : DIGIT_COUNT + 1].T
        ),
        "prediction_feedback_columns": copy_to_cpu(feedback_weight.T),
        "output_head_rows": copy_to_cpu(output_weight),
    }


def analyze_model_parameter_geometry(model, permutations=999, seed=42):
    """Extract and analyze all three learned digit-vector families."""

    return analyze_parameter_geometries(
        extract_digit_parameter_vectors(model),
        permutations=permutations,
        seed=seed,
    )
