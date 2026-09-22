"""Matrix-free eigenvalues with independent residual and derivative checks."""

import time

import numpy as np
import torch
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs


class JacobianOperator:
    def __init__(self, function, state):
        if state.dtype != torch.float64:
            raise ValueError("Eigenvalue estimation requires FP64")
        self.function = function
        self.state = state.detach().clone()
        self.calls = 0

    def apply(self, direction):
        self.calls += 1
        result = torch.func.jvp(self.function, (self.state,), (direction,))[1].detach()
        if not torch.isfinite(result).all():
            raise ValueError("Nonfinite automatic derivative")
        return result

    def matvec(self, vector):
        vector = np.asarray(vector)
        if np.iscomplexobj(vector):
            return self.matvec(vector.real) + 1j * self.matvec(vector.imag)
        direction = torch.as_tensor(vector.copy(), device=self.state.device,
                                    dtype=self.state.dtype).reshape_as(self.state)
        return self.apply(direction).flatten().cpu().numpy()

    def scipy_operator(self):
        size = self.state.numel()
        return LinearOperator((size, size), matvec=self.matvec, dtype=np.float64)


def estimate_eigenvalues(operator, *, seed, k=4, ncv=32, maxiter=80,
                         tolerance=1e-8, residual_tolerance=1e-6):
    size = operator.state.numel()
    if size <= k + 1:
        raise ValueError("ARPACK requires k < state dimension - 1")
    start = time.perf_counter()
    initial_calls = operator.calls
    converged = True
    try:
        values, vectors = eigs(operator.scipy_operator(), k=k, which="LM",
                               v0=np.random.default_rng(seed).normal(size=size),
                               ncv=min(size, max(ncv, 2 * k + 1)), maxiter=maxiter,
                               tol=tolerance)
    except ArpackNoConvergence as error:
        converged = False
        values, vectors = error.eigenvalues, error.eigenvectors
    rows = []
    for index in np.argsort(-np.abs(values)):
        value, vector = values[index], vectors[:, index]
        residual = np.linalg.norm(operator.matvec(vector) - value * vector)
        relative = residual / (max(1.0, abs(value)) * np.linalg.norm(vector))
        rows.append({"real": float(value.real), "imag": float(value.imag),
                     "magnitude": float(abs(value)), "relative_residual": float(relative)})
    valid = converged and len(rows) == k and all(
        row["relative_residual"] <= residual_tolerance for row in rows)
    return {"seed": seed, "converged": converged, "validated": valid,
            "radius": rows[0]["magnitude"] if valid else None,
            "largest_returned_magnitude": rows[0]["magnitude"] if rows else None,
            "eigenvalues": rows, "jvp_calls": operator.calls - initial_calls,
            "seconds": time.perf_counter() - start}


def unit_directions(state, count, seed):
    generator = torch.Generator(device=state.device).manual_seed(seed)
    directions = []
    for _ in range(count):
        direction = torch.randn(state.shape, device=state.device, dtype=state.dtype,
                                generator=generator)
        directions.append(direction / direction.norm())
    return directions


def derivative_checks(function, state, directions, epsilons):
    with torch.no_grad():
        base = function(state)
    rows = []
    for direction in directions:
        derivative = torch.func.jvp(function, (state,), (direction,))[1].detach()
        denominator = derivative.norm().clamp_min(torch.finfo(state.dtype).tiny)
        for epsilon in epsilons:
            with torch.no_grad():
                perturbed = state + epsilon * direction
                forward = (function(perturbed) - base) / epsilon
                central = (function(perturbed) - function(state - epsilon * direction)) / (2 * epsilon)
            rows.append({"epsilon": epsilon, "ad_gain": float(derivative.norm()),
                         "forward_gain": float(forward.norm()),
                         "forward_relative_error": float((forward - derivative).norm() / denominator),
                         "central_relative_error": float((central - derivative).norm() / denominator),
                         "realized_input_norm": float((perturbed - state).norm()),
                         "unchanged_input_fraction": float((perturbed == state).double().mean())})
    return rows


def finite_difference_power(function, state, *, seed, steps=100, epsilon=1e-3):
    """Historical absolute-epsilon method, not a validated eigensolver."""
    vector = unit_directions(state, 1, seed)[0]
    history = []
    with torch.no_grad():
        base = function(state)
        for _ in range(steps):
            product = (function(state + epsilon * vector) - base) / epsilon
            gain = product.norm()
            history.append(float(gain))
            if not torch.isfinite(gain) or gain == 0:
                break
            vector = product / gain
    return {"last_gain": history[-1], "history": history}
