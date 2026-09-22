# Jacobian Precision Check

Protocol fixed September 15, 2026. Compare the local recurrent Jacobians of the original successful LR=2e-3 checkpoint, original collapsing LR=3e-3 checkpoint, and released Sotaku 2 weights using FP64 automatic derivatives.

**Complete:** the [results](RESULTS.md) invalidate the historical large-radius estimates. All three runs finished through iteration 4096, with validated radii for 38 of 60 operating states and 22 unresolved. At iteration 16, every point passed validation and the radii were around 1-2, versus roughly 47-95 from the historical method. [Job records](jobs.json) identify the completed workers.

Use the same five randomly selected development-benchmark puzzles, one per difficulty bucket, excluding the first 200 monitoring rows per bucket. Measure the whole-board recurrent Jacobian at iterations 16, 256, 1024, and 4096. Predictions are recomputed from the perturbed state, so their feedback is included in the derivative. Initial feedback is zero during the first iteration, matching ordinary inference.

The primary estimate uses [PyTorch forward-mode automatic differentiation](https://docs.pytorch.org/docs/2.10/generated/torch.func.jvp.html) and [SciPy's ARPACK eigensolver](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html), targeting the four largest-magnitude eigenvalues. Two independently seeded solves must converge, pass explicit eigenvector residual checks, and agree on the radius. Unresolved points remain unresolved rather than being assigned a radius. All encoding and recurrent arithmetic is FP64 with math SDPA, no compilation or autocast; the stored FP32 weights and RoPE constants are converted to FP64.

Checks include known diagonal, complex-eigenvalue, and nonnormal matrices; exact legacy/current recurrence agreement; automatic-derivative linearity; three random directions checked against central finite differences; and activation dtype verification on the GPU. We also record one-step random-direction gains and motion along the actual update direction. Neither is interchangeable with the spectral radius.

At each fixed FP64 state, repeat the historical absolute-epsilon finite-difference power iteration in FP64 and FP32, with TF32 both disabled and enabled for FP32. The FP32 state is rounded from the same FP64 state; derivatives are compared at each precision's own state. These are controlled numerical checks, not exact replays of historical FP32 trajectories or attention kernels. The old finite-difference result is recorded as a gain estimate, not a validated eigenvalue.

The state is moving, not a known fixed point. Even an accurately measured local radius above one is not a proof that the actual changing trajectory must diverge or lose its answer. Five selected puzzles per model cannot establish population-wide reliability or precision invariance of the full benchmark.

## Run

```sh
source venv/bin/activate
python -m unittest looping.spectral_diagnostics.test_spectral -v
modal run --detach looping/spectral_diagnostics/modal_run.py --key old_stable --smoke
# For a new reproduction, launch each checkpoint separately:
modal run --detach looping/spectral_diagnostics/modal_run.py --key old_stable
modal run --detach looping/spectral_diagnostics/modal_run.py --key old_collapsing
modal run --detach looping/spectral_diagnostics/modal_run.py --key v2
# Read existing jobs; these commands do not launch workers:
python -m looping.spectral_diagnostics.collect
# Summarize the downloaded completed runs:
python -m looping.spectral_diagnostics.report
```

Each point is saved atomically and committed before continuing. Source hashes, weight checksums, sample IDs, and settings are verified on resume. Durable outputs are under `jacobian_fp64_v1_20260915/` on `sudoku-outputs`.
