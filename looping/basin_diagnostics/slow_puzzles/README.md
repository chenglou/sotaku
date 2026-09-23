# Slow-Solving Puzzles in FP64

Protocol fixed September 11, 2026, before screening. Following [Anthony Bao's precision suggestion](https://x.com/AwesomeBao/status/2098541433447706808) and [difficulty suggestion](https://x.com/AwesomeBao/status/2098541906401599604), compare recursive maps of puzzles that take the model more or fewer iterations to settle correctly.

**Complete:** both pipelines finished in 2h44m and 2h47m on separate H200s. The slow puzzles show bands and fine variation through 1024x zoom in FP64; both easy controls settle correctly at iteration 2 everywhere. See the [results and figures](RESULTS.md). Both real-checkpoint preflights passed, all 58 source hashes match the recorded protocol, and all 49 local diagnostic tests passed before launch. Downloaded artifact checksums and all grid summaries were checked against the saved arrays. [Job records](jobs.json) identify the completed workers.

## Sampling and Selection

Use the same healthy and late-collapsing width-128 checkpoints as the earlier recursive study. Sample 250 puzzles, 50 from each benchmark difficulty bucket, with a fixed random seed. Exclude the first 200 monitoring rows per bucket and the previous illustration puzzle. Both checkpoints receive the same sample; this remains a development-benchmark analysis.

For each puzzle, run nine initial states through 2048 iterations: the encoded puzzle and eight signed perturbations along two independently seeded pairs of orthogonal directions. Each direction has unit RMS, scaled by 30% of the encoded state's RMS. These perturbations are the previous map's half-width. They are a sparse screen, not a complete map of robustness.

An eligible puzzle must finish correct and unchanged for at least the last 128 iterations at all nine starts, with finite arithmetic throughout. Rank eligible puzzles by their mean last-answer-change iteration. Separately for each model, choose the slowest and fastest eligible puzzle; ties use the original puzzle index. Require two distinct puzzles. Save the selection before any dense maps are rendered. Report wrong, unsettled, and nonfinite starts separately; a wrong stationary answer is not a slow success. Screening all nine starts cannot guarantee that every start in the subsequent dense grid solves.

## Recursive Views

Render both selected puzzles at 129x129 resolution, with six fresh grids at 1x, 4x, 16x, 64x, 256x, and 1024x magnification. The initial half-width is fixed at 0.3 for both models and both cases. The next box follows the largest mean neighboring settling-time difference among boxes with at least 90% successful adjacent pairs, or the center if no such varying box exists. Every trajectory runs all 2048 iterations. Use a 33x33 unzoomed second-plane view as an orientation control, not as a basis for choosing another puzzle.

All initial encoding, perturbation construction, model parameters, buffers, and recurrent arithmetic use FP64. Stored FP32 weights and positional constants are converted to FP64; conversion does not recover information absent from the checkpoint. The plane is generated directly with FP64 QR. This differs from the earlier spot checks, which retained an FP32-encoded base state.

The plots color the last change in the raw 81-digit prediction. Unsettled or nonfinite starts are gray; settled wrong answers are dark red. A second row shows the maximum neighboring decoded-board separation, following the paper's visualization. That distance depends on digit labels and is not a Jacobian or Lyapunov exponent. Local color ranges are labeled. These selected examples can reveal structure, but do not establish how common that structure is or prove a fractal dimension. Correct answers need not imply a hidden-state fixed point.

## Verification and Execution

The GPU preflight runs focused unit tests, checks real-model FP64 activation dtypes, and tests duplicated starts and overlapping grid rows. Workers verify checkpoint and dataset checksums, record source hashes, and reject resumes with changed identities. Each batch or grid chunk is saved atomically and committed to the Volume. Older study sources and results are retained as recorded.

The completed study used the sources in commit `83050da`. A later portability fix allows `rtol=atol=1e-12` for duplicate FP64 hidden states and probabilities on CPU; decoded answers and GPU controls still require exact agreement. Saved measurements are unchanged, and source checks prevent resuming old runs with modified code.

```sh
source venv/bin/activate
MPLCONFIGDIR=/tmp/sotaku-matplotlib python -m unittest looping.basin_diagnostics.slow_puzzles.test_slow_puzzles -v
modal run --detach looping/basin_diagnostics/slow_puzzles/modal_run.py --key 20k_20260907 --smoke
# After the GPU preflight succeeds, launch each pipeline separately:
modal run --detach looping/basin_diagnostics/slow_puzzles/modal_run.py --key 20k_20260907
modal run --detach looping/basin_diagnostics/slow_puzzles/modal_run.py --key 20k_20260908
# Retrieve committed progress and figures:
python -m looping.basin_diagnostics.slow_puzzles.collect
```

Outputs: `basin_slow_fp64_v1_20260911/` on `sudoku-outputs`. Each worker completes screening and then renders its selected slow and easy puzzles without requiring a connected local client.
